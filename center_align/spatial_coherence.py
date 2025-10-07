import networkx as nx
from scipy.spatial.distance import cdist
import random
import numpy as np
import math
import pandas as pd

def create_graph(adata, degree=4):
    """
    Converts spatial coordinates into graph using networkx library.

    param: adata - ST Slice
    param: degree - number of edges per vertex

    return: 1) G - networkx graph
            2) node_dict - dictionary mapping nodes to spots
    """
    D = cdist(adata.obsm['spatial'], adata.obsm['spatial'])
    # Get column indexes of the degree+1 lowest values per row
    idx = np.argsort(D, 1)[:, 0:degree + 1]
    # Remove first column since it results in self loops
    idx = idx[:, 1:]

    G = nx.Graph()
    for r in range(len(idx)):
        for c in idx[r]:
            G.add_edge(r, c)

    node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
    return G, node_dict


def generate_graph_from_labels(adata, labels_dict):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """

    g, node_to_spot = create_graph(adata)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)

    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels


def spatial_coherence_score(graph, labels):
    g, l = graph, labels
    true_entropy = spatial_entropy(g, l)
    entropies = []
    for i in range(1000):
        new_l = list(l.values())
        random.shuffle(new_l)
        labels = dict(zip(l.keys(), new_l))
        entropies.append(spatial_entropy(g, labels))

    return (true_entropy - np.mean(entropies)) / np.std(entropies)


def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0, index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z / C_sum) * math.log(z / C_sum)
    return H





import torch
import networkx as nx

def create_graph_gpu(adata, degree=4, device=None, batch_q=8192, symmetrize=True):
    """
    Build a kNN graph from adata.obsm['spatial'] using GPU-accelerated kNN.

    Parameters
    ----------
    adata : AnnData with .obsm['spatial'] (n x 2 or n x d)
    degree : int
        Number of neighbors per node (k).
    device : str or None
        'cuda' to force GPU if available, else auto-detect.
    batch_q : int
        Query batch size (tune to your GPU memory).
    symmetrize : bool
        If True, make the graph undirected by adding reverse edges and deduplicating.

    Returns
    -------
    G : networkx.Graph
    node_dict : dict[int -> str]
        Node index -> spot name
    """
    coords_np = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    n = coords_np.shape[0]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    coords = torch.tensor(coords_np, device=device)

    # Batched GPU kNN (via torch.cdist + topk). We drop self (distance 0).
    k = int(degree)
    idx_all = torch.empty((n, k), dtype=torch.int64, device='cpu')  # store on CPU to save GPU RAM

    for start in range(0, n, batch_q):
        end = min(n, start + batch_q)
        qs = coords[start:end]                   # (b, d)
        D = torch.cdist(qs, coords)              # (b, n) on GPU
        nn = torch.topk(D, k=k+1, largest=False) # include self as the nearest
        idx = nn.indices[:, 1:k+1]               # drop self (first column)
        idx_all[start:end] = idx.to('cpu')

    # Build undirected edge list (u,v) without self-loops
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idx_all.reshape(-1).numpy()

    edges = np.vstack([src, dst]).T
    if symmetrize:
        # ensure undirected: add reverse edges then keep unique {min,max}
        edges = np.vstack([edges, edges[:, ::-1]])
    # deduplicate edges by sorting endpoints and uniques
    edges_sorted = np.sort(edges, axis=1)
    edges_unique = np.unique(edges_sorted, axis=0)

    # Construct NetworkX graph on CPU
    G = nx.Graph()
    G.add_edges_from(map(tuple, edges_unique))

    node_dict = dict(zip(range(adata.n_obs), adata.obs.index))
    return G, node_dict


def generate_graph_from_labels_gpu(adata, labels_dict, degree=4, **kwargs):
    """
    GPU kNN graph + keep only nodes present in labels_dict.
    Returns:
      g : networkx.Graph (induced subgraph)
      labels : dict[node -> cluster_label]
    """
    g, node_to_spot = create_graph_gpu(adata, degree=degree, **kwargs)

    # Keep only nodes whose spot id is in labels_dict
    valid_nodes = [n for n in g.nodes if node_to_spot[n] in labels_dict]
    g = g.subgraph(valid_nodes).copy()

    labels = {n: labels_dict[node_to_spot[n]] for n in g.nodes()}
    return g, labels

def _reindex_graph_edges(g):
    """Return contiguous node index array and edges as (E,2) int64."""
    nodes = np.array(list(g.nodes()), dtype=np.int64)
    pos = {n:i for i,n in enumerate(nodes)}
    edges = np.array([(pos[u], pos[v]) for u, v in g.edges()], dtype=np.int64)
    return nodes, edges

def _labels_array(labels, nodes):
    """Map node->label dict into int array aligned with 'nodes'."""
    names = sorted(set(labels.values()))
    name2i = {name:i for i,name in enumerate(names)}
    lab = np.fromiter((name2i[labels[n]] for n in nodes), count=len(nodes), dtype=np.int64)
    return lab, names

def _spatial_entropy_edges(edges, lab_idx, K):
    """Entropy from undirected edge label pairs (vectorized, no pandas)."""
    if edges.size == 0: return 0.0
    li = lab_idx[edges[:, 0]]
    lj = lab_idx[edges[:, 1]]
    i = np.minimum(li, lj)     # upper triangle (i<=j)
    j = np.maximum(li, lj)
    idx = i * K + j            # flatten to KxK bins
    counts = np.bincount(idx, minlength=K*K).astype(np.float64).reshape(K, K)
    tri = np.triu_indices(K)
    z = counts[tri]
    z = z[z > 0]
    p = z / z.sum()
    return -(p * np.log(p)).sum()

def spatial_coherence_score_fast(g, labels, n_perm=1000, seed=0):
    nodes, edges = _reindex_graph_edges(g)
    lab_idx, names = _labels_array(labels, nodes)
    K = len(names)

    true_H = _spatial_entropy_edges(edges, lab_idx, K)

    rng = np.random.default_rng(seed)
    ent = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        lab_perm = lab_idx[rng.permutation(lab_idx.size)]  # shuffle labels across nodes
        ent[t] = _spatial_entropy_edges(edges, lab_perm, K)

    mu = ent.mean()
    sd = ent.std(ddof=0)
    return (true_H - mu) / (sd if sd > 0 else 1.0)