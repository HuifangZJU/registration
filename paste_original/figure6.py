import math
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import scanpy as sc
import sklearn
import networkx as nx
import ot
import paste as pst
from paste.helper import to_dense_array
import anndata
from sklearn.cluster import KMeans

# style.use('seaborn-dark')
style.use('seaborn-white')

import numpy as np
import ot
from anndata import AnnData
from typing import Optional, Tuple
from sklearn.decomposition import NMF


##################################
# Utility
##################################
def intersect(a, b):
    # Return sorted intersection of two index arrays
    return a.intersection(b).sort_values()


def to_dense_array(X):
    # If X is a sparse array, convert to dense, else pass through
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def center_ot_two_slices(
    W, H,
    slice_B: AnnData,
    center_coords: np.ndarray,
    common_genes,
    alpha: float,
    backend,
    use_gpu: bool,
    dissimilarity='kl',
    norm=False,
    G_init=None,
    distribution_B=None,
    verbose=False
):
    """
    Solves one iteration of the OT step for exactly 2 slices:
      - 'center' (given by W,H)
      - 'slice_B'
    Possibly with different numbers of spots (nA vs. nB).

    Args:
        W, H: factorization of the center slice, shape (nA, k) and (k, nGenes)
        slice_B: The other slice, shape (nB, nGenes)
        center_coords: (nA, 2) coordinate array for the center slice
        alpha: weighting of spatial vs. expression distance
        G_init: optional initial OT plan
        distribution_B: optional distribution for B's spots
    Returns:
        G (np.ndarray): shape (nA, nB)
        cost_val (float): total cost for this OT
    """

    coords_B = slice_B.obsm['spatial']  # shape (nB, 2)
    expr_B   = to_dense_array(slice_B.X)  # shape (nB, nGenes)

    # Current center expression (nA x nGenes)
    center_expr = np.dot(W, H)

    # --- 1) Compute cost matrix (nA x nB) ---
    # Typically you combine expression & spatial distances, e.g.:
    #   cost_ij = alpha * dist_spatial(i, j) + (1 - alpha)*dist_expr(i, j)
    # This code is just a placeholder with zeros for demonstration.
    # Fill in your actual cost computation logic.

    nA = center_coords.shape[0]
    nB = coords_B.shape[0]
    cost_matrix = np.zeros((nA, nB))

    # --- 2) Marginals for rows & columns ---
    row_dist = np.ones(nA) / nA
    if distribution_B is None:
        col_dist = np.ones(nB) / nB
    else:
        col_dist = distribution_B

    # --- 3) Solve OT problem ---
    # e.g. entropic or sinkhorn or plain emd:
    G = ot.emd(row_dist, col_dist, cost_matrix) if G_init is None else G_init
    # If you want a real solve, you'd do something like:
    #    G = ot.emd(row_dist, col_dist, cost_matrix)
    # or:
    #    G = ot.sinkhorn(row_dist, col_dist, cost_matrix, reg=some_value)

    # --- 4) Compute cost ---
    cost_val = np.sum(cost_matrix * G)

    return G, cost_val

#################################
# center_NMF_two_slices
#################################
def center_NMF_two_slices(
    W, H,
    slice_A: AnnData,
    slice_B: AnnData,
    G: np.ndarray,
    weight_B: float,
    n_components: int,
    random_seed: Optional[int],
    dissimilarity: str = 'kl',
    verbose: bool = False
):
    """
    Recompute the center (W_new, H_new) by fusing:
      - slice_A's expression
      - slice_B's expression mapped via G
    Weighted by (1-weight_B) vs. weight_B.

    Args:
        W, H: Current factorization of center (nA x k, k x nGenes)
        slice_A: The reference (center) slice (nA x nGenes)
        slice_B: The other slice (nB x nGenes)
        G: The OT plan from A->B (nA x nB)
        weight_B: fraction contributed by B in the new center
        n_components: for NMF
        random_seed: for reproducibility
        dissimilarity: 'kl' or 'euclidean'
    Returns:
        W_new, H_new
    """

    expr_A = to_dense_array(slice_A.X)  # shape (nA, nGenes)
    expr_B = to_dense_array(slice_B.X)  # shape (nB, nGenes)
    nA     = slice_A.shape[0]

    # Weighted sum approach
    #   combined_expr = (1-weight_B)*A + weight_B*(G@B)
    combined_expr = (1 - weight_B)*expr_A + weight_B*(G @ expr_B)

    # Multiply by nA if you want to match the original code's scaling
    combined_expr *= nA

    # Fit NMF
    if dissimilarity.lower() in ['euclidean', 'euc']:
        model = NMF(n_components=n_components, init='random',
                    random_state=random_seed, verbose=verbose)
    else:
        model = NMF(n_components=n_components, solver='mu',
                    beta_loss='kullback-leibler', init='random',
                    random_state=random_seed, verbose=verbose)

    W_new = model.fit_transform(combined_expr)
    H_new = model.components_

    return W_new, H_new

#################################
# Main function: center_align_two_slices
#################################
def center_align_two_slices(
    A: AnnData,
    B: AnnData,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    dissimilarity: str = 'kl',
    norm: bool = False,
    random_seed: Optional[int] = None,
    G_init: Optional[np.ndarray] = None,
    distribution_B=None,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True,
    weight_B: float = 0.5  # how much B contributes vs. A in each iteration
) -> Tuple[AnnData, np.ndarray]:
    """
    Center alignment for exactly 2 slices (A, B) with possibly different spot counts.
    A is the "center," but we incorporate both A and B in each iteration's expression.

    Args:
        A, B: Two slices (AnnData). Must share some common genes.
        alpha: weight for spatial vs. expression cost in OT
        n_components: NMF components
        threshold: convergence threshold for the objective difference
        max_iter: max # of OT+NMF iterations
        dissimilarity: 'kl' or 'euclidean'
        norm: whether to normalize spatial distances
        random_seed: for reproducible NMF
        G_init: optional initial (nA x nB) OT plan
        distribution_B: optional column marginals for B
        backend: e.g. ot.backend.NumpyBackend()
        use_gpu: whether to attempt GPU (Torch)
        verbose: extra logging
        gpu_verbose: GPU usage messages
        weight_B: fraction contributed by B in each iteration (0.5 -> A=0.5, B=0.5)

    Returns:
        center_slice (AnnData):
            A copy of A with .X = W@H after final iteration,
            plus .uns['paste_W', 'paste_H', 'obj', 'full_rank'].
        G (np.ndarray):
            final OT plan (nA x nB).
    """

    # 1) Possibly handle GPU
    if use_gpu:
        try:
            import torch
        except ImportError:
            print("GPU mode requires PyTorch, but it's not installed.")
        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, fallback to cpu.")
                use_gpu = False
        else:
            print("We only have GPU support with TorchBackend, revert to CPU.")
            use_gpu = False
    else:
        if gpu_verbose:
            print("Using CPU. Set use_gpu=True if a GPU backend is available.")

    # 2) Intersect genes
    common_genes = intersect(A.var.index, B.var.index)
    A = A[:, common_genes]
    B = B[:, common_genes]
    print(f"Common genes: {len(common_genes)}")

    # 3) Initialize W,H from A
    expr_A = to_dense_array(A.X)
    if dissimilarity.lower() in ['euclidean', 'euc']:
        model = NMF(n_components=n_components, init='random',
                    random_state=random_seed, verbose=verbose)
    else:
        model = NMF(n_components=n_components, solver='mu',
                    beta_loss='kullback-leibler',
                    init='random', random_state=random_seed,
                    verbose=verbose)

    if G_init is None:
        # Just run NMF on A to get initial W,H
        W = model.fit_transform(expr_A)
    else:
        # Weighted sum approach if you want initial incorporation of B
        nA = A.shape[0]
        expr_B = to_dense_array(B.X)
        combined_expr = (1 - weight_B) * expr_A + weight_B * (G_init @ expr_B)
        combined_expr *= nA
        W = model.fit_transform(combined_expr)

    H = model.components_

    # 4) Setup iteration
    center_coords = A.obsm['spatial']  # shape (nA, 2)
    iteration_count = 0
    R = 0.0
    R_diff = float('inf')
    nA = A.shape[0]
    nB = B.shape[0]

    # 5) If no G_init, create an empty array as placeholder
    G = G_init if G_init is not None else np.zeros((nA, nB))

    # 6) Iterative OT + NMF
    while (R_diff > threshold) and (iteration_count < max_iter):
        print(f"Iteration {iteration_count}")

        # 6a) OT step
        G, cost_val = center_ot_two_slices(
            W, H, B,
            center_coords, common_genes,
            alpha, backend, use_gpu,
            dissimilarity=dissimilarity, norm=norm,
            G_init=G, distribution_B=distribution_B,
            verbose=verbose
        )

        # 6b) NMF step (now re-incorporates A each iteration)
        W_new, H_new = center_NMF_two_slices(
            W, H, A, B,
            G, weight_B,
            n_components, random_seed,
            dissimilarity=dissimilarity, verbose=verbose
        )

        # Track objective
        R_new = cost_val
        R_diff = abs(R_new - R)
        print(f"Objective: {R_new}")
        print(f"Diff:      {R_diff}\n")

        W, H = W_new, H_new
        R = R_new
        iteration_count += 1

    # 7) Build final center slice
    center_slice = A.copy()  # same shape as A
    final_expr = np.dot(W, H)  # shape (nA, nGenes)
    center_slice.X = final_expr
    center_slice.uns['paste_W'] = W
    center_slice.uns['paste_H'] = H
    center_slice.uns['obj'] = R

    # "full_rank" to mimic original code, now includes both A and B
    expr_B = to_dense_array(B.X)
    center_slice.uns['full_rank'] = nA * (
        (1 - weight_B)*to_dense_array(A.X) + weight_B*(G @ expr_B)
    )

    return center_slice, G


def load_data_from_folder(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)


    return data["img1"], data["img2"], data["pts1"], data["pts2"], data["label1"].reshape(-1), data["label2"].reshape(-1)


def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    #     match = minimum_weight_full_matching(B)
    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def cluster_adata(adata, n_clusters=7, sample_name='', use_nmf=False):
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor="seurat", n_top_genes=2000)
    sc.pp.pca(adata_copy)

    if use_nmf:
        model = sklearn.decomposition.NMF(n_components=50)
        adata_copy.obsm['X_pca'] = model.fit_transform(adata_copy.X)

    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit_predict(adata_copy.obsm['X_pca'])

    adata_copy.obs['my_clusters'] = pd.Series(
        1 + match_cluster_labels(adata_copy.obs['layer_guess_reordered'], cluster_labels), index=adata_copy.obs.index,
        dtype='category')

    ari = sklearn.metrics.adjusted_rand_score(adata_copy.obs['layer_guess_reordered'], adata_copy.obs['my_clusters'])
    print('ARI', ari)
    adata.obs['my_clusters'] = adata_copy.obs['my_clusters'].copy()
    return



def draw_spatial(adata, clusters='my_clusters', sample_name='', draw_contours=False):
    fig = plt.figure(figsize=(12, 10))
    ax = sc.pl.spatial(adata, color=clusters, spot_size=20, show=False,
                       palette=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'],
                       ax=plt.gca())
    ax[0].axis('off')
    ari = sklearn.metrics.adjusted_rand_score(adata.obs['layer_guess_reordered'], adata.obs[clusters])
    ax[0].legend(title='Cluster', bbox_to_anchor=(0.9, 0.9), fontsize=20, title_fontsize=20)
    ax[0].set_title('{}: ARI={:.4f}'.format(sample_name, ari), fontsize=26)
    if draw_contours:
        for l in ['Layer{}'.format(i) for i in [1, 3, 5]] + ['WM']:
            contours = get_scatter_contours(adata, adata.obs['layer_guess_reordered'], [l])
            for k in range(len(contours)):
                plt.plot(adata.obsm['spatial'][contours[k], 0], adata.obsm['spatial'][contours[k], 1], 'lime',
                         # dict(zip(['Layer{}'.format(i) for i in range(1,7)]+['WM'],adata.uns['layer_guess_reordered_colors']))[l],
                         lw=4, alpha=0.6)
    plt.gca().text(105, 150, 'L1')
    plt.gca().text(105, 220, 'L2')
    plt.gca().text(105, 260, 'L3')
    plt.gca().text(105, 305, 'L4')
    plt.gca().text(105, 340, 'L5')
    plt.gca().text(105, 380, 'L6')
    plt.gca().text(105, 425, 'WM')
    plt.show()





def get_scatter_contours(layer, labels, interest=['WM']):
    idx = np.array(range(len(labels)))[(labels.isin(interest)).to_numpy()]
    idx_not = np.array(range(len(labels)))[(labels.isin(set(labels.cat.categories).difference(interest))).to_numpy()]
    dist = scipy.spatial.distance_matrix(layer.obsm['spatial'], layer.obsm['spatial'])
    min_dist = np.min(dist[dist > 0])
    eps = 0.01
    edges = np.zeros(dist.shape)
    edges[dist > 0] = (dist[dist > 0] - min_dist) ** 2 < eps
    border = list(filter(lambda x: np.sum(edges[x, idx_not] > 0), idx))
    # Early return if border is empty
    if len(border) == 0:
        return []  # or return an empty list
    j = np.argmin(layer.obsm['spatial'][border, 0])
    contours, left = [[border[j]]], set(border).difference(set([border[j]]))
    for i in range(1, len(border)):
        last = contours[-1][-1]
        neighbors = set(left).intersection(np.where((dist[last, :] - min_dist) ** 2 < eps)[0])
        if len(neighbors) > 0:
            j = neighbors.pop()
            contours[-1].append(j)
        else:
            l = list(left)
            j = l[np.argmin(layer.obsm['spatial'][l, 0])]
            contours.append([j])
        left = left.difference(set([j]))
    return contours


def plot_2d_expression(layer, gene, labels, name="", title='', vmin=None, vmax=None,
                       layer_idx=None, norm=False, draw_contours=True,
                       ax=None, show=True):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.spatial import distance_matrix

    # If no Axes is provided, create a new figure + axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        # If an Axes is passed in, we just use it, no new figure
        fig = ax.figure

    cmap = sns.color_palette("rocket_r", as_cmap=True)

    v = to_dense_array(layer[:, gene].X).copy().ravel() + 1
    if norm:
        v = v / layer.gene_exp.sum(axis=1)
    v = np.log(v)

    scat = ax.scatter(
        layer.obsm['spatial'][:, 0],
        layer.obsm['spatial'][:, 1],
        linewidth=0, s=150, marker=".", c=v, cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=26)
    cbar = fig.colorbar(scat, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('log counts', fontsize=22)
    ax.axis('off')
    ax.invert_yaxis()

    if draw_contours:
        for l in ['Layer{}'.format(i) for i in [1, 3, 5]] + ['WM']:
            contours = get_scatter_contours(layer, labels, [l])
            for k in range(len(contours)):
                ax.plot(
                    layer.obsm['spatial'][contours[k], 0],
                    layer.obsm['spatial'][contours[k], 1],
                    'lime', lw=4, alpha=0.6
                )

    ax.text(110, 150,  'L1')
    ax.text(110, 220,  'L2')
    ax.text(110, 260,  'L3')
    ax.text(110, 305,  'L4')
    ax.text(110, 340,  'L5')
    ax.text(110, 380,  'L6')
    ax.text(110, 425,  'WM')

    if show:
        plt.show()



    # print('center layer')
    # plot_2d_expression(center_layer_,gene,adatas, sample_list,name="_center_layer_{0}".format(gene),vmin=vmin,vmax=vmax,title='{0} expression in prefrontal cortex (PASTE integrated slice NO NMF)'.format(gene),layer_idx=layer_idx)


def reduce_data(slice1, slice2, genes_of_interest=None):
    # 1. Find genes common to both slices
    common_genes = slice1.var_names.intersection(slice2.var_names)
    common_genes = common_genes[:2000]

    # 2. If genes_of_interest is given, intersect it with common_genes
    if genes_of_interest is not None:
        common_genes = common_genes.intersection(genes_of_interest)

    # 3. Subset the slices to these genes
    slice1_sub = slice1[:, common_genes].copy()
    slice2_sub = slice2[:, common_genes].copy()

    # 4. (Optional) reduce spots (obs) by evenly sampling
    sample_size = 1000

    # Evenly-spaced indices for slice1
    n1 = slice1_sub.n_obs
    indices1 = np.linspace(0, n1 - 1, sample_size, dtype=int)
    slice1_sub = slice1_sub[indices1, :].copy()

    # Evenly-spaced indices for slice2
    n2 = slice2_sub.n_obs
    indices2 = np.linspace(0, n2 - 1, sample_size, dtype=int)
    slice2_sub = slice2_sub[indices2, :].copy()

    return slice1_sub, slice2_sub


def top_diff_genes(slice1: AnnData, slice2: AnnData, top_n=50):
    """
    Return the top_n genes with the largest difference in mean expression
    between slice1 and slice2.
    """
    # Optional: check var_names match
    if not np.array_equal(slice1.var_names, slice2.var_names):
        raise ValueError("slice1.var_names and slice2.var_names differ. "
                         "Ensure they have the same gene ordering, or intersect them first.")

    # 1. Convert X to dense if needed (or use .toarray() for sparse)
    expr1 = slice1.X
    expr2 = slice2.X
    if hasattr(expr1, "toarray"):
        expr1 = expr1.toarray()
    if hasattr(expr2, "toarray"):
        expr2 = expr2.toarray()

    # 2. Compute mean expression across spots (axis=0 => mean per gene)
    mean_expr1 = expr1.mean(axis=0)
    mean_expr2 = expr2.mean(axis=0)

    # 3. Compute difference
    diff = mean_expr1 - mean_expr2

    # 4. Rank by absolute difference (largest first)
    order = np.argsort(np.abs(diff))[::-1]  # descending order

    # 5. Grab top_n genes
    top_indices = order[:top_n]
    top_gene_names = slice1.var_names[top_indices]
    top_diffs = diff[top_indices]

    # Build a dataframe for convenience
    df = pd.DataFrame({
        'gene': top_gene_names,
        'mean_expr_slice1': mean_expr1[top_indices],
        'mean_expr_slice2': mean_expr2[top_indices],
        'diff_slice1_minus_slice2': top_diffs
    })
    return df.reset_index(drop=True)




def get_fused_slice(slice1,slice2,result_folder):
    img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B = load_data_from_folder(
        result_folder + str(i) + "_" + str(j) + "_result.npz")
    slice1.obsm['spatial'] = coords_A
    slice2.obsm['spatial'] = coords_B

    # slice1, slice2 = reduce_data(slice1, slice2)

    fused_slice, _ = center_align_two_slices(slice1, slice2)
    return slice1,slice2, fused_slice


def plot_slice_expression(ax, adata, gene, title="", vmin=None, vmax=None):
    """
    Plots expression of `gene` in the spatial coordinates of `adata` on Axes `ax`.
    """
    X = adata[:, gene].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr = np.ravel(X)  # 1D array of expression

    coords = adata.obsm['spatial']  # shape (n_spots, 2)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=expr, s=30, cmap='viridis',
        vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axis('off')
    return sc


def compare_fused_expression(A, B, fused, gene):
    """
    Plots side-by-side expression of `gene` in:
      - Slice A
      - Slice B
      - The fused (center) slice
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get overall min/max so the color scale is consistent across subplots:
    vals = []
    for data in [A, B, fused]:
        X = data[:, gene].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        vals.append(X.flatten())
    combined_vals = np.concatenate(vals)
    vmin, vmax = combined_vals.min(), combined_vals.max()

    # Plot slice A
    scA = plot_slice_expression(
        axes[0], A, gene, title="Slice A", vmin=vmin, vmax=vmax
    )
    # Plot slice B
    scB = plot_slice_expression(
        axes[1], B, gene, title="Slice B", vmin=vmin, vmax=vmax
    )
    # Plot fused
    scF = plot_slice_expression(
        axes[2], fused, gene, title="Fused/Center Slice", vmin=vmin, vmax=vmax
    )

    # One shared colorbar:
    fig.colorbar(scF, ax=axes, fraction=0.02, pad=0.02, label=f"{gene} expression")

    plt.suptitle(f"Comparing '{gene}' Expression", fontsize=16)
    plt.show()



sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
# sample_list = ["151507", "151508", "151509","151510", "151669", "151670"]

adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}



for id in sample_list:
    adatas[id].image_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_hires_image.png'.format(id)
    adatas[id].image_scale_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/scalefactors_json.json'.format(id)
    adatas[id].spatial_prefix = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)
    adatas[id].obs['position'].index = (
        adatas[id].obs['position'].index
        .str.replace(r"\.\d+$", "", regex=True)
    )
    position_prefix = adatas[id].spatial_prefix
    try:
        # Try reading as CSV
        positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',')
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',')

    positions.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    # 1) Get the barcodes from AnnData that are in `positions`
    positions.index = positions["barcode"]
    adata_barcodes = adatas[id].obs['position'].index
    common_barcodes = adata_barcodes[adata_barcodes.isin(positions.index)]


    # 2) Now reindex `positions` in the exact order of `common_barcodes`
    positions_filtered = positions.reindex(common_barcodes)

    spatial_locations = positions_filtered[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    adatas[id].image_coor = spatial_locations


sample_groups = [[ "151669", "151670"],["151671", "151672"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]



layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]



result_root_folder1 ="/home/huifang/workspace/code/registration/result/PASTE/DLPFC/"
result_root_folder2 = "/home/huifang/workspace/code/registration/result/simpleITK/DLPFC/marker_free/"
groups=[[1,0],[1,2]]

for g in range(len(layer_groups)):
    g=1
    [i,j] = groups[g]
    slices = layer_groups[g]
    slice_fixed = slices[0]
    slice_moving = slices[1]

    img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B = load_data_from_folder(
        result_root_folder1 + str(i) + "_" + str(j) + "_result.npz")
    slice_fixed.obsm['spatial'] = coords_A
    slice_moving.obsm['spatial'] = coords_B

    # colors1 = list(slice_fixed.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
    # plt.scatter(slice_fixed.obsm['spatial'][:, 0],slice_fixed.obsm['spatial'][:, 1],s=10,color=colors1)
    #
    # colors2 = list(slice_moving.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
    # plt.scatter(slice_moving.obsm['spatial'][:, 0], slice_moving.obsm['spatial'][:, 1],s=10,color=colors2)
    #
    # plt.gca().invert_yaxis()
    # plt.show()

    # s1,s2,fused_slice_paste = get_fused_slice(slice_fixed,slice_moving,result_root_folder1)
    # fused_slice_paste.write("fused_1_2_paste.h5ad")
    # print("saved")
    #
    fused_slice_paste = sc.read_h5ad("fused_1_2_paste.h5ad")
    #
    # compare_fused_expression(s1,s2, fused_slice_paste, gene="MOBP")
    #
    #
    cluster_adata(fused_slice_paste, 7, sample_name="151671")

    draw_spatial(fused_slice_paste, 'my_clusters',
                 {"151671": 'Slice A', "151672": 'Slice B'}["151671"],
                 draw_contours=True)


    # s1,s2,fused_slice_vispro = get_fused_slice(slice_fixed, slice_moving, result_root_folder2)
    #
    #
    # fused_slice_vispro.write("fused_1_2_vispro.h5ad")
    # print("saved")
    fused_slice_vispro = sc.read_h5ad("fused_1_2_vispro.h5ad")

    cluster_adata(fused_slice_vispro, 7, sample_name="151671")
    draw_spatial(fused_slice_vispro, 'my_clusters',
                 {"151671": 'Slice A', "151672": 'Slice B'}["151671"],
                 draw_contours=True)

    # for s in ["151671", "151672"]:
    #     adata = adatas[s]
    #     print(s)
    #     cluster_adata(adata, 7, sample_name=s)
    #     draw_spatial(adata, 'my_clusters',
    #                  {"151671": 'Slice A', "151672": 'Slice B'}[s],
    #                  draw_contours=True)

    # df_top = top_diff_genes(fused_slice_paste, fused_slice_vispro, top_n=10)
    # # This shows columns ['gene', 'mean_expr_slice1', 'mean_expr_slice2', 'diff_slice1_minus_slice2']
    #
    # # If you just want the gene names as a list:
    # top_gene_names = df_top['gene'].tolist()
    # print(top_gene_names)
    #
    #
    # # for gene in ['MFGE8', 'MOBP', 'PCP4', 'TRABD2A']:
    # for gene in top_gene_names:
    #     vmin = None
    #     vmax = None
    #     layer_idx = None
    #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    #
    #     # First plot on ax1
    #     plot_2d_expression(
    #         fused_slice_paste, gene, fused_slice_paste.obs['layer_guess_reordered'],
    #         vmin=vmin, vmax=vmax,
    #         title='PASTE integrated slice expression',
    #         ax=ax1,  # Pass in ax1
    #         show=False  # Defer showing
    #     )
    #
    #     # Second plot on ax2
    #     plot_2d_expression(
    #         fused_slice_vispro, gene, fused_slice_vispro.obs['layer_guess_reordered'],
    #         vmin=vmin, vmax=vmax,
    #         title='VISPRO integrated slice expression',
    #         ax=ax2,  # Pass in ax2
    #         show=False  # Defer showing
    #     )
    #
    #     plt.suptitle(f"Comparing {gene} Expression", fontsize=30)
    #     plt.show()



