from sklearn.decomposition import NMF
import scipy.sparse
from anndata import AnnData
import numpy as np

import sklearn


import pandas as pd
from scipy.spatial import cKDTree
from anndata import AnnData
import networkx as nx
from sklearn.neighbors import NearestNeighbors
### ----------------------- FILE LOADING FUNCTION -----------------------
from dense_generation import spatial_regrid_fuse_optimized
from dense_generation2 import spatial_regrid_fuse_gpu_robust

from sklearn.cluster import KMeans
from spatial_coherence import generate_graph_from_labels_gpu,spatial_coherence_score_fast
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_and_plot(adata, n_clusters=5, n_top_genes=2000, n_pcs=30, n_neighbors=15):
    """
    Cluster ST data with known number of clusters and plot UMAP & spatial distribution.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with obsm["spatial"] for coordinates.
    n_clusters : int
        Number of clusters (K for KMeans).
    n_top_genes : int
        Number of highly variable genes for preprocessing.
    n_pcs : int
        Number of principal components.
    n_neighbors : int
        Number of neighbors for kNN graph in UMAP.
    """

    # --- Preprocessing ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    adata.obs["kmeans_clusters"] = kmeans.fit_predict(adata.obsm["X_pca"]).astype(str)

    # --- Plot UMAP ---
    sc.pl.umap(adata, color="kmeans_clusters", palette="tab20", legend_loc="on data")

    # --- Plot spatial distribution ---
    coords = adata.obsm["spatial"]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        coords[:, 0], coords[:, 1],
        c=adata.obs["kmeans_clusters"].astype(int),
        cmap="tab20", s=20
    )
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Spatial distribution of clusters")
    plt.show()

    return adata


def load_registration_result(folder_path):
    offset = 5
    data = np.load(folder_path)

    pts1 = data["pts1"]  # shape (4200, 2)
    pts2 = data["pts2"]

    # normalize pts1 to [0, 10]
    min_vals = pts1.min(axis=0)
    max_vals = pts1.max(axis=0)
    scale = max_vals - min_vals
    # if scale[0]>scale[1]:
    #     scale = scale[0]
    # else:
    #     scale = scale[1]


    pts_combined = np.vstack([pts1, pts2])
    # min and max per column (x and y separately)
    min_vals = pts_combined.min(axis=0)

    pts1_norm = (pts1 - min_vals) / scale * 100 + offset
    pts2_norm = (pts2 - min_vals) / scale * 100 + offset

    # combine along rows
    pts_combined = np.vstack([pts1_norm, pts2_norm])
    mask_shape = np.ceil(pts_combined.max(axis=0)).astype(np.uint8) + offset

    return pts1_norm, pts2_norm, data["label1"].reshape(-1), data["label2"].reshape(-1)


def load_gene_expression(adata, genes):
    """
    Extract expression for the given genes (if available in adata),
    apply double log1p transform and per-gene min-max normalization.

    Returns
    -------
    gene_expr_norm : ndarray (n_obs x n_genes_found)
        Normalized expression matrix.
    valid_genes : list of str
        Genes that were present in adata.var_names.
    """
    # keep only genes that exist in adata
    valid_genes = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]

    if not valid_genes:
        raise ValueError("None of the requested genes were found in adata.var_names")

    if missing:
        print(f"[info] Skipped missing genes: {missing}")

    # subset and convert to dense
    gene_expr = adata[:, valid_genes].X
    if hasattr(gene_expr, "toarray"):
        gene_expr = gene_expr.toarray()

    # double log1p
    gene_expr = np.log1p(np.log1p(gene_expr))

    # per-gene min-max normalization
    minv = gene_expr.min(axis=0)
    maxv = gene_expr.max(axis=0)
    gene_expr_norm = (gene_expr - minv) / (maxv - minv + 1e-6)

    return gene_expr_norm, valid_genes

def integrate_slices(fix_adata_path, moving_adata_paths, registration_paths):
    # fix_data = sc.read_h5ad(fix_adata_path)
    # slices=[]
    # for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
    #     coords_fixed, coords_warp, _, _ = load_registration_result(reg_path)
    #     if idx==0:
    #         fix_data.obsm['spatial'] = coords_fixed
    #         slices.append(fix_data)
    #     moving_data = sc.read_h5ad(adata_path)
    #     moving_data.obsm['spatial'] = coords_warp
    #     slices.append(moving_data)
    # adata_combined = sc.concat(slices, join='outer', label='batch', fill_value=0)
    # return fix_data,adata_combined

    """
    Load a fixed Visium AnnData and multiple registered moving AnnData,
    align their gene sets to the intersection (same genes + same order),
    and return (fixed_subset, fused_concat).

    Assumes `load_registration_result(reg_path)` returns:
        coords_fixed, coords_warp, _, _
    where coords_* are Nx2 numpy arrays for spatial coordinates.
    """
    # --- load fixed ---
    fix_data = sc.read_h5ad(fix_adata_path)
    fix_data.var_names_make_unique()

    slices = []
    # --- iterate moving slices & inject registered coords ---
    for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
        coords_fixed, coords_warp, _, _ = load_registration_result(reg_path)

        if idx == 0:
            # ensure ndarray & correct shape
            fix_data.obsm["spatial"] = np.asarray(coords_fixed)
            slices.append(fix_data)

        moving_data = sc.read_h5ad(adata_path)
        moving_data.var_names_make_unique()
        moving_data.obsm["spatial"] = np.asarray(coords_warp)
        slices.append(moving_data)

    # --- compute common genes across ALL slices ---
    common = set(fix_data.var_names)
    for ad in slices:
        common &= set(ad.var_names)

    if len(common) == 0:
        raise ValueError("No common genes across fixed and moving slices.")

    # preserve the fixed-sample gene order
    common_genes_ordered = [g for g in slices[0].var_names if g in common]

    # --- subset EVERY slice to the common genes (same order) ---
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes_ordered].copy()

    # --- build fused AnnData (all registered slices, including fixed as batch 0) ---
    adata_support = sc.concat(
        slices[1:],
        join="inner",  # defensive: should already be intersected
        label="batch",
        fill_value=0
    )
    adata_combined = sc.concat(
        slices,
        join="inner",  # defensive: should already be intersected
        label="batch",
        fill_value=0
    )

    # Also return the fixed subset (same genes/order as fused)
    fix_subset = slices[0]

    print(f"[integrate_slices] Common genes: {len(common_genes_ordered)} "
          f"(from {fix_data.n_vars} in fixed and {[s.n_vars for s in slices[1:]]} in moving)")

    return fix_data[:, common_genes_ordered].copy(), adata_support,adata_combined

import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numpy.linalg import norm

def fuse_to_fixed_barycentric(
        adata_fixed,
        adata_fused,
        *,
        k=3,  # nearest donor spots per fixed spot
        sigma_spatial=2.0,  # good starting point for coords in [0, 105]
        sigma_expr=0.5,  # softness for expression similarity
        use_pca=30,  # PCA dims for expression similarity (fit on fused, transform fixed)
        r_max=None,  # optional radius gate in coord units (e.g. 25.0)
        blend_beta=0.5,  # 0..1: weight for original signal in the final blend
        preserve_counts=True,  # rescale each spot to keep its original library size
        layer_out="reassigned",  # destination layer for fused expression
        overwrite_X=True,  # set True if you want to overwrite adata_fixed.X
        verbose=True
):
    """
    Recalculate expression for fixed Visium spots by barycentric projection from a fused donor set.

    Assumptions:
      - adata_fused DOES NOT contain the fixed slice.
      - adata_fixed and adata_fused have IDENTICAL var_names and order.
      - adata_fixed.obsm["spatial"] and adata_fused.obsm["spatial"] are in the same coordinate system.
    """

    # ---- 0) Protect coordinates & validate gene order ----
    S_fixed_orig = np.asarray(adata_fixed.obsm["spatial"]).copy()
    S_fixed = S_fixed_orig
    S_fused = np.asarray(adata_fused.obsm["spatial"])

    if not np.array_equal(adata_fixed.var_names.values, adata_fused.var_names.values):
        raise ValueError(
            "Gene sets/orders differ between fixed and fused. "
            "Subset/intersect and align var_names BEFORE fusion."
        )

    # ---- 1) Dense matrices for math ----
    Xf_full = adata_fixed.X
    Xu_full = adata_fused.X
    X_fixed = Xf_full.toarray() if scipy.sparse.issparse(Xf_full) else np.asarray(Xf_full)
    X_fused = Xu_full.toarray() if scipy.sparse.issparse(Xu_full) else np.asarray(Xu_full)

    # ---- 2) Low-dim expression for similarity (fit PCA on fused, project fixed) ----
    if use_pca and X_fused.shape[1] > 2:
        n_comp = int(min(use_pca, max(2, X_fused.shape[1] - 1)))
        pca = PCA(n_components=n_comp, random_state=0)
        Z_fused = pca.fit_transform(X_fused)
        Z_fixed = pca.transform(X_fixed)
    else:
        Z_fused, Z_fixed = X_fused, X_fixed

    # ---- 3) Neighbor search on donor coords ----
    n_cand = min(max(5, k * 5), len(S_fused))
    nn = NearestNeighbors(n_neighbors=n_cand, algorithm="auto").fit(S_fused)

    # ---- 4) Allocate result & fuse ----
    reassigned = np.zeros_like(X_fixed, dtype=float)
    n_spots = S_fixed.shape[0]
    fallback = 0

    for i in range(n_spots):
        # candidates by spatial proximity
        d_all, idx_all = nn.kneighbors(S_fixed[i:i + 1], n_neighbors=n_cand, return_distance=True)
        d_all, idx_all = d_all[0], idx_all[0]

        # optional radius gate
        if r_max is not None:
            keep = d_all <= r_max
            idx_all = idx_all[keep]
            d_all = d_all[keep]

        if len(idx_all) == 0:
            # no donors → use original (will still be blended below)
            bary = X_fixed[i].copy()
            fallback += 1
        else:
            idxs = idx_all[:k] if len(idx_all) >= k else idx_all
            d = norm(S_fused[idxs] - S_fixed[i], axis=1)

            # spatial weights (Gaussian)
            w_s = np.exp(-(d ** 2) / (2.0 * sigma_spatial ** 2))

            # expression weights (cosine similarity → softmax-like with sigma_expr)
            zi = Z_fixed[i]
            zj = Z_fused[idxs]
            denom = (norm(zj, axis=1) * (norm(zi) + 1e-8) + 1e-8)
            cos_sim = (zj @ zi) / denom
            w_e = np.exp(cos_sim / max(1e-6, sigma_expr))

            w = w_s * w_e
            sw = w.sum()

            if not np.isfinite(sw) or sw <= 1e-12:
                bary = X_fixed[i].copy()
                fallback += 1
            else:
                w /= sw
                bary = (X_fused[idxs] * w[:, None]).sum(axis=0)

        # ---- 5) Always blend with original so signal never disappears ----
        new_i = blend_beta * X_fixed[i] + (1.0 - blend_beta) * bary

        # ---- 6) Optional: per-spot library size preservation (if counts) ----
        if preserve_counts:
            so = X_fixed[i].sum()
            sn = new_i.sum()
            if sn > 0 and so > 0:
                new_i *= (so / sn)

        reassigned[i] = np.nan_to_num(new_i, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 7) Write out & restore coordinates ----
    adata_fixed.layers[layer_out] = reassigned
    if overwrite_X:
        adata_fixed.X = reassigned

    # ensure coords unchanged
    adata_fixed.obsm["spatial"] = S_fixed_orig
    assert np.allclose(adata_fixed.obsm["spatial"], S_fixed_orig)

    if verbose:
        print(f"[fuse_to_fixed_barycentric] fallbacks={fallback}/{n_spots} "
              f"({fallback / n_spots:.1%}); blend_beta={blend_beta}, "
              f"preserve_counts={preserve_counts}, overwrite_X={overwrite_X}")

    return adata_fixed


def spatial_resample_to_adata(adata, grid_size=4):
    coords = adata.obsm["spatial"]
    expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # Compute spatial bounds
    x_min, y_min = coords.min(axis=0)
    # Map each spot to grid cell
    col_idx = ((coords[:, 0] - x_min) / grid_size).astype(int)
    row_idx = ((coords[:, 1] - y_min) / grid_size).astype(int)

    # Dictionary to accumulate gene reads
    grid_dict = {}
    for i in range(expr.shape[0]):
        key = (row_idx[i], col_idx[i])
        if key not in grid_dict:
            grid_dict[key] = expr[i].copy()
        else:
            grid_dict[key] += expr[i]

    # Build new AnnData
    new_X = []
    new_coords = []
    for (r, c), vals in sorted(grid_dict.items()):
        new_X.append(vals)
        new_coords.append([x_min + c * grid_size, y_min + r * grid_size])

    new_X = np.array(new_X)
    new_coords = np.array(new_coords)

    new_adata = AnnData(X=new_X, var=adata.var.copy())
    new_adata.obsm["spatial"] = new_coords
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]

    return new_adata
# import numpy as np
# import scipy
# from anndata import AnnData
# from scipy.spatial import cKDTree
#
#
# def spatial_resample_to_adata(adata, grid_size=50, k=4, eps=1e-6):
#
#     coords = adata.obsm["spatial"]
#     expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
#
#     # Compute grid centers
#     x_min, y_min = coords.min(axis=0)
#     x_max, y_max = coords.max(axis=0)
#
#     n_cols = int(np.ceil((x_max - x_min) / grid_size)) + 1
#     n_rows = int(np.ceil((y_max - y_min) / grid_size)) + 1
#
#     grid_x = np.arange(x_min, x_min + n_cols * grid_size, grid_size)
#     grid_y = np.arange(y_min, y_min + n_rows * grid_size, grid_size)
#     grid_coords = np.array([(x, y) for y in grid_y for x in grid_x])
#
#     # KDTree for efficient neighbor search
#     tree = cKDTree(grid_coords)
#
#     # Accumulate gene reads into grids
#     new_X = np.zeros((grid_coords.shape[0], expr.shape[1]), dtype=float)
#     counts = np.zeros(grid_coords.shape[0], dtype=float)
#
#     for i, spot in enumerate(coords):
#         dists, idxs = tree.query(spot, k=k)
#         weights = 1.0 / (dists + eps)
#         weights /= weights.sum()
#
#         for w, idx in zip(weights, idxs):
#             new_X[idx] += expr[i] * w
#             counts[idx] += w
#
#     # Keep only grids that actually received reads
#     mask = counts > 0
#     new_X = new_X[mask]
#     new_coords = grid_coords[mask]
#
#     # Build AnnData
#     new_adata = AnnData(X=new_X, var=adata.var.copy())
#     new_adata.obsm["spatial"] = new_coords
#     new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]
#
#     return new_adata

def spatial_resample_nmf(adata: AnnData, grid_size: int = 64, n_components: int = 20) -> AnnData:
    from sklearn.decomposition import NMF

    coords = adata.obsm['spatial']
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Compute grid boundaries
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    # Create regular grid
    x_edges = np.linspace(min_x, max_x, grid_size + 1)
    y_edges = np.linspace(min_y, max_y, grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Assign spots to grid
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)

    grid = np.zeros((grid_size, grid_size, X.shape[1]))
    counts = np.zeros((grid_size, grid_size))

    for i in range(coords.shape[0]):
        grid[y_bin[i], x_bin[i]] += X[i]
        counts[y_bin[i], x_bin[i]] += 1

    # Fill empty grids with nearest non-empty grid values
    filled_grid = grid.copy()
    non_empty_idx = np.argwhere(counts > 0)
    empty_idx = np.argwhere(counts == 0)

    if len(empty_idx) > 0:
        tree = cKDTree(non_empty_idx)
        _, nearest_indices = tree.query(empty_idx)
        for i, empty in enumerate(empty_idx):
            filled_grid[tuple(empty)] = grid[tuple(non_empty_idx[nearest_indices[i]])]

    # Flatten and apply NMF
    flat_grid = filled_grid.reshape(-1, X.shape[1])
    model = NMF(n_components=n_components, solver = 'mu', beta_loss = 'kullback-leibler', init='random', random_state = None, verbose = False)
    W = model.fit_transform(flat_grid)
    H = model.components_
    smoothed_flat = W @ H
    smoothed_grid = smoothed_flat.reshape((grid_size, grid_size, X.shape[1]))

    # Construct new AnnData
    new_coords = grid_coords
    new_expr = smoothed_grid.reshape(-1, X.shape[1])
    new_adata = AnnData(X=new_expr)
    new_adata.obsm['spatial'] = new_coords

    return new_adata

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


from sklearn.cluster import KMeans
def cluster_adata(adata, n_clusters=7, use_nmf=False):

    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor="seurat", n_top_genes=1000)
    sc.pp.pca(adata_copy)

    if use_nmf:
        model = sklearn.decomposition.NMF(n_components=50)
        adata_copy.obsm['X_pca'] = model.fit_transform(adata_copy.X)

    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit_predict(adata_copy.obsm['X_pca'])


    return

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

def draw_spatial(adata, clusters='my_clusters', sample_name='', draw_contours=False):
    fig = plt.figure(figsize=(12, 10))
    ax = sc.pl.spatial(adata, color=clusters, spot_size=5, show=False,
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


def plot_marker_genes(adata, genes, gene_expr_norm):
    n = len(genes)
    # fig, axes = plt.subplots(1, n, figsize=(8 * n, 10), constrained_layout=True)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)

    for i, gene in enumerate(genes):
        ax = axes[i] if n > 1 else axes
        expr = gene_expr_norm[:, i]
        cmap = plt.cm.Reds
        colors = cmap(expr)
        colors[:, -1] = expr  # set alpha as expression level

        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                   color=colors, s=10)
        ax.set_title(gene,fontsize=26)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    plt.show()

def normalize_coords(coords, offset=5):
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    scale = max_vals - min_vals
    return (coords - min_vals) / scale * 100 + offset

def compare_original_vs_fused(adata_original, adata_fused, n_top_genes=2000, n_pcs=30, n_neighbors=15):
    """
    Cluster integrated ST data (adata_fused) into the same number of clusters as adata_original,
    then show side-by-side spatial distributions.

    Parameters
    ----------
    adata_original : AnnData
        Original AnnData with .obs['original_clusters'].
    adata_fused : AnnData
        Integrated AnnData (multiple slices).
    n_top_genes : int
        Number of highly variable genes for preprocessing fused data.
    n_pcs : int
        Number of principal components for fused data.
    n_neighbors : int
        Number of neighbors for fused data UMAP.
    """

    # --- Number of clusters from original ---
    n_clusters = adata_original.obs["clusters_expr_smoothed"].nunique()
    g, l = generate_graph_from_labels_gpu(adata_original, adata_original.obs["clusters_expr_smoothed"])
    single_score = np.abs(spatial_coherence_score_fast(g, l))
    # --- Preprocess fused data ---
    # sc.pp.normalize_total(adata_fused, target_sum=1e4)
    # sc.pp.log1p(adata_fused)

    # sc.pp.highly_variable_genes(adata_fused, flavor="seurat", n_top_genes=n_top_genes)
    # adata_fused = adata_fused[:, adata_fused.var["highly_variable"]].copy()

    sc.pp.scale(adata_fused, max_value=10)
    sc.tl.pca(adata_fused, svd_solver="arpack")
    sc.pp.neighbors(adata_fused, use_rep="X_pca", n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_fused)

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    adata_fused.obs["kmeans_clusters"] = kmeans.fit_predict(adata_fused.obsm["X_pca"]).astype(str)
    g, l = generate_graph_from_labels_gpu(adata_fused, adata_fused.obs["kmeans_clusters"])
    fused_score = np.abs(spatial_coherence_score_fast(g, l))

    coords_orig = normalize_coords(adata_original.obsm["spatial"])
    coords_fused = adata_fused.obsm["spatial"]

    # --- Plot side-by-side spatial distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original
    axes[0].scatter(
        coords_orig[:, 0], coords_orig[:, 1],
        c=adata_original.obs["clusters_expr_smoothed"].astype("category").cat.codes,
        cmap="tab20", s=20
    )
    axes[0].set_title(f"Original clusters (normalized), Score: {single_score}")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Fused
    axes[1].scatter(
        coords_fused[:, 0], coords_fused[:, 1],
        c=adata_fused.obs["kmeans_clusters"].astype(int),
        cmap="tab20", s=20
    )
    axes[1].set_title(f"Fused clusters (KMeans, normalized), Score: {fused_score}")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("X")

    plt.tight_layout()
    plt.show()



def plot_stl_score(adata):
    marker_sets = {
        "TLS": ["CD4","CD8A","CD74","CD79A","IL7R","ITGAE","CD1D","CD3D","CD3E","CD8B",
                "CD19","CD22","CD52","CD79B","CR2","CXCL13","CXCR5","FCER2","MS4A1",
                "PDCD1","PTGDS","TRBC2"],
        "T_cells": ["CD3D","CD3E","CD2","CD5","CD28","TRBC2"],
        "B_cells": ["CD19","MS4A1","CD79A","CD79B","CD22","CR2"],
        "Myeloid": ["CD14","CD68","S100A8","S100A9","FCGR3A"],
    }

    coords = adata.obsm["spatial"]
    # keep only rows with finite coords
    finite_coord_mask = np.isfinite(coords).all(axis=1)
    if not finite_coord_mask.all():
        print(f"[warn] Dropping {np.sum(~finite_coord_mask)} cells with non-finite spatial coords.")
    coords_ok = coords[finite_coord_mask]

    # score each set if it has valid genes present
    scored_sets = []
    for set_name, genes in marker_sets.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if not valid_genes:
            print(f"[info] No valid genes for {set_name}; skipping.")
            continue
        # Use raw if you have it and it’s up to date; otherwise keep use_raw=False
        sc.tl.score_genes(
            adata, gene_list=valid_genes,
            score_name=f"{set_name}_Score",
            use_raw=False, random_state=0
        )
        scored_sets.append(set_name)

    if not scored_sets:
        print("[error] No marker sets could be scored.")
        return

    # make subplots only for scored sets
    n_sets = len(scored_sets)
    fig, axs = plt.subplots(1, n_sets, figsize=(5 * n_sets, 5), squeeze=False)

    for i, set_name in enumerate(scored_sets):
        score_col = f"{set_name}_Score"
        scores = adata.obs[score_col].to_numpy()

        # align scores with coords_ok (drop rows with bad coords)
        scores_ok = scores[finite_coord_mask]

        # Handle NaNs in scores: Option A (filter), Option B (impute)
        nan_mask = ~np.isfinite(scores_ok)
        if nan_mask.any():
            print(f"[warn] {set_name}: {nan_mask.sum()} cells have NaN scores.")
            # Option A: filter those points
            keep = ~nan_mask
            coords_plot = coords_ok[keep]
            c_plot = scores_ok[keep]
            # Option B: (alternative) impute:
            # fill_val = np.nanmin(scores_ok) if np.isfinite(scores_ok).any() else 0.0
            # c_plot = np.nan_to_num(scores_ok, nan=fill_val)
            # coords_plot = coords_ok
        else:
            coords_plot = coords_ok
            c_plot = scores_ok

        ax = axs[0, i]
        sca = ax.scatter(
            coords_plot[:, 0],  # Visium: often (y, x) to match image orientation
            coords_plot[:, 1],
            c=c_plot, s=30, cmap='viridis', alpha=0.9, edgecolor='none'
        )
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        ax.set_title(f"{set_name} Score", fontsize=12)
        ax.set_xlabel("Spatial Y"); ax.set_ylabel("Spatial X")
        cbar = fig.colorbar(sca, ax=ax, shrink=0.8)
        cbar.set_label(f"{set_name}_Score", fontsize=10)

    plt.tight_layout()
    plt.show()

root_folder = "/media/huifang/data/registration/result/pairwise_align/mouse/ours"
gene_folder="/media/huifang/data/registration/mouse/huifang/h5ad/nouns"
for i in [1]:
    fix_adata_path = f"{gene_folder}/group_{i}_slice_v1_clusters_smoothed.h5ad"
    registration_paths = [
        f"{root_folder}/{i}_result.npz",
    ]
    moving_adata_paths = [
        f"{gene_folder}/group_{i}_slice_v2_clusters_smoothed.h5ad"
    ]

    genes = ['Mbp','Prox1','Snap25','Gfap']

    adata_single = sc.read_h5ad(fix_adata_path)



    exprs,valid_genes = load_gene_expression(adata_single, genes)
    plot_marker_genes(adata_single, valid_genes, exprs)

    # cluster_and_plot(adata_single,n_clusters=8)



    adata_fixed,adata_pool,adata_full = integrate_slices(fix_adata_path, moving_adata_paths, registration_paths)

    exprs = load_gene_expression(adata_full, genes)
    plot_marker_genes(adata_full, genes, exprs)
    #
    # plot_stl_score(adata_fixed)

    # exprs = load_gene_expression(adata_full, genes)
    # plot_marker_genes(adata_full, genes, exprs)


    # adata_fused = fuse_to_fixed_barycentric(adata_fixed,adata_full)
    # adata_combined = spatial_resample_to_adata(adata_combined,grid_size=4)
    print(adata_fixed)
    print(adata_full)
    adata_fused = spatial_regrid_fuse_optimized(adata_full,grid_size=1.5,alpha=0.6)
    # adata_fused = spatial_regrid_fuse_gpu_robust(adata_full,grid_size=1.5,alpha=0.6)


    exprs, valid_genes = load_gene_expression(adata_fused, genes)
    plot_marker_genes(adata_fused, valid_genes, exprs)
    print(adata_fused)



    # exprs = load_gene_expression(adata_fused, genes)
    # plot_marker_genes(adata_fused, genes, exprs)
    #
    #
    compare_original_vs_fused(adata_single,adata_fused)
    # adata_combined = spatial_resample_nmf(adata_combined,grid_size=50)

    # print(adata_combined)
    # print(adata_combined.shape)

    # print(adata_combined)
    # print(new_adata)
    # test = input()
    # exprs = load_gene_expression(adata_combined, genes)
    # plot_marker_genes(adata_combined, genes, exprs)

    # cmap = plt.cm.Reds
    # colors = cmap(norm_expr)
    # colors[:, -1] = norm_expr  # transparency reflects expression level
    # plt.scatter(adata_combined.obsm['spatial'][:, 0], adata_combined.obsm['spatial'][:, 1], color=colors, s=10)
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal')
    # plt.show()
