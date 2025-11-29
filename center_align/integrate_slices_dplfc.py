from sklearn.decomposition import NMF
import scipy.sparse
from anndata import AnnData
import numpy as np
import scanpy as sc
import sklearn
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.spatial import cKDTree
from anndata import AnnData
import networkx as nx
from dense_generation import spatial_regrid_fuse_optimized
# from dense_generation2 import spatial_regrid_fuse_gpu_robust
from dense_generation_gpu_efficient import spatial_regrid_fuse_gpu_robust
from sklearn.neighbors import NearestNeighbors
### ----------------------- FILE LOADING FUNCTION -----------------------

def load_registration_result(folder_path):
    """Load image and spot data from a given folder."""



    offset = 5
    data = np.load(folder_path)

    pts1 = data["pts1"]  # shape (4200, 2)
    pts2 = data["pts2"]

    # normalize pts1 to [0, 10]
    min_vals = pts1.min(axis=0)
    max_vals = pts1.max(axis=0)
    scale = max_vals - min_vals

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
    gene_expr = adata[:, genes].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()

    # Double log1p transform
    gene_expr = np.log1p(np.log1p(gene_expr))

    # Min-max normalization per gene
    gene_expr_norm = (gene_expr - gene_expr.min(axis=0)) / (gene_expr.max(axis=0) - gene_expr.min(axis=0) + 1e-6)
    return gene_expr_norm

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
        k=6,  # nearest donor spots per fixed spot
        sigma_spatial=15,  # good starting point for coords in [0, 105]
        sigma_expr=0.5,  # softness for expression similarity
        use_pca=1000,  # PCA dims for expression similarity (fit on fused, transform fixed)
        r_max=None,  # optional radius gate in coord units (e.g. 25.0)
        blend_beta=1,  # 0..1: weight for original signal in the final blend
        preserve_counts=False,  # rescale each spot to keep its original library size
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
        new_i = blend_beta * X_fixed[i] + bary
        # new_i = X_fixed[i] + bary

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
#         adata_fixed,
#         adata_fused,
#         k=4,  # donors per center spot
#         sigma_spatial=5.0,  # good for coords normalized ~[0,105]
#         sigma_expr=0.25,  # softness for expression similarity
#         use_pca=30,  # fit PCA on fused, project center each iter
#         layer_out="reassigned",
#         blend_beta=0.0,  # set 0.0 for fully unbiased; >0 softly mix original back in
#         overwrite_X=True,
#         preserve_counts=True
# ):
#     """
#     Unbiased center-building on fixed spots: iteratively re-estimates expression on the
#     fixed coordinates by using ONLY fused donors (no self-copy). Keeps the same input/output
#     contract as before. Set blend_beta=0 for a fully unbiased center; >0 to softly anchor.
#     """
#
#     # ----- internal iteration/denoising knobs (kept here to avoid changing signature) -----
#     N_ITER = 5  # total outer iterations
#     NMF_COMPONENTS = None  # e.g., 15 for low-rank denoising; None = off
#     R_MAX = None  # optional max donor radius (in coord units), e.g., 25.0
#
#     # ----- protect coordinates & validate gene order -----
#     S_fixed_orig = np.asarray(adata_fixed.obsm["spatial"]).copy()
#     S_fixed = S_fixed_orig
#     S_fused = np.asarray(adata_fused.obsm["spatial"])
#
#     if not np.array_equal(adata_fixed.var_names.values, adata_fused.var_names.values):
#         raise ValueError("Gene sets/orders differ between fixed and fused. Align var_names before fusion.")
#
#     # ----- dense matrices -----
#     Xf_full = adata_fixed.X
#     Xu_full = adata_fused.X
#     X_fixed = Xf_full.toarray() if scipy.sparse.issparse(Xf_full) else np.asarray(Xf_full)
#     X_fused = Xu_full.toarray() if scipy.sparse.issparse(Xu_full) else np.asarray(Xu_full)
#
#     n_center, n_genes = X_fixed.shape
#
#     # ----- expression similarity space (fit on fused only) -----
#     if use_pca and X_fused.shape[1] > 2:
#         n_comp = int(min(use_pca, max(2, X_fused.shape[1] - 1)))
#         pca = PCA(n_components=n_comp, random_state=0)
#         Z_fused = pca.fit_transform(X_fused)
#         # center (fixed) will be projected each iteration via pca.transform(Xc)
#     else:
#         pca = None
#         Z_fused = X_fused  # fall back to raw features for similarity
#
#     # ----- neighbors on donors -----
#     n_cand = min(max(5, k * 5), len(S_fused))
#     nn = NearestNeighbors(n_neighbors=n_cand, algorithm="auto").fit(S_fused)
#
#     # ----- initialize center expression Xc from fused donors (spatial only) -----
#     Xc = np.zeros_like(X_fixed, dtype=float)
#     for i in range(n_center):
#         d_all, idx_all = nn.kneighbors(S_fixed[i:i + 1], n_neighbors=n_cand, return_distance=True)
#         d_all, idx_all = d_all[0], idx_all[0]
#         if R_MAX is not None:
#             keep = d_all <= R_MAX
#             idx_all = idx_all[keep]
#             d_all = d_all[keep]
#         if len(idx_all) == 0:
#             Xc[i] = X_fixed[i]  # if no donors, start from original
#             continue
#         idxs = idx_all[:k] if len(idx_all) >= k else idx_all
#         d = norm(S_fused[idxs] - S_fixed[i], axis=1)
#         w = np.exp(-(d ** 2) / (2.0 * sigma_spatial ** 2))
#         w = w / max(w.sum(), 1e-12)
#         Xc[i] = (X_fused[idxs] * w[:, None]).sum(axis=0)
#
#     # optional low-rank denoising at init
#     if NMF_COMPONENTS is not None:
#         nmf = NMF(n_components=NMF_COMPONENTS, init="random", random_state=0,
#                   solver="mu", beta_loss="kullback-leibler")
#         W = nmf.fit_transform(np.maximum(Xc, 0))
#         H = nmf.components_
#         Xc = W @ H
#
#     # ----- iterate: pull center toward fused using spatial × expression kernels -----
#     for _ in range(N_ITER):
#         if pca is not None:
#             Zc = pca.transform(Xc)  # project current center to PCA space
#         else:
#             Zc = Xc
#
#         Xc_new = np.zeros_like(Xc)
#         for i in range(n_center):
#             d_all, idx_all = nn.kneighbors(S_fixed[i:i + 1], n_neighbors=n_cand, return_distance=True)
#             d_all, idx_all = d_all[0], idx_all[0]
#             if R_MAX is not None:
#                 keep = d_all <= R_MAX
#                 idx_all = idx_all[keep]
#                 d_all = d_all[keep]
#             if len(idx_all) == 0:
#                 Xc_new[i] = Xc[i]  # keep previous estimate if no donors
#                 continue
#
#             idxs = idx_all[:k] if len(idx_all) >= k else idx_all
#
#             # spatial kernel
#             d = norm(S_fused[idxs] - S_fixed[i], axis=1)
#             w_s = np.exp(-(d ** 2) / (2.0 * sigma_spatial ** 2))
#
#             # expression kernel (cosine between current center and donor fused)
#             zi = Zc[i]
#             zj = Z_fused[idxs]
#             denom = (norm(zj, axis=1) * (norm(zi) + 1e-8) + 1e-8)
#             cos_sim = (zj @ zi) / denom
#             w_e = np.exp(cos_sim / max(1e-6, sigma_expr))
#
#             w = w_s * w_e
#             sw = w.sum()
#             if not np.isfinite(sw) or sw <= 1e-12:
#                 Xc_new[i] = Xc[i]
#             else:
#                 w /= sw
#                 Xc_new[i] = (X_fused[idxs] * w[:, None]).sum(axis=0)
#
#         # optional denoising per iteration
#         if NMF_COMPONENTS is not None:
#             W = nmf.fit_transform(np.maximum(Xc_new, 0))
#             H = nmf.components_
#             Xc = W @ H
#         else:
#             Xc = Xc_new
#
#     # ----- final mix with original (kept for backward-compat; set blend_beta=0 for unbiased) -----
#     X_final = blend_beta * X_fixed + (1.0 - blend_beta) * Xc
#
#     # optional per-spot library-size preservation (if counts)
#     if preserve_counts:
#         sums_old = X_fixed.sum(axis=1, keepdims=True)
#         sums_new = X_final.sum(axis=1, keepdims=True)
#         scale = np.divide(sums_old, np.maximum(sums_new, 1e-12))
#         X_final = X_final * scale
#
#     # ----- write out & restore coordinates -----
#     adata_fixed.layers[layer_out] = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
#     if overwrite_X:
#         adata_fixed.X = adata_fixed.layers[layer_out]
#
#     adata_fixed.obsm["spatial"] = S_fixed_orig  # never change coords
#     return adata_fixed

def spatial_resample_to_adata(adata, grid_size=50.):
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

def spatial_resample_to_adata_smoothed(adata, grid_size=50., k=4, eps=1e-6):

    coords = adata.obsm["spatial"]
    expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

    # Compute grid centers
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    n_cols = int(np.ceil((x_max - x_min) / grid_size)) + 1
    n_rows = int(np.ceil((y_max - y_min) / grid_size)) + 1

    grid_x = np.arange(x_min, x_min + n_cols * grid_size, grid_size)
    grid_y = np.arange(y_min, y_min + n_rows * grid_size, grid_size)
    grid_coords = np.array([(x, y) for y in grid_y for x in grid_x])

    # KDTree for efficient neighbor search
    tree = cKDTree(grid_coords)

    # Accumulate gene reads into grids
    new_X = np.zeros((grid_coords.shape[0], expr.shape[1]), dtype=float)
    counts = np.zeros(grid_coords.shape[0], dtype=float)

    for i, spot in enumerate(coords):
        dists, idxs = tree.query(spot, k=k)
        weights = 1.0 / (dists + eps)
        weights /= weights.sum()

        for w, idx in zip(weights, idxs):
            new_X[idx] += expr[i] * w
            counts[idx] += w

    # Keep only grids that actually received reads
    mask = counts > 0
    new_X = new_X[mask]
    new_coords = grid_coords[mask]

    # Build AnnData
    new_adata = AnnData(X=new_X, var=adata.var.copy())
    new_adata.obsm["spatial"] = new_coords
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]

    return new_adata

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


def cluster_adata(adata, n_clusters=7, sample_name='', use_nmf=False):
    from sklearn.cluster import KMeans
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor="seurat", n_top_genes=1000)
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
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)

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





gene_folder = "/home/huifang/workspace/code/registration/data/DLPFC"
sampleids=[['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]

registration_folder = "/media/huifang/data/registration/result/center_align/DLPFC/ours/"


for i in range(1,3):

    fix_adata_path = f"{gene_folder}/{sampleids[i][0]}_preprocessed.h5"
    moving_adata_paths = [
        f"{gene_folder}/{sampleids[i][1]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][2]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][3]}_preprocessed.h5"
    ]
    registration_paths = [
        f"{registration_folder}/{i}_0_result.npz",
        f"{registration_folder}/{i}_1_result.npz",
        f"{registration_folder}/{i}_2_result.npz"
    ]
    # genes = ['MFGE8','MOBP','PCP4','TRABD2A']
    genes = ['MFGE8', 'MOBP', 'PCP4']


    adata_fixed,adata_pool,adata_full = integrate_slices(fix_adata_path, moving_adata_paths, registration_paths)

    exprs = load_gene_expression(adata_fixed, genes)
    plot_marker_genes(adata_fixed, genes, exprs)
    # print(adata_full)
    #
    exprs = load_gene_expression(adata_full, genes)
    plot_marker_genes(adata_full, genes, exprs)
    # test = input()
    # adata_fused = spatial_regrid_fuse_optimized(adata_full,grid_size=1.5,alpha=0.9,fuse_radius=3.0)
    print(adata_fixed)
    print(adata_full)
    adata_fused = spatial_regrid_fuse_gpu_robust(
        adata_full,
        grid_size=1.5,
        fuse_radius=5,  # ~1.5–2.0 * grid_size works well
        keep_nuc_frac=0.6,
        knee_mode=False,
        alpha=0.6,  # prioritize geometry a bit for deformation
        pca_dim=32,
        device="cuda",
        use_raw=False
    )
    print(adata_fused)

    # adata_fused = fuse_to_fixed_barycentric(adata_fixed,adata_pool)
    # adata_combined = spatial_resample_to_adata_smoothed(adata_combined,grid_size=1,k=4)
    # print(adata_combined)
    # adata_combined = spatial_resample_nmf(adata_combined,grid_size=100)
    # print(adata_combined.shape)

    # print(adata_combined)
    # print(new_adata)
    # test = input()
    exprs = load_gene_expression(adata_fused, genes)
    plot_marker_genes(adata_fused, genes, exprs)

    # cmap = plt.cm.Reds
    # colors = cmap(norm_expr)
    # colors[:, -1] = norm_expr  # transparency reflects expression level
    # plt.scatter(adata_combined.obsm['spatial'][:, 0], adata_combined.obsm['spatial'][:, 1], color=colors, s=10)
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal')
    # plt.show()
