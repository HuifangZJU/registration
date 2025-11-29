import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
# Assume you already have your Visium-like AnnData in `result`
# with `result.obs_names` holding cell_ids (strings)

def fuse_morphology_into_adata(
    adata,
    morpho_h5_path,
    obsm_key: str = "X_morphology",
    target_key: str = "morphology_target",
    save_path: str = None,
):
    # ---------- 1) Read HDF5 ----------
    with h5py.File(morpho_h5_path, "r") as f:
        feats = f["features"][:]                      # (M, D)
        cell_ids_raw = f["cell_ids"][:]               # (M,)
        targets = f["targets"][:] if "targets" in f else None  # (M,) or (M, K)

    # Decode cell_ids safely to Python str
    def _to_str(x):
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        return str(x)

    cell_ids = np.array([_to_str(x) for x in cell_ids_raw], dtype=object)

    # ---------- 2) Build tables, resolve duplicates ----------
    # Features
    df_feats = pd.DataFrame(feats, index=cell_ids)
    # If duplicate cell_ids exist in HDF5, average them
    if df_feats.index.duplicated().any():
        df_feats = df_feats.groupby(level=0).mean()

    # Targets (optional)
    df_tgt = None
    if targets is not None:
        # If targets is 1D, make a Series; if 2D, make a DataFrame
        if targets.ndim == 1:
            s = pd.Series(targets, index=cell_ids)
            if s.index.duplicated().any():
                # For duplicates, keep first (or you could s.groupby(level=0).mean())
                s = s[~s.index.duplicated(keep="first")]
            df_tgt = s
        else:
            df_tgt = pd.DataFrame(targets, index=cell_ids)
            if df_tgt.index.duplicated().any():
                df_tgt = df_tgt.groupby(level=0).mean()

    # ---------- 3) Align to AnnData.obs_names ----------
    # Ensure obs_names are strings
    adata.obs_names = adata.obs_names.astype(str)

    # Reindex features to all cells (fill missing with NaN)
    feats_aligned = df_feats.reindex(adata.obs_names)
    X_morph = feats_aligned.to_numpy(dtype=np.float32)  # shape = (n_obs, D)

    # Attach to obsm
    adata.obsm[obsm_key] = X_morph

    # Attach targets if present
    if df_tgt is not None:
        if isinstance(df_tgt, pd.Series):
            tgt_aligned = df_tgt.reindex(adata.obs_names)
            # Try to make categorical if looks like strings
            if tgt_aligned.dtype == object:
                adata.obs[target_key] = pd.Categorical(tgt_aligned)
            else:
                adata.obs[target_key] = tgt_aligned
        else:
            # Multidim targets -> put in obsm
            adata.obsm[f"{target_key}_matrix"] = df_tgt.reindex(adata.obs_names).to_numpy()

    # ---------- 4) Report and save ----------
    # matched = feats_aligned.notna().any(axis=1).sum()
    # total = adata.n_obs
    # D = X_morph.shape[1]
    # print(f"Fused morphology features: matched {matched} / {total} cells; feature dim = {D}")
    # print(adata)
    # test = input()

    # if save_path is None:
    #     # save alongside the morphology H5
    #     base_dir = os.path.dirname(morpho_h5_path)
    #     save_path = os.path.join(base_dir, "xenium_visium_with_morphology.h5ad")
    #
    # adata.write(save_path)
    # print(f"Saved updated AnnData to: {save_path}")

    return adata

def get_combined_data(data_name):
    # --- Load data ---
    result = sc.read_10x_h5(
        f"/media/huifang/data/Xenium/xenium_data/{data_name}/cell_feature_matrix.h5"
    )
    centroids_file = (
        f"/media/huifang/data/Xenium/xenium_data/{data_name}/preprocessing/cell_centroids.csv"
    )

    centroids_data = pd.read_csv(centroids_file)
    # --- Align and merge ---
    centroids_data = centroids_data.set_index("cell_id")

    # Keep only overlapping cell IDs
    common_ids = result.obs_names.intersection(centroids_data.index)

    # Subset AnnData and metadata
    result = result[common_ids].copy()
    meta_aligned = centroids_data.loc[common_ids]

    # Add metadata to AnnData
    result.obs = result.obs.join(meta_aligned)
    # --- Step 1: Construct Visium-style spatial coordinates ---
    # Visium expects .obsm["spatial"] as an (n_cells, 2) array
    if "centroid_x" in result.obs.columns and "centroid_y" in result.obs.columns:
        result.obsm["spatial"] = result.obs[["centroid_x", "centroid_y"]].to_numpy()

    # --- Step 2: Optional — clean obs column names ---
    # Visium usually has simpler metadata (optional)
    result.obs = result.obs.rename(columns={
        "centroid_x": "x",
        "centroid_y": "y",
        "cell_type": "cell_type"
    })

    # --- Step 3: Add basic metadata for consistency ---
    result.uns["spatial"] = {
        "library_id": {
            "images": {},
            "scalefactors": {
                "spot_diameter_fullres": 1.0,  # dummy for compatibility
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 1.0,
            }
        }
    }

    return result

from sklearn.cluster import KMeans

def plot_kmeans_similarity(adata, obsm_key="X_morphology", K=3, invert_y=True, seed=0):
    X = adata.obsm[obsm_key].astype(np.float32)
    XY = adata.obsm["spatial"][:, :2]
    mask = np.isfinite(X).all(axis=1)
    Xv, XYv = X[mask], XY[mask]

    Xv = StandardScaler().fit_transform(Xv)
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(Xv)
    C = km.cluster_centers_
    # soft assignment via RBF to centroids
    # scale by median pairwise dist to centroids
    d2 = np.sum((Xv[:, None, :] - C[None, :, :])**2, axis=2)
    tau2 = np.median(d2) + 1e-8
    P = np.exp(-d2 / (2*tau2))
    P /= P.sum(axis=1, keepdims=True)

    ncols = 3; nrows = int(np.ceil(K/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()
    for k in range(K):
        ax = axes[k]
        v = P[:, k]
        sc = ax.scatter(XYv[:,0], XYv[:,1], c=v, s=7, cmap="gray", edgecolors="none")
        if invert_y: ax.invert_yaxis()
        ax.set_aspect("equal","box"); ax.axis("off")
        ax.set_title(f"Prototype {k+1}")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    for j in range(k+1, len(axes)): axes[j].axis("off")
    plt.tight_layout(); plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def plot_kmeans_similarity_gridded(
    adata,
    obsm_key="X_morphology",
    K=3,
    grid_size="auto",                 # ("auto" or (H, W)) – use "auto" to avoid blank images
    target_cells_per_pixel=5.0,       # for "auto": average cells per pixel
    seed=0,
    n_init="auto",
    cmap="gray",
    origin="upper"                    # use "lower" if you prefer Cartesian
):
    # ------------------ 1) data & soft assignment (same as before) ------------------
    X = adata.obsm[obsm_key].astype(np.float32)
    XY = adata.obsm["spatial"][:, :2].astype(np.float32)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(XY).all(axis=1)
    Xv, XYv = X[mask], XY[mask]

    Xv = StandardScaler().fit_transform(Xv)
    km = KMeans(n_clusters=K, random_state=seed, n_init=n_init).fit(Xv)
    C  = km.cluster_centers_

    d2   = np.sum((Xv[:, None, :] - C[None, :, :])**2, axis=2)
    tau2 = np.median(d2) + 1e-8
    P    = np.exp(-d2 / (2 * tau2))
    P   /= (P.sum(axis=1, keepdims=True) + 1e-12)  # (N, K)

    # ------------------ 2) choose grid size ------------------
    x_min, y_min = XYv.min(axis=0)
    x_max, y_max = XYv.max(axis=0)
    width  = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    aspect = width / height

    if grid_size == "auto":
        N = XYv.shape[0]
        # want H*W ≈ N / target_cells_per_pixel, preserving aspect
        total_pixels = max(int(N / max(target_cells_per_pixel, 1e-3)), 1)
        W = max(int(np.sqrt(total_pixels * aspect)), 8)
        H = max(int(total_pixels // max(W, 1)), 8)
    else:
        H, W = grid_size

    # ------------------ 3) map cells to grid pixels ------------------
    gx = ((XYv[:, 0] - x_min) / width  * (W - 1)).astype(np.int32)
    gy = ((XYv[:, 1] - y_min) / height * (H - 1)).astype(np.int32)
    gx = np.clip(gx, 0, W - 1)
    gy = np.clip(gy, 0, H - 1)
    flat = gy * W + gx  # (N,)

    # ------------------ 4) accumulate sums & counts, then average ------------------
    counts = np.zeros(H * W, dtype=np.int32)
    np.add.at(counts, flat, 1)

    sums = np.zeros((K, H * W), dtype=np.float32)
    for k in range(K):
        np.add.at(sums[k], flat, P[:, k].astype(np.float32))

    grids = np.full((K, H * W), np.nan, dtype=np.float32)
    nz = counts > 0
    grids[:, nz] = sums[:, nz] / counts[nz]

    grid_maps = np.transpose(grids.reshape(K, H, W), (1, 2, 0))  # (H, W, K)
    valid_mask = nz.reshape(H, W)

    # ------------------ 5) plot ------------------
    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    fill_ratio = valid_mask.mean()
    print(f"Grid size: {H}×{W} | filled pixels: {fill_ratio*100:.1f}%")

    for k in range(K):
        ax = axes[k]
        ch = grid_maps[..., k]
        # robust range ignoring NaNs; fallback if too sparse
        if np.isfinite(ch).sum() > 10:
            vmin = np.nanpercentile(ch, 1)
            vmax = np.nanpercentile(ch, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = np.nanmin(ch), np.nanmax(ch)
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(ch, cmap=cmap, origin=origin, interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Prototype {k+1}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(k + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return grid_maps, valid_mask, (H, W)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------- tiny edge-preserving smoother on the grid ----------
def edge_preserving_smooth_grid(grid_maps, valid_mask, iterations=2,
                                edge_percentile=75, center_weight=2.0,
                                use_diagonals=True):
    """
    grid_maps: (H, W, K) with NaNs where empty
    valid_mask: (H, W) bool
    Returns smoothed (H, W, K) in-place copy.
    """
    H, W, K = grid_maps.shape
    out = grid_maps.copy()

    # neighbor shifts (dy, dx)
    nbr4 = [(-1,0),(1,0),(0,-1),(0,1)]
    nbr8 = nbr4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    nbrs = nbr8 if use_diagonals else nbr4

    # helper: shift array with NaN padding
    def shift(a, dy, dx):
        b = np.full_like(a, np.nan)
        y0 = max(0, dy); y1 = H + min(0, dy)
        x0 = max(0, dx); x1 = W + min(0, dx)
        b[y0:y1, x0:x1] = a[y0-dy:y1-dy, x0-dx:x1-dx]
        return b

    vm = valid_mask

    for k in range(K):
        ch = out[..., k]
        cm = vm & np.isfinite(ch)

        # global robust scale for differences (controls edge stopping)
        diffs = []
        for dy, dx in [( -1,0),(1,0),(0,-1),(0,1)]:  # use 4-neigh for scale
            n = shift(ch, dy, dx)
            m = cm & np.isfinite(n)
            if m.any():
                diffs.append(np.abs(ch[m] - n[m]))
        if len(diffs) == 0:
            continue
        diffs = np.concatenate(diffs)
        sigma = np.percentile(diffs, edge_percentile) + 1e-8  # smaller -> stronger edge keeping

        for _ in range(iterations):
            num = center_weight * ch
            den = center_weight * np.ones_like(ch)

            for dy, dx in nbrs:
                n = shift(ch, dy, dx)
                m = cm & np.isfinite(n)
                w = np.zeros_like(ch)
                # bilateral-like weight by intensity difference
                w[m] = np.exp(-((ch[m] - n[m])**2) / (sigma**2))
                num += w * np.nan_to_num(n, nan=0.0)
                den += w

            # update only where center valid
            upd = num / np.maximum(den, 1e-12)
            ch[cm] = upd[cm]

        out[..., k] = ch

    return out

# ---------- your pipeline: same phenotype, then grid, then smooth ----------
def plot_kmeans_similarity_gridded_coherent(
    adata,
    obsm_key="X_morphology",
    K=3,
    grid_size="auto",               # or (H, W)
    target_cells_per_pixel=2.0,     # used when grid_size="auto"
    seed=0,
    n_init="auto",
    cmap="gray",
    origin="upper",
    smooth=True,
    smooth_iters=2,
    edge_percentile=75,             # 60–80 keeps boundaries; lower => stricter edges
    center_weight=2.0
):
    X = adata.obsm[obsm_key].astype(np.float32)
    XY = adata.obsm["spatial"][:, :2].astype(np.float32)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(XY).all(axis=1)
    Xv, XYv = X[mask], XY[mask]

    # phenotype (unchanged)
    Xv = StandardScaler().fit_transform(Xv)
    km = KMeans(n_clusters=K, random_state=seed, n_init=n_init).fit(Xv)
    C  = km.cluster_centers_
    d2 = np.sum((Xv[:, None, :] - C[None, :, :])**2, axis=2)
    tau2 = np.median(d2) + 1e-8
    P = np.exp(-d2 / (2 * tau2))
    P /= (P.sum(axis=1, keepdims=True) + 1e-12)  # (N, K)

    # choose grid
    x_min, y_min = XYv.min(axis=0); x_max, y_max = XYv.max(axis=0)
    width  = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    aspect = width / height
    if grid_size == "auto":
        N = XYv.shape[0]
        total_pixels = max(int(N / max(target_cells_per_pixel, 1e-3)), 1)
        W = max(int(np.sqrt(total_pixels * aspect)), 16)
        H = max(int(total_pixels // max(W, 1)), 16)
    else:
        H, W = grid_size

    # bin to grid (average)
    gx = ((XYv[:, 0] - x_min) / width  * (W - 1)).astype(np.int32)
    gy = ((XYv[:, 1] - y_min) / height * (H - 1)).astype(np.int32)
    gx = np.clip(gx, 0, W - 1); gy = np.clip(gy, 0, H - 1)
    flat = gy * W + gx
    counts = np.zeros(H * W, dtype=np.int32)
    np.add.at(counts, flat, 1)
    sums = np.zeros((K, H * W), dtype=np.float32)
    for k in range(K): np.add.at(sums[k], flat, P[:, k].astype(np.float32))
    grids = np.full((K, H * W), np.nan, dtype=np.float32)
    nz = counts > 0
    grids[:, nz] = sums[:, nz] / counts[nz]
    grid_maps = np.transpose(grids.reshape(K, H, W), (1, 2, 0))  # (H, W, K)
    valid_mask = nz.reshape(H, W)

    # edge-preserving smoothing on the grid
    if smooth:
        grid_maps = edge_preserving_smooth_grid(
            grid_maps, valid_mask,
            iterations=smooth_iters,
            edge_percentile=edge_percentile,
            center_weight=center_weight,
            use_diagonals=True
        )

    # plot
    ncols = min(3, K); nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()
    print(f"Grid size: {H}×{W}  |  filled: {valid_mask.mean()*100:.1f}%")

    for k in range(K):
        ax = axes[k]
        ch = grid_maps[..., k]
        if np.isfinite(ch).sum() > 10:
            vmin = np.nanpercentile(ch, 1); vmax = np.nanpercentile(ch, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = np.nanmin(ch), np.nanmax(ch)
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(ch, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"Prototype {k+1}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(k+1, len(axes)): axes[j].axis("off")
    plt.tight_layout(); plt.show()

    return grid_maps, valid_mask, (H, W)
# ---------------- Usage ----------------
# Path to your morphology HDF5 (same folder as cell_feature_matrix.h5)
datasets=['Xenium_V1_FFPE_TgCRND8_2_5_months','Xenium_V1_FFPE_TgCRND8_5_7_months','Xenium_V1_FFPE_TgCRND8_17_9_months',
          'Xenium_V1_FFPE_wildtype_2_5_months','Xenium_V1_FFPE_wildtype_5_7_months','Xenium_V1_FFPE_wildtype_13_4_months']
for dataset in datasets:
    morpho_h5 = f"/media/huifang/data/Xenium/xenium_data/{dataset}/cell_features.h5"  # <-- adjust if different
    result = get_combined_data(dataset)
    # If your AnnData is named `result`:
    adata = fuse_morphology_into_adata(
        result,
        morpho_h5_path=morpho_h5,
        obsm_key="X_morphology",
        target_key="morphology_target",
        save_path=None,  # auto path next to H5
    )



    plot_kmeans_similarity(adata,K=3)
    plot_kmeans_similarity_gridded(
        adata,
        obsm_key="X_morphology",
        K=3,
        grid_size=(256, 256),  # or (2000, 2000) if you insist
        target_cells_per_pixel=0.9,  # increase to coarsen further
        seed=0
    )

    # grid_maps, valid_mask, (H, W) = plot_kmeans_similarity_gridded_coherent(
    #     adata,
    #     obsm_key="X_morphology",  # your 128-D SimCLR features
    #     K=3,  # number of prototypes to visualize
    #     grid_size="auto",  # auto-choose grid so it’s not too sparse
    #     target_cells_per_pixel=2.0,  # ~2 cells per pixel on average
    #     seed=0,
    #     smooth=True,  # turn on light edge-preserving smoothing
    #     smooth_iters=2,  # 2–3 is usually good
    #     edge_percentile=75,  # lower → sharper boundaries (e.g., 65–70)
    #     center_weight=2.0,  # higher → stays closer to original values
    #     cmap="gray",
    #     origin="upper"  # image-style top-left origin
    # )