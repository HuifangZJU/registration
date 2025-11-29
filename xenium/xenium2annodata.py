import numpy as np
import scanpy as sc
import pandas as pd
import h5py
import os
import re
def normalize_to_str_index(idx):
    """Cast an index to clean string IDs like '1','2',... (no spaces)."""
    # 1) convert to string
    s = pd.Index(idx.astype(str) if isinstance(idx, pd.Index) else pd.Index(idx).astype(str))
    # 2) strip whitespace
    s = s.str.strip()
    # 3) optional: normalize leading zeros (e.g., '0003' -> '3')
    # comment this block out if your IDs intentionally keep zero padding
    s = s.map(lambda x: re.sub(r'^0+(\d+)$', r'\1', x))
    return s


def get_combined_data(data_name):
    # --- Load data ---
    result = sc.read_10x_h5(
        f"/media/huifang/data/Xenium/xenium_data/{data_name}/cell_feature_matrix.h5"
    )
    centroids_data = pd.read_csv(f"/media/huifang/data/Xenium/xenium_data/{data_name}/preprocessing/cell_centroids.csv")
    centroids_data = centroids_data.set_index("cell_id")
    # --- Align and merge ---

    result.obs_names = normalize_to_str_index(result.obs_names)

    # Make centroids index strings to match
    centroids_data.index = normalize_to_str_index(centroids_data.index)

    # Find overlapping cell IDs
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
    return result

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
    return adata

#
# datasets = ['Xenium_V1_FFPE_TgCRND8_2_5_months', 'Xenium_V1_FFPE_TgCRND8_5_7_months',
#              'Xenium_V1_FFPE_TgCRND8_17_9_months',
#             'Xenium_V1_FFPE_wildtype_2_5_months', 'Xenium_V1_FFPE_wildtype_5_7_months',
#              'Xenium_V1_FFPE_wildtype_13_4_months']
datasets=['Xenium_V1_FFPE_Human_Breast_ILC','Xenium_V1_FFPE_Human_Breast_ILC_With_Addon']
for data in datasets:
    print(data)
    result = get_combined_data(data)


    morpho_h5 = f"/media/huifang/data/Xenium/xenium_data/{data}/cell_features.h5"
    adata = fuse_morphology_into_adata(
        result,
        morpho_h5_path=morpho_h5,
        obsm_key="X_morphology",
        target_key="morphology_target",
        save_path=None,  # auto path next to H5
    )
    image_path = f"/media/huifang/data/Xenium/xenium_data/{data}/morphology_down10x.png"
    adata.uns["image_path"] = image_path

    base_dir = f"/media/huifang/data/Xenium/xenium_data/{data}/"
    save_path = os.path.join(base_dir, "xenium_annodata_with_morphology.h5ad")
    adata.write(save_path)
    print('done')
