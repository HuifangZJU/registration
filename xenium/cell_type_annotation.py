import scanpy as sc
import numpy as np
import pandas as pd
import celltypist
from celltypist import models
from sklearn.neighbors import NearestNeighbors
import os
import anndata as ad
from matplotlib import pyplot as plt
ad.settings.allow_write_nullable_strings = True
# -----------------------------
# Helper: get 2D spatial coords
# -----------------------------

def build_proba_from_ct_proba(adata,use_proba=True):

    if use_proba:
        P = np.asarray(adata.obsm["ct_proba"], dtype=float)  # (n_cells, C_full)
        # 1) Get class names for columns, if we saved them earlier.
        all_classes = np.array(list(adata.uns["ct_classes"]))

        classes = np.array(sorted(pd.unique(adata.obs["ct_celltypist"].astype(str))))

        # 4) Align P to desired_classes order (add zero columns if a desired class was not in ct_proba)
        label2col = {lab: j for j, lab in enumerate(all_classes)}
        proba = np.zeros((adata.n_obs, len(classes)), float)
        for k, c in enumerate(classes):
            j = label2col.get(c, None)
            proba[:, k] = P[:, j]
        # optional safety: renormalize rows (they often already sum≈1)
        rowsum = proba.sum(axis=1, keepdims=True)
        nz = rowsum.squeeze() > 0
        proba[nz] /= rowsum[nz]
    else:
        classes = np.array(sorted(adata.obs["ct_celltypist"].unique()))
        proba = np.zeros((adata.n_obs, len(classes)), float)
        label_to_col = {c: i for i, c in enumerate(classes)}
        for i, lab in enumerate(adata.obs["ct_celltypist"].astype(str).values):
            proba[i, label_to_col[lab]] = 1.0
    return classes, proba




# for sid in ["Tg_2.5m", "Tg_5.7m", "Tg_17.9m", "WT_2.5m", "WT_5.7m", "WT_13.4m"]:
for sid in ["ILC", "ILC_addon"]:
    fn = os.path.join("/media/huifang/data/registration/xenium/cell_typing", f"xenium_{sid}_scvi.h5ad")
    adata = sc.read_h5ad(fn)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_cells(adata, min_counts=100)  # tune if needed
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---------- 3A) Quick cell types via CellTypist ----------
    # model = models.Model.load(model="Mouse_Whole_Brain.pkl")  # placeholder; pick the exact mouse brain model you want
    model = models.Model.load(model="Cells_Adult_Breast.pkl")  # placeholder; pick the exact mouse brain model you want
    pred = celltypist.annotate(adata, model=model, majority_voting=True, p_thres=0.3)
    adata.obs["ct_celltypist"] = pred.predicted_labels.iloc[:, 0]
    # optional confidence for smoothing later
    proba_df = pred.probability_matrix
    common = adata.obs_names.intersection(proba_df.index)
    adata.obsm["ct_proba"] = proba_df.loc[common].reindex(adata.obs_names, fill_value=0).to_numpy()
    adata.uns["ct_classes"] = list(proba_df.columns)

    # coords = adata.obsm["spatial"]
    # labels = adata.obs["ct_celltypist"]
    # plt.scatter(coords[:, 0], coords[:, 1], c=labels.cat.codes, cmap="tab20", s=2)
    # plt.show()

    coords = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=8).fit(coords)
    nn = nbrs.kneighbors(return_distance=False)

    classes, proba = build_proba_from_ct_proba(adata)  # returns (C-names, n_cells x C) aligned
    smooth = np.zeros_like(proba)
    for i in range(adata.n_obs):
        smooth[i] = 0.5 * proba[nn[i]].mean(axis=0) + 0.5 * proba[i]
    # write smoothed labels
    adata.obs["ct_celltypist_smooth"] = pd.Series(classes[smooth.argmax(axis=1)], index=adata.obs_names)
    # make categorical (important!)
    adata.obs["ct_celltypist_smooth"] = (
        adata.obs["ct_celltypist_smooth"].astype("string").fillna("Unknown").astype("category")
    )
    # # --- plotting (simple) ---
    labels = adata.obs["ct_celltypist_smooth"]
    codes = labels.cat.codes.to_numpy()  # categorical -> integer codes
    plt.scatter(coords[:, 0], coords[:, 1], c=codes, cmap="tab20", s=2)
    plt.show()

    fn = os.path.join("/media/huifang/data/registration/xenium/cell_typing", f"xenium_{sid}_scvi_celltypist.h5ad")
    adata.write_h5ad(fn)
    print(f"Saved: {fn}")



