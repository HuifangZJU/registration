import matplotlib.pyplot as plt
import scanpy as sc
from skimage.restoration import denoise_tv_chambolle
import os
import cv2
import imageio.v2 as imageio
import numpy as np
from sklearn.decomposition import PCA

def donwsample_adata(adata,dsrate):
    n_sub = max(1, int(adata.n_obs / dsrate))
     # deterministic: evenly spaced indices
    step = max(1, int(np.floor(adata.n_obs / n_sub)))
    subset_idx = np.arange(0, adata.n_obs, step)[:n_sub]

    adata_sub = adata[subset_idx].copy()
    return adata_sub



def img_processing(adata):
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def paste_processing(adata):

    return adata

def gpsa_processing(adata):
    adata = donwsample_adata(adata, 2)
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=10)
    # 2. Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def santo_processing(adata):
    adata = donwsample_adata(adata, 4)
    return adata


def get_xenium_data(case):
    data_dir = "/media/huifang/data/registration/xenium/cell_typing/joint_domains"
    sample_ids = [["ILC", "ILC_addon"]]
    groups = []
    for group_data in sample_ids:
        group = []
        for sid in group_data:
            fn = os.path.join(data_dir, f"xenium_{sid}_domains_shared.h5ad")
            adata = sc.read_h5ad(fn)
            if case == "img":
                adata = img_processing(adata)
            if case == "paste":
                adata = paste_processing(adata)
            if case == "gpsa":
                adata = gpsa_processing(adata)
            if case == "santo":
                adata = santo_processing(adata)
            group.append(adata)
        groups.append(group)
    return groups


cases=['img','paste','gpsa','santo']
for case in cases:
    print(case)
    groups = get_xenium_data(case)
    for k,slices in enumerate(groups):
        for i, sl in enumerate(slices):
            labels = sl.obs["marker_id"].astype("category").cat.codes.to_numpy()
            labels = np.asarray(labels)
            print(len(labels))
            np.savez(
                f"/media/huifang/data/registration/breast/{k}_{i}_marker_label_{case}", label=labels)


