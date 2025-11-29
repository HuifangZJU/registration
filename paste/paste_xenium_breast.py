import pandas as pd
import scipy
import seaborn as sns
import os
import scanpy as sc
import paste as pst
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ot

def donwsample_adata(adata,dsrate):
    # assume your object is named `adata`
    frac = 1 / dsrate  # downsample ratio
    n_sub = max(1, int(adata.n_obs * frac))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # reproducible sampling
    rng = np.random.default_rng(seed=0)
    subset_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)

    adata_sub = adata[subset_idx].copy()
    return adata_sub

def resize_and_flip_image_coords(
    coords: np.ndarray,
    size,
    flip_para
):

    # Retrieve flip parameters for this image
    flip_lr, flip_ud = flip_para
    new_coords = coords.copy().astype(np.float32)

    # --- Step 3: Apply flips ---
    if flip_lr:
        new_coords[:, 0] = size[0] - 1 - new_coords[:, 0]

    if flip_ud:
        new_coords[:, 1] = size[1] - 1 - new_coords[:, 1]

    return new_coords
def get_xenium_data():
    data_dir = "/media/huifang/data/registration/xenium/cell_typing/joint_domains"
    sample_ids = [["ILC", "ILC_addon"]]

    groups = []
    for group_data in sample_ids:
        group = []
        for sid in group_data:
            fn = os.path.join(data_dir, f"xenium_{sid}_domains_shared.h5ad")
            adata = sc.read_h5ad(fn)
            # sc.pp.filter_cells(adata, min_counts=100)
            # sc.pp.filter_genes(adata, min_cells=10)
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # sc.pp.log1p(adata)
            group.append(adata)
        groups.append(group)
    return groups

groups = get_xenium_data()
visualization = True

pairs=[[0,1],[1,0]]

for group_id, slices in enumerate(groups):
    for pair in pairs:
        pair_id1 = pair[0]
        pair_id2 = pair[1]
        slice1 = slices[pair_id1]
        slice2 = slices[pair_id2]

        slice1_sub = donwsample_adata(slice1,100)
        slice2_sub = donwsample_adata(slice2,100)

        print('calculaing correlation')
        pi = pst.pairwise_align(slice1_sub, slice2_sub, alpha=0.1, backend=ot.backend.TorchBackend(), use_gpu=True)


        print('estimating R and T')
        _, _, rY, tX, tY = pst.visualization.generalized_procrustes_analysis(slice1_sub.obsm['spatial'], slice2_sub.obsm['spatial'], pi,
                                                          output_params=True, matrix=True)

        pts_slice1 = slice1.obsm['spatial']
        pts_slice2 = slice2.obsm['spatial']
        labels1 = slice1.obs["cluster_id"]
        labels2 = slice2.obs["cluster_id"]
        # plt.scatter(warped_slice1[:,0],warped_slice1[:,1],s=10)
        warped_slice1 = pts_slice1-tX
        # plt.scatter(test[:,0],test[:,1],s=5)
        # plt.show()
        #
        # plt.scatter(warped_slice2[:,0],warped_slice2[:,1],s=10)
        warped_slice2 = rY.dot((pts_slice2-tY).T).T
        # plt.scatter(test[:,0],test[:,1],s=5)
        # plt.show()



        # np.savez(f"/media/huifang/data/registration/result/xenium/breast/paste/{group_id}_{pair_id1}_{pair_id2}_result", pts1=warped_slice1,pts2=warped_slice2,label1=labels1.astype(int).to_numpy(),label2=labels2.astype(int).to_numpy())
        if visualization:
            # make sure labels are plain strings
            labels1_str = labels1.astype(str)
            labels2_str = labels2.astype(str)

            # build categories across both
            all_labels = pd.concat([labels1_str, labels2_str]).astype("category")
            categories = all_labels.cat.categories

            # build palette mapping
            import seaborn as sns

            palette = dict(zip(categories, sns.color_palette("tab20", len(categories))))

            # map to colors
            colors1 = labels1_str.map(palette)
            colors2 = labels2_str.map(palette)


            # Create subplots
            f, a = plt.subplots(1, 2, figsize=(20,10))

            # Scatter plot of original points
            a[0].scatter(
                pts_slice1[:, 0], pts_slice1[:, 1],
                c=colors1, label="Slice 1",
                alpha=0.6, marker="o" ,s=0.5 # circle
            )
            a[0].scatter(
                pts_slice2[:, 0], pts_slice2[:, 1],
                c=colors2, label="Slice 2",
                alpha=0.6, marker="^" ,s=0.5 # triangle
            )
            a[0].invert_yaxis()  # Match image coordinate system
            a[0].set_title("Original Points")
            a[0].set_aspect('equal')
            a[0].legend()

            # Scatter plot of warped (transformed) points
            a[1].scatter(warped_slice1[:, 0], warped_slice1[:, 1],c=colors1, label="Warped Slice 1", alpha=0.6, marker="o",s=0.5)
            a[1].scatter(warped_slice2[:, 0], warped_slice2[:, 1],c=colors2, label="Warped Slice 2", alpha=0.6, marker="^" ,s=0.5)
            a[1].invert_yaxis()
            a[1].set_title("Warped Points")
            a[1].set_aspect('equal')
            a[1].legend()

            plt.tight_layout()
            plt.show()