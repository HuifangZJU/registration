import scanpy as sc
from pathlib import Path
import squidpy as sq
import matplotlib.pyplot as plt

import scanpy as sc
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smooth_clusters(adata, cluster_key="clusters_expr", k=6):
    """
    Spatially smooth cluster assignments with kNN majority vote.
    """
    coords = adata.obsm["spatial"]
    labels = adata.obs[cluster_key].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)[:, 1:]

    smoothed = []
    for i, neigh in enumerate(indices):
        neigh_labels = labels[neigh]
        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        smoothed.append(vals[np.argmax(counts)])
    adata.obs[cluster_key + "_smoothed"] = smoothed

def get_mouse_data(
    base_dir: str = "/media/huifang/data/registration/mouse",
    sample_groups = None,
) -> list[list[sc.AnnData]]:

    # ---------- defaults ----------
    sample_list = [
        "anterior_v1", "anterior_v2",
        "posterior_v1", "posterior_v2",
    ]
    sample_groups = [
        ["anterior_v1", "anterior_v2"],
        ["posterior_v1", "posterior_v2"],
    ]
    # ---------- load all adatas ----------
    adatas: dict[str, sc.AnnData] = {}
    for sid in sample_list:
        adata = sq.read.visium(Path(base_dir) / f"{sid}/")
        adata.var_names_make_unique()  # this modifies in-place
        adatas[sid] = adata

    # ---------- enrich with spatial info ----------
    for sid, ad in adatas.items():
        spatial_dir = Path(base_dir) / sid / "spatial"
        # --- put auxiliary files in .uns so they persist on disk ---
        ad.uns.setdefault("spatial_meta", {})
        ad.uns["spatial_meta"] = [
            str(spatial_dir / "tissue_hires_image_image_0.png"),
            str(spatial_dir / "scalefactors_json.json"),
        ]

    # ---------- assemble requested groups ----------
    layer_groups = [[adatas[sid] for sid in grp] for grp in sample_groups]
    return layer_groups

def visualization():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for i, (adata_slice, label, ax) in enumerate(zip(slices, ["v1", "v2"], axes)):
        mask = adata_joint.obs["slice_id"] == label
        coords = adata_joint.obsm["spatial"][mask, :]
        clusters = adata_joint.obs.loc[mask, "clusters_expr_smoothed"]

        scx = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=clusters.astype("category").cat.codes,
            cmap="tab20", s=10, alpha=0.8
        )
        ax.set_title(f"Slice {label}")
        ax.set_xlabel("X")
        if i == 0:
            ax.set_ylabel("Y")

        # enforce inverted y-axis
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)

    # shared legend
    handles, _ = scx.legend_elements()
    fig.legend(handles, sorted(adata_joint.obs["clusters_expr_smoothed"].unique()),
               title="Clusters", bbox_to_anchor=(1.05, 0.5), loc="center left")

    plt.tight_layout()
    plt.show()

def clean_uns(adata):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        elif obj is None:
            return None
        else:
            # fallback: cast to string
            return str(obj)
    adata.uns = sanitize(adata.uns)

layer_groups = get_mouse_data()
for k, slices in enumerate(layer_groups):

    # 1. Concatenate slices into one AnnData
    adata_joint = slices[0].concatenate(
        slices[1],
        batch_key="slice_id",  # store which slice each spot came from
        batch_categories=["v1", "v2"]
    )
    adata_joint.uns["spatial_meta"] = {
        "v1": slices[0].uns["spatial_meta"],
        "v2": slices[1].uns["spatial_meta"],
    }


    # 2. Preprocessing (expression only)
    sc.pp.normalize_total(adata_joint, target_sum=1e4)
    sc.pp.log1p(adata_joint)
    sc.pp.highly_variable_genes(adata_joint, flavor="seurat", n_top_genes=2000)
    adata_joint = adata_joint[:, adata_joint.var["highly_variable"]].copy()

    sc.pp.scale(adata_joint, max_value=10)
    sc.tl.pca(adata_joint, svd_solver="arpack")
    sc.pp.neighbors(adata_joint, use_rep="X_pca", n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_joint)

    # 3. Leiden clustering
    sc.tl.leiden(adata_joint, resolution=0.5, key_added="clusters_expr")
    adata_joint.obs["clusters_expr"] = adata_joint.obs["clusters_expr"].astype(str)

    # 4. Spatial smoothing within each slice
    for i, adata in enumerate(slices):
        label = adata_joint.obs["slice_id"].cat.categories[i]
        mask = adata_joint.obs["slice_id"] == label
        adata_slice = adata_joint[mask].copy()
        smooth_clusters(adata_slice, cluster_key="clusters_expr", k=6)
        # push smoothed labels back into the joint object
        adata_joint.obs.loc[mask, "clusters_expr_smoothed"] = adata_slice.obs["clusters_expr_smoothed"].values

    # Plot UMAP to see shared embedding
    sc.pl.umap(adata_joint, color=["clusters_expr", "clusters_expr_smoothed", "slice_id"], wspace=0.4)
    visualization()

    # Extract spatial coordinates separately for each slice (for plotting)
    coords = []
    for i, adata in enumerate(slices):
        mask = adata_joint.obs["slice_id"] == adata_joint.obs["slice_id"].cat.categories[i]
        coords.append(adata_joint.obsm["spatial"][mask, :])

    for i, slice_id in enumerate(adata_joint.obs["slice_id"].cat.categories):
        mask = adata_joint.obs["slice_id"] == slice_id
        adata_slice = adata_joint[mask].copy()

        # keep metadata again just in case
        adata_slice.uns["spatial_meta"] = slices[i].uns.get("spatial_meta", [])
        adata_slice.uns = {}

        adata_slice.write(
            f"/media/huifang/data/registration/mouse/huifang/h5ad/nouns/"
            f"group_{k}_slice_{slice_id}_clusters_smoothed.h5ad"
        )














    # for adata in slices:
    #     sc.pp.normalize_total(adata, target_sum=1e4)
    #     sc.pp.log1p(adata)
    #     sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    #     adata = adata[:, adata.var['highly_variable']].copy()
    #
    #
    #     # 2. Dimensionality reduction
    #     sc.pp.scale(adata, max_value=10)
    #     sc.tl.pca(adata, svd_solver="arpack")
    #
    #     # 3. Neighborhood graph and clustering
    #     sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    #     sc.tl.umap(adata)
    #     sc.tl.leiden(adata, resolution=1.0, key_added="clusters")
    #
    #     # 4. Plot clusters in UMAP space
    #     # sc.pl.umap(adata, color="clusters")
    #
    #     # 4. Store cluster labels in obs (string type is safer for plotting)
    #     adata.obs["clusters"] = adata.obs["clusters"].astype(str)
    #     # get spatial coordinates
    #     coords = adata.obsm["spatial"]  # shape (n_spots, 2)
    #     clusters = adata.obs["clusters"].astype(str)
    #
    #     # make a scatter plot of spots
    #     plt.figure(figsize=(6, 6))
    #     scatter = plt.scatter(
    #         coords[:, 0], coords[:, 1],
    #         c=clusters.astype("category").cat.codes,  # color by cluster
    #         cmap="tab20", s=10
    #     )
    #
    #     # invert y-axis so orientation matches tissue image
    #     plt.gca().invert_yaxis()
    #
    #     # add legend mapping cluster IDs to colors
    #     handles, _ = scatter.legend_elements()
    #     plt.legend(handles, sorted(clusters.unique()), title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    #
    #     plt.xlabel("X coordinate (pixels)")
    #     plt.ylabel("Y coordinate (pixels)")
    #     plt.title("Spatial distribution of clusters")
    #     plt.tight_layout()
    #     plt.show()