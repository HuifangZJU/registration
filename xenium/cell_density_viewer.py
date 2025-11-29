import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd

def generate_pseudo_tissue_image():
    # --- Assume you already have coordinates ---
    coords = result.obs[['centroid_x', 'centroid_y']].to_numpy()


    # --- Step 1: Define output image size ---
    img_size = (1024, 1024)

    # --- Step 2: Compute the spatial range ---
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # --- Step 3: Map coordinates to pixel indices ---
    x_scaled = ((coords[:, 0] - x_min) / (x_max - x_min) * (img_size[1] - 1)).astype(int)
    y_scaled = ((coords[:, 1] - y_min) / (y_max - y_min) * (img_size[0] - 1)).astype(int)

    # --- Step 4: Initialize and fill the density map ---
    density = np.zeros(img_size, dtype=np.float32)
    for x, y in zip(x_scaled, y_scaled):
        density[y, x] += 1  # Note: (y, x) order

    # --- Step 5: Smooth slightly for nicer visualization (optional) ---
    from scipy.ndimage import gaussian_filter
    density_smooth = gaussian_filter(density, sigma=1)

    # --- Step 6: Normalize brightness ---
    density_norm = density_smooth / np.percentile(density_smooth, 99)  # normalize to 99th percentile
    density_norm = np.clip(density_norm, 0, 1)

    # --- Step 7: Show and save ---
    plt.figure(figsize=(8, 8))
    plt.imshow(density, cmap='gray', origin='upper')
    plt.axis('off')
    plt.title(f"Pseudo Cell Density Image: {img_size}", fontsize=14)
    plt.show()







    # --- Step 8: Plot histogram (with numerical labels) ---
    counts = density.flatten()
    counts_nonzero = counts[counts > 0]  # only non-empty grids

    fig, ax = plt.subplots(figsize=(7, 4))
    n, bins, patches = ax.hist(counts_nonzero, bins=50, color='steelblue', edgecolor='black')

    # Annotate each bar with its count
    for i in range(len(n)):
        if n[i] > 0:
            ax.text(
                (bins[i] + bins[i + 1]) / 2,  # x position (center of bar)
                n[i],  # y position (height)
                f"{int(n[i])}",  # text (integer count)
                ha='center', va='bottom', fontsize=8, rotation=90
            )

    ax.set_xlabel("Cells per grid (pixel)")
    ax.set_ylabel("Number of grids")
    ax.set_title("Distribution of Cell Counts per Grid", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


datasets=['Xenium_V1_FFPE_TgCRND8_2_5_months','Xenium_V1_FFPE_TgCRND8_5_7_months','Xenium_V1_FFPE_TgCRND8_17_9_months',
          'Xenium_V1_FFPE_wildtype_2_5_months','Xenium_V1_FFPE_wildtype_5_7_months','Xenium_V1_FFPE_wildtype_13_4_months']
for dataset in datasets:
    # --- Load data ---
    result = sc.read_10x_h5(
        f"/media/huifang/data/Xenium/xenium_data/{dataset}/cell_feature_matrix.h5"
    )
    centroids_file = (
        f"/media/huifang/data/Xenium/xenium_data/{dataset}/preprocessing/cell_centroids.csv"
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

    generate_pseudo_tissue_image()


#
# if "centroid_x" in result.obs.columns and "centroid_y" in result.obs.columns:
#     result.obsm["spatial"] = result.obs[["centroid_x", "centroid_y"]].to_numpy()
#
# # --- Step 2: Optional — clean obs column names ---
# # Visium usually has simpler metadata (optional)
# result.obs = result.obs.rename(columns={
#     "centroid_x": "x",
#     "centroid_y": "y",
#     "cell_type": "cell_type"
# })
#
# # --- Step 3: Add basic metadata for consistency ---
# result.uns["spatial"] = {
#     "library_id": {
#         "images": {},
#         "scalefactors": {
#             "spot_diameter_fullres": 1.0,  # dummy for compatibility
#             "tissue_hires_scalef": 1.0,
#             "tissue_lowres_scalef": 1.0,
#         }
#     }
# }
#
# # --- Step 4: Save to .h5ad file ---
# output_path = "/media/huifang/data/Xenium/xenium_data/Xenium_mouse_brain_visium_style.h5ad"
# result.write(output_path)
# print(f"✅ Saved Visium-style AnnData to:\n{output_path}")