import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from functools import reduce
import cv2
from skimage.restoration import denoise_tv_chambolle
import os
import scipy.sparse as sp


def extract_trailing_digits(s):
    out = pd.Series(s).astype(str).str.extract(r'(\d+)$')[0]
    return out

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
    result.obs_names = result.obs_names.astype(str)
    centroids_data = centroids_data.copy()
    centroids_data.index = centroids_data.index.astype(str)

    # (optional but safe) drop duplicate ids in centroids, keep first
    centroids_data = centroids_data[~centroids_data.index.duplicated(keep="first")]

    # 2) Find common ids (order follows result.obs_names)
    common_ids = result.obs_names.intersection(centroids_data.index)
    print(f"Matched {len(common_ids)} cells")

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
    # base_dir = f"/media/huifang/data/Xenium/xenium_data/{data}/"
    # save_path = os.path.join(base_dir, "xenium_annodata.h5ad")
    # result.write(save_path)
    # print('saved')

    return result
from sklearn.decomposition import PCA
def get_gene_feature_matrix(coords: np.ndarray,
                            reduced_data: np.ndarray,
                            image_size=(2016, 2016),
                            patch_size=32):            # how many smoothing passes if local

    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    n_spots, n_dims = reduced_data.shape
    height, width = image_size
    out_height = height // patch_size
    out_width  = width  // patch_size

    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) bin spots to patches
    for i in range(n_spots):
        x, y = coords[i]  # expected pixel coords
        px = int(x) // patch_size
        py = int(y) // patch_size
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]
            count_array[py, px] += 1

    # 2) per-patch averages where we have data
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = sum_array[valid_mask, :] / count_array[valid_mask, None]
    return patch_matrix, valid_mask


def get_gene_feature_matrix_soft(
    coords: np.ndarray,
    reduced_data: np.ndarray,
    image_size =(2016, 2016),
    patch_size: int = 32,
    kernel: str = "gaussian",         # "gaussian" or "inverse_distance"
    sigma: float = None,              # if None, defaults to patch_size / 2 for gaussian
    eps: float = 1e-6,                # small constant for inverse-distance
    normalize_per_cell: bool = True   # normalize 9 weights to sum=1 per cell
):
    """
    Soft-assign each cell's vector to its 3x3 neighborhood of grid patches,
    with weights based on distance to patch centers.

    Returns:
        patch_matrix: (H_out, W_out, n_dims) averaged features per patch
        valid_mask:   (H_out, W_out) mask where weight > 0
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    n_spots, n_dims = reduced_data.shape
    height, width = image_size
    out_height = height // patch_size
    out_width  = width  // patch_size

    if kernel not in {"gaussian", "inverse_distance"}:
        raise ValueError("kernel must be 'gaussian' or 'inverse_distance'.")

    if sigma is None and kernel == "gaussian":
        sigma = patch_size / 2.0

    # Accumulators
    sum_array   = np.zeros((out_height, out_width, n_dims), dtype=np.float64)
    weight_sum  = np.zeros((out_height, out_width), dtype=np.float64)

    # Precompute neighbor offsets (3x3 neighborhood)
    nbr_offsets = [(-1,-1), (-1,0), (-1,1),
                   ( 0,-1), ( 0,0), ( 0,1),
                   ( 1,-1), ( 1,0), ( 1,1)]

    for i in range(n_spots):
        x, y = coords[i]  # pixel coordinates (float)
        if not (0 <= x < width and 0 <= y < height):
            continue  # skip cells outside the image bounds

        # Patch indices of the cell's containing grid
        px = int(x) // patch_size
        py = int(y) // patch_size

        # Collect neighbor centers and distances
        neigh_idxs = []
        dists = []

        for dy, dx in nbr_offsets:
            ny = py + dy
            nx = px + dx
            if 0 <= nx < out_width and 0 <= ny < out_height:
                # center of the neighbor patch
                cx = (nx + 0.5) * patch_size
                cy = (ny + 0.5) * patch_size
                dist = np.hypot(x - cx, y - cy)
                neigh_idxs.append((ny, nx))
                dists.append(dist)

        if not neigh_idxs:
            continue

        dists = np.asarray(dists, dtype=np.float64)

        # Compute weights
        if kernel == "gaussian":
            # w = exp(-d^2 / (2*sigma^2))
            denom = 2.0 * (sigma ** 2) + 1e-12
            w = np.exp(-(dists ** 2) / denom)
        else:  # inverse_distance
            w = 1.0 / (dists + eps)

        if normalize_per_cell:
            s = w.sum()
            if s > 0:
                w = w / s

        # Accumulate
        v = reduced_data[i].astype(np.float64)
        for (ny, nx), ww in zip(neigh_idxs, w):
            if ww <= 0:
                continue
            sum_array[ny, nx, :] += ww * v
            weight_sum[ny, nx]   += ww

    # Average per patch where weight > 0
    patch_matrix = np.zeros_like(sum_array, dtype=np.float64)
    valid_mask = weight_sum > 0
    patch_matrix[valid_mask, :] = sum_array[valid_mask, :] / weight_sum[valid_mask, None]

    # Cast back to float32 if you prefer
    patch_matrix = patch_matrix.astype(np.float32)

    return patch_matrix, valid_mask


from scipy.ndimage import median_filter
def remove_salt_pepper(feature_matrix, size=4):
    """
    Apply a median filter to remove isolated spikes in each channel.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (H, W, C)
        Input array with possible single-pixel high/low outliers.
    size : int
        Size of the filtering window. Usually 3 or 5.

    Returns
    -------
    np.ndarray
        Filtered array of the same shape.
    """
    filtered = np.zeros_like(feature_matrix)
    for c in range(feature_matrix.shape[2]):
        filtered[..., c] = median_filter(feature_matrix[..., c], size=size)
    return filtered

def preprocess_feature_matrix(feature_matrix):
    """
        Preprocess a 64x64x10 feature matrix using bilateral filtering
        (to preserve edges) and mild total variation (TV) denoising.

        Parameters
        ----------
        feature_matrix : np.ndarray, shape (64, 64, 10)
            Input array with values in [0, 1].

        Returns
        -------
        np.ndarray
            Preprocessed array of the same shape (64, 64, 10).
        """
    # assert feature_matrix.shape == (64, 64, 10), "Expected shape (64,64,10)."
    # assert 0 <= feature_matrix.min() and feature_matrix.max() <= 1, \
    #     "Values should be in [0,1]."
    if feature_matrix.max()>1:
        feature_matrix = (feature_matrix-feature_matrix.min())/feature_matrix.max()
    # We'll store our results in a new array
    preprocessed = np.zeros_like(feature_matrix)


    # To apply the OpenCV bilateral filter, we need 8-bit or float32
    # We'll convert each slice to float32 [0,1] for processing
    for i in range(feature_matrix.shape[2]):
        channel = feature_matrix[..., i].astype(np.float32)

        # ---------------------------------------------------------------------
        # 1. Bilateral filtering for edge-preserving smoothing
        # d: Filter diameter (pixel neighborhood).
        # sigmaColor: Larger value -> more influence of intensity difference.
        # sigmaSpace: Larger value -> more influence of distant pixels.
        # Try adjusting these parameters to see how strongly edges are preserved.
        # ---------------------------------------------------------------------
        #feature paras
        channel_bilat = cv2.bilateralFilter(
            channel,  # source image
            d=10,  # diameter of the pixel neighborhood
            sigmaColor=0.1,  # range sigma for color
            sigmaSpace=25  # range sigma for spatial distance
        )



        # Convert back to numpy float64 to feed into TV denoising
        channel_bilat = channel_bilat.astype(np.float64)

        # ---------------------------------------------------------------------
        # 2. Mild Total Variation denoising
        # weight: controls amount of denoising; too high can soften edges
        # For strong edge preservation, keep this small.
        # ---------------------------------------------------------------------
        channel_denoised = denoise_tv_chambolle(
            channel_bilat,
            weight=0.02,  # small weight for gentle smoothing
            eps=1e-4,
        )

        preprocessed[..., i] = channel_denoised
    return preprocessed

def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    ncols = 5
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")

    plt.tight_layout()
    plt.show()

def reduce_gene_reads(gene_reads: np.ndarray, method: str = 'pca', n_components: int = 10) -> np.ndarray:

    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")

    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(gene_reads)

    elif method.lower() == 'umap':
        # UMAP
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(gene_reads)
    else:
        raise ValueError("method must be one of ['pca', 'umap'].")

    return reduced_data

def channelwise_min_max_normalize(data):
    """
    data: shape (N, C)
      N = number of samples (e.g., 17952)
      C = number of channels/features (e.g., 10)

    Returns:
      normalized_data of the same shape, where each channel is mapped to [0, 1]
      across the N samples.
    """
    mins = data.min(axis=0)   # shape (C,)
    maxs = data.max(axis=0)   # shape (C,)
    ranges = maxs - mins

    # Avoid division by zero (when all values in a channel are the same)
    ranges[ranges == 0] = 1e-8

    normalized_data = (data - mins) / ranges
    return normalized_data

root_folder="/media/huifang/data/registration/phenocycler/"
datasets=[['LUAD_2_A']]

groups=[]
for group_data in datasets:
    group=[]
    for data in group_data:
        base_dir = root_folder+data+'/visium'
        save_path = os.path.join(base_dir, "filtered_feature_bc_matrix.h5")
        adata = sc.read_10x_h5(save_path)
        print(adata)
        test = input()

        sc.pp.filter_cells(adata, min_genes=20)
        sc.pp.filter_genes(adata, min_cells=30)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=5)

        group.append(adata)
    groups.append(group)

for k,slices in enumerate(groups):

    all_gene_lists = [sl.var.index for sl in slices]
    common_genes = reduce(np.intersect1d, all_gene_lists)
    # 2. Subset each slice to the common genes, gather coordinates & data
    gene_data_list = []
    coords_list = []
    for i, sl in enumerate(slices):
        # Focus on common genes only
        # sl_sub = sl[:, common_genes]
        # Convert to a NumPy array
        gene_data = np.array(sl.X)  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        coords_list.append(sl.obsm['spatial'])

    # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)

    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)

    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        coords = coords_list[i]

        feature_matrix, gene_mask = get_gene_feature_matrix(coords, reduced_slice_data, (1024, 1024),
                                                            patch_size=8)
        # feature_matrix = preprocess_feature_matrix(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)

        plot_dimensional_images_side_by_side(feature_matrix)
