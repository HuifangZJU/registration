import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import json
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import SimpleITK as sitk
from scipy.spatial import cKDTree
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from functools import reduce
from scipy.spatial import cKDTree
from PIL import Image
# style.use('seaborn-white')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import squidpy as sq


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))
def create_overlay(fixed, moving):
    fixed = normalize(fixed)
    moving = normalize(moving)
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay



def get_gene_feature_matrix(coords: np.ndarray,
                                         reduced_data: np.ndarray,
                                         image_size=(2016, 2016),
                                         patch_size=32):
    """
    Given:
      - coords:        (N, 2) 2D positions (x, y) for each data point
      - reduced_data:  (N, D) data values at each coordinate (N points, D dims)
      - image_size:    (height, width) of the *original* large image
      - patch_size:    size of the patch to downsample into

    We'll create a patch grid of shape:
       out_height = image_size[0] // patch_size
       out_width  = image_size[1] // patch_size
      and accumulate data from reduced_data into that grid.

    Steps:
      1) For each point, compute which patch (px, py) it belongs to.
      2) Accumulate the reduced_data values into sum_array[py, px, :].
      3) Keep track of the number of points in each patch (count_array).
      4) patch_matrix = sum_array / count_array (elementwise), ignoring patches with zero count.
      5) Plot each dimension side by side using subplots.
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    # Number of points (N) and number of dimensions (D)
    n_spots, n_dims = reduced_data.shape

    # Image size
    height, width = image_size

    # Compute the shape of the patch matrix
    out_height = height // patch_size
    out_width  = width  // patch_size

    # We'll accumulate sums in sum_array and the count of points in count_array
    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) Assign each data point to its corresponding patch
    for i in range(n_spots):
        x, y = coords[i]  # e.g., coords might be (x, y) in [0..width, 0..height]
        px = int(x) // patch_size
        py = int(y) // patch_size

        # Check if we're within valid patch bounds
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]  # Accumulate the data
            count_array[py, px] += 1

    # 2) Compute the average (or keep as sum if you prefer) for each patch
    #    We'll avoid division by zero by clipping count_array
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = (
        sum_array[valid_mask, :] / count_array[valid_mask, np.newaxis]
    )

    return patch_matrix

def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray):
    print(patch_matrix.shape)
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    ncols = 8
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

def get_uv_coordinates(slice):
    scale_path = slice.uns["spatial_meta"]["scalefactor_path"]
    image = Image.open(slice.uns["spatial_meta"]["image_path"])
    with open(scale_path, 'r') as f:
        data = json.load(f)
        res_scale = data['tissue_hires_scalef']

    uv_coords = slice.obsm["spatial"] * res_scale

    return uv_coords, image


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

def crop_square_then_resize_square(
    img: Image.Image,
    original_uv: np.ndarray,
    crop_para,
    final_size
):
    # 1) Crop the square from the original image
    #    crop box = (left, top, right, bottom)
    left,top,side_length,_ = crop_para
    right  = left + side_length
    bottom = top  + side_length
    crop_box = (left, top, right, bottom)

    cropped_img = img.crop(crop_box)

    # 2) Resize the cropped square to (final_size, final_size)
    final_img = cropped_img.resize((final_size, final_size), Image.LANCZOS)

    # 3) Transform the coordinates
    # Step A: shift by subtracting (left, top)
    shifted_uv = original_uv - np.array([left, top])  # shape (N, 2)

    # Step B: scale factor for both x and y
    scale = final_size / side_length
    final_uv = shifted_uv * scale

    return final_img, final_uv




import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian


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
            d=5,  # diameter of the pixel neighborhood
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
            weight=0.01,  # small weight for gentle smoothing
            eps=1e-4,
        )

        preprocessed[..., i] = channel_denoised

    return preprocessed
from scipy.ndimage import median_filter
def remove_salt_pepper(feature_matrix, size=1):
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
    if len(feature_matrix.shape)==2:
        filtered = median_filter(feature_matrix, size=size)
    else:
        for c in range(feature_matrix.shape[2]):
            filtered[..., c] = median_filter(feature_matrix[..., c], size=size)
    return filtered


def save_DLPFC_to_file():
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}

    for id in sample_list:
        adatas[id].obs['position'].index = (
            adatas[id].obs['position'].index
            .str.replace(r"\.\d+$", "", regex=True)
        )
        position_prefix = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)
        try:
            # Try reading as CSV
            positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',')
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',')

        positions.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]
        # 1) Get the barcodes from AnnData that are in `positions`
        positions.index = positions["barcode"]
        adata_barcodes = adatas[id].obs['position'].index
        common_barcodes = adata_barcodes[adata_barcodes.isin(positions.index)]
        # 2) Now reindex `positions` in the exact order of `common_barcodes`
        positions_filtered = positions.reindex(common_barcodes)

        spatial_locations = positions_filtered[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()

        adatas[id].obsm['spatial'] = spatial_locations
        adatas[id].write("/media/huifang/data/registration/humanpilot/precessed_feature_matrix/"+id+".h5ad")
    print("all saved")

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from sklearn.cluster import AgglomerativeClustering


def cluster_svg_modules(slices, K=64):
    # 1. Get union of top genes (svg) and intersect of all available genes
    top_genes_union = set()
    all_varnames = [set(sl.var_names) for sl in slices]

    for sl in slices:
        top_genes_union.update(sl.uns['svgs'])

    intersect_genes = set.intersection(*all_varnames)
    selected_genes = sorted(top_genes_union.intersection(intersect_genes))

    # 2. Concatenate expression across slices
    all_expr = []
    for sl in slices:
        expr = sl[:, selected_genes].X.toarray() if hasattr(sl[:, selected_genes].X, 'toarray') else sl[:, selected_genes].X
        all_expr.append(expr)
    all_expr = np.vstack(all_expr)

    # 3. Cluster based on gene-gene Pearson correlation
    corr = np.corrcoef(all_expr.T)
    labels = AgglomerativeClustering(n_clusters=K, linkage='average').fit_predict(1 - corr)

    # 4. Assign module scores per slice
    for sl in slices:
        for k in range(K):
            gene_group = np.array(selected_genes)[labels == k].tolist()
            sc.tl.score_genes(sl, gene_group, score_name=f"module_{k}",ctrl_as_ref=False)
        # sc.pl.spatial(sl, color=[f"module_{k}" for k in range(K)],
        #               ncols=8, spot_size=1.5, img_key=None)

    # 5. Return the module expression matrices per slice (shape: [spots, K])
    feature_maps = {}
    for i, sl in enumerate(slices):
        maps = np.vstack([sl.obs[f"module_{k}"].values for k in range(K)]).T
        feature_maps[f"slice_{i}"] = maps

    return feature_maps





def get_heart_data():
    layer_groups=[]
    for slice_id in ['3d','7d','14d','21d']:
        adata = sc.read_h5ad(f"/media/huifang/data/registration/heart/stomics/{slice_id}/10x_processed.h5ad")
        adata_in = adata[adata.obs['in_tissue'] == 1].copy()
        layer_groups.append(adata_in)
    return layer_groups



slices = get_heart_data()

gene_data_list = []
coords_list = []

for i, sl in enumerate(slices):

    sc.pp.filter_cells(sl, min_genes=200)
    sc.pp.filter_genes(sl, min_cells=3)
    sc.pp.normalize_total(sl, target_sum=1e4)
    sc.pp.log1p(sl)
    sc.pp.highly_variable_genes(sl, flavor='seurat_v3', n_top_genes=3000)
    sl = sl[:, sl.var['highly_variable']]
    sc.pp.scale(sl, max_value=5)

all_gene_lists = [sl.var.index for sl in slices]

common_genes = reduce(np.intersect1d, all_gene_lists)
# 2. Subset each slice to the common genes, gather coordinates & data
gene_data_list = []
coords_list = []
for i, sl in enumerate(slices):
    # Focus on common genes only
    sl_sub = sl[:, common_genes]
    # Convert to a NumPy array
    gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
    gene_data_list.append(gene_data)
    # Extract coordinates from the slice

    coords = sl.obsm['spatial']
    coords_list.append(coords)
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
    reduced_slice_data = channelwise_min_max_normalize(reduced_slice_data)
    index_start = index_end
    # Get the corresponding coordinates
    coords = coords_list[i]



    feature_matrix = get_gene_feature_matrix(coords, reduced_slice_data, (2048, 2048), patch_size=32)
    # feature_matrix = remove_salt_pepper(feature_matrix)
    valid_mask = (feature_matrix > 0)
    gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)
    gene_mask = remove_salt_pepper(gene_mask,size=2)



    # plt.imshow(gene_mask)
    # plt.show()
    #

    plot_dimensional_images_side_by_side(feature_matrix)
    #
    # plt.imsave("/media/huifang/data/registration/mouse/huifang/" + str(k) + "_" + str(i) + "_image_512.png", image)
    # np.save("/media/huifang/data/registration/mouse/huifang/" + str(k) + "_" + str(i) + "_pca_out.npy",
    #         feature_matrix)
    # np.save("/media/huifang/data/registration/mouse/huifang/" + str(k) + "_" + str(i) + "_pca_mask.npy", gene_mask)
    # np.savez("/media/huifang/data/registration/mouse/huifang/" + str(k) + "_" + str(i) + "_validation",
    #          coord=coords, label=labels)
