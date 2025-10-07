import pandas as pd
import seaborn as sns
import json
import cv2
import scanpy as sc
import SimpleITK as sitk
from sklearn.decomposition import PCA
from functools import reduce
from PIL import Image

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def run_simpleITK(fixed_image,moving_image):
    # Set up the B-spline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, [3, 3], order=3)  # Control points grid size

    # Registration setup
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(transform)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-3, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform

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

def get_simpleITK_transformation(image1,image2):
    image1 = image1.convert('L')
    image2 = image2.convert('L')

    fixed_image = sitk.GetImageFromArray(np.array(image1).astype(np.float32))

    moving_image = sitk.GetImageFromArray(np.array(image2).astype(np.float32))

    # Perform registration (B-spline or non-linear)
    itk_transform = run_simpleITK(fixed_image, moving_image)

    # Generate displacement field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(moving_image)
    displacement_field = displacement_filter.Execute(itk_transform)

    # Resample moving image with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(itk_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered_image = resampler.Execute(moving_image)

    # Convert images to numpy arrays for visualization
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    registered_array = sitk.GetArrayViewFromImage(registered_image)

    fixed_array = normalize(fixed_array)
    moving_array = normalize(moving_array)
    registered_array = normalize(registered_array)

    # f,a = plt.subplots(1,2)
    # a[0].imshow(create_overlay(fixed_array,moving_array))
    # a[1].imshow(create_overlay(fixed_array,registered_array))
    #
    # plt.show()

    return fixed_image, moving_image,fixed_array, moving_array, registered_array, displacement_field

def transform_uv_with_displacement(uv_coords, deformation_field):
    deformation_np = sitk.GetArrayFromImage(deformation_field)
    deformation_size = deformation_field.GetSize()
    transformed_coords = []
    for u, v in uv_coords:
        u_int, v_int = int(u), int(v)

        # Ensure UV coordinates are within bounds
        if 0 <= u_int < deformation_size[0] and 0 <= v_int < deformation_size[1]:
            # Sample displacement at (u, v)
            displacement = deformation_np[v_int,u_int]  # (v, u) - numpy row-major

            # Apply displacement directly to UV
            u_transformed = u - displacement[0]  # x-component
            v_transformed = v - displacement[1]  # y-component
            transformed_coords.append([u_transformed, v_transformed])
        else:
            # If out of bounds, keep original point
            transformed_coords.append([u, v])
    return np.array(transformed_coords)


def warp_coords_moving_to_fixed(uv_coords, moving_image, fixed_image, itk_transform):
    """
    uv_coords: array/list of (u,v) pixel coordinates in the moving image domain
    moving_image, fixed_image: SimpleITK images
    itk_transform: transform from moving->fixed
    returns: array of (u_fixed, v_fixed) in fixed pixel space
    """
    warped_coords = []
    for (u, v) in uv_coords:
        # 1) moving pixel -> physical
        phys_moving = moving_image.TransformIndexToPhysicalPoint([int(u), int(v)])

        # 2) apply transform (moving->fixed)
        phys_fixed = itk_transform.TransformPoint(phys_moving)

        # 3) physical -> fixed pixel index
        uv_fixed = fixed_image.TransformPhysicalPointToIndex(phys_fixed)

        warped_coords.append([uv_fixed[0], uv_fixed[1]])

    return np.array(warped_coords)


def simpleITK_align_to_center(image_list,coords_list):
    image_0 = image_list[0]
    registered_image_list=[np.asarray(image_0.convert('L'))]
    registered_coords_list=[coords_list[0]]
    for i in range(1,len(image_list)):
    # for i in range(1, 2):
        fixed_itk,moving_itk,_,_,new_image,transform = get_simpleITK_transformation(image_0,image_list[i])
        transformed_coords = transform_uv_with_displacement(coords_list[i],transform)
        # transformed_coords = warp_coords_moving_to_fixed(coords_list[i], moving_itk, fixed_itk, transform)
        registered_image_list.append(new_image)
        registered_coords_list.append(transformed_coords)

    return registered_image_list,registered_coords_list


from scipy.ndimage import gaussian_filter, distance_transform_edt
def get_gene_feature_matrix_backup(coords: np.ndarray,
                            reduced_data: np.ndarray,
                            image_size=(2016, 2016),
                            patch_size=32,
                            feather_width=2,     # width (in patches) of the seam to soften
                            blur_sigma=1.0):     # sigma (in patches) for masked blur
    """
    Build a (H/ps, W/ps, D) patch feature matrix. Empty patches can be filled to soften borders.

    fill_mode:
      - "global_mean": fill empties with per-feature global mean across valid patches.
      - "local_mean" : iteratively fill empties with mean of 8-neighbors (per feature).
    """
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


    # per-feature mean over valid patches
    total_counts = count_array.sum()
    if total_counts > 0:
        global_mean = sum_array.sum(axis=(0, 1)) / total_counts  # shape (D,)
    else:
        global_mean = np.zeros((n_dims,), dtype=float)
    patch_matrix[~valid_mask, :] = global_mean[None, :]
    out = patch_matrix
    base = out.copy()
    valid = valid_mask

    # ---- Build a thin boundary band to soften only the seam ----
    # Distance to nearest empty (inside valid) and to nearest valid (inside empty)
    dist_valid_side = distance_transform_edt(valid)      # >0 inside valid, 0 at boundary
    dist_empty_side = distance_transform_edt(~valid)     # >0 inside empty, 0 at boundary
    # Pixels within 'feather_width' of either side are in the seam:
    seam = (dist_valid_side <= feather_width) | (dist_empty_side <= feather_width)

    if not seam.any():
        return base  # nothing to soften

    # ---- Masked Gaussian smooth (only used for blending in the seam) ----
    # Classic masked blur: smooth = G*(X*M) / (G*M), extending valid values across boundary
    M = valid.astype(float)
    eps = 1e-8
    smooth = np.empty_like(base)
    for d in range(n_dims):
        num = gaussian_filter(out[..., d] * M, sigma=blur_sigma, mode="nearest")
        den = gaussian_filter(M,           sigma=blur_sigma, mode="nearest") + eps
        smooth[..., d] = num / den  # smooth continuation of valid region

    # ---- Blend ONLY inside the seam between base (global-mean fill) and smooth ----
    # Weight ramps from 1 at the boundary to 0 at the seam edge on both sides
    # (so changes are confined to a thin band)
    min_dist = np.minimum(dist_valid_side, dist_empty_side)  # 0 at boundary, grows outward
    w_seam = np.clip(1.0 - (min_dist / max(feather_width, 1e-6)), 0.0, 1.0)  # (h, w)
    w3 = w_seam[..., None]  # (h, w, 1)

    result = base.copy()
    # Only modify seam cells; elsewhere keep strict global-mean + original valid means
    result[seam, :] = (1.0 - w3[seam, :]) * base[seam, :] + w3[seam, :] * smooth[seam, :]

    return result

from scipy.ndimage import convolve
def get_gene_feature_matrix(coords: np.ndarray,
                            reduced_data: np.ndarray,
                            image_size=(2016, 2016),
                            patch_size=32):            # how many smoothing passes if local
    """
    Build a (H/ps, W/ps, D) patch feature matrix. Empty patches can be filled to soften borders.

    fill_mode:
      - "global_mean": fill empties with per-feature global mean across valid patches.
      - "local_mean" : iteratively fill empties with mean of 8-neighbors (per feature).
    """
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

def get_uv_coordinates(slice):
    scale_path = slice.image_scale_path
    image = Image.open(slice.image_path)
    with open(scale_path, 'r') as f:
        data = json.load(f)
        low_res_scale = data['tissue_hires_scalef']
    uv_coords = slice.obsm['spatial'] * low_res_scale
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
            weight=0.02,  # small weight for gentle smoothing
            eps=1e-4,
        )

        preprocessed[..., i] = channel_denoised
    return preprocessed
from scipy.ndimage import median_filter
def remove_salt_pepper(feature_matrix, size=2):
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







def get_DLPFC_data():
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}

    for id in sample_list:
        adatas[id].image_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_hires_image_image_0.png'.format(id)
        adatas[id].image_scale_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/scalefactors_json.json'.format(id)
        adatas[id].spatial_prefix = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)


        adatas[id].obs['position'].index = (
            adatas[id].obs['position'].index
            .str.replace(r"\.\d+$", "", regex=True)
        )
        position_prefix = adatas[id].spatial_prefix
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

        adatas[id].obsm['spatial']=spatial_locations


    sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],
                     ["151673", "151674", "151675", "151676"]]
    layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in
                    range(len(sample_groups))]
    return layer_groups






def build_knn_edges(adata, k=6):
    """
    Returns:
      edges: (E,2) int array of undirected edges (i,j), i<j
      dists: (E,) float distances for each edge
    """
    XY = np.asarray(adata.obsm["spatial"])  # shape (N,2) or (N,>=2)
    XY = XY[:, :2]  # keep x,y
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(XY)
    dists, idx = nbrs.kneighbors(XY)  # idx: (N, k+1) including self at [:,0]
    src = np.repeat(np.arange(XY.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)  # drop self
    # make undirected & unique
    edges = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    uniq, uniq_idx = np.unique(edges, axis=0, return_index=True)
    edges = uniq
    dists = np.sqrt(((XY[edges[:,0]] - XY[edges[:,1]])**2).sum(axis=1))
    return edges, dists

def plot_connectivity(adata, edges, s=6, alpha_pts=0.8, lw=0.5, max_edges=20000):
    """
    Quick sanity plot: points + thin lines for edges.
    """
    XY = np.asarray(adata.obsm["spatial"])[:, :2]
    x, y = XY[:,0], XY[:,1]

    # Downsample edges if too many (to keep it snappy)
    if edges.shape[0] > max_edges:
        sel = np.random.choice(edges.shape[0], max_edges, replace=False)
        edges = edges[sel]

    plt.figure(figsize=(6,6))
    # draw edges
    for i, j in edges:
        plt.plot([x[i], x[j]], [y[i], y[j]], linewidth=lw, alpha=0.3, color="tab:gray")
    # draw points
    plt.scatter(x, y, s=s, c="tab:blue", alpha=alpha_pts, edgecolors="none")
    plt.gca().invert_yaxis()  # Visium-like orientation (optional)
    plt.axis("equal"); plt.axis("off")
    plt.title(f"k-NN connectivity (k≈{int(edges.shape[0]*2/len(x))})")
    plt.show()





def show_clusters(coors,labels):
    unique_labels = np.unique(labels)  # sorted unique label values
    palette = sns.color_palette("deep", len(unique_labels))  # e.g. 'deep', 'tab10', etc.
    color_index = np.searchsorted(unique_labels, labels)
    colors = np.array(palette)[color_index]

    plt.scatter(coors[:, 0], coors[:, 1], s=10, color=colors)
    plt.show()

import squidpy as sq
def get_svgs(sl):
    sq.gr.spatial_neighbors(
        sl,
        coord_type="generic",  # or "grid" for regular 10x layout
        n_rings=50,  # k‐NN; use `n_rings=` for Visium rings
    )

    sq.gr.spatial_autocorr(
        sl,
        mode="moran",  # or "geary"
        corr_method="fdr_bh",  # FDR correction
        n_jobs=8  # speed-up on multicore
    )

    # filter
    moran = sl.uns["moranI"]
    sig = (
        moran.sort_values(by=["pval_norm_fdr_bh", "I"], ascending=[True, False])
    )

    top_genes = sig.head(5000).index.tolist()
    return top_genes


layer_groups = get_DLPFC_data()
crop_paras = [[(50,0,1850,1024),(50,50,1900,1024),(50,120,1896,1024),(50,155,1861,1024)],
              [(300,200,1650,1024), (350,220,1550,1024),(350,290,1600,1024),(360,230,1600,1024)],
              [(160,10,1770,1024),(160,50,1770,1024),(160,120,1750,1024),(180,20,1770,1024)]]
image_size = 512

layer_to_color_map = {'Layer{0}'.format(i+1):i for i in range(6)}
layer_to_color_map['WM'] = 6





# for k,slices in enumerate(layer_groups):
#     # if k==0 or k ==1:
#     #     continue
#     for i, sl in enumerate(slices):
#         sq.gr.spatial_neighbors(
#             sl,
#             coord_type="grid",  # or "generic"
#             n_rings=1,  # number of hex rings for Visium; try 1–3
#             # n_neighs=6,          # use this for 'generic' (kNN)
#         )
#
#         # Matrices:
#         G = sl.obsp["spatial_connectivities"]  # adjacency (CSR)
#         D = sl.obsp.get("spatial_distances", None)  # (optional) distances
#
#
#         rows, cols = G.nonzero()
#         mask = rows < cols  # undirected unique edges
#         edges_sq = np.stack([rows[mask], cols[mask]], axis=1)
#         # np.save("/media/huifang/data/registration/DLPFC/" + str(k) + "_" + str(i) + "_spots_connection.npy", edges_sq)
#
#         # plot
#         plot_connectivity(sl, edges_sq, s=6)
#
#
#
# print('done')






for k,slices in enumerate(layer_groups):
    # if k==0 or k ==1:
    #     continue
    for sl in slices:
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
    image_list = []
    label_list=[]
    for i, sl in enumerate(slices):

        # Focus on common genes only
        sl_sub = sl[:, common_genes]
        # Convert to a NumPy array
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        # Extract coordinates from the slice

        labels = list(sl.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
        labels = np.asarray(labels)
        label_list.append(labels)

        sl_sub.image_scale_path = sl.image_scale_path
        sl_sub.image_path = sl.image_path
        sl_sub.spatial_prefix = sl.spatial_prefix

        coords, image = get_uv_coordinates(sl_sub)  # your custom function

        cropped_image, cropped_coor = crop_square_then_resize_square(image, coords,crop_paras[k][i],image_size)
        image_list.append(cropped_image)
        coords_list.append(cropped_coor)

    # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)

    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    # 5. Split the reduced data back for each slice and plot


    # f, a = plt.subplots(1, 4)
    # id = [0, 1, 2, 3]
    # for i, img, coors in zip(id, image_list, coords_list):
    #     a[i].imshow(img)
    #     a[i].scatter(coors[:, 0], coors[:, 1])
    # plt.show()
    #
    #
    # image_list, coords_list = simpleITK_align_to_center(image_list,coords_list)
    #
    # f,a = plt.subplots(1,4)
    # id = [0,1,2,3]
    # for i,img, coors in zip(id,image_list,coords_list):
    #     a[i].imshow(img)
    #     a[i].scatter(coors[:, 0], coors[:, 1])
    # plt.show()

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
        image = np.asarray(image_list[i])
        labels = label_list[i]

        feature_matrix, gene_mask = get_gene_feature_matrix(coords, reduced_slice_data, (image_size, image_size), patch_size=8)
        feature_matrix = preprocess_feature_matrix(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)

        plot_dimensional_images_side_by_side(gene_mask)

        # plt.imsave("../data/DLPFC/huifang/" + str(k) + "_" + str(i) + "_image_512.png", image)
        # np.save("/media/huifang/data/registration/DLPFC/" + str(k) + "_" + str(i) + "_pca_out.npy", feature_matrix)
        # np.save("/media/huifang/data/registration/DLPFC/" + str(k) + "_" + str(i) + "_pca_mask.npy", gene_mask)
        # np.savez("../data/DLPFC/huifang/" + str(k) + "_" + str(i) + "_validation", coord=coords,label = labels)