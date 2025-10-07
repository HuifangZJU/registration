import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import json
from scipy import sparse, stats
import cv2
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import SimpleITK as sitk
from sklearn.preprocessing import normalize
from scipy.spatial import cKDTree
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from functools import reduce
from scipy.spatial import cKDTree
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
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
    print(patch_matrix.shape)
    return patch_matrix

def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    ncols = 4
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
        low_res_scale = data['tissue_hires_scalef']
    uv_coords = slice.obsm["spatial"] * low_res_scale
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

def clip_and_min_max_normalize(data, lower_percentile=3, upper_percentile=97):
    """
    Clips each channel based on empirical percentiles, then performs min-max normalization.

    Parameters:
    - data: np.ndarray of shape (N, C)
        N = number of samples (e.g., 17952)
        C = number of channels/features (e.g., 10)
    - lower_percentile: float
        Lower clipping percentile (e.g., 1 for 1st percentile)
    - upper_percentile: float
        Upper clipping percentile (e.g., 99 for 99th percentile)

    Returns:
    - normalized_data: np.ndarray of same shape as input, with values in [0, 1]
    """
    data = data.copy()  # avoid modifying original
    N, C = data.shape

    for c in range(C):
        channel = data[:, c]
        lower = np.percentile(channel, lower_percentile)
        upper = np.percentile(channel, upper_percentile)
        data[:, c] = np.clip(channel, lower, upper)

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-8  # avoid division by zero

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







def get_DLPFC_data(
    base_dir: str = "../data/DLPFC",
    raw_prefix: str = "/media/huifang/data/registration/humanpilot",
    sample_list = None,
    sample_groups = None,
) -> list[list[sc.AnnData]]:

    # ---------- defaults ----------
    if sample_list is None:
        sample_list = [
            "151507", "151508", "151509", "151510",
            "151669", "151670", "151671", "151672",
            "151673", "151674", "151675", "151676",
        ]

    if sample_groups is None:
        sample_groups = [
            ["151507", "151508", "151509", "151510"],
            ["151669", "151670", "151671", "151672"],
            ["151673", "151674", "151675", "151676"],
        ]

    # ---------- load all adatas ----------
    adatas: dict[str, sc.AnnData] = {
        sid: sc.read_h5ad(Path(base_dir) / f"{sid}_preprocessed.h5")
        for sid in sample_list
    }

    # ---------- enrich with spatial info ----------
    for sid, ad in adatas.items():
        spatial_dir = Path(raw_prefix) / sid / "spatial"

        # --- put auxiliary files in .uns so they persist on disk ---
        ad.uns.setdefault("spatial_meta", {})
        ad.uns["spatial_meta"].update(
            {
                "image_path":      str(spatial_dir / "tissue_hires_image_image_0.png"),
                "scalefactor_path": str(spatial_dir / "scalefactors_json.json"),
                "positions_path":   str(spatial_dir / "tissue_positions_list"),
                "library_id": sid,
            }
        )

        # --- load barcodes & pixel coords ---
        pos_csv = spatial_dir / "tissue_positions_list.csv"
        pos_txt = spatial_dir / "tissue_positions_list.txt"
        if pos_csv.exists():
            positions = pd.read_csv(pos_csv, header=None)
        else:
            positions = pd.read_csv(pos_txt, header=None)

        positions.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]
        positions.index = positions["barcode"].astype(str)

        # --- harmonise AnnData barcodes (drop trailing ".1", ".2", ...) ---
        ad.obs_names = ad.obs_names.str.replace(r"\.\d+$", "", regex=True)

        # --- keep only barcodes present in both objects ---
        common = ad.obs_names.intersection(positions.index)
        if len(common) == 0:
            raise ValueError(f"No matching barcodes found for sample {sid}")

        coords = positions.loc[common, ["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
        ad.obsm["spatial"] = coords      # (n_obs, 2) float64

    # ---------- assemble requested groups ----------
    layer_groups = [[adatas[sid] for sid in grp] for grp in sample_groups]
    return layer_groups

def show_clusters(coors,labels):
    unique_labels = np.unique(labels)  # sorted unique label values
    palette = sns.color_palette("deep", len(unique_labels))  # e.g. 'deep', 'tab10', etc.
    color_index = np.searchsorted(unique_labels, labels)
    colors = np.array(palette)[color_index]

    plt.scatter(coors[:, 0], coors[:, 1], s=10, color=colors)
    plt.show()

def gaussian_clip_per_gene(adata, lower_tail_p=0.025, upper_tail_p=0.025):
    """
    Clip each gene by fitting a Gaussian to nonzero entries, then clipping
    lower and upper tails using z-score quantiles.

    Parameters
    ----------
    adata : AnnData
        AnnData object with .X (dense or sparse).
    lower_tail_p : float
        Fraction of lower tail to clip (e.g., 0.025 for 2.5%).
    upper_tail_p : float
        Fraction of upper tail to clip (e.g., 0.025 for 2.5%).
    """

    X = adata.X

    if not sparse.issparse(X):
        # ----- Dense matrix -----
        for i in range(X.shape[1]):
            col = X[:, i]
            nonzero = col[col > 0]
            if len(nonzero) == 0:
                continue

            mu, sigma = nonzero.mean(), nonzero.std()
            if sigma == 0:
                continue

            # Compute Gaussian-based cutoffs
            lower = stats.norm.ppf(lower_tail_p, loc=mu, scale=sigma)
            upper = stats.norm.ppf(1 - upper_tail_p, loc=mu, scale=sigma)

            # Clip in place
            col = np.clip(col, lower, upper)
            X[:, i] = col

        adata.X = X

    else:
        # ----- Sparse matrix (CSC for column access) -----
        X = X.tocsc()
        X_clipped = X.copy()

        for i in range(X.shape[1]):
            start, end = X.indptr[i], X.indptr[i + 1]
            data = X.data[start:end]

            if len(data) == 0:
                continue

            nonzero = data[data > 0]
            if len(nonzero) == 0:
                continue

            mu, sigma = nonzero.mean(), nonzero.std()
            if sigma == 0:
                continue

            lower = stats.norm.ppf(lower_tail_p, loc=mu, scale=sigma)
            upper = stats.norm.ppf(1 - upper_tail_p, loc=mu, scale=sigma)

            X_clipped.data[start:end] = np.clip(data, lower, upper)

        adata.X = X_clipped.tocsr()


def zscore_per_gene(adata):
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    gene_min = X.min(axis=0)
    gene_max = X.max(axis=0)
    denom = gene_max - gene_min
    denom[denom == 0] = 1  # avoid divide-by-zero

    X_norm = (X - gene_min) / denom
    adata.X = sparse.csr_matrix(X_norm) if sparse.issparse(adata.X) else X_norm

import scipy
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

    # Count items per cluster
    counts = np.bincount(labels)
    titles = [f"Module {k} ({counts[k]} genes)" for k in range(K)]


    # # 4. Assign module scores per slice
    # for sl in slices:
    #     for k in range(K):
    #         gene_group = np.array(selected_genes)[labels == k].tolist()
    #         sc.tl.score_genes(sl, gene_group, score_name=f"module_{k}",ctrl_as_ref=False)
    #     sc.pl.spatial(sl, color=[f"module_{k}" for k in range(K)],
    #                   ncols=4, spot_size=150, img_key=None,title=titles)
    #
    gene_pool = list(set.intersection(*[set(sl.var_names) for sl in slices]))

    for k in range(K):
        gene_group = np.array(selected_genes)[labels == k].tolist()
        gene_group = [g for g in gene_group if g in gene_pool]  # ensure availability

        for sl in slices:
            sc.tl.score_genes(
                sl,
                gene_group,
                score_name=f"module_{k}",
                gene_pool=gene_pool,  # fixed pool ensures stable background
                ctrl_size=50,  # optional tuning
                ctrl_as_ref=True  # use background genes instead of random control
            )

    # 5. Visualize modules
    for i, sl in enumerate(slices):
        sc.pl.spatial(
            sl,
            color=[f"module_{k}" for k in range(K)],
            ncols=8,
            spot_size=150,
            img_key=None,
            title=[f"Slice {i} - Module {k}" for k in range(K)]
        )

    # 5. Return the module expression matrices per slice (shape: [spots, K])
    feature_maps = {}
    for i, sl in enumerate(slices):
        maps = np.vstack([sl.obs[f"module_{k}"].values for k in range(K)]).T
        feature_maps[f"slice_{i}"] = maps


    return feature_maps


def spatial_clip_gene_values(sl, lower_tail=0.1, upper_tail=0.9):
    X = sl.X.toarray() if not isinstance(sl.X, np.ndarray) else sl.X
    clipped = np.zeros_like(X)

    for i in range(X.shape[1]):  # iterate over genes
        gene_values = X[:, i]

        # Exclude zeros if needed (optional)
        non_zero = gene_values[gene_values > 0]
        if len(non_zero) < 10:
            clipped[:, i] = gene_values  # skip if too sparse
            continue
        # Fit normal distribution or use empirical quantiles
        mean, std = np.mean(non_zero), np.std(non_zero)
        lower_bound = stats.norm.ppf(lower_tail, loc=mean, scale=std)
        upper_bound = stats.norm.ppf(upper_tail, loc=mean, scale=std)

        # Clip values
        clipped[:, i] = np.clip(gene_values, lower_bound, upper_bound)

    sl.X = clipped
    return sl


layer_groups = get_DLPFC_data()
crop_paras = [[(50,0,1850,1024),(50,50,1900,1024),(50,120,1896,1024),(50,155,1861,1024)],
              [(300,200,1650,1024), (350,220,1550,1024),(350,290,1600,1024),(360,230,1600,1024)],
              [(160,10,1770,1024),(160,50,1770,1024),(160,120,1750,1024),(180,20,1770,1024)]]
image_size = 512

layer_to_color_map = {'Layer{0}'.format(i+1):i for i in range(6)}
layer_to_color_map['WM'] = 6


for k,original_slices in enumerate(layer_groups):
    # if k <2:
    #     continue

    slices=[]

    for sl in original_slices:
        sc.pp.filter_genes(sl, min_cells=3)
        # sc.pp.normalize_total(sl, target_sum=1e4)
        # sc.pp.log1p(sl)
        # sc.pp.scale(sl, max_value=10)


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
            moran.sort_values(by=["pval_norm_fdr_bh","I"], ascending=[True, False])
        )

        top_genes = sig.head(2000).index.tolist()
        # # print(top_genes)
        # # test = input()
        # for iter in range(10):
        #     sc.pl.spatial(
        #         sl,
        #         color=top_genes[iter*64:(iter+1)*64],
        #         spot_size=200,
        #         img_key=None,  # omit histology overlay if not present
        #         ncols=8
        #     )


        sl.uns['svgs']=top_genes

        print(sig.head())
        slices.append(sl)
    print('Clustering spatially variable gene data')
    reduced_data = cluster_svg_modules(slices,64)
    print('done')
    coords_list = []
    image_list = []
    label_list=[]
    for i, sl in enumerate(slices):
        labels = list(sl.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
        labels = np.asarray(labels)
        label_list.append(labels)
        coords, image = get_uv_coordinates(sl)  # your custom function
        cropped_image, cropped_coor = crop_square_then_resize_square(image, coords,crop_paras[k][i],image_size)
        image_list.append(cropped_image)
        coords_list.append(cropped_coor)

    for i in range(len(reduced_data)):
        reduced_slice_data = reduced_data[f"slice_{i}"]

        reduced_slice_data = channelwise_min_max_normalize(reduced_slice_data)
        # reduced_slice_data = clip_and_min_max_normalize(reduced_slice_data)
        # Get the corresponding coordinates
        coords = coords_list[i]
        image = np.asarray(image_list[i])
        labels = label_list[i]

        feature_matrix = get_gene_feature_matrix(coords, reduced_slice_data, (image_size, image_size), patch_size=8)
        # feature_matrix = preprocess_feature_matrix(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)
        # plot_dimensional_images_side_by_side(feature_matrix)
        feature_matrix = preprocess_feature_matrix(feature_matrix)
        # plot_dimensional_images_side_by_side(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)
        plot_dimensional_images_side_by_side(feature_matrix)

        # plt.imsave("/media/huifang/data/registration/DLPFC/" + str(k) + "_" + str(i) + "_image_512.png", image)
        # np.save("/media/huifang/data/registration/DLPFC/top2000_clustered_svg/" + str(k) + "_" + str(i) + "_clustered_svg_smoothed.npy", feature_matrix)
        # np.savez("/media/huifang/data/registration/DLPFC/" + str(k) + "_" + str(i) + "_validation", coord=coords,label = labels)
