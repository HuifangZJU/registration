import matplotlib.pyplot as plt
import scanpy as sc
from skimage.restoration import denoise_tv_chambolle
import os
import cv2
import imageio.v2 as imageio
import numpy as np
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

def get_xenium_data():
    data_dir = "/media/huifang/data/registration/xenium/cell_typing/joint_domains"
    sample_ids = [["Tg_2.5m", "Tg_5.7m", "Tg_17.9m"], ["WT_2.5m", "WT_5.7m", "WT_13.4m"]]
    image_paths = [["Xenium_V1_FFPE_TgCRND8_2_5_months", "Xenium_V1_FFPE_TgCRND8_5_7_months", "Xenium_V1_FFPE_TgCRND8_17_9_months"],
                  ["Xenium_V1_FFPE_wildtype_2_5_months", "Xenium_V1_FFPE_wildtype_5_7_months", "Xenium_V1_FFPE_wildtype_13_4_months"]]
    groups = []
    for group_data,image_data in zip(sample_ids,image_paths):
        group = []
        for sid, img in zip(group_data,image_data):
            fn = os.path.join(data_dir, f"xenium_{sid}_domains_shared.h5ad")
            adata = sc.read_h5ad(fn)
            adata.uns['image_path'] = f"/media/huifang/data/Xenium/xenium_data/{img}/morphology_down10x.png"
            group.append(adata)
        groups.append(group)
    return groups


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

    return patch_matrix


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
    if feature_matrix.max()>1:
        feature_matrix = (feature_matrix-feature_matrix.min())/feature_matrix.max()
    preprocessed = np.zeros_like(feature_matrix)

    for i in range(feature_matrix.shape[2]):
        channel = feature_matrix[..., i].astype(np.float32)

        channel_bilat = cv2.bilateralFilter(
            channel,  # source image
            d=10,  # diameter of the pixel neighborhood
            sigmaColor=0.1,  # range sigma for color
            sigmaSpace=25  # range sigma for spatial distance
        )
        channel_bilat = channel_bilat.astype(np.float64)
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

def clahe(dapi_img):
    # Convert to uint8 if needed
    img_uint8 = (dapi_img * 255).astype('uint8') if dapi_img.max() <= 1.0 else dapi_img

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(img_clahe, kernel, iterations=2)

    return dilated


def resize_and_flip_image_coords(
    image: np.ndarray,
    coords: np.ndarray,
    new_size: int,
    flip_paras: list,
    group_idx: int = 0,
    image_idx: int = 0
):

    # Retrieve flip parameters for this image
    flip_lr, flip_ud = flip_paras[group_idx][image_idx]

    # --- Step 1: Resize ---
    h, w = image.shape[:2]
    scale_x = new_size / w
    scale_y = new_size / h
    new_img = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    # --- Step 2: Scale coordinates ---
    new_coords = coords.copy().astype(np.float32)
    new_coords[:, 0] *= scale_x
    new_coords[:, 1] *= scale_y

    # --- Step 3: Apply flips ---
    if flip_lr:
        new_img = cv2.flip(new_img, 1)
        new_coords[:, 0] = new_size - 1 - new_coords[:, 0]

    if flip_ud:
        new_img = cv2.flip(new_img, 0)
        new_coords[:, 1] = new_size - 1 - new_coords[:, 1]
    new_img = clahe(new_img)
    return new_img, new_coords


def pca_noarmalize_gene_list():
    # # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)
    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    return reduced_data



groups = get_xenium_data()
flip_paras = [
    [ [False, False], [False, False], [True, False] ],
    [[False, False], [True, False], [False, False] ]
]
image_size = 1024
for k,slices in enumerate(groups):
    for sl in slices:
        # 1. Filter low-quality cells and genes
        sc.pp.filter_cells(sl, min_counts=100)
        sc.pp.filter_genes(sl, min_cells=10)
        # 2. Normalize and log-transform
        sc.pp.normalize_total(sl, target_sum=1e4)
        sc.pp.log1p(sl)

    # 2. Subset each slice to the common genes, gather coordinates & data


    coords_list = []
    image_list = []
    label_list=[]
    gene_data_list = []

    morpho_list = []
    for i, sl in enumerate(slices):
        morph_data = np.array(sl.obsm["X_morphology"])
        morpho_list.append(morph_data)

        gene_data = np.array(sl.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)

        # cell_type = sl.obs['cell_type'].tolist()
        # label_list.append(cell_type)
        labels = sl.obs["domain_id_shared"].astype("category").cat.codes.to_numpy()
        labels = np.asarray(labels)
        label_list.append(labels)

        coords = sl.obsm['spatial']/10.
        # imagepath = sl.uns['image_path'][:-7]+"20x.png"
        imagepath = sl.uns['image_path']
        image = plt.imread(imagepath)
        resized_image,resized_coords = resize_and_flip_image_coords(image,coords,image_size,flip_paras, k, i)
        coords_list.append(resized_coords)
        image_list.append(resized_image)


    new_label_list = label_list
    new_gene_data = pca_noarmalize_gene_list()


    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        coords = coords_list[i]
        image = image_list[i]
        labels = new_label_list[i]

        # fig, axes = plt.subplots(1, 3)
        # axes = np.atleast_1d(axes).ravel()
        # for k in range(3):
        #     ax = axes[k]
        #     v = morphofeature[:, k]
        #     sc = ax.scatter(coords[:, 0], coords[:, 1], c=v, s=7, cmap="gray", edgecolors="none")
        #     ax.invert_yaxis()
        #     ax.set_aspect("equal", "box");
        #     ax.axis("off")
        #     ax.set_title(f"Prototype {k + 1}")
        #     plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # for j in range(k + 1, len(axes)): axes[j].axis("off")
        # plt.tight_layout();
        # plt.show()
        # continue
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots
        # Slice out the portion that belongs to this slice
        reduced_slice_data = new_gene_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        # plt.imshow(image)
        # plt.scatter(coords[:,0],coords[:,1],c=labels,s=5, cmap="tab20")
        # plt.show()

        feature_matrix= get_gene_feature_matrix_soft(coords, reduced_slice_data, (image_size, image_size),
                                                            patch_size=8)


        feature_matrix = preprocess_feature_matrix(feature_matrix)
        feature_matrix = remove_salt_pepper(feature_matrix)
        feature_matrix[feature_matrix < 0.01] = 0
        valid_mask = (feature_matrix > 0)
        gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)
        # plot_dimensional_images_side_by_side(feature_matrix)

        imageio.imwrite(
            f"/media/huifang/data/registration/xenium/{k}_{i}_image_{image_size}.png",
            image.astype(np.uint8)  # scale to 8-bit if image is 0–1 float
        )
        # plt.imsave("/media/huifang/data/registration/xenium/" + str(k) + "_" + str(i) + f"_image_{image_size}.png", image)
        # np.save("/media/huifang/data/registration/xenium/" + str(k) + "_" + str(i) + f"_pca_out_{image_size}.npy", feature_matrix)
        # np.save("/media/huifang/data/registration/xenium/" + str(k) + "_" + str(i) + f"_pca_mask_{image_size}.npy", gene_mask)
        # np.savez("/media/huifang/data/registration/xenium/" + str(k) + "_" + str(i) + f"_validation_{image_size}", coord=coords,label = labels)


