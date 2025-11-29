import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from skimage.morphology import remove_small_objects, binary_closing, disk

xenium_root_folder = '/media/huifang/data/sennet/xenium/'
codex_root_folder = '/media/huifang/data/sennet/codex/'

def get_data_list(root):
    data = pd.read_csv(root +'data_list.txt', sep=None, engine='python', header=None)
    data.columns = ['subfolder', 'sampleid']
    return data

def get_subfolder_by_sampleid(data,sampleid):
    result = data.loc[data['sampleid'] == sampleid, 'subfolder']



    return result.values[0] if not result.empty else None

def get_xenium_data(xenium_list, xenium_sampleid,xenium_regionid):
    xenium_subfolder_name = get_subfolder_by_sampleid(xenium_list, xenium_sampleid)
    subfolder = os.path.join(xenium_root_folder, xenium_subfolder_name)
    if os.path.exists(os.path.join(subfolder, 'outs')):
        subfolder = os.path.join(subfolder, 'outs')
    if not os.path.exists(subfolder + '/morphology_focus/regional_images/'):
        img_path = subfolder + '/morphology_focus/channel0_quarter.png'
    else:
        img_path = subfolder + '/morphology_focus/regional_images' + f"/channel0_region_{xenium_regionid}_quarter.png"

    cell_path = '/media/huifang/data/sennet/xenium/regional_data/'+ f"{xenium_sampleid}_{xenium_regionid}.h5ad"
    df = sc.read_h5ad(cell_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = plt.imread(img_path)
    # plt.imshow(img)
    # plt.scatter(coordinate[:,0],coordinate[:,1])
    # plt.show()

    return img,df



def get_codex_data(codex_sampleid,codex_regionid):

    subfolder = os.path.join(codex_root_folder, codex_sampleid, 'per_tissue_region-selected')
    img_path = subfolder + f"/{codex_regionid}_X01_Y01_Z01_channel0_half.png"


    cell_path = '/media/huifang/data/sennet/codex/regional_data/' + f"{codex_sampleid}_{codex_regionid}.h5ad"
    df = sc.read_h5ad(cell_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img)
    # plt.scatter(coordinate[:,0],coordinate[:,1])
    # plt.show()
    return img,df

def flip_image(img,coordinate):
    # Invert both x and y axes of the image
    img_flipped = np.flipud(np.fliplr(img))  # y-flip then x-flip

    # Invert coordinates based on the original image shape
    height, width = img.shape[:2]

    # x' = width - x, y' = height - y
    coordinate_flipped = np.empty_like(coordinate)
    coordinate_flipped[:, 0] = width - coordinate[:, 0]
    coordinate_flipped[:, 1] = height - coordinate[:, 1]
    return img_flipped,coordinate_flipped

def resize_preserve_aspect(image, target_edge):
    h, w = image.shape
    scale = target_edge / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale, new_w, new_h

def prepare_images_on_shared_canvas(img1, coords1, img2, coords2):
    target_edge = max(xenium_img.shape + codex_img.shape)
    # Resize both images
    img1_resized, scale1, w1, h1 = resize_preserve_aspect(img1, target_edge)
    img2_resized, scale2, w2, h2 = resize_preserve_aspect(img2, target_edge)

    # Determine shared canvas size with minimal margin
    canvas_h = max(h1, h2)
    canvas_w = max(w1, w2)

    def place_image_and_transform_coords(img_resized, coords, scale, canvas_h, canvas_w):
        h, w = img_resized.shape
        y_off = (canvas_h - h) // 2
        x_off = (canvas_w - w) // 2
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas[y_off:y_off + h, x_off:x_off + w] = img_resized
        transformed_coords = coords * scale + np.array([x_off, y_off])
        return canvas, transformed_coords

    canvas1, coords1_trans = place_image_and_transform_coords(img1_resized, coords1, scale1, canvas_h, canvas_w)
    canvas2, coords2_trans = place_image_and_transform_coords(img2_resized, coords2, scale2, canvas_h, canvas_w)

    return canvas1, coords1_trans, canvas2, coords2_trans


def create_overlay(img1,img2):
    img1_rgb = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
    # Image 2 → Cyan (G + B)
    img2_rgb = np.stack([np.zeros_like(img2), img2, img2], axis=-1)

    # Blend with maximum pixel value
    overlay = np.maximum(img1_rgb, img2_rgb)
    return overlay

def save_single_dataset(image, gene_data, datatype, sampleid, regionid, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    # Create base name
    base_name = f"{datatype}_{sampleid}_{regionid}"
    # Save AnnData object
    # gene_data.write(os.path.join(output_dir, f"{base_name}.h5ad"))
    # Save aligned image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.png"), image)
    print(f"Saved: {base_name} to {output_dir}")

def show_pixel_value_distribution(img):
    plt.figure(figsize=(6, 4))
    plt.hist(img.ravel(), bins=256, range=(0, 255), color='gray')
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def binarize_image(img):
    threshold = 0.5 if img.max() <= 1.0 else 128  # pick based on range

    # Binarize
    binary_img = (img > threshold).astype(np.uint8)
    binary_img = binary_img*255
    binary_img = binary_img.astype(np.uint8)
    return binary_img



def clahe(dapi_img):
    # Convert to uint8 if needed
    img_uint8 = (dapi_img * 255).astype('uint8') if dapi_img.max() <= 1.0 else dapi_img

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(img_clahe, kernel, iterations=2)
    # Gaussian blur to smooth
    # blurred = cv2.GaussianBlur(dilated, (3, 3), 0)

    # # Optional: morphological closing to fill small holes
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    #
    # # Normalize to 0–1 for visualization or model input
    # enhanced = cv2.normalize(closed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return dilated


def generate_pseudo_tissue_image(coords):
    # --- Step 1: Define output image size ---
    img_size = (512, 512)

    # --- Step 2: Compute spatial range ---
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # --- Step 3: Map coordinates to pixel indices ---
    x_scaled = ((coords[:, 0] - x_min) / (x_max - x_min) * (img_size[1] - 1)).astype(int)
    y_scaled = ((coords[:, 1] - y_min) / (y_max - y_min) * (img_size[0] - 1)).astype(int)

    # --- Step 4: Count number of cells per grid (no blur) ---
    density = np.zeros(img_size, dtype=np.int32)
    for x, y in zip(x_scaled, y_scaled):
        density[y, x] += 1  # note (y, x) order for image array

    # --- Step 5: Normalize for visualization ---
    density_norm = density / np.percentile(density, 99)
    density_norm = np.clip(density_norm, 0, 1)

    # --- Step 6: Show pseudo cell density image ---
    plt.figure(figsize=(8, 8))
    plt.imshow(density_norm, cmap='gray', origin='lower')
    plt.axis('off')
    plt.title("Pseudo Cell Density Map (No Smoothing)", fontsize=14)
    plt.show()

    # --- Step 8: Plot histogram of cell counts per pixel ---
    counts = density.flatten()
    counts_nonzero = counts[counts > 0]  # ignore empty pixels

    plt.figure(figsize=(6, 4))
    plt.hist(counts_nonzero, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel("Cells per grid (pixel)")
    plt.ylabel("Number of grids")
    plt.title("Distribution of Cell Counts per Grid", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# 2) helper normalization
def minlog1p(x):
    x = np.asarray(x, dtype=float)
    x = x - np.min(x)
    return np.log1p(x)

def build_aligned_modal_channels_simple(
    xenium_gene_data,
    codex_protein_data,
    marker_map,
    agg="mean"               # "mean" or "max" over genes per marker
):

    marker_order, used_genes = [], {}
    for m, genes in marker_map.items():
        if m not in codex_protein_data.obs.columns:
            continue
        avail = [g for g in genes if g in xenium_gene_data.var_names]
        if not avail:
            continue
        marker_order.append(m)
        used_genes[m] = avail



    # 3) Xenium gene channels
    n_x, D = xenium_gene_data.n_obs, len(marker_order)
    gene_mat = np.zeros((n_x, D), dtype=float)
    for j, m in enumerate(marker_order):
        genes = used_genes[m]
        X = xenium_gene_data[:, genes].X
        if hasattr(X, "toarray"):  # sparse small slice -> dense
            X = X.toarray()
        score = X.mean(axis=1) if agg == "mean" else X.max(axis=1)
        gene_mat[:, j] = np.log1p(minlog1p(score))

    # 4) CODEX protein channels
    n_p = codex_protein_data.n_obs
    prot_mat = np.zeros((n_p, D), dtype=float)
    for j, m in enumerate(marker_order):
        vals = codex_protein_data.obs[m].astype(float).values
        prot_mat[:, j] = minlog1p(vals)

    return gene_mat, prot_mat, marker_order, used_genes
from sklearn.decomposition import PCA
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
    mins = data.min(axis=0)   # shape (C,)
    maxs = data.max(axis=0)   # shape (C,)
    ranges = maxs - mins

    # Avoid division by zero (when all values in a channel are the same)
    ranges[ranges == 0] = 1e-8

    normalized_data = (data - mins) / ranges
    return normalized_data
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


def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray,ncols=5,prefix=None):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]

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
    if prefix:
        plt.savefig(f"/media/huifang/data/sennet/gene_pca/{prefix}.png", dpi=300)
    else:
        plt.show()

from scipy.stats import spearmanr
def find_top_genes_per_protein(xenium_gene_data, codex_protein_data, candidate_genes, proteins, topk=3):
    # coords → nearest-neighbor match CODEX cells to Xenium cells
    coords_gene = xenium_gene_data.obs[['x_trans','y_trans']].values
    coords_prot = codex_protein_data.obs[['x_trans','y_trans']].values
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords_gene)
    _, idx = nbrs.kneighbors(coords_prot)
    idx = idx[:,0]

    # helpers
    varset = set(xenium_gene_data.var_names)
    cand = [g for g in candidate_genes if g in varset]

    def minlog1p(x):
        x = np.asarray(x, dtype=float)
        x = x - np.min(x)
        return np.log1p(x)

    def get_gene_vec(g):
        X = xenium_gene_data[:, g].X
        if hasattr(X, 'toarray'): X = X.toarray().ravel()
        return minlog1p(X[idx])  # reorder to CODEX cells

    results = []
    for p in proteins:
        if p not in codex_protein_data.obs.columns:
            continue
        prot_vec = codex_protein_data.obs[p].astype(float).values
        prot_vec = minlog1p(prot_vec)

        stats = []
        for g in cand:
            gx = get_gene_vec(g)
            # guard against constant vectors
            if np.std(gx)==0 or np.std(prot_vec)==0:
                pear, spear = np.nan, np.nan
            else:
                pear = float(np.corrcoef(gx, prot_vec)[0,1])
                spear = float(spearmanr(gx, prot_vec).correlation)
            stats.append((g, pear, spear))

        # sort by |Pearson|, take top-k
        top = sorted(stats, key=lambda t: (0 if np.isnan(t[1]) else abs(t[1])), reverse=True)[:topk]
        for rank, (g, pear, spear) in enumerate(top, 1):
            results.append({'protein': p, 'gene': g, 'rank': rank, 'pearson_r': pear, 'spearman_r': spear})

    return pd.DataFrame(results).sort_values(['protein','rank'])


def resize_image_coords(
    image: np.ndarray,
    coords: np.ndarray,
    new_size: int,
):
    # --- Step 1: Resize ---
    h, w = image.shape[:2]
    scale_x = new_size / w
    scale_y = new_size / h
    new_img = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    # --- Step 2: Scale coordinates ---
    new_coords = coords.copy().astype(np.float32)
    new_coords[:, 0] *= scale_x
    new_coords[:, 1] *= scale_y

    return new_img, new_coords
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian


def preprocess_feature_matrix(feature_matrix):

    if feature_matrix.max()>1:
        feature_matrix = (feature_matrix-feature_matrix.min())/feature_matrix.max()
    # We'll store our results in a new array
    preprocessed = np.zeros_like(feature_matrix)
    # To apply the OpenCV bilateral filter, we need 8-bit or float32
    # We'll convert each slice to float32 [0,1] for processing
    for i in range(feature_matrix.shape[2]):
        channel = feature_matrix[..., i].astype(np.float32)
        #feature paras
        channel_bilat = cv2.bilateralFilter(
            channel,  # source image
            d=5,  # diameter of the pixel neighborhood
            sigmaColor=0.1,  # range sigma for color
            sigmaSpace=25  # range sigma for spatial distance
        )
        # Convert back to numpy float64 to feed into TV denoising
        channel_bilat = channel_bilat.astype(np.float64)

        channel_denoised = denoise_tv_chambolle(
            channel_bilat,
            weight=0.02,  # small weight for gentle smoothing
            eps=1e-4,
        )

        preprocessed[..., i] = channel_denoised
    return preprocessed
from scipy.ndimage import median_filter
def remove_salt_pepper(feature_matrix, size=2):
    filtered = np.zeros_like(feature_matrix)
    for c in range(feature_matrix.shape[2]):
        filtered[..., c] = median_filter(feature_matrix[..., c], size=size)
    return filtered

from scipy import sparse
def add_binary_senescence_labels(xenium_gene_data,codex_protein_data,xenium_topk=0.1, codex_topn=0.1,
                                 plot=True):
    # ---- helper function ----
    def binarize_label(adata, feature_name, top_value, apply_log1p=False):
        lower_vars = [v.lower() for v in adata.var_names]
        lower_obs = [v.lower() for v in adata.obs.columns]
        target = feature_name.lower()

        # extract expression
        if target in lower_vars:
            gene = adata.var_names[lower_vars.index(target)]
            expr = adata[:, gene].X
            if sparse.issparse(expr):
                expr = expr.toarray().flatten()
            else:
                expr = np.array(expr).flatten()
        elif target in lower_obs:
            obs_col = adata.obs.columns[lower_obs.index(target)]
            expr = adata.obs[obs_col].astype(float).values
        else:
            print(f"[Warning] {feature_name} not found.")
            adata.obs[f"{feature_name}_binary"] = np.nan
            return adata

        expr = np.nan_to_num(expr.astype(float))
        if apply_log1p:
            expr = np.log1p(expr)

        # determine threshold
        if top_value < 1:  # top percentage
            cutoff = np.quantile(expr, 1 - top_value)
        else:  # top N cells
            cutoff = np.partition(expr, -int(top_value))[-int(top_value)]

        labels = (expr >= cutoff).astype(int)
        adata.obs[f"{feature_name}_binary"] = labels
        return adata,labels

    # ---- compute ----
    xenium_gene_data,xenium_labels = binarize_label(xenium_gene_data, "CDKN1A", xenium_topk, apply_log1p=True)
    codex_protein_data,codex_labels = binarize_label(codex_protein_data, "p16", codex_topn, apply_log1p=False)

    # ---- plot ----
    if plot:
        def plot_binary(adata, feature, ax, title):
            x, y = adata.obs["x_trans"], adata.obs["y_trans"]
            label = adata.obs[f"{feature}_binary"]
            sca = ax.scatter(x, y, c=label, cmap="coolwarm", s=8, vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("x_trans");
            ax.set_ylabel("y_trans")
            ax.set_aspect("equal")
            plt.colorbar(sca, ax=ax, ticks=[0, 1], label="Senescence Label")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_binary(xenium_gene_data, "CDKN1A", axes[0], "Xenium - CDKN1A (binary)")
        plot_binary(codex_protein_data, "p16", axes[1], "CODEX - p16 (binary)")
        plt.tight_layout()
        plt.show()

    return xenium_gene_data, codex_protein_data,xenium_labels,codex_labels

xenium_list = get_data_list(xenium_root_folder)
# Replace with your actual file path
file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
root_path = '/media/huifang/data/sennet/hf_aligned_data/'
# Read the text file without header
df = pd.read_csv(file_path, sep=None, engine='python', header=None)
image_size=1024
start_line = 0  # zero-based line number
for k, row in enumerate(df.iloc[start_line:].itertuples(index=False, name=None)):
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = row

    xenium_gene_data = sc.read_h5ad(root_path + f"xenium_{xenium_sampleid}_{xenium_regionid}.h5ad")
    codex_protein_data = sc.read_h5ad(root_path + f"codex_{codex_sampleid}_{codex_regionid}.h5ad")


    xenium_gene_data, codex_protein_data,xenium_labels,codex_labels = add_binary_senescence_labels(xenium_gene_data, codex_protein_data,plot=False)
    label_list = [xenium_labels,codex_labels]



    xenium_img = cv2.imread(root_path + f"xenium_{xenium_sampleid}_{xenium_regionid}_enhanced.png", cv2.IMREAD_GRAYSCALE)


    codex_img = cv2.imread(root_path + f"codex_{codex_sampleid}_{codex_regionid}_enhanced.png",
                            cv2.IMREAD_GRAYSCALE)


    coords_xenium = np.array(xenium_gene_data.obs[['x_trans', 'y_trans']].values)
    coords_codex = np.array(codex_protein_data.obs[['x_trans', 'y_trans']].values)

    xenium_img, coords_xenium = resize_image_coords(xenium_img, coords_xenium, image_size)
    codex_img, coords_codex = resize_image_coords(codex_img, coords_codex, image_size)

    coords_list = [coords_xenium,coords_codex]
    image_list = [xenium_img,codex_img]


    # marker_map = {
    # 'p53':   ['TP53','CDKN1A','MDM2'],
    #  'p16':   ['CDKN1A','CDKN1B','TP53'],
    #  'pH2AX': ['ATM','RAD50','CDKN1A'],
    #  'HSP47': ['COL17A1','THBS1','FERMT1'],
    #  'PD1':   ['JAK1','STAT3','MAF'],
    #  'PDL1':  ['JAK1','STAT3','EGFR'],
    #  'CD32b': ['ID2','STAT3','NT5E'],
    #  'SPP1':  ['ITGAV','ITGB1','ITGB5'],
    #  'CD23':  ['ID2','STAT3','NT5E'],
    # }
    marker_map = {
        'p53': ['TP53'],
        'p16': ['CDKN2A'],
        # 'pH2AX': ['ATM'],
        # 'HSP47': ['COL17A1'],
        # 'PD1': ['JAK1'],
        # 'PDL1': ['JAK1'],
        # 'CD32b': ['ID2'],
        # 'SPP1': ['ITGAV'],
        # 'CD23': ['ID2'],
    }

    gene_mat, prot_mat, marker_order, used_genes = build_aligned_modal_channels_simple(xenium_gene_data,codex_protein_data,marker_map)
    gene_mat =gene_mat+0.5

    gene_data_list=[gene_mat,prot_mat]



    # 4. Reduce dimensionality (e.g., PCA)
    combined_data = np.vstack(gene_data_list)
    reduced_data =combined_data
    # reduced_data = reduce_gene_reads(
    #     combined_data,
    #     method='pca',
    #     n_components=2
    # )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    # reduced_data = combined_data
    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end

        coords = coords_list[i]
        image = image_list[i]
        labels = label_list[i]


        feature_matrix, gene_mask = get_gene_feature_matrix(coords, reduced_slice_data, (image.shape[0], image.shape[1]),
                                                            patch_size=8)

        feature_matrix = preprocess_feature_matrix(feature_matrix)
        feature_matrix = remove_salt_pepper(feature_matrix)
        valid_mask = (feature_matrix > 0.01)
        gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)


        # prefix=f"{k}_{i}"
        plot_dimensional_images_side_by_side(feature_matrix,ncols=2)
        #
        # cv2.imwrite("/media/huifang/data/registration/sennet/" + str(k) + "_" + str(i) + f"_image_{image_size}.png",
        #            image)
        # np.save("/media/huifang/data/registration/sennet/" + str(k) + "_" + str(i) + f"_pca_out_{image_size}.npy",
        #         feature_matrix)
        # np.save("/media/huifang/data/registration/sennet/" + str(k) + "_" + str(i) + f"_pca_mask_{image_size}.npy",
        #         gene_mask)
        # np.savez("/media/huifang/data/registration/sennet/" + str(k) + "_" + str(i) + f"_validation_{image_size}",
        #          coord=coords, label=labels)


