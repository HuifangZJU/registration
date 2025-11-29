import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import scanpy as sc
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

xenium_list = get_data_list(xenium_root_folder)
# Replace with your actual file path
file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'

# Read the text file without header
df = pd.read_csv(file_path, sep=None, engine='python', header=None)

start_line = 0  # zero-based line number
for i, row in enumerate(df.iloc[start_line:].itertuples(index=False, name=None)):
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = row

    xenium_img,xenium_gene_data = get_xenium_data(xenium_list, xenium_sampleid, xenium_regionid)
    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_centroid'].values / 4,
        xenium_gene_data.obs['y_centroid'].values / 4
    ], axis=1)

    codex_img,codex_gene_data = get_codex_data(codex_sampleid, codex_regionid)

    codex_coordinate = np.stack([
        codex_gene_data.obs['x'].values / 2,
        codex_gene_data.obs['y'].values / 2
    ], axis=1)

    plt.imshow(codex_img)
    # plt.scatter(codex_coordinate[:,0],codex_coordinate[:,1])
    plt.show()

    # generate_pseudo_tissue_image(codex_coordinate)

    xenium_img,xenium_coordinate = flip_image(xenium_img,xenium_coordinate)
    xenium_img, xenium_coordinate, codex_img, codex_coordinate = prepare_images_on_shared_canvas(
        xenium_img, xenium_coordinate, codex_img, codex_coordinate)

    # show_pixel_value_distribution(xenium_img)
    generate_pseudo_tissue_image(xenium_coordinate)


    # === Step 1: Overwrite coordinates in .obs ===
    xenium_gene_data.obs['x_trans'] = xenium_coordinate[:, 0]
    xenium_gene_data.obs['y_trans'] = xenium_coordinate[:, 1]
    codex_gene_data.obs['x_trans'] = codex_coordinate[:, 0]
    codex_gene_data.obs['y_trans'] = codex_coordinate[:, 1]

    # xenium_img_enhanced = clahe(xenium_img)
    # codex_img_enhanced = clahe(codex_img)
    xenium_img_enhanced = clahe(xenium_img)
    plt.imshow(xenium_img)
    plt.show()


    save_single_dataset(xenium_img_enhanced, xenium_gene_data, 'xenium', xenium_sampleid, xenium_regionid,
                        "/media/huifang/data/sennet/hf_aligned_data")
    # save_single_dataset(codex_img_enhanced, codex_gene_data, 'codex', codex_sampleid, codex_regionid,
    #                     "/media/huifang/data/sennet/hf_aligned_data")




    # f,a = plt.subplots(1,2,figsize=(15,6))
    # a[0].imshow(create_overlay(xenium_img,codex_img))
    # a[1].imshow(create_overlay(xenium_img_enhanced, codex_img_enhanced))
    # plt.show()


    # f,a = plt.subplots(1,2,figsize=(15,6))
    # a[0].imshow(create_overlay(xenium_img,codex_img))
    # # a[0].imshow(codex_img)
    #
    # a[1].scatter(xenium_coordinate[:,0],xenium_coordinate[:,1],s=2)
    # a[1].scatter(codex_coordinate[:,0],codex_coordinate[:,1],s=2)
    # a[1].axis('equal')
    # a[1].invert_yaxis()
    # plt.show()


