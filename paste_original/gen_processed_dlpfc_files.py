import pandas as pd
import seaborn as sns
import json
import cv2
import scanpy as sc
from sklearn.decomposition import PCA
from functools import reduce
from PIL import Image
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import median_filter
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def get_uv_coordinates(slice):
    scale_path = slice.image_scale_path
    image = Image.open(slice.image_path)
    with open(scale_path, 'r') as f:
        data = json.load(f)
        low_res_scale = data['tissue_hires_scalef']
    uv_coords = slice.obsm['spatial'] * low_res_scale
    return uv_coords, image



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



def get_DLPFC_data():
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}

    for id in sample_list:
        adatas[id].image_path = '/media/huifang/data1/registration/humanpilot/{0}/spatial/tissue_hires_image_image_0.png'.format(id)
        adatas[id].image_scale_path = '/media/huifang/data1/registration/humanpilot/{0}/spatial/scalefactors_json.json'.format(id)
        adatas[id].spatial_prefix = '/media/huifang/data1/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)


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

def preprocess_data(layer_groups):
    crop_paras = [[(50, 0, 1850, 1024), (50, 50, 1900, 1024), (50, 120, 1896, 1024), (50, 155, 1861, 1024)],
                  [(300, 200, 1650, 1024), (350, 220, 1550, 1024), (350, 290, 1600, 1024), (360, 230, 1600, 1024)],
                  [(160, 10, 1770, 1024), (160, 50, 1770, 1024), (160, 120, 1750, 1024), (180, 20, 1770, 1024)]]
    image_size = 512
    layer_to_color_map = {'Layer{0}'.format(i + 1): i for i in range(6)}
    layer_to_color_map['WM'] = 6

    for k, slices in enumerate(layer_groups):
        for i, sl in enumerate(slices):
            labels = list(sl.obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
            sl.obs['layer_labels'] = labels
            labels = np.asarray(labels)
            coords, image = get_uv_coordinates(sl)  # your custom function
            cropped_image, cropped_coor = crop_square_then_resize_square(image, coords, crop_paras[k][i], image_size)
            sl.obsm['spatial'] = cropped_coor
            sl.uns['512_image'] = np.asarray(cropped_image)
    return layer_groups

def show_clusters(coors,labels):
    unique_labels = np.unique(labels)  # sorted unique label values
    palette = sns.color_palette("deep", len(unique_labels))  # e.g. 'deep', 'tab10', etc.
    color_index = np.searchsorted(unique_labels, labels)
    colors = np.array(palette)[color_index]

    plt.scatter(coors[:, 0], coors[:, 1], s=10, color=colors)
    plt.show()


layer_groups = get_DLPFC_data()
layer_groups = preprocess_data(layer_groups)

sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],
                 ["151673", "151674", "151675", "151676"]]

for j in range(len(sample_groups)):
    for i in range(len(sample_groups[j])):
        sl = layer_groups[j][i]
        region_label = np.load("/media/huifang/data1/registration/DLPFC/huifang/"+str(j)+"_"+str(i)+"_region_label.npy")
        sl.obs["region_labels"] = region_label


        layer_groups[j][i].write("/media/huifang/data1/registration/DLPFC/huifang/h5adfile/"+sample_groups[j][i]+".h5ad")

        # sl = sc.read_h5ad("/media/huifang/data1/registration/DLPFC/huifang/h5adfile/"+sample_groups[j][i]+".h5ad")
        # plt.imshow(sl.uns['512_image'])
        # plt.scatter(sl.obsm['spatial'][:, 0], sl.obsm['spatial'][:, 1],c=sl.obs["region_labels"], cmap='tab10')
        # plt.show()



