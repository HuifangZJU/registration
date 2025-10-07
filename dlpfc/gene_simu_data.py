import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
import matplotlib
import time
import json
import matplotlib.patches as patches
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
style.use('seaborn-white')
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import scanpy as sc

def get_DLPFC_data():
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}

    for id in sample_list:
        adatas[id].image_path = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_hires_image_0.png'.format(id)
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


        adatas[id].image_coor = spatial_locations

    sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],
                     ["151673", "151674", "151675", "151676"]]
    layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in
                    range(len(sample_groups))]
    return layer_groups


def split_visium_coords(coords,row_tol=5):
    coords = np.asarray(coords)
    # 1) Sort by y, then x
    sorted_idx = np.lexsort((coords[:, 0], coords[:, 1]))
    coords = coords[sorted_idx]

    # 2) Group rows by y using a tolerance
    rows = []
    current_row = [coords[0]]
    for pt in coords[1:]:
        if abs(pt[1] - current_row[-1][1]) < row_tol:  # same row
            current_row.append(pt)
        else:  # new row
            rows.append(np.vstack(current_row))
            current_row = [pt]
    rows.append(np.vstack(current_row))

    # 3) Build four slices
    slices = [[] for _ in range(4)]
    for r, row in enumerate(rows):
        # sort this row by x
        row = row[np.argsort(row[:, 0])]
        # even / odd columns in this row
        even_cols = row[::2]
        odd_cols = row[1::2]
        if r % 2 == 0:  # even row index
            slices[0].append(even_cols)  # Slice 1
            slices[1].append(odd_cols)  # Slice 2
        else:  # odd row index
            slices[2].append(even_cols)  # Slice 3
            slices[3].append(odd_cols)  # Slice 4

    slices = [np.vstack(s) if len(s) else np.empty((0, 2)) for s in slices]
    downsampled_slices=[]
    # ratio = 3 / 4  # 0.8
    for s in slices:
        ratio = np.random.uniform(0.4, 0.9)
        n = int(np.floor(s.shape[0] * ratio))  # 856 rows

        idx = np.random.choice(s.shape[0], n, replace=False)
        downsampled_slices.append(s[idx])
    # return slices
    return downsampled_slices

def get_uv_coordinates(slice):
    scale_path = slice.image_scale_path
    image = plt.imread(slice.image_path)
    with open(scale_path, 'r') as f:
        data = json.load(f)
        low_res_scale = data['tissue_hires_scalef']
    uv_coords = slice.image_coor * low_res_scale
    return uv_coords, image


def deform_affine(
        img: np.ndarray,
        coords: np.ndarray,
        *,
        angle_deg: float = 0,
        shear_deg: float = 0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        tx: float = 0,
        ty: float = 0,
        fill_val: Tuple[int, int, int] = (255, 255, 255)
    ):
    """
    Apply an affine deformation (scale_x/scale_y, shear, rotation, translation)
    about the image origin, then re-centre and (if necessary) isotropically
    down-scale so the warped content fits back into the original image
    canvas (same H×W).  Coordinates are transformed by the *same* composite
    matrix.

    Returns
    -------
    warped_img    : np.ndarray
    warped_coords : np.ndarray        # shape = coords.shape
    """
    H, W = img.shape[:2]

    # ----------------- build user-requested affine -------------------
    ang  = np.deg2rad(angle_deg)
    shr  = np.deg2rad(shear_deg)

    D = np.diag([scale_x, scale_y])        # anisotropic scale
    R = np.array([[ np.cos(ang),  np.sin(ang)],
                  [-np.sin(ang),  np.cos(ang)]])
    S = np.array([[1, np.tan(shr)],
                  [0, 1]])

    A_user = R @ S @ D                      # 2×2
    T_user = np.array([tx, ty])             # translation

    # ----------------------------------------------------------------
    # Transform the four image corners to find new bounding box
    corners = np.array([[0, 0, 1],
                        [W, 0, 1],
                        [0, H, 1],
                        [W, H, 1]], dtype=float)
    M_user  = np.hstack([A_user, T_user.reshape(2, 1)])  # 2×3
    proj    = (corners @ np.vstack([M_user, [0, 0, 1]]).T)[:, :2]

    xmin, ymin = proj.min(axis=0)
    xmax, ymax = proj.max(axis=0)
    out_w, out_h = xmax - xmin, ymax - ymin

    # ----------------- centre and scale to fit original canvas ------
    # translation to bring top-left to (0,0)
    shift = np.array([-xmin, -ymin])
    # optional isotropic down-scale if content grew beyond canvas
    scale_fit = min(W / out_w, H / out_h, 1.0)   # ≤1 keeps within canvas

    A_fit = np.eye(2) * scale_fit
    T_fit = shift * scale_fit

    # composite: first user affine, then fit (scale+shift)
    M_comp = np.hstack([A_fit @ A_user,
                        (A_fit @ T_user + T_fit).reshape(2, 1)])  # ← reshape!

    # ----------------- warp image -----------------------------------
    warped_img = cv2.warpAffine(
        img, M_comp, dsize=(W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_val)

    # ----------------- warp coordinates -----------------------------
    hom_c   = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N,3)
    warped_coords = (hom_c @ np.vstack([M_comp, [0, 0, 1]]).T)[:, :2]

    return warped_img, warped_coords

γ          = 0.7           # < 1 ⇒ brighter      (γ-correction)
edge_lw    = 0.4           # marker-edge width
face_alpha = 0.5          # 0‒1 transparency for marker *faces*
marker_sz  = 6          # scatter‐point size
colors      = plt.cm.get_cmap('tab10').colors
# gamma-brighten helper (works for uint8 or float images)
def brighten(im, gamma=0.7):
    im = im.astype(float) / (255. if im.dtype == np.uint8 else 1.0)
    return np.clip(im**gamma, 0, 1)


layer_groups = get_DLPFC_data()

for k,slices in enumerate(layer_groups):
    for sl in slices:
        coords, img = get_uv_coordinates(sl)
        # coords = sl.obsm['spatial']

        subsets = split_visium_coords(coords)
        # colors = ['r', 'g', 'b', 'm']
        #
        # plt.imshow(img)
        # for i, subs in enumerate(subsets):
        #     plt.scatter(subs[:, 0], subs[:, 1], s=10, label=f'Slice {i + 1}', color=colors[i], alpha=0.7)
        # plt.legend()
        # plt.title("Subsampled Non-Overlapping Regularized Slices")
        # plt.show()
        # Plot original + 4 slices
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))  # <- 2 rows × 5 cols
        titles = [f'Original  ({coords.shape[0]} spots)'] + \
                 [f'Slice {i + 1}  ({subsets[i].shape[0]} spots)' for i in range(4)]

        for i in range(5):

            # ----- pick the slice data -------------------------------------
            if i == 0:
                bg = brighten(img, γ)
                xy = coords
            else:
                # random affine to each subslice
                angle = np.random.uniform(-3, 3)
                scalex = np.random.uniform(0.95, 1.35)
                scaley = np.random.uniform(0.95, 1.35)
                shear = np.random.uniform(-2, 2)
                tx, ty = np.random.randint(-1, 1, size=2)

                bg, xy = deform_affine(
                    img, subsets[i - 1],
                    angle_deg=angle,
                    scale_x=scalex,
                    scale_y=scaley,
                    shear_deg=shear,
                    tx=tx, ty=ty,
                    fill_val=(200, 200, 200))
                bg = brighten(bg, γ)

            # ----- row 0: background image only ----------------------------
            ax_img = axes[1, i]
            ax_img.imshow(bg)
            ax_img.axis("off")

            # ----- row 1: spot cloud on white ------------------------------
            ax_pts = axes[0, i]
            ax_pts.imshow(np.ones_like(bg))  # plain white canvas
            ax_pts.scatter(
                xy[:, 0], xy[:, 1],
                s=marker_sz,
                facecolors=(*colors[i % 10][:3], face_alpha),
                edgecolors=colors[i % 10][:3],
                linewidths=edge_lw)
            # ax_pts.invert_yaxis()
            ax_pts.set_aspect("equal")
            ax_pts.axis("off")
            ax_pts.set_title(titles[i], fontsize=20)

        plt.tight_layout()
        plt.show()
