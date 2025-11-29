import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
import scanpy as sc
import json
from scipy.sparse import issparse
from imageio.v2 import imwrite
from scipy import sparse
from skimage.color import rgb2gray
from imageio.v2 import imwrite
import cv2

def read_protein():
    out_h5ad = datadir + f"{data}_protein.h5ad"
    adata = sc.read_h5ad(out_h5ad)
    stain_image = plt.imread(datadir + f"{data}_protein_DAPI.png")
    coords = adata.obsm['spatial']

    vals = adata[:, "DAPI"].X.astype(float)
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo), 0, 1)

    return coords, stain_image, vals,adata


def read_xenium():
    out_h5ad = datadir + f"{data}_xenium.h5ad"
    adata = sc.read_h5ad(out_h5ad)
    coords = adata.obsm["spatial"]

    dapi_img = plt.imread(datadir + f"{data}_xenium_DAPI.png")
    he_img = plt.imread(datadir + f"{data}_xenium_HE.png")


    gene = "ACTA2"
    vals = adata[:, gene].X
    if issparse(vals):
        vals = vals.toarray().flatten()
    else:
        vals = np.asarray(vals).flatten()

    # Robust intensity normalization
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo), 0, 1)

    return coords,dapi_img,he_img,vals,adata


def read_visium():
    out_h5ad = datadir + f"{data}_visium.h5ad"
    adata = sc.read_h5ad(out_h5ad)

    he = plt.imread(datadir + f"{data}_visium_HE.png")
    xy = adata.obsm["spatial"]

    gene_name = "ACTA2"  # choose your gene of interest



    vals = adata[:, gene_name].X.toarray().flatten()
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo + 1e-6), 0, 1)

    return xy,he,vals,adata

def norm_grayimg(img_gray,lo_th,hi_th):
    lo, hi = np.nanpercentile(img_gray, [lo_th, hi_th])  # bottom/top 2%
    img_gray = np.clip(img_gray, lo, hi)
    img_gray = (img_gray - lo) / (hi - lo + 1e-6)

    return img_gray

def clahe(dapi_img):
    # Convert to uint8 if needed
    img_uint8 = (dapi_img * 255).astype('uint8') if dapi_img.max() <= 1.0 else dapi_img

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    img_clahe = clahe.apply(img_uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(img_clahe, kernel, iterations=2)

    return dilated

def save_image(img_to_save,out_img_path):
    # Preserve appearance: only coerce dtype if needed (no contrast scaling)
    if img_to_save.dtype.kind in ("f",):  # float image
        # if in [0,1], scale to 0–255; otherwise clamp to 0–255
        mx = float(np.nanmax(img_to_save)) if np.isfinite(img_to_save).any() else 1.0
        if mx <= 1.0:
            img_to_save = (np.clip(img_to_save, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            img_to_save = np.clip(img_to_save, 0.0, 255.0).round().astype(np.uint8)
    elif img_to_save.dtype != np.uint8:
        # integer but not uint8 → clamp to 0–255
        img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)

    imwrite(out_img_path, img_to_save)
    print(f"✅ Saved final stain image: {out_img_path}")


def keep_foreground_to_black(
    img: np.ndarray,
    method: str = "otsu",            # 'fixed' | 'otsu' | 'adaptive'
    thresh: float = 128,             # used if method='fixed'
    adaptive_block_size: int = 31,   # odd, >=3
    adaptive_C: int = 5,
    morph_open: int = 0,             # e.g., 3 to remove speckles
    feather_px: int = 0              # soft edge; 0 = hard mask
) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("img must be 2D grayscale")

    # --- prepare uint8 for thresholding ---
    if img.dtype == np.uint8:
        img_u8 = img
        imin, imax = 0.0, 255.0
    else:
        imin, imax = float(np.nanmin(img)), float(np.nanmax(img))
        if imax > imin:
            img_u8 = ((img - imin) / (imax - imin) * 255).clip(0, 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img, dtype=np.uint8)

    # --- make a foreground mask ---
    if method == "fixed":
        _, mask = cv2.threshold(img_u8, float(thresh), 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif method == "adaptive":
        if adaptive_block_size < 3 or adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be odd and >=3")
        mask = cv2.adaptiveThreshold(
            img_u8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size, adaptive_C
        )
    else:
        raise ValueError("method must be 'fixed', 'otsu', or 'adaptive'")

    # --- optional cleanup ---
    if morph_open and morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # --- optional feather (soft mask) ---
    if feather_px and feather_px > 0:
        # distance from background -> foreground edge
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        soft = np.clip(dist / float(feather_px), 0.0, 1.0)
        # scale to original dtype
        if np.issubdtype(img.dtype, np.integer):
            out = (img.astype(np.float32) * soft).astype(img.dtype)
        else:
            out = img * soft.astype(img.dtype)
        return out

    # --- hard mask ---
    if np.issubdtype(img.dtype, np.integer):
        out = img.copy()
        out[mask == 0] = 0
        return out
    else:
        out = img.copy()
        out[mask == 0] = img.dtype.type(0.0)
        return out


def standardize_image_and_coords(
    img: np.ndarray,
    coords: np.ndarray,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    out_size: tuple = (1024, 1024),
    center: tuple = None,
    interp: int = cv2.INTER_LINEAR,
) -> tuple:

    if img.ndim != 2:
        raise ValueError("`img` must be a 2D grayscale array")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must be (N, 2) array of (x, y) points")

    in_h, in_w = img.shape[:2]
    out_w, out_h = int(out_size[0]), int(out_size[1])

    # Rotation/scaling about the chosen center (default: image center)
    if center is None:
        cx, cy = (in_w - 1) / 2.0, (in_h - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    # 2x3 affine: rotation + scale around (cx, cy)
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)  # keeps (cx, cy) fixed

    # Translate so that the rotation center lands at the center of the output canvas
    out_cx, out_cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    dx, dy = out_cx - cx, out_cy - cy

    # Compose translation with rotation/scale: M_total = T * M_rot
    M_total = M_rot.copy()
    M_total[0, 2] += dx
    M_total[1, 2] += dy

    # Warp the image onto the black canvas
    out_img = cv2.warpAffine(
        img,
        M_total,
        (out_w, out_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Transform coordinates (x, y, 1) -> (x', y')
    ones = np.ones((coords.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([coords.astype(np.float64), ones])  # (N, 3)
    new_xy = pts_h @ M_total.T  # (N, 2)

    return out_img, new_xy.astype(np.float64)

datasets=["LUAD_2_A", "TSU_20_1",  "TSU_33",
             "LUAD_3_A", "TSU_24", "TSU_35"]

paras=[[-20,0.36,0.0,0.2,3.0,0.54],
       [0, 1, -10, 0.55, -15, 1],
       [0, 0.5, -30, 0.28, 0, 0.55],
       [-20, 0.45, 8, 0.24, 5, 0.58],
       [-30, 0.48, 10, 0.28, -3, 0.6],
       [0, 0.38, 5, 0.20, -30, 0.4]]

root = Path("/media/huifang/data/registration/phenocycler/")
datadir = "/media/huifang/data/registration/phenocycler/H5ADs/"
start_idx = 0
for data,para in zip (datasets[start_idx:],paras[start_idx:]):

    print(data)

    protein_coords,protein_dapi,protein_signal,protein_adata= read_protein()
    visium_coords,visium_he,visium_signal,visium_adata = read_visium()
    xenium_coords, xenium_dapi,xenium_he,xenium_signal,xenium_adata = read_xenium()




    visium_gray = rgb2gray(visium_he).astype(np.float32)

    protein_dapi = norm_grayimg(protein_dapi,5,85)
    xenium_dapi = norm_grayimg(xenium_dapi,5,90)
    visium_gray = norm_grayimg(visium_gray,1,99)
    visium_gray = 1 - visium_gray

    protein_dapi = keep_foreground_to_black(protein_dapi)
    xenium_dapi = keep_foreground_to_black(xenium_dapi)
    visium_gray = keep_foreground_to_black(visium_gray)

    protein_dapi_enhanced = clahe(protein_dapi)
    xenium_dapi_enhanced = clahe(xenium_dapi)
    visium_gray_enhanced = clahe(visium_gray)

    save_image(protein_dapi_enhanced, datadir + f"{data}_protein_DAPI_enhanced.png")
    save_image(xenium_dapi_enhanced, datadir + f"{data}_xenium_DAPI_enhanced.png")
    save_image(visium_gray_enhanced, datadir + f"{data}_visium_GRAY_enhanced.png")







    # protein_dapi_trans, protein_coords_trans = standardize_image_and_coords(
    #     protein_dapi,
    #     protein_coords,
    #     angle_deg=para[0],
    #     scale=para[1],
    #     out_size=(1024, 1024),
    # )
    # xenium_dapi_trans, xenium_coords_trans = standardize_image_and_coords(
    #     xenium_dapi,
    #     xenium_coords,
    #     angle_deg=para[2],
    #     scale=para[3],
    #     out_size=(1024, 1024),
    # )
    # visium_gray_trans, visium_coords_trans = standardize_image_and_coords(
    #     visium_gray,
    #     visium_coords,
    #     angle_deg=para[4],
    #     scale=para[5],
    #     out_size=(1024, 1024),
    # )
    #
    # protein_adata.obsm['spatial']= protein_coords_trans
    # xenium_adata.obsm['spatial']=xenium_coords_trans
    # visium_adata.obsm['spatial'] = visium_coords_trans
    #
    #
    # protein_dapi_trans = keep_foreground_to_black(protein_dapi_trans)
    # xenium_dapi_trans = keep_foreground_to_black(xenium_dapi_trans)
    # visium_gray_trans = keep_foreground_to_black(visium_gray_trans)
    #
    # protein_dapi_trans_enhanced = clahe(protein_dapi_trans)
    # xenium_dapi_trans_enhanced = clahe(xenium_dapi_trans)
    # visium_gray_trans_enhanced = clahe(visium_gray_trans)
    #
    # protein_adata.write(Path(datadir +f"{data}_protein_trans.h5ad"))
    # xenium_adata.write(Path(datadir + f"{data}_xenium_trans.h5ad"))
    # visium_adata.write(Path(datadir + f"{data}_visium_trans.h5ad"))
    #
    # save_image(protein_dapi_trans, datadir + f"{data}_protein_DAPI_trans.png")
    # save_image(xenium_dapi_trans, datadir + f"{data}_xenium_DAPI_trans.png")
    # save_image(visium_gray_trans, datadir + f"{data}_visium_GRAY_trans.png")
    # # # #
    # save_image(protein_dapi_trans_enhanced, datadir + f"{data}_protein_DAPI_trans_enhanced.png")
    # save_image(xenium_dapi_trans_enhanced, datadir + f"{data}_xenium_DAPI_trans_enhanced.png")
    # save_image(visium_gray_trans_enhanced, datadir + f"{data}_visium_GRAY_trans_enhanced.png")


    # xenium_coords_trans = xenium_coords_trans[::2]
    # xenium_signal = xenium_signal[::2]
    # f,a = plt.subplots(2,2,figsize=(16,16))
    # a[0,0].imshow(protein_dapi_trans,cmap='gray')
    # a[0,0].scatter(protein_coords_trans[:,0], protein_coords_trans[:,1], c=protein_signal, s=5, cmap="magma",
    #          alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[0, 1].imshow(visium_gray_trans)
    # a[0, 1].scatter(visium_coords_trans[:, 0], visium_coords_trans[:, 1], c=visium_signal, s=2.5, cmap="magma",
    #                 alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[1,0].imshow(xenium_dapi_trans, cmap='gray')
    # a[1,0].scatter(xenium_coords_trans[:, 0], xenium_coords_trans[:, 1], c=xenium_signal, s=0.5, cmap="magma",
    #              alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[1, 1].imshow(xenium_he)
    #
    # plt.show()