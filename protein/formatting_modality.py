from tifffile import TiffFile
from skimage.measure import block_reduce
import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
import scanpy as sc
import json
from scipy.sparse import issparse
from imageio.v2 import imwrite
from scipy import sparse
import cv2

def transform_image_and_coords(img, coords, rotate_k=0, flip_ud=False, flip_lr=False, show_preview=False):
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be (N,2)"
    img_t = img.copy()
    coords_t = coords.astype(float).copy()

    H, W = img_t.shape[:2]

    # --- rotate in 90° CCW steps ---
    k = int(rotate_k) % 4
    for _ in range(k):
        # before rotation, current dims are (H, W)
        x = coords_t[:, 0]
        y = coords_t[:, 1]
        x_new = y
        y_new = (W - 1) - x
        coords_t = np.column_stack([x_new, y_new])

        img_t = np.rot90(img_t, k=1)  # CCW
        H, W = W, H  # swap dims after 90° rotation

    # --- flips (apply on current dims H,W) ---
    if flip_ud:
        # up-down flip: y -> H-1-y
        coords_t[:, 1] = (H - 1) - coords_t[:, 1]
        img_t = np.flipud(img_t)

    if flip_lr:
        # left-right flip: x -> W-1-x
        coords_t[:, 0] = (W - 1) - coords_t[:, 0]
        img_t = np.fliplr(img_t)

    # Optional: sanity clamp to [0, W-1] & [0, H-1]
    coords_t[:, 0] = np.clip(coords_t[:, 0], 0, W - 1)
    coords_t[:, 1] = np.clip(coords_t[:, 1], 0, H - 1)

    if show_preview:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.scatter(coords[:, 0], coords[:, 1], s=0.1, c='lime', alpha=0.6)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_t)
        plt.scatter(coords_t[:, 0], coords_t[:, 1], s=0.1, c='cyan', alpha=0.6)
        plt.title(f"Rot k={k}, UD={flip_ud}, LR={flip_lr}")
        plt.axis("off")
        plt.show()

    return img_t, coords_t


def transform_image(img, rotate_k=0, flip_ud=False, flip_lr=False):
    img_t = img.copy()
    # --- rotate in 90° CCW steps ---
    k = int(rotate_k) % 4
    for _ in range(k):
        img_t = np.rot90(img_t, k=1)  # CCW
    # --- flips (apply on current dims H,W) ---
    if flip_ud:
        img_t = np.flipud(img_t)
    if flip_lr:
        img_t = np.fliplr(img_t)
    return img_t

def crop_image_and_coords(img, coords, x0, y0, x1, y1):
    H, W = img.shape[:2]

    # Normalize and clamp ROI to image bounds
    xa, xb = sorted([int(np.floor(x0)), int(np.ceil(x1))])
    ya, yb = sorted([int(np.floor(y0)), int(np.ceil(y1))])
    xa = max(10, min(xa, W-10))
    xb = max(10, min(xb, W-10))
    ya = max(10, min(ya, H-10))
    yb = max(10, min(yb, H-10))

    # Crop image
    img_crop = img[ya:yb, xa:xb].copy()

    # Select points inside ROI and shift
    x = coords[:, 0];
    y = coords[:, 1]
    mask = (x >= xa) & (x < xb) & (y >= ya) & (y < yb)
    keep_idx = np.flatnonzero(mask)

    coords_crop = coords[mask].astype(float).copy()
    coords_crop[:, 0] -= xa
    coords_crop[:, 1] -= ya

    return img_crop, coords_crop, keep_idx

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


def read_protein(para1,para2,para3,save=False):
    data_path = root / data / "protein"
    IMG_DIR = data_path / "preview_pngs"

    csv_files = list(data_path.glob("*.csv"))
    SEGMENTATION_CSV = csv_files[0]
    seg = pd.read_csv(SEGMENTATION_CSV)
    x = seg["Centroid X um"].to_numpy() / 3.775
    y = seg["Centroid Y um"].to_numpy() / 3.775
    coords = np.column_stack((x, y))
    col = 'DAPI: Cell: Mean'
    stain_image = plt.imread(IMG_DIR / "ch00_DAPI.png")
    vals = seg[col].to_numpy().astype(float)
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo), 0, 1)

    stain_image,coords = transform_image_and_coords(stain_image,coords, rotate_k=para1, flip_ud=para2, flip_lr=para3,
                               show_preview=False)
    if not save:
        return coords, stain_image, vals
    else:
        save_image(stain_image, outdir + f"{data}_protein_DAPI.png")
        # 1) pick the per-cell protein intensity columns (": Cell: Mean")
        intensity_cols = [c for c in seg.columns if c.endswith(": Cell: Mean")]
        proteins = [c.split(":")[0].strip() for c in intensity_cols]

        # 2) X matrix (cells × proteins)
        X = seg[intensity_cols].to_numpy().astype(np.float32)
        # 4) var (per-protein)
        var = pd.DataFrame(index=pd.Index(proteins, name="protein"))

        # 5) AnnData
        adata = sc.AnnData(X=X, var=var)
        adata.obsm["spatial"] = coords  # <-- your final processed coordinates

        # (optional) store raw intensities as .raw for later normalization
        adata.raw = adata
        # (optional) stash some provenance
        adata.uns["phenocycler"] = {
            "source_csv": str(SEGMENTATION_CSV),
            "intensity_suffix": ": Cell: Mean",
            "transform": {"rotate_k": int(para1), "flip_ud": bool(para2), "flip_lr": bool(para3)},
            "coord_units": "pixels_after_scaling"
        }
        out_h5ad = outdir + f"{data}_protein.h5ad"
        adata.write(out_h5ad)
        print(f"✅ Saved: {out_h5ad}  shape={adata.shape}, spatial={adata.obsm['spatial'].shape}")
        return coords, stain_image, vals



def read_xenium(para1,para2,para3,para4,para5,para6,save=False):
    data_path = root / data / "xenium"
    para_path = data_path / "experiment.xenium"

    dapi_img = plt.imread(data_path / "morphology_focus.ome_down10x.png")
    he_img = plt.imread(next(data_path.glob("XeniumHE*_down5x.png")))

    adata = sc.read_h5ad(data_path / "adata_xenium_cell_level.h5ad")

    with open(para_path, "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size")

    # === 5. Plot the marker genes ===
    x = adata.obsm["spatial"][:, 0] / pixel_size
    y = adata.obsm["spatial"][:, 1] / pixel_size
    coords = np.column_stack((x, y))
    coords = coords/10
    dapi_img, coords = transform_image_and_coords(dapi_img, coords, rotate_k=para1, flip_ud=para2, flip_lr=para3,
                                                     show_preview=False)
    he_img = transform_image(he_img, rotate_k=para4, flip_ud=para5, flip_lr=para6)

    # Extract values safely as dense
    gene = "ACTA2"
    vals = adata[:, gene].X
    if issparse(vals):
        vals = vals.toarray().flatten()
    else:
        vals = np.asarray(vals).flatten()

    # Robust intensity normalization
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo), 0, 1)

    if not save:
        return coords,dapi_img,he_img,vals
    else:
        save_image(dapi_img, outdir + f"{data}_xenium_DAPI.png")
        save_image(he_img, outdir + f"{data}_xenium_HE.png")

        adata.X = sparse.csr_matrix(adata.X)
        adata.obsm["spatial"] = coords
        out_h5ad = outdir + f"{data}_xenium.h5ad"
        adata.write(out_h5ad)
        print(f"✅ Saved: {out_h5ad}  shape={adata.shape}, spatial={adata.obsm['spatial'].shape}")
        return coords,dapi_img,he_img,vals

def read_visium(para1,para2,para3,para4,para5,para6,para7,save=False):
    data_path = root / data / "visium"
    spatial_dir = data_path / "spatial"
    gene_name = "ACTA2"  # choose your gene of interest

    adata = sc.read_h5ad(data_path / "adata_visium.h5ad")
    scales = adata.uns['spatial_scalefactors']
    # Usually named "tissue_hires_image.png" or ".tif"
    img_path = next(spatial_dir.glob("tissue_hires_image_1.png"))
    he = plt.imread(img_path)

    # --- Compute scaled spot coordinates ---
    scale = scales["tissue_hires_scalef"]
    xy = adata.obsm["spatial"] * scale  # scale Visium spot positions to match hires image

    # --- Get gene expression values ---
    he, xy = transform_image_and_coords(he, xy, rotate_k=para1, flip_ud=para2, flip_lr=para3,
                                                     show_preview=False)
    he, xy,keep_idx = crop_image_and_coords(he, xy, para4, para5, para6, para7)

    vals = adata[:, gene_name].X.toarray().flatten()
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo + 1e-6), 0, 1)
    vals = vals[keep_idx]

    if not save:
        return xy,he,vals
    else:
        save_image(he, outdir + f"{data}_visium_HE.png")
        adata = adata[keep_idx].copy()
        adata.X = sparse.csr_matrix(adata.X)
        adata.obsm["spatial"] = xy
        out_h5ad = outdir + f"{data}_visium.h5ad"
        adata.write(out_h5ad)
        print(f"✅ Saved: {out_h5ad}  shape={adata.shape}, spatial={adata.obsm['spatial'].shape}")
        return xy, he, vals


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
             "LUAD_3_A", "TSU_21", "TSU_24", "TSU_30", "TSU_35"]

formatting_paras=[
[0,False,False,-1,False,False,-1,False,False,0,False,True,120,450,2100,2100],
[0,False,False,0,False,False,0,True,True,1,False,True,550,400,1450,1450],
[0,False,True,0,False,False,0,True,False,1,False,False,400,0,1900,2000],
[0,False,False,1,True,True,0,False,False,-1,False,True,400,600,2000,2000],
[0,False,True,0,False,True,0,False,True,-1,False,False,0,0,2000,2000],
[0,False,False,1,False,False,0,True,True,1,False,True,100,500,1700,1300],
[0,False,False,0,False,False,1,False,True,0,True,True,300,0,1600,2000],
[0,False,False,0,False,False,1,False,False,0,True,False,250,0,1700,2000]]


# ------------------ user paths ------------------
root = Path("/media/huifang/data/registration/phenocycler/")
outdir = "/media/huifang/data/registration/phenocycler/H5ADs/"
start_idx = 0
for data, para in zip(datasets[start_idx:], formatting_paras[start_idx:]):

    print(data)

    protein_coords,protein_img,protein_signal= read_protein(para[0], para[1], para[2],save=True)
    visium_coords,visium_he,visium_signal = read_visium(para[3], para[4], para[5],para[12], para[13], para[14], para[15],save=True)
    xenium_coords, xenium_morph,xenium_he,xenium_signal = read_xenium(para[6], para[7], para[8],para[9], para[10], para[11],save=True)


    # xenium_coords = xenium_coords[::2]
    # xenium_signal = xenium_signal[::2]
    #
    #
    # f,a = plt.subplots(2,2,figsize=(16,16))
    # a[0,0].imshow(protein_img,cmap='gray')
    # a[0,0].scatter(protein_coords[:,0], protein_coords[:,1], c=protein_signal, s=0.5, cmap="magma",
    #          alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[0, 1].imshow(visium_he)
    # a[0, 1].scatter(visium_coords[:, 0], visium_coords[:, 1], c=visium_signal, s=2.5, cmap="magma",
    #                 alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[1,0].imshow(xenium_morph, cmap='gray')
    # a[1,0].scatter(xenium_coords[:, 0], xenium_coords[:, 1], c=xenium_signal, s=0.5, cmap="magma",
    #              alpha=0.9, edgecolors="none", rasterized=True)
    #
    # a[1, 1].imshow(xenium_he)
    #
    # plt.show()
