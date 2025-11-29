from tifffile import TiffFile
from skimage.measure import block_reduce
import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
# ------------------------------------------------

def get_color_bar(vals):
    # Keep only finite values for stats
    finite = np.isfinite(vals)
    if not finite.any():
        print(f"[SKIP] {name}: no finite values")
        return
    # Robust limits
    lo, hi = np.nanpercentile(vals[finite], [2, 98])
    if not np.isfinite(lo): lo = np.nanmin(vals[finite])
    if not np.isfinite(hi): hi = np.nanmax(vals[finite])

    # Handle flat/degenerate cases
    if hi <= lo:
        hi = lo + 1e-6

    # Choose normalization: log if very skewed and positive
    use_log = (hi > 0) and (lo > 0) and (hi / max(lo, 1e-12) > 1e3)
    if use_log:
        norm = LogNorm(vmin=lo, vmax=hi)
        cbar_label = f"Intensity (log, clipped @ {lo:.3g}-{hi:.3g})"
    else:
        norm = Normalize(vmin=lo, vmax=hi)
        cbar_label = f"Intensity (clipped @ {lo:.3g}-{hi:.3g})"
    return norm,cbar_label

def scale01(img, p_lo=1, p_hi=99.8):
    """Scale image intensities to 0–1 using robust percentiles."""
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def save_staining_images(OUT_DIR):
    qptiff_files = list(data_path.glob("*.qptiff"))
    QPTIFF_PATH = qptiff_files[0]

    channel_names = [c.split(':')[0] for c in seg.columns.tolist() if 'Mean' in c]
    # downsample factor (20 000 → 2 000)
    DOWNSAMPLE = 10
    with TiffFile(QPTIFF_PATH) as tif:
        s = tif.series[0]  # fluorescence series (CYX)
        z = s.asarray(out="memmap")  # shape (C, Y, X)
        nC = z.shape[0]
        print(f"Loaded {nC} channels, size={z.shape[1]}×{z.shape[2]}")

        for ci in range(nC):
            name = channel_names[ci] if ci < len(channel_names) else f"Ch{ci:02d}"
            print(f"Processing {ci:02d}: {name}")
            img = np.asarray(z[ci], dtype=np.float32)

            # downsample for faster plotting
            if DOWNSAMPLE > 1:
                img = block_reduce(img, block_size=(DOWNSAMPLE, DOWNSAMPLE), func=np.mean)

            # rescale for visibility
            vis = scale01(img)

            # save to png
            png_path = OUT_DIR / f"ch{ci:02d}_{name}.png"
            imageio.imwrite(png_path, (vis * 255).astype(np.uint8))

            # # optional quick show (comment out if batch running)
            # plt.figure(figsize=(6,6))
            # plt.imshow(vis, cmap="gray")
            # plt.title(f"{ci:02d} – {name}")
            # plt.axis("off")
            # plt.show()

    print(f"✅  Done. Visible PNGs saved in: {OUT_DIR.resolve()}")

def save_reads_images(OUT_DIR):
    x = seg["Centroid X um"].to_numpy()
    y = seg["Centroid Y um"].to_numpy()

    ci = 0
    for col in seg.columns:
        if "Mean" in col:
            name = col.split(":")[0]  # extract channel name like "CD4"
            print(f"Processing {name}")
            vals = seg[col].to_numpy().astype(float)

            norm, cbar_label = get_color_bar(vals)
            # Plot using raw vals + norm (so colorbar shows real units)
            fig = plt.figure(figsize=(16, 16))
            ax = plt.gca()
            sc = ax.scatter(x, y, c=vals, norm=norm, s=2, cmap="viridis",
                            alpha=0.9, edgecolors="none", rasterized=True)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.axis("off")
            plt.title(f"ch{ci:02d}_{name}")

            cb = plt.colorbar(sc, shrink=0.75)
            cb.set_label(cbar_label)
            # plt.show()
            #
            fn = OUT_DIR / f"ch{ci:02d}_{name}_reads.png"
            plt.savefig(fn, dpi=150, bbox_inches="tight")
            plt.close(fig)
            ci = ci + 1
# ------------------ user paths ------------------
root = Path("/media/huifang/data/registration/phenocycler/")
h5ad_path = "/media/huifang/data/registration/phenocycler/H5ADs/"
for data in ["LUAD_2_A", "TSU_20_1", "TSU_23", "TSU_28", "TSU_33",
             "LUAD_3_A", "TSU_21", "TSU_24", "TSU_30", "TSU_35"]:

    print(data)
    data_path = root / data / "protein"
    IMG_DIR = data_path / "preview_pngs"
    # OUT_DIR.mkdir(exist_ok=True)
    # Find qptiff and csv files automatically

    csv_files = list(data_path.glob("*.csv"))
    SEGMENTATION_CSV = csv_files[0]
    seg = pd.read_csv(SEGMENTATION_CSV)
    # print(seg.columns.tolist())
    # test = input()

    x = seg["Centroid X um"].to_numpy()/3.775
    y = seg["Centroid Y um"].to_numpy()/3.775

    ci = 0
    for col in seg.columns:
        if ci>0:
            continue
        if "Mean" in col:
            name = col.split(":")[0]  # extract channel name like "CD4"
            stain_image = plt.imread(IMG_DIR / f"ch{ci:02d}_{name}.png")
            vals = seg[col].to_numpy().astype(float)
            norm, cbar_label = get_color_bar(vals)
            plt.figure(figsize=(8, 8))
            plt.imshow(stain_image,cmap='gray')
            plt.scatter(x, y, c=vals, norm=norm, s=0.5, cmap="magma",
                     alpha=0.9, edgecolors="none", rasterized=True)
            plt.show()
            ci = ci+1



