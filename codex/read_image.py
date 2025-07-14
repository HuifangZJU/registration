import tifffile as  tiff
from matplotlib import pyplot as plt
import scanpy as sc
import pandas as pd
import os
from PIL import Image
from itertools import islice
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
def save_channle_0(img):
    img0 = img[0, :, :]

    # Convert to 8-bit image if needed
    if img0.dtype != 'uint8':
        img0 = (255 * (img0 - img0.min()) / (img0.ptp() + 1e-8)).astype('uint8')

    # Convert to PIL Image
    im = Image.fromarray(img0)

    # Output directory
    out_dir = "/media/huifang/data/sennet/codex/20250314_Yang_SenNet_S4/per_tissue_region-selected/reg001"
    os.makedirs(out_dir, exist_ok=True)

    # Save in original size
    im.save(os.path.join(out_dir, "channel0_original.png"))

    # Save half-size
    im_half = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    im_half.save(os.path.join(out_dir, "channel0_half.png"))

    # Save quarter-size
    im_quarter = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)
    im_quarter.save(os.path.join(out_dir, "channel0_quarter.png"))


# img = tiff.imread("/media/huifang/data/sennet/codex/20250314_Yang_SenNet_S4/per_tissue_region-selected/reg001_X01_Y01_Z01.tif")
# out_dir = "/media/huifang/data/sennet/codex/20250314_Yang_SenNet_S4/per_tissue_region-selected/reg001"
img = tiff.imread("/media/huifang/data/sennet/xenium/1812/morphology_focus/morphology_focus_0000.ome.tif")
print(img.shape)
out_dir = "/media/huifang/data/sennet/xenium/1812/image_channel/"
# for i in range(img.shape[0]):
for i in [0]:
    # Extract channel
    channel_img = img[i, :, :]

    # Convert to 8-bit with contrast stretching
    if channel_img.dtype != 'uint8':
        p1, p99 = np.percentile(channel_img, [1, 99])
        channel_img = np.clip(channel_img, p1, p99)
        channel_img = (255 * (channel_img - p1) / (p99 - p1 + 1e-8)).astype('uint8')
    # Convert to PIL Image
    im = Image.fromarray(channel_img)


    # Resize to quarter size
    im_quarter = im.resize((im.width // 8, im.height // 8), Image.LANCZOS)
    im_quarter_rotated = im_quarter.rotate(180)

    # Save
    im_quarter_rotated.save(os.path.join(out_dir, f"channel{i:02d}_one_eighth.png"))

