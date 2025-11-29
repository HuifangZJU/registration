import os
import tifffile
import numpy as np
import cv2
from PIL import Image
# Define root directory
from pathlib import Path


root = Path("/media/huifang/data/registration/phenocycler/")

for data in ["LUAD_2_A", "TSU_20_1", "TSU_23", "TSU_28", "TSU_33",
             "LUAD_3_A", "TSU_21", "TSU_24", "TSU_30", "TSU_35"]:

    print(data)
    # data_path = root / data / "xenium"
    # tiff_files = list(data_path.glob("morphology_focus*.tif"))
    data_path = root / data / "visium"
    tiff_files = list(data_path.glob("*.tif"))

    for file_path in tiff_files:
        channel_img = tifffile.imread(file_path)
        # Convert to 8-bit with contrast stretching
        if channel_img.dtype != 'uint8':
            p1, p99 = np.percentile(channel_img, [1, 99])
            channel_img = np.clip(channel_img, p1, p99)
            channel_img = (255 * (channel_img - p1) / (p99 - p1 + 1e-8)).astype('uint8')
        # Convert to PIL Image
        im = Image.fromarray(channel_img)
        # Resize to quarter size
        im_fifth = im.resize((im.width // 5, im.height // 5), Image.LANCZOS)
        im_tenth = im.resize((im.width // 10, im.height // 10), Image.LANCZOS)
        im_fifth.save(str(file_path)[:-4] + "_down5x.png")
        im_tenth.save(str(file_path)[:-4] + "_down10x.png")
        print(f"Processed {file_path} → saved down5x and down10x.")
