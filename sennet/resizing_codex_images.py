import os
import tifffile
import numpy as np
import cv2
from PIL import Image
# Define root directory
root_folder = '/media/huifang/data/sennet/codex'
# Traverse and process all .tif/.tiff images
for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            # Load image
            img = tifffile.imread(file_path)
            # Extract channel
            channel_img = img[0, :, :]

            # Convert to 8-bit with contrast stretching
            if channel_img.dtype != 'uint8':
                p1, p99 = np.percentile(channel_img, [1, 99])
                channel_img = np.clip(channel_img, p1, p99)
                channel_img = (255 * (channel_img - p1) / (p99 - p1 + 1e-8)).astype('uint8')
            # Convert to PIL Image
            im = Image.fromarray(channel_img)
            # Resize to quarter size
            im_half = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
            im_quarter = im.resize((im.width // 4, im.height // 4), Image.LANCZOS)

            # Build new filenames
            base, ext = os.path.splitext(filename)
            half_path = os.path.join(dirpath, f"{base}_channel0_half.png")
            quarter_path = os.path.join(dirpath, f"{base}_channel0_quarter.png")

            im_half.save(half_path)
            im_quarter.save(quarter_path)

            print(f"Processed {filename} → saved half and quarter images.")
            # test = input()
