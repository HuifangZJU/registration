import os

import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2
from PIL import Image
# Define root directory
root_folder = '/media/huifang/data/Xenium/xenium_data'
# Traverse and process all .tif/.tiff images
# datasets=['Xenium_V1_FFPE_TgCRND8_2_5_months','Xenium_V1_FFPE_TgCRND8_5_7_months','Xenium_V1_FFPE_TgCRND8_17_9_months',
#           'Xenium_V1_FFPE_wildtype_2_5_months','Xenium_V1_FFPE_wildtype_5_7_months','Xenium_V1_FFPE_wildtype_13_4_months']
datasets=['Xenium_V1_FFPE_Human_Breast_ILC','Xenium_V1_FFPE_Human_Breast_ILC_With_Addon']
for data in datasets:
    file_path = os.path.join(root_folder, data,'morphology_focus.ome.tif')

    channel_img = tifffile.imread(file_path)



    # Convert to 8-bit with contrast stretching
    if channel_img.dtype != 'uint8':
        p1, p99 = np.percentile(channel_img, [1, 99])
        channel_img = np.clip(channel_img, p1, p99)
        channel_img = (255 * (channel_img - p1) / (p99 - p1 + 1e-8)).astype('uint8')
    # Convert to PIL Image
    im = Image.fromarray(channel_img)
    # Resize to quarter size
    im_half = im.resize((im.width // 10, im.height // 10), Image.LANCZOS)
    im_quarter = im.resize((im.width // 20, im.height // 20), Image.LANCZOS)

    # Build new filenames
    half_path = os.path.join(root_folder, data, "morphology_down10x.png")
    quarter_path = os.path.join(root_folder, data, "morphology_down20x.png")

    im_half.save(half_path)
    im_quarter.save(quarter_path)
    print(f"Processed {file_path} → saved down10x and down20x images.")
