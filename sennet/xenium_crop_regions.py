import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import tifffile as tiff
import os
from PIL import Image
import time
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

def annotate_boxes():
    # --- Storage for boxes ---
    boxes = []

    # --- Callback function to store box coordinates ---
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        boxes.append([x_min, y_min, x_max, y_max])
        print(f"Box added: {x_min, y_min, x_max, y_max}")

    # --- Start interactive plot ---
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Draw boxes on sample {sample_id}. Close window to finish.")

    toggle_selector = RectangleSelector(ax, onselect,
                                        useblit=True,
                                        button=[1],
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)

    plt.show()

    # --- Save boxes to CSV ---
    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    df.to_csv(subfolder + '/crop_boxes.csv', index=False)
    print(f"Saved {len(boxes)} boxes to crop_boxes.csv")

def visualize_boxes():
    crop_df = pd.read_csv(subfolder + '/crop_boxes.csv')
    crop_section = crop_df[['x1', 'y1', 'x2', 'y2']].values.tolist()
    # --- Visualize crop regions ---
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img, cmap='gray')
    colors = ['red', 'green', 'blue', 'yellow']
    for i, (x1, y1, x2, y2) in enumerate(crop_section):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'Region {i}', color=colors[i % len(colors)], fontsize=12)
    plt.title(f"Crop Regions on Tissue Image - Sample {sample_id}")
    plt.axis('off')
    plt.show()

def save_regional_image():
    outdir = subfolder + "/morphology_focus/regional_images"
    os.makedirs(outdir, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(crop_section):
        crop_img = img[y1:y2, x1:x2]  # Note: image is indexed as [y, x]
        if crop_img.dtype != 'uint8':
            p1, p99 = np.percentile(crop_img, [1, 99])
            crop_img = np.clip(crop_img, p1, p99)
            crop_img = (255 * (crop_img - crop_img.min()) / (p99 - p1 + 1e-8)).astype('uint8')
        im = Image.fromarray(crop_img)
        im.save(outdir + f"/channel0_region_{i}_one_sixteenth.png")
    print('saved')

root_path = "/media/huifang/data/sennet/xenium/"

dataset = open(root_path+'data_list.txt')
lines = dataset.readlines()
for i in range(0,len(lines)):
    print(i)
    line = lines[i].rstrip().split(' ')
    subfolder = os.path.join(root_path,line[0])
    if os.path.exists(os.path.join(subfolder,'outs')):
        subfolder = os.path.join(subfolder,'outs')
    print(subfolder)
    sample_id = line[1]

    crop_df = pd.read_csv(subfolder + '/crop_boxes.csv')
    crop_section = crop_df[['x1', 'y1', 'x2', 'y2']].values.tolist()
    # if not crop_section:
    #     continue

    img_path = subfolder + '/morphology_focus/channel0_quarter.png'
    img = plt.imread(img_path)
    visualize_boxes()
    # annotate_boxes()
    # ome_tiff_path = subfolder + '/morphology_focus/morphology_focus_0000.ome.tif'
    # image_data = tiff.imread(ome_tiff_path)
    # img = image_data[0, :, :]
    # crop_section = [[v*4 for v in box] for box in crop_section]




    # test = input()



