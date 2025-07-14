import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

xenium_root_folder = '/media/huifang/data/sennet/xenium/'
codex_root_folder = '/media/huifang/data/sennet/codex/'

def get_data_list(root):
    data = pd.read_csv(root +'data_list.txt', sep=None, engine='python', header=None)
    data.columns = ['subfolder', 'sampleid']
    return data

def get_subfolder_by_sampleid(data,sampleid):
    result = data.loc[data['sampleid'] == sampleid, 'subfolder']
    return result.values[0] if not result.empty else None

def get_xenium_image(xenium_list, xenium_sampleid,xenium_regionid):
    xenium_subfolder_name = get_subfolder_by_sampleid(xenium_list, xenium_sampleid)
    subfolder = os.path.join(xenium_root_folder, xenium_subfolder_name)
    if os.path.exists(os.path.join(subfolder, 'outs')):
        subfolder = os.path.join(subfolder, 'outs')
    if not os.path.exists(subfolder + '/morphology_focus/regional_images/'):
        img_path = subfolder + '/morphology_focus/channel0_quarter.png'
    else:
        img_path = subfolder + '/morphology_focus/regional_images' + f"/channel0_region_{xenium_regionid}_quarter.png"
    # img = plt.imread(img_path)
    return img_path
def get_codex_image(codex_list,codex_sampleid,codex_regionid):
    codex_subfolder_name = get_subfolder_by_sampleid(codex_list, codex_sampleid)
    subfolder = os.path.join(codex_root_folder, codex_subfolder_name, 'per_tissue_region-selected')
    img_path = subfolder + f"/{codex_regionid}_X01_Y01_Z01_channel0_quarter.png"
    # img = plt.imread(img_path)
    return img_path


def visualize_overlap(img1_path,img2_path,id):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.rotate(img1, cv2.ROTATE_180)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    def resize_preserve_aspect(img, max_edge):
        h, w = img.shape
        scale = max_edge / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Determine the largest edge across both images
    max_edge = max(img1.shape + img2.shape)

    # Resize both images using the shared max_edge
    img1_resized = resize_preserve_aspect(img1, max_edge)
    img2_resized = resize_preserve_aspect(img2, max_edge)


    # Create white canvas large enough to hold both
    h1, w1 = img1_resized.shape
    h2, w2 = img2_resized.shape
    canvas_h = max(h1, h2)
    canvas_w = max(w1, w2)

    def place_on_canvas(img, canvas_h, canvas_w):
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)  # black background
        h, w = img.shape
        y_off = (canvas_h - h) // 2
        x_off = (canvas_w - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = img
        return canvas

    img1_canvas = place_on_canvas(img1_resized, canvas_h, canvas_w)
    img2_canvas = place_on_canvas(img2_resized, canvas_h, canvas_w)

    # print(img1_canvas.shape)
    # print(img2_canvas.shape)
    # f,a = plt.subplots(1,2)
    # a[0].imshow(img1_canvas)
    # a[1].imshow(img2_canvas)
    # plt.show()
    # test = input()


    # Convert to RGB using magenta and cyan
    # Image 1 → Magenta (R + B)
    img1_rgb = np.stack([img1_canvas, np.zeros_like(img1_canvas), img1_canvas], axis=-1)
    # Image 2 → Cyan (G + B)
    img2_rgb = np.stack([np.zeros_like(img2_canvas), img2_canvas, img2_canvas], axis=-1)

    # Blend with maximum pixel value
    overlay = np.maximum(img1_rgb, img2_rgb)


    # Show overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Magenta (Image 1) vs Cyan (Image 2)')
    plt.gca().invert_yaxis()
    plt.show()
    # flipped_overlay = np.flipud(overlay)
    #
    # # Save using OpenCV
    # cv2.imwrite('/media/huifang/data/sennet/paired_images/'+str(id)+'.png', cv2.cvtColor(flipped_overlay, cv2.COLOR_RGB2BGR))
    # # test = input()


xenium_list = get_data_list(xenium_root_folder)
codex_list = get_data_list(codex_root_folder)
# Replace with your actual file path
file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'

# Read the text file without header
df = pd.read_csv(file_path, sep=None, engine='python', header=None)
start_line = 0  # zero-based line number
for i, row in enumerate(df.iloc[start_line:].itertuples(index=False, name=None)):

    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = row

    xenium_img = get_xenium_image(xenium_list, xenium_sampleid,xenium_regionid)
    codex_img =get_codex_image(codex_list,codex_sampleid,codex_regionid)

    # Define metadata
    datatype = "Xenium-Codex"
    sampleid = xenium_sampleid  # assuming both share the same sample ID
    regionid = xenium_regionid  # assuming both share the same region ID

    # Plot
    f, a = plt.subplots(1, 2, figsize=(10, 5))
    a[0].imshow(plt.imread(xenium_img))
    a[0].set_title("Xenium",fontsize=18)
    a[1].imshow(plt.imread(codex_img))
    a[1].set_title("Codex",fontsize=18)

    # Set figure-level title with both Xenium and Codex metadata
    plt.suptitle(
        f"Xenium - Sample ID: {xenium_sampleid}, Region ID: {xenium_regionid} | "
        f"Codex - Sample ID: {codex_sampleid}, Region ID: {codex_regionid}",
        fontsize=24
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    plt.show()

    visualize_overlap(xenium_img,codex_img,i)


    # f,a = plt.subplots(1,2)
    # a[0].imshow(xenium_img)
    # a[1].imshow(codex_img)
    # plt.show()
