import math
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from skimage.draw import polygon
import alphashape
import seaborn as sns
### ----------------------- FILE LOADING FUNCTION -----------------------

def load_data_from_folder(folder_path):
    """Load image and spot data from a given folder."""
    offset=5
    data = np.load(folder_path)

    pts1 = data["pts1"]  # shape (4200, 2)
    pts2 = data["pts2"]

    # normalize pts1 to [0, 10]
    min_vals = pts1.min(axis=0)
    max_vals = pts1.max(axis=0)
    scale = max_vals-min_vals

    pts_combined = np.vstack([pts1, pts2])
    # min and max per column (x and y separately)
    min_vals = pts_combined.min(axis=0)


    pts1_norm = (pts1 - min_vals) / scale * 100+offset
    pts2_norm = (pts2 - min_vals) / scale * 100+offset

    # combine along rows
    pts_combined = np.vstack([pts1_norm, pts2_norm])
    mask_shape = np.ceil(pts_combined.max(axis=0)).astype(np.uint8)+offset

    return pts1_norm, pts2_norm, data["label1"].reshape(-1), data["label2"].reshape(-1),mask_shape

def plot_label_compare(coords_A, labels_A,
                       coords_B, labels_B,
                       coords_C, labels_C,
                       target_label=3,  # choose your label
                       point_size=3):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    datasets = [
        ("Dataset A", coords_A, labels_A),
        ("Dataset B", coords_B, labels_B),
        ("Dataset C", coords_C, labels_C)
    ]

    for ax, (title, coords, labels) in zip(axes, datasets):

        # Boolean mask for the target label
        mask = labels == target_label

        # Plot all points (gray)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="lightgray", s=point_size, alpha=0.6)

        # Plot selected label (colored)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c="red", s=point_size+4, alpha=0.9,
                   label=f"Label {target_label}")

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def evaluate_multiple_pairs(root_folder,keys,suffix):
    for group in range(0,1):
        data_path = root_folder + str(group) + "_0_1_result.npz"
        data_path2 = root_folder + str(group) + "_0_2_result.npz"
        coords_A, coords_B, labels_A, labels_B, mask_shape_ = load_data_from_folder(data_path)
        _, coords_C, _, labels_C, mask_shape_ = load_data_from_folder(data_path2)

        for i in range(14):
            plot_label_compare(coords_A, labels_A,
                               coords_B, labels_B,
                               coords_C, labels_C,
                               target_label=i,  # choose your label
                               point_size=3)


        plt.scatter(coords_A[:, 0], coords_A[:, 1], s=0.5,alpha=0.8)
        plt.scatter(coords_B[:, 0], coords_B[:, 1], s=0.5,alpha=0.5)
        plt.scatter(coords_C[:, 0], coords_C[:, 1], s=0.5,color='purple',alpha=0.8)
        plt.show()


### ----------------------- USAGE EXAMPLE -----------------------

# Example: Single pair evaluation
# single_results = evaluate_single_pair("/home/huifang/workspace/code/registration/result/original/DLPFC/0_0_result.npz")
# print("Single Pair Evaluation Results:", single_results)
result_root='/media/huifang/data/registration/result/xenium/mouse_brain/figures/'
# # Example: Multiple pairs evaluation
keys=["Class-wise Dice Coefficient","Spatial Cross-Correlation", "Mean Centroid Shift"]

# print('Unaligned')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/initial/1024/",keys,"initial_1024")
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/initial/2048/",keys,"initial_2048")
# print('SimpleITK')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/simpleitk/1024/",keys,"simpleitk_1024")
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/simpleitk/2048/",keys,"simpleitk_2048")
# print('PASTE')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/PASTE/",keys,'paste')
# print('GPSA')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/gpsa/",keys,'gpsa')
# print('SANTO')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/SANTO/",keys,'santo')
# print('Voxelmorph')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/vxm/1024/",keys,'vxm_1024')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/vxm/2048/",keys,'vxm_2048')
# print('Nicetrans')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/nicetrans/1024/",keys,'nicetrans_1024')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/xenium/mouse_brain/nicetrans/2048/",keys,'nicetrans_2048')
print('Ours')
average_results = evaluate_multiple_pairs("//media/huifang/data/registration/result/xenium/mouse_brain/ours/1024/",keys,"Ours_1024")
# average_results = evaluate_multiple_pairs("//media/huifang/data/registration/result/xenium/mouse_brain/ours/2048/",keys,"Ours_2048")
