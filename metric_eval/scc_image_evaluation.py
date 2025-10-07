import numpy as np
import os
import cv2
import math
from scipy.signal import correlate2d
from scipy.spatial import KDTree
from scipy.spatial.distance import jensenshannon, euclidean, dice
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy, label
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import shift
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from sklearn.cluster import DBSCAN
### ----------------------- FILE LOADING FUNCTION -----------------------



def load_data_from_folder(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)
    return data["img1"], data["img2"]


### ----------------------- IMAGE EVALUATION FUNCTIONS -----------------------
def normalized_cross_correlation(img_fixed, img_moving):

    img_fixed = img_fixed - np.mean(img_fixed)
    img_moving = img_moving - np.mean(img_moving)
    numerator = np.sum(img_fixed * img_moving)
    denominator = np.sqrt(np.sum(img_fixed ** 2) * np.sum(img_moving ** 2))
    return numerator / denominator


def visualize_joint_histogram(img1, img2, bins=128):
    """
    Visualizes the joint histogram with supporting visualizations.

    Args:
        img1: First RGB image (H, W, 3)
        img2: Second RGB image (H, W, 3)
        bins: Number of bins for histogram (default: 128)
    """
    # Convert images to grayscale for easier metric interpretation
    img1_gray = np.mean(img1, axis=-1)
    img2_gray = np.mean(img2, axis=-1)

    # Compute joint histogram for grayscale
    joint_hist, x_edges, y_edges = np.histogram2d(img1_gray.ravel(), img2_gray.ravel(), bins=bins)

    # Compute joint histograms for RGB channels separately
    joint_hist_r, _, _ = np.histogram2d(img1[:, :, 0].ravel(), img2[:, :, 0].ravel(), bins=bins)
    joint_hist_g, _, _ = np.histogram2d(img1[:, :, 1].ravel(), img2[:, :, 1].ravel(), bins=bins)
    joint_hist_b, _, _ = np.histogram2d(img1[:, :, 2].ravel(), img2[:, :, 2].ravel(), bins=bins)

    # Scatter plot of pixel intensity pairs
    scatter_sample = 50000  # Sample to prevent excessive scatter points
    img1_sample = img1_gray.ravel()[::len(img1_gray.ravel()) // scatter_sample]
    img2_sample = img2_gray.ravel()[::len(img2_gray.ravel()) // scatter_sample]

    # Set up subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Show images side by side
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Image 1 (Fixed)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("Image 2 (Moving)")
    axes[0, 1].axis("off")

    # Overlay images for visual alignment check
    overlay = 0.5 * img1 + 0.5 * img2
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("Overlay of Image 1 & 2")
    axes[0, 2].axis("off")

    # Plot joint histogram (grayscale intensity correlation)
    axes[1, 0].imshow(joint_hist.T, origin="lower", cmap="hot", norm=LogNorm())
    axes[1, 0].set_xlabel("Intensity - Image 1 (Grayscale)")
    axes[1, 0].set_ylabel("Intensity - Image 2 (Grayscale)")
    axes[1, 0].set_title("Joint Histogram (Grayscale)")
    fig.colorbar(axes[1, 0].imshow(joint_hist.T, origin="lower", cmap="hot", norm=LogNorm()), ax=axes[1, 0])

    # Scatter plot of intensity pairs
    axes[1, 1].scatter(img1_sample, img2_sample, alpha=0.3, s=1)
    axes[1, 1].set_xlabel("Intensity - Image 1")
    axes[1, 1].set_ylabel("Intensity - Image 2")
    axes[1, 1].set_title("Scatter Plot of Intensity Correspondences")

    # RGB joint histogram combined
    axes[1, 2].imshow(joint_hist_r.T, origin="lower", cmap="Reds", norm=LogNorm(), alpha=0.5)
    axes[1, 2].imshow(joint_hist_g.T, origin="lower", cmap="Greens", norm=LogNorm(), alpha=0.5)
    axes[1, 2].imshow(joint_hist_b.T, origin="lower", cmap="Blues", norm=LogNorm(), alpha=0.5)
    axes[1, 2].set_xlabel("Intensity - Image 1 (RGB Combined)")
    axes[1, 2].set_ylabel("Intensity - Image 2 (RGB Combined)")
    axes[1, 2].set_title("Joint Histogram (RGB Channels Combined)")

    plt.tight_layout()
    plt.show()


def mutual_information(img_fixed, img_moving, bins=256,visualize=True):
    """
    Computes the Mutual Information (MI) between two images.

    Args:
        img_fixed: Fixed/reference image.
        img_moving: Moving/transformed image.
        bins: Number of histogram bins (default: 256).

    Returns:
        Mutual Information (MI) score.
    """
    if visualize:
        visualize_joint_histogram(img_fixed, img_moving, bins=256)

    # Step 1: Compute Joint Histogram
    joint_hist, x_edges, y_edges = np.histogram2d(img_fixed.ravel(), img_moving.ravel(), bins=bins)

    # Step 2: Normalize to Get Joint Probability Distribution
    joint_prob = joint_hist / joint_hist.sum()

    # Step 3: Compute Marginal Distributions
    prob_fixed = joint_prob.sum(axis=1)  # Summing along y-axis gives P(fixed)
    prob_moving = joint_prob.sum(axis=0)  # Summing along x-axis gives P(moving)

    # Step 4: Compute Entropies from Probability Distributions
    h_fixed = -np.sum(prob_fixed * np.log2(prob_fixed + 1e-9))  # Entropy of fixed image
    h_moving = -np.sum(prob_moving * np.log2(prob_moving + 1e-9))  # Entropy of moving image
    h_joint = -np.sum(joint_prob * np.log2(joint_prob + 1e-9))  # Joint entropy

    # Step 5: Compute Mutual Information
    mi = h_fixed + h_moving - h_joint

    return mi


def compute_ssim(img_fixed, img_moving):
    return ssim(img_fixed, img_moving, data_range=img_moving.max() - img_moving.min(),channel_axis=-1)

def check_labels(coords,labels):

    plt.figure()

    # Scatter all points, colored by their label
    scatter_plot = plt.scatter(coords[:, 0], coords[:, 1], s=10,c=labels)

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        # Find all points belonging to this label
        mask = (labels == lab)
        # Compute centroid (mean x and y) for this label
        x_mean = coords[mask, 0].mean()
        y_mean = coords[mask, 1].mean()
        # Place a text annotation at the centroid
        plt.text(x_mean, y_mean, str(lab), fontsize=12,
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.gca().invert_yaxis()
    plt.title("2D Points Colored by Label, with Centroid Annotations")
    plt.colorbar(scatter_plot, label="Label")
    plt.show()
def evaluate_single_pair(folder_path,i,j,suffix):
    img_fixed, img_moving = load_data_from_folder(folder_path)
    results = {
        "Mutual Information": mutual_information(img_fixed, img_moving,visualize=False),
        "SSIM": compute_ssim(img_fixed, img_moving),
        "NCC": normalized_cross_correlation(img_fixed, img_moving),
    }
    # print(results)
    return results


def evaluate_multiple_pairs(root_folder,keys,suffix):
    all_results = {key: [] for key in keys}
    for i in [2,5,9,10]:
        for j in [0,1]:
            data_path = root_folder + str(i) + "_" + str(j) + "_result.npz"
            results = evaluate_single_pair(data_path, i, j, suffix)
            for key in all_results:
                all_results[key].append(results[key])
    avg_results = {key: np.mean(values) for key, values in all_results.items()}
    print(avg_results)
    return avg_results


### ----------------------- USAGE EXAMPLE -----------------------

# Example: Single pair evaluation
# single_results = evaluate_single_pair("/home/huifang/workspace/code/registration/result/original/DLPFC/0_0_result.npz")
# print("Single Pair Evaluation Results:", single_results)
result_root='/media/huifang/data/registration/result/pairwise_align/SCC/figures/'
# # Example: Multiple pairs evaluation
keys=["Mutual Information","SSIM","NCC"]
print('Unaligned')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/initial/",keys,"initial")
print('SimpleITK')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/simpleitk/",keys,"simpleitk")
print('Voxelmorph')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/voxelmorph/",keys,'vxm')
print('Nicetrans')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/nicetrans/",keys,'nicetrans')
print('Ours')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/img_win9_gene_win5_frac0.95_gene0.5_generegu0/",keys,"Ours")

