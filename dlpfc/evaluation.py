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

result_root='/home/huifang/workspace/code/registration/result/'

def load_data_from_folder(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)
    image = data["img1"]
    # print(image.shape)
    # print(np.max(data["pts1"]))
    # test = input()

    if image.shape[0]>512 and image.shape[0]<1500:
        img1 = cv2.resize(data["img1"],(512,512),interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(data["img2"], (512, 512), interpolation=cv2.INTER_AREA)
        pts1 = data["pts1"]/2
        pts2 = data["pts2"]/2
        return img1,img2,pts1,pts2,data["label1"].reshape(-1), data["label2"].reshape(-1)
    elif image.shape[0] >1500 and image.shape[0]<2500:
        img1 = cv2.resize(data["img1"],(512,512),interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(data["img2"], (512, 512), interpolation=cv2.INTER_AREA)
        pts1 = data["pts1"]/4
        pts2 = data["pts2"]/4
        return img1,img2,pts1,pts2,data["label1"].reshape(-1), data["label2"].reshape(-1)
    else:
        return data["img1"], data["img2"], data["pts1"], data["pts2"], data["label1"].reshape(-1), data["label2"].reshape(-1)


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

### ----------------------- SPOT EVALUATION FUNCTIONS -----------------------

def plot_spatial_correlation(coords_A, labels_A, coords_B, labels_B, density_A, density_B, grid_size,
                             correlation_scores):
    """
    Visualizes the spatial distribution, density maps, and correlation between two datasets.

    Args:
        coords_A, labels_A: Normalized coordinates and labels for dataset A.
        coords_B, labels_B: Normalized coordinates and labels for dataset B.
        density_A, density_B: Density maps for each label.
        grid_size: Size of the density map.
        correlation_scores: Computed correlation per class.
    """
    fig, axes = plt.subplots(4, len(density_A), figsize=(15, 12))

    unique_classes = list(density_A.keys())
    cmap = plt.get_cmap("coolwarm")

    for idx, c in enumerate(unique_classes):
        # Scatter plots of raw points
        axes[0, idx].scatter(coords_A[labels_A == c][:, 0], coords_A[labels_A == c][:, 1], color='red', s=10,
                             label=f"Class {c} (A)")
        axes[0, idx].scatter(coords_B[labels_B == c][:, 0], coords_B[labels_B == c][:, 1], color='blue', s=10,
                             label=f"Class {c} (B)")
        axes[0, idx].set_xlim([0, grid_size])
        axes[0, idx].set_ylim([0, grid_size])
        axes[0, idx].set_title(f"Class {c} - Raw Points")
        axes[0, idx].legend()
        axes[0, idx].invert_yaxis()

        # Heatmap of density_A
        im1 = axes[1, idx].imshow(density_A[c], cmap='Reds', interpolation='nearest', origin='lower')
        axes[1, idx].set_title(f"Class {c} - Density (A)")
        axes[1, idx].invert_yaxis()
        fig.colorbar(im1, ax=axes[1, idx])


        # Heatmap of density_B
        im2 = axes[2, idx].imshow(density_B[c], cmap='Blues', interpolation='nearest', origin='lower')
        axes[2, idx].set_title(f"Class {c} - Density (B)")
        axes[2, idx].invert_yaxis()
        fig.colorbar(im2, ax=axes[2, idx])

        # Compute Normalized Spatial Correlation Heatmap
        # numerator = correlate2d(density_A[c], density_B[c], mode='same')
        # denominator = np.sqrt(np.sum(density_A[c] ** 2) * np.sum(density_B[c] ** 2)) + 1e-9
        #
        # normalized_correlation = numerator / denominator
        numerator = density_A[c] * density_B[c]
        denominator = np.sqrt(np.sum(density_A[c] ** 2) * np.sum(density_B[c] ** 2))
        normalized_correlation = numerator / (denominator + 1e-9)  # Avoid division by zero

        # Heatmap of spatial correlation
        im3 = axes[3, idx].imshow(normalized_correlation, cmap='coolwarm', interpolation='nearest', origin='lower')
        axes[3, idx].set_title(f"Class {c} - Correlation: {correlation_scores[c]:.3f}")
        axes[3, idx].invert_yaxis()
        fig.colorbar(im3, ax=axes[3, idx])

    plt.tight_layout()
    plt.show()

def spatial_cross_correlation(coords_A, labels_A, coords_B, labels_B, grid_size=200, sigma=2, visualize=True):
    """
    Computes and visualizes spatial cross-correlation between two sets of labeled points.

    Args:
        coords_A, labels_A: Coordinates and labels for dataset A.
        coords_B, labels_B: Coordinates and labels for dataset B.
        grid_size: Size of the grid for density estimation.
        sigma: Standard deviation for Gaussian smoothing.
        visualize: Whether to generate visualization plots.

    Returns:
        Mean normalized spatial cross-correlation score.
    """
    # Step 1: Normalize coordinates to fit within [0, grid_size-1]
    all_coords = np.vstack((coords_A, coords_B))
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)

    def normalize(coords):
        return ((coords - [x_min, y_min]) / ([x_max - x_min, y_max - y_min]) * (grid_size - 1)).astype(int)
    norm_coords_A = normalize(coords_A)
    norm_coords_B = normalize(coords_B)

    # Step 2: Compute density maps and correlation scores
    unique_classes = np.unique(labels_A)
    correlation_scores = {}
    density_maps_A = {}
    density_maps_B = {}

    for c in unique_classes:
        density_A = np.zeros((grid_size, grid_size))
        density_B = np.zeros((grid_size, grid_size))

        for (x, y) in norm_coords_A[labels_A == c]:
            density_A[y, x] += 1  # Now safely within [0, grid_size-1]
        for (x, y) in norm_coords_B[labels_B == c]:
            density_B[y, x] += 1  # Now safely within [0, grid_size-1]

        # Apply Gaussian smoothing
        density_A = scipy.ndimage.gaussian_filter(density_A, sigma=sigma)
        density_B = scipy.ndimage.gaussian_filter(density_B, sigma=sigma)

        # Normalize correlation using energy normalization

        numerator = np.sum(density_A * density_B)
        denominator = np.sqrt(np.sum(density_A ** 2) * np.sum(density_B ** 2))
        normalized_correlation = numerator / (denominator + 1e-9)  # Avoid division by zero

        # Store results
        correlation_scores[c] = normalized_correlation
        density_maps_A[c] = density_A
        density_maps_B[c] = density_B

    # Visualization
    if visualize:
        plot_spatial_correlation(norm_coords_A, labels_A, norm_coords_B, labels_B,
                                 density_maps_A, density_maps_B, grid_size, correlation_scores)


    return np.mean(list(correlation_scores.values()))


def get_connected_regions(coords, labels, threshold=50):
    """
    Identifies connected regions in labeled spatial data using distance-based clustering.

    Args:
        coords (np.ndarray): N x 2 array of point coordinates.
        labels (np.ndarray): N array of class labels.
        threshold (float): Maximum distance between points to be considered connected.

    Returns:
        dict: Dictionary where keys are class labels and values are arrays with region labels.
    """
    unique_labels = np.unique(labels)
    region_masks = {}

    for class_label in unique_labels:
        # Extract points for the current class
        class_coords = coords[labels == class_label]

        if len(class_coords) == 0:
            continue

        # Build KDTree for efficient neighbor searching
        tree = KDTree(class_coords)

        # Compute adjacency matrix: find points within `threshold`
        adjacency_matrix = tree.sparse_distance_matrix(tree, max_distance=threshold)

        # Compute connected components (region labels)
        n_components, region_labels = connected_components(adjacency_matrix, directed=False)

        # Store results
        region_masks[class_label] = (class_coords, region_labels)

    # visualize_region_labels(region_masks)
    return region_masks

def compute_region_based_centroid_shift_balanced(region_masks_A, region_masks_B):

    shifts = []

    # Iterate over class labels in region_masks_A
    for class_label in region_masks_A.keys():

        coords_A = region_masks_A[class_label]
        coords_B = region_masks_B[class_label]

        centroids_A = np.mean(coords_A, axis=0)
        centroids_B = np.mean(coords_B, axis=0)

        dx = centroids_A[0] - centroids_B[0]  # 1488.26794226 - 1544.3325
        dy = centroids_A[1] - centroids_B[1]  # 418.61993267 - 432.871365
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Compute the centroid shifts for matched pairs
        shifts.append(dist)

    return np.mean(np.asarray(shifts)) if shifts else 0.0


def visualize_centroid_shift(region_masks_A, region_masks_B,region_mean_shift,save_suffix,save=True):
    """
    Plots centroid shifts between two region masks, using different colors for each class.
    Overlays the original scatter points with centroids.

    Args:
        region_masks_A: Dictionary mapping class labels to (coordinates, region labels) for dataset A.
        region_masks_B: Dictionary mapping class labels to (coordinates, region labels) for dataset B.
    """
    plt.figure(figsize=(12, 9))
    # colors = plt.cm.get_cmap("tab10", len(region_masks_A))  # Unique color for each class
    import seaborn as sns
    colors = sns.color_palette("tab10", n_colors=len(region_masks_A))

    for idx, class_label in enumerate(region_masks_A.keys()):

        coords_A= region_masks_A[class_label]
        coords_B= region_masks_B[class_label]

        centroids_A = np.mean(coords_A, axis=0)
        centroids_B = np.mean(coords_B, axis=0)

        dx = centroids_A[0] - centroids_B[0]  # 1488.26794226 - 1544.3325
        dy = centroids_A[1] - centroids_B[1]  # 418.61993267 - 432.871365
        dist = math.sqrt(dx ** 2 + dy ** 2)


        # class_color = colors(idx % 10)  # Assign unique color to each class
        color = colors[idx]

        # Plot original scatter points
        plt.scatter(coords_A[:, 0], coords_A[:, 1], color=color, marker="o",  s=10, alpha=0.6)
        plt.scatter(coords_B[:, 0], coords_B[:, 1], color=color,  marker="s",s=10, alpha=0.6)

        # Plot centroids
        plt.scatter(centroids_A[0], centroids_A[1], edgecolor='black', color=color, marker="o", s=100,
                    label=f"Region {class_label} - Centroid A")
        plt.scatter(centroids_B[0], centroids_B[1], edgecolor='black', color=color, marker="s", s=100,
                    label=f"Region {class_label} - Centroid B")


        plt.arrow(centroids_A[0], centroids_A[1],
                  centroids_B[0] - centroids_A[0],
                  centroids_B[1] - centroids_A[1],
                  color=color, width=0.5, head_width=3, alpha=0.8)

        shift_value = np.round(dist, 2)
        mid_x = (centroids_A[0] + centroids_B[ 0]) / 2
        mid_y = (centroids_A[1] + centroids_B[ 1]) / 2
        plt.text(mid_x, mid_y, f"{shift_value}", fontsize=16, ha="center", color="black")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # plt.title("Region-Based Centroid Shift Visualization")
    plt.title(str(region_mean_shift))
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=16)
    # plt.legend(loc="best", fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(result_root + "centroid_shift_" + save_suffix + ".png", dpi=300, bbox_inches='tight')
    else:
        plt.show()



def mean_centroid_shift(coords_A, labels_A, coords_B, labels_B,figure_suffix,visualize=True,save=True):

    regions_A = {}
    regions_B = {}
    labels = np.unique(labels_A)
    for class_label in labels:
        region_coords_A = coords_A[(labels_A == class_label)]
        region_coords_B = coords_B[(labels_B == class_label)]

        regions_A[class_label] = (region_coords_A)
        regions_B[class_label] = (region_coords_B)

    region_mean_shift = compute_region_based_centroid_shift_balanced(regions_A, regions_B)
    if visualize:
        visualize_centroid_shift(regions_A, regions_B,region_mean_shift,figure_suffix,save)

    return region_mean_shift

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from skimage.draw import polygon
from scipy.spatial.distance import dice
import alphashape  # Install with `pip install alphashape`


def alpha_shape(points, alpha=0.5):
    """
    Computes the alpha shape (concave hull) of a set of points.

    Args:
        points (np.ndarray): Array of (x, y) coordinates.
        alpha (float): Alpha parameter controlling shape tightness.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: The outer boundary polygon.
    """
    if len(points) < 3:
        return None  # A polygon cannot be formed

    return alphashape.alphashape(points, alpha)


def create_boundary_mask(coords, shape, alpha=0.5):
    mask = np.zeros(shape, dtype=np.uint8)
    boundary = alpha_shape(coords, alpha)
    if boundary is None:
        return mask  # No valid boundary found

    # Handle different geometry types
    if isinstance(boundary, Polygon):
        boundary = [np.array(boundary.exterior.coords)]
    elif isinstance(boundary, MultiPolygon):
        boundary = [np.array(poly.exterior.coords) for poly in boundary.geoms]
    elif isinstance(boundary, GeometryCollection):
        boundary = [np.array(geom.exterior.coords) for geom in boundary.geoms if isinstance(geom, Polygon)]

    if not boundary:  # If no valid polygons were extracted
        return mask

    for poly in boundary:
        rr, cc = polygon(poly[:, 1], poly[:, 0], shape)
        mask[rr, cc] = 1  # Fill the region

    return mask


def dice_coefficient(mask_A, mask_B):
    """
    Computes the Dice coefficient between two binary masks.

    Args:
        mask_A (np.ndarray): Binary mask for dataset A.
        mask_B (np.ndarray): Binary mask for dataset B.

    Returns:
        float: Dice similarity coefficient.
    """
    intersection = np.sum(mask_A & mask_B)
    sum_masks = np.sum(mask_A) + np.sum(mask_B)

    return (2. * intersection) / sum_masks if sum_masks > 0 else 0.0


def create_boundary_polygon(coords, alpha=0.15):
    """
    Generates the boundary polygon from coordinates using alphashape.

    Args:
        coords (np.ndarray): (N,2) array of (x,y) coordinates.
        alpha (float): Alpha parameter controlling shape tightness.

    Returns:
        shapely.geometry.Polygon: The boundary polygon.
    """
    if len(coords) < 3:
        return None  # A polygon cannot be formed

    boundary = alphashape.alphashape(coords, alpha)
    if isinstance(boundary, MultiPolygon):
        return max(boundary.geoms, key=lambda p: p.area)  # Take the largest shape
    return boundary


def visualize_dice_regions(region_masks_A, region_masks_B, dice_scores, average_dice_score,save_suffix,save,alpha=0.15):
    """
    Plots the region boundaries for dataset A and B overlaid on their original points.

    Args:
        region_masks_A: Dictionary of class labels to region coordinates (dataset A).
        region_masks_B: Dictionary of class labels to region coordinates (dataset B).
        dice_scores: Dictionary of Dice scores for each matched region pair.
        alpha (float): Alpha parameter for boundary tightness.
    """
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 20})
    import seaborn as sns
    colors = sns.color_palette("tab10", n_colors=len(region_masks_A))

    for idx, class_label in enumerate(region_masks_A.keys()):


        region_coords_A, region_labels_A = region_masks_A[class_label]
        region_coords_B, region_labels_B = region_masks_B[class_label]

        boundary_A = create_boundary_polygon(region_coords_A, alpha)
        boundary_B = create_boundary_polygon(region_coords_B, alpha)


        color = colors[idx]  # Assign different colors for each class

        # Scatter plot of original points (A and B with different markers)
        plt.scatter(region_coords_A[:, 0], region_coords_A[:, 1], color=color, alpha=0.5, s=10, label=f"Region {class_label} (A)")
        plt.scatter(region_coords_B[:, 0], region_coords_B[:, 1], color=color, alpha=0.5, s=10, marker="s", label=f"Region {class_label} (B)")

        # Overlay boundaries with transparency

        x_A, y_A = boundary_A.exterior.xy
        plt.plot(x_A, y_A, color=color, linestyle="--", linewidth=2, alpha=1)
        x_B, y_B = boundary_B.exterior.xy
        plt.plot(x_B, y_B, color=color, linestyle="-", linewidth=2, alpha=1)


        centroid_A = np.mean(region_coords_A, axis=0)
        centroid_B = np.mean(region_coords_B, axis=0)
        midpoint = (centroid_A + centroid_B) / 2  # Position for text annotation

        dice_value = dice_scores[class_label]
        plt.text(midpoint[0], midpoint[1], f"{dice_value:.2f}", ha="center", color="black", bbox=dict(facecolor='white', alpha=0.5))


    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.title(str(average_dice_score))
    plt.axis('equal')
    # plt.title( "Overlay of Region Boundaries, Dice Scores, and Original Points")
    plt.legend( bbox_to_anchor=(1.0, 1.0))
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(result_root + "dice_"+save_suffix+".png", dpi=300, bbox_inches='tight')
    else:
        # plt.savefig('/home/huifang/workspace/code/fiducial_remover/paper_figures/figures/20.png', dpi=300)
        plt.show()





def compute_region_dice_score(coords_A, labels_A, coords_B, labels_B, save_suffix,mask_shape=(512, 512), alpha=0.01,visualize=True,save=True):
    # alpha = 0.05
    """
    Computes the Dice score for each class label using the exact outer boundaries.

    Args:
        region_masks_A: Dictionary mapping class labels to (coordinates, region labels) for dataset A.
        region_masks_B: Dictionary mapping class labels to (coordinates, region labels) for dataset B.
        mask_shape (tuple): Fixed shape for binary masks (height, width).
        alpha (float): Alpha parameter for boundary tightness.

    Returns:
        dict: Class-wise Dice scores.
        float: Average Dice score across all classes.
    """

    region_dice_scores = {}
    labels = np.unique(labels_B)

    regions_A={}
    regions_B={}
    for class_label in labels:
        # class_label=6
        classid_A = (labels_A == class_label)
        region_coords_A = coords_A[classid_A]

        classid_B = (labels_B == class_label)
        region_coords_B = coords_B[classid_B]


        mask_A = create_boundary_mask(region_coords_A, mask_shape, alpha)
        mask_B = create_boundary_mask(region_coords_B, mask_shape, alpha)

        # f, a = plt.subplots(1, 2)
        # a[0].scatter(region_coords_A[:,0],region_coords_A[:,1])
        # a[1].scatter(region_coords_B[:,0],region_coords_B[:,1])
        # plt.show()
        #
        # f,a = plt.subplots(1,2)
        # a[0].imshow(mask_A)
        # a[1].imshow(mask_B)
        # plt.show()

        dice_score = dice_coefficient(mask_A, mask_B)
        region_dice_scores[class_label] = dice_score

        regions_A[class_label] = (region_coords_A, class_label)
        regions_B[class_label] = (region_coords_B, class_label)
    average_dice_score = np.mean(list(region_dice_scores.values())) if region_dice_scores else 0.0
    if visualize:
        visualize_dice_regions(regions_A, regions_B, region_dice_scores,average_dice_score, save_suffix,save,alpha)
    return average_dice_score


### ----------------------- MAIN EVALUATION FUNCTIONS -----------------------
def combine_disconnected_regions(regions_A):
    """
    regions_A: a dictionary or list-like structure
               with regions_A[i] = (class_coords, region_labels)
               for i in 1..len(regions_A).

    Returns:
        newcoords: (M, 2) array of all points, from all regions.
        newlabels: (M,)   array of unique label IDs, so that each
                          disconnected region has a distinct label.
    """
    all_coords = []
    all_labels = []

    offset = 0  # Keep track of global label IDs
    # If your indexing truly goes from 1..len(regions_A),
    # then iterate accordingly:
    for i in range(1, len(regions_A) + 1):
        class_coords, region_labels = regions_A[i]

        # Make sure region_labels starts from 0 (or 1).
        # If region_labels is already 0-based, you can skip or adapt this step.
        # region_labels = region_labels - 1  # only if it's 1-based

        # Shift these subcluster labels by 'offset'
        shifted_labels = region_labels + offset

        # Accumulate
        all_coords.append(class_coords)
        all_labels.append(shifted_labels)

        # Update offset so that the next region set
        # doesn't overlap these labels
        # (We assume region_labels is 0..something.)
        offset += region_labels.max() + 1

    # Combine everything into arrays
    newcoords = np.vstack(all_coords)  # shape (M, 2)
    newlabels = np.concatenate(all_labels)  # shape (M,)

    return newcoords, newlabels

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
    figure_suffix = str(i)+"_"+str(j)+"_"+suffix
    img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B = load_data_from_folder(folder_path)
    overlay = 0.5*img_fixed+0.5*img_moving
    overlay = overlay.astype(np.uint8)
    plt.imshow(overlay)
    # plt.imshow(0.5*img_fixed+0.5*img_moving)
    plt.show()


    # coords_A = coords_A*2
    # coords_B = coords_B*2
    if i==0 and j==0:
        labels_A = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_0_region_label.npy")
        labels_B = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_1_region_label.npy")
    if i==0 and j==1:
        labels_A = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_1_region_label.npy")
        labels_B = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_2_region_label.npy")
    if i==0 and j==2:
        labels_A = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_2_region_label.npy")
        labels_B = np.load("/home/huifang/workspace/code/registration/data/DLPFC/huifang/0_3_region_label.npy")


    # check_labels(coords_A,labels_A)
    # check_labels(coords_B,labels_B)
    if i ==0:
        alpha_value = 0.01
    else:
        alpha_value = 0.04

    results = {
        "Class-wise Dice Coefficient": compute_region_dice_score(coords_A, labels_A, coords_B, labels_B,figure_suffix,alpha=alpha_value,visualize=True,save=True),
        "Spatial Cross-Correlation": spatial_cross_correlation(coords_A, labels_A, coords_B, labels_B, visualize=False),
        "Mean Centroid Shift": mean_centroid_shift(coords_A, labels_A, coords_B, labels_B,figure_suffix,visualize=True,save=True),
        "Mutual Information": mutual_information(img_fixed, img_moving,visualize=False),
        "SSIM": compute_ssim(img_fixed, img_moving),
        "NCC": normalized_cross_correlation(img_fixed, img_moving),
    }
    print(results)
    return results


def evaluate_multiple_pairs(root_folder,keys,suffix):
    all_results = {key: [] for key in keys}
    for i in range(3):
        for j in range(1,3):
            # i=2
            # j=1
            data_path = root_folder+str(i)+"_"+str(j)+"_result.npz"
            print(data_path)
            results = evaluate_single_pair(data_path,i,j,suffix)
            for key in all_results:
                all_results[key].append(results[key])

    avg_results = {key: np.mean(values) for key, values in all_results.items()}
    return avg_results


### ----------------------- USAGE EXAMPLE -----------------------

# Example: Single pair evaluation
# single_results = evaluate_single_pair("/home/huifang/workspace/code/registration/result/original/DLPFC/0_0_result.npz")
# print("Single Pair Evaluation Results:", single_results)

# # Example: Multiple pairs evaluation
keys=["Class-wise Dice Coefficient","Spatial Cross-Correlation", "Mean Centroid Shift","Mutual Information","SSIM","NCC"]
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/original/DLPFC/",keys,"orig")
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/simpleITK/DLPFC/marker_free/",keys,"vispro_itk")
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/simpleITK/DLPFC/with_marker/",keys,"original_itk")
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/nicetrans/DLPFC/",keys,"nicetrans")
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/ours/DLPFC/attention_fusion/regu2/",keys,"attentionfusion")
average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/PASTE/DLPFC/",keys,'paste')
# average_results = evaluate_multiple_pairs("/home/huifang/workspace/code/registration/result/ours/DLPFC/hvg_smoothed/",keys,'smoothed')
print("Average Evaluation Results Across Multiple Pairs:", average_results)
