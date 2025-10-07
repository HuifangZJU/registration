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
    fig, axes = plt.subplots(2, len(density_A), figsize=(15, 12))

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

        # # Heatmap of density_A
        # im1 = axes[1, idx].imshow(density_A[c], cmap='Reds', interpolation='nearest', origin='lower')
        # axes[1, idx].set_title(f"Class {c} - Density (A)")
        # axes[1, idx].invert_yaxis()
        # fig.colorbar(im1, ax=axes[1, idx])
        #
        #
        # # Heatmap of density_B
        # im2 = axes[2, idx].imshow(density_B[c], cmap='Blues', interpolation='nearest', origin='lower')
        # axes[2, idx].set_title(f"Class {c} - Density (B)")
        # axes[2, idx].invert_yaxis()
        # fig.colorbar(im2, ax=axes[2, idx])

        # Compute Normalized Spatial Correlation Heatmap
        # numerator = correlate2d(density_A[c], density_B[c], mode='same')
        # denominator = np.sqrt(np.sum(density_A[c] ** 2) * np.sum(density_B[c] ** 2)) + 1e-9
        #
        # normalized_correlation = numerator / denominator
        numerator = density_A[c] * density_B[c]
        denominator = np.sqrt(np.sum(density_A[c] ** 2) * np.sum(density_B[c] ** 2))
        normalized_correlation = numerator / (denominator + 1e-9)  # Avoid division by zero

        # Heatmap of spatial correlation
        im3 = axes[1, idx].imshow(normalized_correlation, cmap='coolwarm', interpolation='nearest', origin='lower')
        axes[1, idx].set_title(f"Class {c} - Correlation: {correlation_scores[c]:.3f}")
        axes[1, idx].invert_yaxis()
    fig.colorbar(im3, ax=axes[1, idx])

    plt.tight_layout()
    plt.show()

def spatial_cross_correlation(coords_A, labels_A, coords_B, labels_B, grid_size=100, sigma=2, visualize=True):
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





def get_centroid(mask):
    ys, xs = np.nonzero(mask)  # coordinates of foreground pixels
    cx = xs.mean()
    cy = ys.mean()
    return cx, cy

def get_mask(boundary,shape):
    if isinstance(boundary, Polygon):
        boundary = [np.array(boundary.exterior.coords)]
    elif isinstance(boundary, MultiPolygon):
        boundary = [np.array(poly.exterior.coords) for poly in boundary.geoms]
    elif isinstance(boundary, GeometryCollection):
        boundary = [np.array(geom.exterior.coords) for geom in boundary.geoms if isinstance(geom, Polygon)]

    mask = np.zeros(shape, dtype=np.uint8)
    for poly in boundary:
        rr, cc = polygon(poly[:, 1], poly[:, 0], shape)
        mask[rr, cc] = 1  # Fill the region
    return mask

def create_boundary_mask(coords, shape, alpha=0.5):
    if len(coords) < 3:
        return None  # A polygon cannot be formed
    boundary = alphashape.alphashape(coords, alpha)
    # Handle different geometry types
    mask = get_mask(boundary,shape)

    return boundary,mask

def dice_coefficient(mask_A, mask_B):
    intersection = np.sum(mask_A & mask_B)
    sum_masks = np.sum(mask_A) + np.sum(mask_B)

    return (2. * intersection) / sum_masks if sum_masks > 0 else 0.0

def get_region_meta_data(coords, labels, mask_shape, alpha=0.01):
    labels_ = np.unique(labels)
    regions={}
    for class_label in labels_:
        if np.isnan(class_label):
            continue
        classid = (labels == class_label)
        region_coords = coords[classid]
        boundary, mask = create_boundary_mask(region_coords, mask_shape, alpha)

        # compute centers
        cx, cy = get_centroid(mask)

        # visualize
        #
        # plt.imshow(mask, cmap="gray")
        # plt.scatter(region_coords[:,0],region_coords[:,1])
        # plt.scatter(cx, cy, c="red", s=50, marker="x")
        # plt.show()

        regions[class_label] = (region_coords, mask,boundary, cx,cy)

    return regions


def visualize_centroid_shift(region_A, region_B, shifts,region_mean_shift,save_suffix,save):
    """
    Plots centroid shifts between two region masks, using different colors for each class.
    Overlays the original scatter points with centroids.

    Args:
        region_masks_A: Dictionary mapping class labels to (coordinates, region labels) for dataset A.
        region_masks_B: Dictionary mapping class labels to (coordinates, region labels) for dataset B.
    """
    plt.figure(figsize=(12, 9))
    common_labels = set(region_A.keys()) & set(region_B.keys())
    colors = sns.color_palette("tab10", n_colors=len(common_labels))

    for idx, label in enumerate(common_labels):
        coords_a, mask_a, boundary_a, cx_a, cy_a = region_A[label]
        coords_b, mask_b, boundary_b, cx_b, cy_b = region_B[label]


        color = colors[idx]

        # Plot original scatter points
        plt.scatter(coords_a[:, 0], coords_a[:, 1], color=color, marker="o",  s=10, alpha=0.6)
        plt.scatter(coords_b[:, 0], coords_b[:, 1], color=color,  marker="s",s=10, alpha=0.6)
        # plot_boundary(boundary_a, color)
        # plot_boundary(boundary_b, color)

        # Plot centroids
        plt.scatter(cx_a, cy_a, edgecolor='black', color=color, marker="o", s=200,
                    label=f"Region {label} - Centroid A")
        plt.scatter(cx_b, cy_b, edgecolor='black', color=color, marker="s", s=200,
                    label=f"Region {label} - Centroid B")


        plt.arrow(cx_a, cy_a,
                  cx_b - cx_a,
                  cy_b - cy_a,
                  color=color, width=0.2, head_width=1, alpha=0.8)


        _, dist = shifts[label]
        shift_value = np.round(dist, 2)
        mid_x = (cx_a + cx_b) / 2
        mid_y = (cy_a + cy_b) / 2
        plt.text(mid_x, mid_y, f"{shift_value}", fontsize=24, ha="center", color="black")

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
        plt.close()
    else:
        plt.show()


def plot_boundary(boundary, color="red"):
    if isinstance(boundary, Polygon):
        x, y = boundary.exterior.xy
        plt.plot(x, y, color=color, linestyle="--", linewidth=2, alpha=1)
    elif isinstance(boundary, MultiPolygon):
        for poly in boundary.geoms:  # loop through each polygon
            x, y = poly.exterior.xy
            plt.plot(x, y, color=color, linestyle="--", linewidth=2, alpha=1)
    else:
        raise TypeError(f"Unsupported geometry type: {type(boundary)}")

def visualize_dice_regions(region_A, region_B, dice_scores, average_dice_score,save_suffix,save):
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


    common_labels = set(region_A.keys()) & set(region_B.keys())
    colors = sns.color_palette("tab10", n_colors=len(common_labels))
    for idx, label in enumerate(common_labels):
        coords_a, mask_a, boundary_a, cx_a, cy_a = region_A[label]
        coords_b, mask_b, boundary_b, cx_b, cy_b = region_B[label]


        color = colors[idx]  # Assign different colors for each class

        # Scatter plot of original points (A and B with different markers)
        plt.scatter(coords_a[:, 0], coords_a[:, 1], color=color, alpha=0.5, s=10, label=f"Region {label} (A)")
        plt.scatter(coords_b[:, 0], coords_b[:, 1], color=color, alpha=0.5, s=10, marker="x", label=f"Region {label} (B)")

        # Overlay boundaries with transparency

        # x_A, y_A = boundary_a.exterior.xy
        # plt.plot(x_A, y_A, color=color, linestyle="--", linewidth=2, alpha=1)
        # x_B, y_B = boundary_b.exterior.xy
        # plt.plot(x_B, y_B, color=color, linestyle="-", linewidth=2, alpha=1)
        plot_boundary(boundary_a,color)
        plot_boundary(boundary_b,color)

        midx = (cx_a + cx_b) / 2  # Position for text annotation
        midy = (cy_a + cy_b) / 2  # Position for text annotation

        dice_value,_ = dice_scores[label]
        plt.text(midx, midy, f"{dice_value:.2f}", ha="center", color="black", bbox=dict(facecolor='white', alpha=0.5))

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
        plt.close()
    else:
        # plt.savefig('/home/huifang/workspace/code/fiducial_remover/paper_figures/figures/20.png', dpi=300)
        plt.show()


### ----------------------- MAIN EVALUATION FUNCTIONS -----------------------
def compute_region_dice_score(coords_A, labels_A, coords_B, labels_B, save_suffix,mask_shape=(100, 100), alpha=0.01,visualize=True,save=True):

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
    region_meta_a = get_region_meta_data(coords_A, labels_A, mask_shape, alpha)
    region_meta_b = get_region_meta_data(coords_B, labels_B, mask_shape, alpha)
    common_labels = set(region_meta_a.keys()) & set(region_meta_b.keys())

    region_metrics = {}
    for label in common_labels:
        coords_a, mask_a,_, cx_a, cy_a = region_meta_a[label]
        coords_b, mask_b,_, cx_b, cy_b = region_meta_b[label]

        dice_score = dice_coefficient(mask_a, mask_b)

        dx = cx_a - cx_b
        dy = cy_a - cy_b
        dist = math.sqrt(dx ** 2 + dy ** 2)

        region_metrics[label] = (dice_score,dist)

    vals = np.array(list(region_metrics.values()))
    avg_dice, avg_dist = vals.mean(axis=0)
    if visualize:
        visualize_dice_regions(region_meta_a, region_meta_b, region_metrics,avg_dice, save_suffix,save)
        visualize_centroid_shift(region_meta_a, region_meta_b, region_metrics, avg_dist, save_suffix, save)
    return avg_dice,avg_dist






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
    coords_A, coords_B, labels_A, labels_B,mask_shape_ = load_data_from_folder(folder_path)

    if suffix == 'gpsa' or suffix =='santo':
        pass
    else:
        labels_A = np.load("/media/huifang/data/registration/DLPFC/huifang/"+str(i)+"_"+str(j)+"_"+"region_label.npy")
        labels_B = np.load("/media/huifang/data/registration/DLPFC/huifang/"+str(i)+"_"+str(j+1)+"_"+"region_label.npy")

    # check_labels(coords_A,labels_A)
    # check_labels(coords_B,labels_B)
    if i ==0:
        alpha_value = 0.05
    else:
        alpha_value = 0.1

    scc_scores = spatial_cross_correlation(
        coords_A, labels_A, coords_B, labels_B, visualize=False
    )

    avg_dice, avg_shift = compute_region_dice_score(
        coords_A, labels_A, coords_B, labels_B,
        figure_suffix,
        mask_shape=mask_shape_,
        alpha=alpha_value,
        visualize=True,
        save=True
    )

    results = {
        "Class-wise Dice Coefficient": avg_dice,
        "Mean Centroid Shift": avg_shift,
        "Spatial Cross-Correlation": scc_scores
    }
    print(results)
    return results


def evaluate_multiple_pairs(root_folder,keys,suffix):
    all_results = {key: [] for key in keys}
    for i in range(3):
        for j in range(3):
            data_path = root_folder+str(i)+"_"+str(j)+"_result.npz"
            results = evaluate_single_pair(data_path,i,j,suffix)
            for key in all_results:
                all_results[key].append(results[key])

    avg_results = {key: np.mean(values) for key, values in all_results.items()}
    print(avg_results)
    return avg_results


### ----------------------- USAGE EXAMPLE -----------------------

# Example: Single pair evaluation
# single_results = evaluate_single_pair("/home/huifang/workspace/code/registration/result/original/DLPFC/0_0_result.npz")
# print("Single Pair Evaluation Results:", single_results)
result_root='/media/huifang/data/registration/result/pairwise_align/DLPFC/figures/'
# # Example: Multiple pairs evaluation
keys=["Class-wise Dice Coefficient","Spatial Cross-Correlation", "Mean Centroid Shift"]
# print('Unaligned')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/initial/",keys,"initial")
# print('SimpleITK')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/simpleitk/",keys,"simpleitk")
# print('PASTE')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/paste/",keys,'paste')
# print('GPSA')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/GPSA/",keys,'gpsa')
# print('SANTO')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/SANTO/",keys,'santo')
# print('Voxelmorph')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/voxelmorph/",keys,'vxm')
# print('Nicetrans')
# average_results = evaluate_multiple_pairs("/media/huifang/data1/registration/result/pairwise_align/DLPFC/nicetrans/",keys,'nicetrans')
# print('Ours')
# average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/DLPFC/ours2_vis_best/",keys,"Ours2")

