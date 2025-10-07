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
    data = np.load(folder_path, allow_pickle=True)

    pts1 = data["pts1"]  # shape (4200, 2)
    pts2 = data["pts2"]

    # normalize pts1 to [0, 10]
    min_vals = pts1.min(axis=0)
    max_vals = pts1.max(axis=0)
    scale = max_vals-min_vals


    pts1_norm = (pts1 - min_vals) / scale * 100+offset
    pts2_norm = (pts2 - min_vals) / scale * 100+offset

    # combine along rows
    pts_combined = np.vstack([pts1_norm, pts2_norm])
    mask_shape = np.ceil(pts_combined.max(axis=0)).astype(np.uint8)+offset

    return pts1_norm, pts2_norm, data["label1"].reshape(-1), data["label2"].reshape(-1),mask_shape

### ----------------------- SPOT EVALUATION FUNCTIONS -----------------------

def plot_spatial_correlation(coords_A, labels_A, coords_B, labels_B, density_A, density_B, grid_size,
                             correlation_scores):

    fig, axes = plt.subplots(2, len(density_A), figsize=(3*len(density_A), 12))

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

        class_coords_A = norm_coords_A[labels_A == c]
        class_coords_B = norm_coords_B[labels_B == c]

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





def get_centroid(coords):
    center =coords.mean(axis=0)
    return center[0], center[1]




def get_region_meta_data(coords, labels):
    labels_ = np.unique(labels)
    regions={}
    for class_label in labels_:
        if np.isnan(class_label):
            continue
        classid = (labels == class_label)
        region_coords = coords[classid]

        # compute centers
        cx, cy = get_centroid(region_coords)
        regions[class_label] = (region_coords, cx,cy)

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
        coords_a,  cx_a, cy_a = region_A[label]
        coords_b,  cx_b, cy_b = region_B[label]


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


        dist = shifts[label]
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




### ----------------------- MAIN EVALUATION FUNCTIONS -----------------------
def compute_region_score(coords_A, labels_A, coords_B, labels_B, save_suffix,visualize=True,save=True):

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
    region_meta_a = get_region_meta_data(coords_A, labels_A)
    region_meta_b = get_region_meta_data(coords_B, labels_B)
    common_labels = set(region_meta_a.keys()) & set(region_meta_b.keys())

    region_metrics = {}
    for label in common_labels:
        coords_a, cx_a, cy_a = region_meta_a[label]

        coords_b, cx_b, cy_b = region_meta_b[label]

        dx = cx_a - cx_b
        dy = cy_a - cy_b
        dist = math.sqrt(dx ** 2 + dy ** 2)
        region_metrics[label] = dist

    vals = np.array(list(region_metrics.values()))
    avg_dist = vals.mean(axis=0)
    if visualize:

        visualize_centroid_shift(region_meta_a, region_meta_b, region_metrics, avg_dist, save_suffix, save)
    return avg_dist






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


from sklearn.neighbors import NearestNeighbors
def spotwise_topk_accuracy(coords_A, coords_B, labels_A, labels_B, dist_thresh=5.0):
    """
    Compute spot-wise label consistency between dataset A and B
    using a distance threshold instead of k-NN.

    For each spot in B, find all spots in A within dist_thresh,
    then check whether the true label of B matches the majority of A's labels.

    Parameters
    ----------
    coords_A : (nA, 2) np.ndarray
        Spatial coordinates of dataset A
    coords_B : (nB, 2) np.ndarray
        Spatial coordinates of dataset B
    labels_A : (nA,) np.ndarray
        Integer labels for dataset A
    labels_B : (nB,) np.ndarray
        Integer labels for dataset B
    dist_thresh : float
        Distance threshold for defining neighbors

    Returns
    -------
    acc_majority : float
        Majority-vote accuracy (more intuitive for biologists)
    acc_any : float
        Relaxed accuracy (true label appears among neighbor labels)
    """
    coords_A = np.asarray(coords_A)
    coords_B = np.asarray(coords_B)
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)

    nn = NearestNeighbors(radius=dist_thresh, algorithm="auto").fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)

    nB = coords_B.shape[0]
    correct_majority, correct_any, count_valid = 0, 0, 0

    for i in range(nB):
        idxs = neighbors[i]
        if len(idxs) == 0:
            # no neighbors in threshold → skip this spot
            continue
        neigh_labels = labels_A[idxs]
        true_label = labels_B[i]

        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        maj_label = vals[np.argmax(counts)]
        if maj_label == true_label:
            correct_majority += 1

        # relaxed: if true label appears among neighbors
        if true_label in neigh_labels:
            correct_any += 1

        count_valid += 1

    acc_majority = correct_majority / count_valid if count_valid > 0 else np.nan
    return acc_majority

def evaluate_single_pair(folder_path,i,j,suffix):

    figure_suffix = str(i)+"_"+str(j)+"_"+suffix
    coords_A, coords_B, labels_A, labels_B,mask_shape_ = load_data_from_folder(folder_path)


    # check_labels(coords_B,labels_B)



    # avg_shift = compute_region_score(
    #     coords_A, labels_A, coords_B, labels_B,
    #     figure_suffix,
    #     visualize=True,
    #     save=True
    # )
    # scc_scores = spatial_cross_correlation(
    #     coords_A, labels_A, coords_B, labels_B, visualize=False
    # )

    topk_acc = spotwise_topk_accuracy(coords_A, coords_B, labels_A, labels_B)


    results = {
        # "Mean Centroid Shift": avg_shift,
        # "Spatial Cross-Correlation": scc_scores,
        "Label spatial consistency":topk_acc
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
# keys=["Spatial Cross-Correlation", "Mean Centroid Shift", "Label spatial consistency"]
keys=["Label spatial consistency"]
# for ablation_level in [1,2,3,4,5,8]:
#     _ = evaluate_multiple_pairs(f"/media/huifang/data/registration/result/ablation/scc/{ablation_level}/", keys, f"scc_{ablation_level}")

print('Unaligned')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/initial/",keys,"initial")
print('SimpleITK')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/simpleitk/",keys,"simpleitk")
print('PASTE')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/paste/",keys,'paste')
print('GPSA')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/GPSA/",keys,'gpsa')
print('SANTO')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/SANTO/",keys,'santo')
print('Voxelmorph')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/voxelmorph/",keys,'vxm')
print('Nicetrans')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/nicetrans/",keys,'nicetrans')
print('Ours')
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/img_win9_gene_win5_frac0.95_gene0.5_generegu0/",keys,"Ours")

