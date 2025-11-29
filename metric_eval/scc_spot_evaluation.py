import math
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from skimage.draw import polygon
import alphashape
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors
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

    topk_acc = spotwise_topk_accuracy(coords_A, coords_B, labels_A, labels_B,dist_thresh=10)


    results = {
        # "Mean Centroid Shift": avg_shift,
        # "Spatial Cross-Correlation": scc_scores,
        "Label spatial consistency":topk_acc
    }
    # print(results)
    return results


def evaluate_multiple_pairs(root_folder,keys,suffix):
    all_results = {key: [] for key in keys}
    for i in [10]:
        for j in [1]:
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
average_results = evaluate_multiple_pairs("/media/huifang/data/registration/result/pairwise_align/SCC/ours/",keys,"Ours")

