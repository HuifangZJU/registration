import numpy as np
import os
import cv2
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
def load_data_from_folder(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)


    return data["img1"], data["img2"], data["pts1"], data["pts2"], data["label1"].reshape(-1), data["label2"].reshape(-1)


def get_connected_regions(coords, labels, threshold=50):
    """
    Finds disconnected sub-regions within each class label.
    Returns a dict mapping `class_label -> (indices_for_label, region_labels_for_those_indices)`.

    coords: (N, 2) array, the original coordinates (never reordered here).
    labels: (N,)   array of integer labels.
    threshold: distance threshold for connecting two points in a sub-region.
    """
    unique_labels = np.unique(labels)
    region_dict = {}

    for class_label in unique_labels:
        # Indices of all points belonging to this class_label
        class_inds = np.where(labels == class_label)[0]
        if len(class_inds) == 0:
            continue

        class_coords = coords[class_inds]  # The actual points for this label

        # Build KDTree for efficient neighbor searching
        tree = KDTree(class_coords)

        # Build adjacency (sparse) matrix for points within `threshold` distance
        adjacency_matrix = tree.sparse_distance_matrix(tree, max_distance=threshold)

        # Find connected components
        n_components, region_labels = connected_components(adjacency_matrix, directed=False)

        # Store:
        #   - the original indices (class_inds),
        #   - the sub-region labels for these indices
        region_dict[class_label] = (class_inds, region_labels)

    return region_dict


def combine_disconnected_regions(region_dict, total_points):
    """
    Builds a single (N,) label array assigning each point a unique ID
    for its connected subregion (across all classes).

    region_dict:  dict returned by get_connected_regions
                  {class_label -> (class_inds, region_labels_for_this_class)}
    total_points: total number of points (N)
    """
    new_labels = np.full(total_points, -1, dtype=int)

    offset = 1  # so our subregion IDs start at 1 (or 0 if you prefer)
    # If your keys are numeric class labels, we sort them just to have a consistent order.
    for class_label in sorted(region_dict.keys()):
        class_inds, region_labels = region_dict[class_label]

        # region_labels is typically 0..(n_components-1), so offset them
        shifted_labels = region_labels + offset

        # Assign them into the global array for exactly those indices
        new_labels[class_inds] = shifted_labels

        # Increase the offset so the next set of subregions doesn’t overlap
        offset += region_labels.max() + 1

    return new_labels

def switch_label_ids(labels,id1,id2):
    mask_id1 = (labels == id1)
    mask_id2 = (labels == id2)

    # Swap them in the copy
    new_labels = labels.copy()
    new_labels[mask_id1] = id2
    new_labels[mask_id2] = id1
    labels = new_labels
    return labels


def merge_label_ids(labels, id1, id2):
    """
    Merge two label IDs into one and reindex labels contiguously.

    Args:
        labels (np.ndarray): array of integer labels
        id1, id2 (int): the two labels to merge

    Returns:
        np.ndarray: new label array with merged and reindexed labels
    """
    labels = labels.copy()

    # merge: replace all id2 with id1
    labels[labels == id2] = id1

    # get unique labels and sort
    unique = np.unique(labels)

    # build mapping to contiguous range 1..K
    mapping = {old: new for new, old in enumerate(unique, start=1)}

    # apply mapping
    new_labels = np.array([mapping[l] for l in labels])

    return new_labels


def check_labels(coords,labels):

    plt.figure()

    # Scatter all points, colored by their label
    scatter_plot = plt.scatter(coords[:, 0], coords[:, 1], c=labels)

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


def switch_labels(i,j,labels):
    if i==0 and j ==0 or j==1:
        labels = switch_label_ids(labels, 1, 5)
        labels = switch_label_ids(labels, 3, 5)
        labels = switch_label_ids(labels, 4, 5)
    if i==0 and j ==2 or j==3:
        labels = switch_label_ids(labels, 1, 4)
        labels = switch_label_ids(labels, 2, 3)
        labels = switch_label_ids(labels, 3, 4)

    if i == 1 and j == 0 or j==2:
        pass
    if i == 1 and j == 1:
        labels = switch_label_ids(labels, 1,3)
        labels = switch_label_ids(labels, 3, 4)
        labels = switch_label_ids(labels, 4, 5)
    if i == 1 and j == 3:
        labels = switch_label_ids(labels, 1, 3)
        labels = switch_label_ids(labels, 2, 4)
        labels = switch_label_ids(labels, 3, 4)


    if i == 2 and j == 0:
        # labels = switch_label_ids(labels, 1, 3)
        labels = merge_label_ids(labels,1,2)
    if i==2 and j==1:
        labels = switch_label_ids(labels, 1, 3)
        labels = switch_label_ids(labels, 3, 4)
        labels = switch_label_ids(labels, 4, 5)
    if i==2 and j==2:
        pass
    if i==2 and j==3:
        labels = switch_label_ids(labels, 1, 3)
        labels = switch_label_ids(labels, 2, 4)
        labels = switch_label_ids(labels, 3, 4)

    return labels


root_folder = "/mnt/data/registration/DLPFC/huifang/"
for i in range(3):
    for j in range(4):

        data = np.load(root_folder+str(i)+"_"+str(j)+"_validation.npz")
        coords = data["coord"]
        layer_labels = data["label"]
        regions = get_connected_regions(coords, layer_labels)

        labels = combine_disconnected_regions(regions,len(layer_labels))

        labels = switch_labels(i,j,labels)

        np.save(root_folder +str(i)+"_"+str(j) + "_region_label.npy", labels)
        # check_labels(coords,labels)



