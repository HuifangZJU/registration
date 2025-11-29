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
import os
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



def plot_registration_topk_visual(
        coords_A, coords_B, labels_A, labels_B,
        spot_index_B=200,
        dist_thresh=5.0,
        label_cmap = plt.get_cmap("tab10"),
        prefix=None
    ):
    coords_A = np.asarray(coords_A)
    coords_B = np.asarray(coords_B)
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)

    # nearest neighbor search (same logic as your metric)
    nn = NearestNeighbors(radius=dist_thresh).fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)
    idxs = neighbors[spot_index_B]

    spot_B = coords_B[spot_index_B]
    true_label = labels_B[spot_index_B]

    # ---------------------------
    # Create side-by-side axes
    # ---------------------------
    fig, (ax_zoom, ax_main) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 2]}
    )



    # ---------------------------
    # Right panel: full slices
    # ---------------------------
    scA = ax_main.scatter(
        coords_A[:, 0], coords_A[:, 1],
        s=30,
        c=labels_A,
        marker="X",
        cmap=label_cmap,
        alpha=0.3,
        label="Slice A (reference)"
    )

    scB = ax_main.scatter(
        coords_B[:, 0], coords_B[:, 1],
        s=30,
        c=labels_B,
        marker='o',
        cmap=label_cmap,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.6,
        label="Slice B (moving)"
    )



    # Reuse these in zoomed view
    cmap = scA.cmap
    norm = scA.norm

    ax_main.set_title("Cross-slice registration: all spots + labels")
    ax_main.set_aspect('equal')
    ax_main.legend(loc="lower left", frameon=True,fontsize=14)

    # draw rectangle showing zoom area on main plot
    margin = dist_thresh * 2
    zoom_rect = Rectangle(
        (spot_B[0] - margin, spot_B[1] - margin),
        2 * margin,
        2 * margin,
        linewidth=1.2,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.8
    )
    ax_main.add_patch(zoom_rect)

    # ---------------------------
    # Left panel: zoomed neighborhood
    # ---------------------------


    # (1) Background: all spots with original color/shape but faded
    ax_zoom.scatter(
        coords_A[:, 0], coords_A[:, 1],
        s=200,
        c=labels_A,
        marker="X",
        cmap=label_cmap,
        alpha=0.3
    )
    ax_zoom.scatter(
        coords_B[:, 0], coords_B[:, 1],
        s=200,
        c=labels_B,
        marker='o',
        cmap=label_cmap,
        alpha=0.6,
        edgecolor='none'
    )

    # (3) Highlight: neighbors in A with solid colors and lines
    true=0
    for ni in idxs:
        coord_A = coords_A[ni]
        label_A = labels_A[ni]

        # match vs mismatch
        is_match = (label_A == true_label)

        line_color = "green" if is_match else "red"
        true = true+1 if is_match else true


        # connecting line
        ax_zoom.plot(
            [spot_B[0], coord_A[0]],
            [spot_B[1], coord_A[1]],
            c=line_color,
            linewidth=3.5,
            alpha=0.9,
            zorder=2
        )



        # neighbor point: original shape, label color, solid
        ax_zoom.scatter(
            coord_A[0], coord_A[1],
            s=200,
            c=[cmap(norm(label_A))],  # <-- actual RGBA color
            marker="X",
            edgecolor='black',
            linewidth=0.8,
            alpha=1.0,
            zorder=3
        )

    # (2) Highlight: chosen B spot with solid color and edge
    if len(idxs) == 0:
        acc=None
    else:
        acc = true/len(idxs)
    ax_zoom.set_title(f"Acc={acc:.3f}", fontsize=14)

    ax_zoom.scatter(
        spot_B[0], spot_B[1],
        s=200,
        c=[cmap(norm(true_label))],  # <-- same mapping as main plot
        marker='o',
        edgecolor='black',
        linewidth=1.0,
        alpha=1.0,
        label="Selected spot (B)",
        zorder=4
    )

    # (4) distance threshold circle around selected B spot
    circ = Circle(
        (spot_B[0], spot_B[1]),
        dist_thresh,
        color='blue',
        fill=False,
        linestyle='--',
        alpha=0.6
    )
    ax_zoom.add_patch(circ)

    # set zoom limits
    ax_zoom.set_xlim(spot_B[0] - margin, spot_B[0] + margin)
    ax_zoom.set_ylim(spot_B[1] - margin, spot_B[1] + margin)
    ax_zoom.set_aspect('equal')

    # custom legend: match vs mismatch
    match_line = Line2D([0], [0], color='green', lw=1.5, label='Label match')
    mismatch_line = Line2D([0], [0], color='red', lw=1.5, label='Label mismatch')

    ax_zoom.legend(
        handles=[match_line, mismatch_line],
        loc="upper left",
        frameon=True,
        fontsize=14
    )
    for ax in [ax_zoom,ax_main]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.invert_yaxis()
    plt.tight_layout()

    if not prefix:
        plt.show()
    else:
        outdir = f"/media/huifang/data/registration/result/pairwise_align/SCC/figures/{prefix}/"
        os.makedirs(outdir,exist_ok=True)
        plt.savefig(outdir+method)


def spotwise_label_consistency_per_spot(coords_A, coords_B, labels_A, labels_B, dist_thresh=5.0):
    """
    Compute per-spot label consistency for each spot in B
    based on the label composition of its neighbors in A.

    Returns
    -------
    per_spot_consistency : np.ndarray of shape (nB,)
        Each value ∈ [0, 1] indicates the fraction of neighbors in A
        that share the same label as labels_B[i].
        Spots with no neighbors within radius → np.nan.
    """
    coords_A = np.asarray(coords_A)
    coords_B = np.asarray(coords_B)
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)

    nn = NearestNeighbors(radius=dist_thresh, algorithm="auto").fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)

    nB = coords_B.shape[0]
    per_spot_consistency = np.full(nB, np.nan, dtype=float)

    for i in range(nB):
        idxs = neighbors[i]
        if len(idxs) == 0:
            continue  # no neighbors → leave as NaN

        neigh_labels = labels_A[idxs]
        true_label = labels_B[i]

        # compute fraction of matching labels
        match_rate = np.mean(neigh_labels == true_label)
        per_spot_consistency[i] = match_rate

    return per_spot_consistency





def find_best_spot(i,j):
    scores_by_method = {}
    coords_B_ref = None
    methods = ['initial','simpleitk','paste','SANTO','voxelmorph','nicetrans','ours']
    for method in methods:
        root_folder = f"/media/huifang/data/registration/result/pairwise_align/SCC/{method}/"
        data_path = root_folder + f"{i}_{j}_result.npz"
        coords_A, coords_B, labels_A, labels_B, mask_shape_ = load_data_from_folder(data_path)

        per_spot_scores = spotwise_label_consistency_per_spot(
            coords_A, coords_B, labels_A, labels_B, dist_thresh=5.0
        )
        scores_by_method[method] = per_spot_scores

        if coords_B_ref is None:
            coords_B_ref = coords_B
        else:
            assert coords_B.shape[0] == coords_B_ref.shape[0], "Mismatch in number of B spots across methods."

    def safe_score(arr):
        out = np.array(arr, copy=True)
        out[np.isnan(out)] = -np.inf  # NaNs can never be winners
        return out

    urs_scores = scores_by_method['ours']
    urs_safe = safe_score(urs_scores)

    other_methods = [m for m in methods if m != 'ours']
    others_safe = np.stack([safe_score(scores_by_method[m]) for m in other_methods], axis=0)  # (n_methods-1, nB)

    # best competing score per spot (across all other methods)
    best_other_per_spot = np.max(others_safe, axis=0)  # shape (nB,)

    # margin = urs - best_other
    margins = urs_safe - best_other_per_spot  # shape (nB,)

    # candidate spots where urs strictly beats all others
    urs_better_mask = urs_safe > best_other_per_spot
    candidate_indices = np.where(urs_better_mask)[0]

    if len(candidate_indices) == 0:
        print("No spot where 'urs' strictly beats all other methods.")
        ordered_best_idx = np.array([], dtype=int)
    else:
        # sort candidates by margin (descending)
        candidate_margins = margins[candidate_indices]
        order = np.argsort(candidate_margins)[::-1]  # largest margin first
        ordered_best_idx = candidate_indices[order]

        print(f"Number of spots where 'urs' beats all others: {len(ordered_best_idx)}")
        print("Top 10 indices (strongest wins):", ordered_best_idx[:10])

        # # optional: print scores for the very best one
        # for idx in range(10):
        #     top_idx = ordered_best_idx[idx]
        #     print(f"\nStrongest example at spot index {top_idx}:")
        #     for m in methods:
        #         print(f"  {m:10s}: {scores_by_method[m][top_idx]}")

    return ordered_best_idx

methods = ['nicetrans', 'ours','initial', 'simpleitk', 'paste', 'SANTO', 'voxelmorph']
for i in [10]:
    print(i)
    for j in [0]:
        print(j)
        index = find_best_spot(i,j)
        # if len(index)>3:
        #     index = index[:3]
        for id in index:
            id=6
            print(id)
            for method in methods:
                root_folder = f"/media/huifang/data/registration/result/pairwise_align/SCC/{method}/"
                data_path = root_folder + f"{i}_{j}_result.npz"
                coords_A, coords_B, labels_A, labels_B, mask_shape_ = load_data_from_folder(data_path)
                prefix = f"{i}_{j}_{id}"
                plot_registration_topk_visual(
                    coords_A, coords_B,
                    labels_A, labels_B,
                    spot_index_B=id,  # choose any spot in slice B
                    dist_thresh=5.0,
                    prefix=prefix
                )
            print('saved')
            test = input()



