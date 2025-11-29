import cv2
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
### ----------------------- FILE LOADING FUNCTION -----------------------
import matplotlib.cm as cm
from coordinate_utils import find_inverse_warp_coords
from matplotlib.patches import ConnectionPatch, Rectangle

def load_registration_result(folder_path):
    """Load image and spot data from a given folder."""

    data = np.load(folder_path)

    pts1 = data["pts1"]  # shape (4200, 2)
    pts2 = data["pts2"]

    img1 = data["img1"]  # shape (4200, 2)
    img2 = data["img2"]

    return pts1, pts2,img1,img2


def load_gene_expression(adata_path, gene):
    adata = sc.read_h5ad(adata_path)
    gene_expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
    gene_expr = np.log1p(gene_expr)
    gene_expr = np.log1p(gene_expr)
    norm_expr = (gene_expr - gene_expr.min()) / (gene_expr.max() - gene_expr.min() + 1e-6)
    return norm_expr




def load_scc_data():
    path_to_output_dir = '/media/huifang/data/registration/SCC/huifang/'
    path_to_h5ads = path_to_output_dir + 'H5ADs/'

    patient_2 = []
    patient_5 = []
    patient_9 = []
    patient_10 = []

    patients = {
        # "patient_2": patient_2,
        # "patient_5": patient_5,
        "patient_9": patient_9,
        # "patient_10": patient_10,
    }
    for k in patients.keys():
        for i in range(3):
            data = sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad')
            patients[k].append(data)


    return patients


def load_aligned_coordinates(patients_data, result_folder, fixed_slice_id=0):
    for key, slices in patients_data.items():
        patient_id = key.split('_')[1]
        for data_id, adata in enumerate(slices):
            if data_id == fixed_slice_id:
                continue
            else:
                fixed_coor, registered_coor,fixed_image,registered_image = load_registration_result(
                    f"{result_folder}/{patient_id}_{fixed_slice_id}_{data_id}_result.npz"
                )
                slices[fixed_slice_id].obsm['spatial_registered'] = fixed_coor
                slices[fixed_slice_id].uns['image_array_registered'] = fixed_image
                slices[fixed_slice_id].uns[
                    'high_res_img_path'] = f"/home/huifang/workspace/code/registration/data/SCC/scc_p{patient_id}_layer{fixed_slice_id+1}_cropped.jpg"
                adata.uns['high_res_img_path']=f"/home/huifang/workspace/code/registration/data/SCC/scc_p{patient_id}_layer{data_id+1}_cropped.jpg"
                adata.obsm['spatial_registered'] = registered_coor
                adata.uns['image_array_registered'] = registered_image
    return patients_data



def plot_spot_registration_triptych(
    fixed_adata,
    moving_adatas,
    sample_n=200,
    zoom_center=(220,330),   # (x, y) in fixed image coords; if None, use mean of all registered spots
    zoom_size=80,      # side length of zoom window in pixels
    figsize=(16, 10),
):
    assert len(moving_adatas) >= 2, "moving_adatas must contain at least two AnnData objects."

    mov0, mov1 = moving_adatas[0], moving_adatas[1]

    # ---------- helper to get image ----------
    def get_image(adata):
        img = adata.uns.get("image_array", None)
        if img is None:
            img = adata.uns["image_array"]
        return img

    fixed_img = get_image(fixed_adata)
    mov0_img = get_image(mov0)
    mov1_img = get_image(mov1)


    fixed_hi_res_image = plt.imread(fixed_adata.uns['high_res_img_path'])
    ratio = fixed_hi_res_image.shape[0] / fixed_img.shape[0]
    # spot coordinates
    fixed_spots = fixed_adata.obsm.get("spatial_image_coor", None)
    if fixed_spots is not None:
        fixed_spots = np.asarray(fixed_spots)





    mov0_spots_orig = np.asarray(mov0.obsm["spatial_image_coor"])
    mov0_spots_reg = np.asarray(mov0.obsm["spatial_registered"])
    mov1_spots_orig = np.asarray(mov1.obsm["spatial_image_coor"])
    mov1_spots_reg = np.asarray(mov1.obsm["spatial_registered"])

    plt.imshow(fixed_hi_res_image, cmap="gray")
    plt.scatter(
        ratio*fixed_spots[:, 0], ratio*fixed_spots[:, 1],
        s=100, c="yellow", marker="o", alpha=0.8,
        label="Fixed spots"
    )
    plt.scatter(
        ratio*mov0_spots_reg[:, 0], ratio*mov0_spots_reg[:, 1],
        s=100, c="cyan", marker="s", alpha=0.9,
        edgecolor="k", linewidth=0.2, label="Warped (mov0)"
    )

    # warped mov1
    plt.scatter(
        ratio*mov1_spots_reg[:, 0], ratio*mov1_spots_reg[:, 1],
        s=100, c="magenta", marker="^", alpha=0.9,
        edgecolor="k", linewidth=0.2, label="Warped (mov1)"
    )

    plt.gca().set_title("Integrated slice (dense)")
    plt.gca().set_xticks([]);
    plt.gca().set_yticks([])
    # if fixed_spots is not None:
    #     plt.legend(loc="upper right", frameon=True, fontsize=16)
    plt.show()

    # all registered spots (for zoom center)
    all_reg = np.vstack([mov0_spots_reg, mov1_spots_reg])
    if zoom_center is None:
        zoom_center = all_reg.mean(axis=0)
    zx, zy = zoom_center
    half = zoom_size / 2.0

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax00, ax01, ax02 = axes[0]
    ax10, ax11, ax12 = axes[1]

    # RNG for sampling arrows
    rng = np.random.default_rng(0)

    def sample_indices(n_total, n_sample):
        n = min(n_total, n_sample)
        if n == 0:
            return np.array([], dtype=int)
        return rng.choice(n_total, size=n, replace=False)

    # RNG for sampling arrows
    rng = np.random.default_rng(0)

    def sample_indices(n_total, n_sample):
        n = min(n_total, n_sample)
        if n == 0:
            return np.array([], dtype=int)
        return rng.choice(n_total, size=n, replace=False)

    # --------------------------------------------------
    # Row 0, col 0: moving0 – original spots + arrows to registered
    # --------------------------------------------------
    ax00.imshow(mov0_img, cmap="gray")
    dx0 = mov0_spots_reg[:, 0] - mov0_spots_orig[:, 0]
    dy0 = mov0_spots_reg[:, 1] - mov0_spots_orig[:, 1]

    ax00.quiver(
        mov0_spots_orig[:, 0], mov0_spots_orig[:, 1],
        dx0, dy0,
        angles="xy",
        scale_units="xy",
        scale=2,
        color="cyan",
        alpha=0.9,
        width=0.004,  # ↑ make the shaft thicker
        headwidth=4,  # optional: bigger arrow head
        headlength=6,  # optional: longer head
        headaxislength=5,  # optional: head axis length
        zorder=2,
    )

    ax00.set_title("Moving slice 0\noriginal → registered (arrows)")
    ax00.set_xticks([]);
    ax00.set_yticks([])
    ax00.legend(loc="lower left", frameon=True, fontsize=8)

    # --------------------------------------------------
    # Row 0, col 1: fixed image + fixed spots
    # --------------------------------------------------
    ax01.imshow(fixed_img, cmap="gray")
    if fixed_spots is not None:
        ax01.scatter(
            fixed_spots[:, 0], fixed_spots[:, 1],
            s=10, c="yellow", marker="o", alpha=0.8,
            label="Fixed spots"
        )
    ax01.set_title("Fixed slice (reference)")
    ax01.set_xticks([]);
    ax01.set_yticks([])
    if fixed_spots is not None:
        ax01.legend(loc="lower left", frameon=True, fontsize=8)

    # --------------------------------------------------
    # Row 0, col 2: moving1 – original spots + arrows to registered
    # --------------------------------------------------
    ax02.imshow(mov1_img, cmap="gray")
    dx1 = mov1_spots_reg[:, 0] - mov1_spots_orig[:, 0]
    dy1 = mov1_spots_reg[:, 1] - mov1_spots_orig[:, 1]

    ax02.quiver(
        mov1_spots_orig[:, 0], mov1_spots_orig[:, 1],
        dx1, dy1,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="magenta",
        alpha=0.9,
        width=0.004,  # ↑ make the shaft thicker
        headwidth=4,  # optional: bigger arrow head
        headlength=6,  # optional: longer head
        headaxislength=5,  # optional: head axis length
        zorder=2,
    )

    ax02.set_title("Moving slice 1\noriginal → registered (arrows)")
    ax02.set_xticks([]);
    ax02.set_yticks([])
    ax02.legend(loc="lower left", frameon=True, fontsize=8)

    # --------------------------------------------------
    # Row 1, col 1: joint slice (fixed + all three sets of spots)
    # --------------------------------------------------
    ax11.imshow(fixed_img, cmap="gray")

    # fixed spots
    if fixed_spots is not None:
        ax11.scatter(
            fixed_spots[:, 0], fixed_spots[:, 1],
            s=8, c="yellow", marker="o", alpha=0.8,
            label="Fixed spots"
        )

    # warped mov0
    ax11.scatter(
        mov0_spots_reg[:, 0], mov0_spots_reg[:, 1],
        s=10, c="cyan", marker="s", alpha=0.9,
        edgecolor="k", linewidth=0.2, label="Warped (mov0)"
    )

    # warped mov1
    ax11.scatter(
        mov1_spots_reg[:, 0], mov1_spots_reg[:, 1],
        s=10, c="magenta", marker="^", alpha=0.9,
        edgecolor="k", linewidth=0.2, label="Warped (mov1)"
    )

    ax11.set_title("Fixed slice with\nfixed + warped spots")
    ax11.set_xticks([]);
    ax11.set_yticks([])
    ax11.legend(loc="lower left", frameon=True, fontsize=8)

    # draw rectangle showing zoom area
    zoom_rect = Rectangle(
        (zx - half, zy - half),
        2 * half,
        2 * half,
        linewidth=2.0,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.9,
    )
    ax11.add_patch(zoom_rect)

    # --------------------------------------------------
    # Row 1, col 0: zoomed region on fixed slice (all three sets)
    # --------------------------------------------------


    # ax12.imshow(fixed_img, cmap="gray")
    ax12.imshow(fixed_hi_res_image)
    def plot_zoom_spots(coords, color, marker, label=None):
        if coords is None:
            return
        coords = np.asarray(coords)
        mask = (
                (coords[:, 0] >= zx - half) & (coords[:, 0] <= zx + half) &
                (coords[:, 1] >= zy - half) & (coords[:, 1] <= zy + half)
        )
        sel = coords[mask]
        if sel.size == 0:
            return
        sel = sel*ratio
        ax12.scatter(
            sel[:, 0], sel[:, 1],
            s=100, c=color, marker=marker,
            alpha=0.95, edgecolor="k", linewidth=0.3,
            label=label
        )

    # fixed spots
    plot_zoom_spots(fixed_spots, "yellow", "o", label="Fixed spots")

    # warped mov0
    plot_zoom_spots(mov0_spots_reg, "cyan", "s", label="Warped (mov0)")

    # warped mov1
    plot_zoom_spots(mov1_spots_reg, "magenta", "^", label="Warped (mov1)")

    ax12.set_xlim(ratio*(zx - half), ratio*(zx + half))
    ax12.set_ylim(ratio*(zy + half), ratio*(zy - half))  # same orientation as imshow


    ax12.set_xticks([]);
    ax12.set_yticks([])
    ax12.set_title("Zoom: fixed + warped spots\nin a vacant region")
    ax12.legend(loc="lower right", frameon=True, fontsize=14)

    # --------------------------------------------------
    # Row 1, col 2: explanation (optional)
    # --------------------------------------------------
    ax10.axis("off")
    ax10.text(
        0.5, 0.5,
        "Top row: per-slice displacement fields\n"
        "Bottom center: all spots on fixed slice\n"
        "Bottom left: zoomed region with\n"
        "fixed + warped spots",
        ha="center", va="center", fontsize=10,
        transform=ax10.transAxes,
    )

    # keep frames, hide ticks nicely
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    plt.show()




root_folder = "/media/huifang/data/registration/result/center_align/SCC/ours"
patients_data = load_scc_data()

fixed_id = 2
aligned_patients_data = load_aligned_coordinates(patients_data,root_folder,fixed_slice_id=fixed_id)
for key, slices in patients_data.items():

    fixed_slice = slices[fixed_id]
    moving_slices = [a for i, a in enumerate(slices) if i != fixed_id]



    plot_spot_registration_triptych(
        fixed_adata=fixed_slice,
        moving_adatas=moving_slices,

    )



    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_img, ax_reg = axes





    fixed_image = slices[fixed_id].uns['image_array']
    ax_img.imshow(fixed_image)
    ax_reg.imshow(fixed_image)

    fig.suptitle(f"Patient: {key}", y=0.98)

    # Optional: color per slice
    cmap = plt.get_cmap("tab10")

    for i, adata in enumerate(slices):
        img_coor = adata.obsm['spatial_image_coor']
        reg_coor = adata.obsm['spatial_registered']

        color = cmap(i % 10)

        # Overlay all original image coordinates
        if i==fixed_id:
            ax_img.scatter(
                img_coor[:, 0], img_coor[:, 1],
                s=30,  color=color, label=f"Slice {i}" if i == 0 else None
            )

        # Overlay all registered coordinates
        ax_reg.scatter(
            reg_coor[:, 0], reg_coor[:, 1],
            s=30,  color=color
        )

    # Formatting
    ax_img.set_title("Overlay: spatial_image_coor (all slices)")
    ax_reg.set_title("Overlay: spatial_registered (all slices)")

    for ax in (ax_img, ax_reg):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        # ax.invert_yaxis()
    # If you want a legend for slices, uncomment:
    # ax_img.legend(loc="upper right", fontsize=8, title="Slices")

    plt.tight_layout()
    plt.show()

