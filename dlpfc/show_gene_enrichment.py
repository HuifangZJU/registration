import cv2
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
### ----------------------- FILE LOADING FUNCTION -----------------------
import matplotlib.cm as cm

import pathlib

def load_registration_result(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)
    image = data["img_fix"]
    if image.shape[0]>512 and image.shape[0]<1500:
        pts1 = data["pts_fix"]/2
        pts2 = data["pts_warp"]/2
        pts3 = data["pts_moving"] / 2
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    elif image.shape[0] >1500 and image.shape[0]<2500:
        pts1 = data["pts_fix"]/4
        pts2 = data["pts_warp"] / 4
        pts3 = data["pts_moving"] / 4
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    else:
        return data["pts_fix"], data["pts_warp"], data["pts_moving"],data["fixed_label"],data['moving_label']


def load_registration_image(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)

    return data["img_fix"], data["img_warp"], data["img_moving"]


def visualize_gene_overlay_with_colors(fix_adata_path, moving_adata_paths, registration_paths, gene):
    # Load fixed slice
    fix_adata = sc.read_h5ad(fix_adata_path)
    # print(fix_adata.var_names)
    # test = input()



    gene_expr_fix = fix_adata[:, gene].X.toarray().flatten() if hasattr(fix_adata[:, gene].X, 'toarray') else fix_adata[
                                                                                                              :,
                                                                                                              gene].X.flatten()
    log_expr_fix = np.log1p(gene_expr_fix)
    norm_fix = (log_expr_fix - log_expr_fix.min()) / (log_expr_fix.max() - log_expr_fix.min() + 1e-8)
    coords_fix, _, _ = load_registration_result(registration_paths[0])
    cmap_fix = plt.cm.Reds
    colors_fix = cmap_fix(norm_fix)
    colors_fix[:, -1] = norm_fix

    # Color maps for moving slices
    colors_list = ['Blues', 'Greens', 'Purples']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Fixed slice
    axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18)
    axes[0].set_title(f"Fixed Slice: {gene}")
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Panel 2: Overlay of fixed + raw moving
    axes[1].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18, label='Fixed')
    for idx, moving_path in enumerate(moving_adata_paths):
        moving_adata = sc.read_h5ad(moving_path)
        gene_expr = moving_adata[:, gene].X.toarray().flatten() if hasattr(moving_adata[:, gene].X,
                                                                           'toarray') else moving_adata[:,
                                                                                           gene].X.flatten()
        log_expr = np.log1p(gene_expr)
        norm_expr = (log_expr - log_expr.min()) / (log_expr.max() - log_expr.min() + 1e-8)
        _, _, coords_moving = load_registration_result(registration_paths[idx])
        cmap = plt.get_cmap(colors_list[idx])
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr
        axes[1].scatter(coords_moving[:, 0], coords_moving[:, 1], color=colors, s=10, label=f'Moving {idx + 1}')
    axes[1].set_title(f"Overlay of Raw Moving Slices: {gene}")
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].legend()

    # Panel 3: Overlay of fixed + warped moving
    axes[2].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18, label='Fixed')
    for idx, moving_path in enumerate(moving_adata_paths):
        moving_adata = sc.read_h5ad(moving_path)
        gene_expr = moving_adata[:, gene].X.toarray().flatten() if hasattr(moving_adata[:, gene].X,
                                                                           'toarray') else moving_adata[:,
                                                                                           gene].X.flatten()
        log_expr = np.log1p(gene_expr)
        norm_expr = (log_expr - log_expr.min()) / (log_expr.max() - log_expr.min() + 1e-8)
        _, coords_warp, _ = load_registration_result(registration_paths[idx])
        cmap = plt.get_cmap(colors_list[idx])
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr
        axes[2].scatter(coords_warp[:, 0], coords_warp[:, 1], color=colors, s=18, label=f'Warped {idx + 1}')
    axes[2].set_title(f"Overlay of Warped Slices: {gene}")
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()
    axes[2].legend()

    plt.tight_layout()
    plt.show()



def load_gene_expression(adata_path, gene):
    adata = sc.read_h5ad(adata_path)
    gene_expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
    gene_expr = np.log1p(gene_expr)
    gene_expr = np.log1p(gene_expr)
    norm_expr = (gene_expr - gene_expr.min()) / (gene_expr.max() - gene_expr.min() + 1e-6)
    return norm_expr

def visualize_gene_overlay(fix_adata_path, moving_adata_paths, registration_paths, gene='MFGE8'):
    # Load fixed slice
    norm_fix = load_gene_expression(fix_adata_path, gene)
    coords_fix, _, _ ,_,_= load_registration_result(registration_paths[0])
    cmap = plt.cm.Reds
    colors_fix = cmap(norm_fix)
    colors_fix[:, -1] = norm_fix

    # # Prepare figure
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #
    # # Panel 1: Fixed slice gene expression
    # axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18)
    # axes[0].set_title(f'Fixed Slice: {gene} Expression')
    # axes[0].set_aspect('equal')
    # axes[0].invert_yaxis()
    #
    # # Panel 2: Overlay of raw moving slices
    # for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
    #     norm_expr = load_gene_expression(adata_path, gene)
    #     _, _, coords_moving,_,_ = load_registration_result(reg_path)
    #     colors = cmap(norm_expr)
    #     colors[:, -1] = norm_expr
    #     axes[1].scatter(coords_moving[:, 0], coords_moving[:, 1], color=colors, s=18)
    # axes[1].set_title('Overlay of Raw Moving Slices')
    # axes[1].set_aspect('equal')
    # axes[1].invert_yaxis()
    #
    # # Panel 3: Overlay of warped slices
    # for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
    #     norm_expr = load_gene_expression(adata_path, gene)
    #     _, coords_warp, _ ,_,_= load_registration_result(reg_path)
    #     colors = cmap(norm_expr)
    #     colors[:, -1] = norm_expr
    #     axes[2].scatter(coords_warp[:, 0], coords_warp[:, 1], color=colors, s=18)
    # axes[2].set_title('Overlay of Warped Slices')
    # axes[2].set_aspect('equal')
    # axes[2].invert_yaxis()
    #
    # plt.tight_layout()
    # plt.show()
    # Prepare 5 subplots
    # Prepare 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    # Panel 1: Fixed slice gene expression
    axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=10)
    # axes[0].set_title('Slice1')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    axes[0].invert_yaxis()

    # Panels 2–4: Individual moving slices
    for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
        norm_expr = load_gene_expression(adata_path, gene)
        _, _, coords_moving,_,_ = load_registration_result(reg_path)
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr  # transparency reflects expression level
        axes[idx + 1].scatter(coords_moving[:, 0], coords_moving[:, 1], color=colors, s=10)
        # axes[idx + 1].set_title(f'Slice {idx + 1}')
        axes[idx + 1].set_aspect('equal')
        axes[idx + 1].axis('off')
        axes[idx + 1].invert_yaxis()

    # Panel 5: Overlay of warped slices
    for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
        norm_expr = load_gene_expression(adata_path, gene)
        _, coords_warp, _,_,_ = load_registration_result(reg_path)
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr
        axes[4].scatter(coords_warp[:, 0], coords_warp[:, 1], color=colors, s=10)
    axes[4].set_title('Overlay of Warped Slices')
    axes[4].set_aspect('equal')
    axes[4].axis('off')
    axes[4].invert_yaxis()

    # fig.canvas.draw()
    # out_dir = pathlib.Path("/home/huifang/workspace/grant/k99/resubmission/figures")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for idx, ax in enumerate(fig.axes, start=1):
    #     # tight bounding box of *this* axes in figure coordinates
    #     bbox = ax.get_tightbbox(fig.canvas.get_renderer()) \
    #         .transformed(fig.dpi_scale_trans.inverted())
    #
    #     # build filename subplot_1.png, subplot_2.png, ...
    #     fname = out_dir / f"subplot_{idx}.png"
    #
    #     # save only the region inside bbox
    #     fig.savefig(fname, dpi=300, bbox_inches=bbox)
    #     print(f"Saved {fname}")
    #
    # plt.close(fig)  # optional: free memory

    plt.tight_layout()
    plt.show()

def simulate_visualize_gene_overlay(fix_adata_path, moving_adata_paths, registration_paths, gene='MFGE8'):
    # Load fixed slice
    norm_fix = load_gene_expression(fix_adata_path, gene)
    coords_fix, _, _ = load_registration_result(registration_paths[0])
    cmap = plt.cm.Reds
    colors_fix = cmap(norm_fix)
    colors_fix[:, -1] = norm_fix

    # Initialize accumulators
    coords_moving_all = [coords_fix]
    norm_moving_all = [norm_fix]
    coords_warp_all = [coords_fix]
    norm_warp_all = [norm_fix]

    # Accumulate moving slice data
    for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
        norm_expr = load_gene_expression(adata_path, gene)
        _, coords_warp, coords_moving = load_registration_result(reg_path)
        coords_moving_all.append(coords_moving)
        norm_moving_all.append(norm_expr)
        coords_warp_all.append(coords_warp)
        norm_warp_all.append(norm_expr)

    # Concatenate
    all_coords_moving = np.concatenate(coords_moving_all, axis=0)
    all_norm_moving = np.concatenate(norm_moving_all, axis=0)
    all_colors_moving = cmap(all_norm_moving)
    all_colors_moving[:, -1] = all_norm_moving

    all_coords_warp = np.concatenate(coords_warp_all, axis=0)
    all_norm_warp = np.concatenate(norm_warp_all, axis=0)
    all_colors_warp = cmap(all_norm_warp)
    all_colors_warp[:, -1] = all_norm_warp

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Fixed
    axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=10)
    axes[0].set_title(f'Fixed Slice: {gene} Expression')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Panel 2: Raw Moving + Fixed
    axes[1].scatter(all_coords_moving[:, 0], all_coords_moving[:, 1], color=all_colors_moving, s=10)
    axes[1].set_title('Overlay: Fixed + Raw Moving')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()

    # Panel 3: Warped + Fixed
    axes[2].scatter(all_coords_warp[:, 0], all_coords_warp[:, 1], color=all_colors_warp, s=10)
    axes[2].set_title('Overlay: Fixed + Warped Moving')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.show()

def visualize_gene_overlay_no_alpha(fix_adata_path, moving_adata_paths, registration_paths, gene='MFGE8'):
    # Load fixed data
    norm_fix = load_gene_expression(fix_adata_path, gene)
    coords_fix, _, _ = load_registration_result(registration_paths[0])

    # Initialize lists for concatenation
    all_coords_moving = [coords_fix]
    all_expr_moving = [norm_fix]
    all_coords_warp = [coords_fix]
    all_expr_warp = [norm_fix]

    # Load moving slices
    for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
        norm_expr = load_gene_expression(adata_path, gene)
        _, coords_warp, coords_moving = load_registration_result(reg_path)
        all_coords_moving.append(coords_moving)
        all_expr_moving.append(norm_expr)
        all_coords_warp.append(coords_warp)
        all_expr_warp.append(norm_expr)

    # Concatenate all data
    all_coords_moving = np.vstack(all_coords_moving)
    all_expr_moving = np.concatenate(all_expr_moving)
    all_coords_warp = np.vstack(all_coords_warp)
    all_expr_warp = np.concatenate(all_expr_warp)

    cmap = plt.cm.Reds
    colors_fix = cmap(norm_fix)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Fixed only
    axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=12)
    axes[0].set_title(f'Fixed Slice: {gene}')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Raw moving + fixed
    axes[1].scatter(all_coords_moving[:, 0], all_coords_moving[:, 1], color=cmap(all_expr_moving), s=12)
    axes[1].set_title(f'Overlay: Fixed + Raw Moving')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()

    # Warped + fixed
    axes[2].scatter(all_coords_warp[:, 0], all_coords_warp[:, 1], color=cmap(all_expr_warp), s=12)
    axes[2].set_title(f'Overlay: Fixed + Warped')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.show()



def visualize_coordinate_overlay(fix_coords, moving_coords_list, warped_coords_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Define color list
    color_list = ['red', 'blue', 'green', 'purple']

    # Panel 1: Fixed + raw moving slices
    axes[0].scatter(fix_coords[:, 0], fix_coords[:, 1], color=color_list[0], s=8, label='Fixed')
    for idx, coords in enumerate(moving_coords_list):
        axes[0].scatter(coords[:, 0], coords[:, 1], color=color_list[idx + 1], s=8, label=f'Moving {idx + 1}')
    axes[0].set_title('Overlay of Fixed + Raw Moving Slices')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].legend()

    # Panel 2: Fixed + warped slices
    axes[1].scatter(fix_coords[:, 0], fix_coords[:, 1], color=color_list[0], s=8, label='Fixed')
    for idx, coords in enumerate(warped_coords_list):
        axes[1].scatter(coords[:, 0], coords[:, 1], color=color_list[idx + 1], s=8, label=f'Warped {idx + 1}')
    axes[1].set_title('Overlay of Fixed + Warped Slices')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def visualize_overlay_coords_with_labels(registration_paths):
    # Prepare figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: Overlay of raw coordinates (fixed + moving)
    all_coords_raw = []
    all_labels_raw = []
    all_coords_warp=[]

    for idx, reg_path in enumerate(registration_paths):
        coords_fix, coords_warp, coords_moving, label_fix,label_moving = load_registration_result(reg_path)
        if idx == 0:
            all_coords_raw.append(coords_fix)
            all_coords_warp.append(coords_fix)
            all_labels_raw.append(label_fix.squeeze())  # Assume labels apply to fix in the first one


        all_coords_raw.append(coords_moving)
        all_coords_warp.append(coords_warp)
        all_labels_raw.append(label_moving.squeeze())

    coords_raw_concat = np.concatenate(all_coords_raw, axis=0)
    labels_raw_concat = np.concatenate(all_labels_raw, axis=0)
    coords_warp_concat = np.concatenate(all_coords_warp, axis=0)
    # coords_raw_concat = all_coords_raw[3]
    # labels_raw_concat = all_labels_raw[3]
    # coords_warp_concat = all_coords_warp[3]

    cmap = cm.get_cmap('tab10', len(np.unique(labels_raw_concat)))
    colors_raw = cmap(labels_raw_concat)

    colors_warp = cmap(labels_raw_concat)

    axes[0].scatter(coords_raw_concat[:, 0], coords_raw_concat[:, 1], c=colors_raw, s=10,alpha=0.5)
    # axes[0].set_title('Overlay of Raw Coordinates by Label')
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    axes[1].scatter(coords_warp_concat[:, 0], coords_warp_concat[:, 1], c=colors_warp, s=10,alpha=0.4)
    # axes[1].set_title('Overlay of Warped Coordinates by Label')
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.tight_layout()

    # fig.canvas.draw()
    # out_dir = pathlib.Path("/home/huifang/workspace/grant/k99/resubmission/figures")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for idx, ax in enumerate(fig.axes, start=1):
    #     # tight bounding box of *this* axes in figure coordinates
    #     bbox = ax.get_tightbbox(fig.canvas.get_renderer()) \
    #         .transformed(fig.dpi_scale_trans.inverted())
    #
    #     # build filename subplot_1.png, subplot_2.png, ...
    #     fname = out_dir / f"subplot_{idx}.png"
    #
    #     # save only the region inside bbox
    #     fig.savefig(fname, dpi=300, bbox_inches=bbox)
    #     print(f"Saved {fname}")

    # plt.close(fig)  # optional: free memory


    plt.show()

def visualize_overlay_image(registration_paths):
    # Prepare figure with 2 subplots

    # Panel 1: Overlay of raw coordinates (fixed + moving)
    all_images_raw = []
    all_images_warp=[]

    for idx, reg_path in enumerate(registration_paths):
        img_fix, img_warp, img_moving = load_registration_image(reg_path)
        if idx == 0:
            all_images_raw.append(img_fix)
            all_images_warp.append(img_fix)
        all_images_raw.append(img_moving)
        all_images_warp.append(img_warp)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # --------- Overlay of all RAW (fixed + every moving) --------------
    # # axes[0].set_title('Fixed + All Raw Moving')
    # for k, img in enumerate(all_images_raw):
    #     axes[0].imshow(img, alpha=1.0 if k == 0 else 0.30)  # 30 % opacity for movers
    # axes[0].axis('off')
    #
    # # --------- Overlay of all WARPED (fixed + every warped) -----------
    # # axes[1].set_title('Fixed + All Warped')
    # for k, img in enumerate(all_images_warp):
    #     axes[1].imshow(img, alpha=1.0 if k == 0 else 0.30)  # same strategy
    # axes[1].axis('off')
    #
    # plt.tight_layout()

    n_img = len(all_images_raw)  # how many raw images
    fig, axes = plt.subplots(1, n_img, figsize=(4 * n_img, 4))

    # axes is a single Axes object if n_img == 1 → wrap into a list
    if n_img == 1:
        axes = [axes]

    for ax, img in zip(axes, all_images_raw):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    #
    fig.canvas.draw()
    out_dir = pathlib.Path("/home/huifang/workspace/grant/k99/resubmission/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, ax in enumerate(fig.axes, start=1):
        # tight bounding box of *this* axes in figure coordinates
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()) \
            .transformed(fig.dpi_scale_trans.inverted())

        # build filename subplot_1.png, subplot_2.png, ...
        fname = out_dir / f"subplot_{idx}.png"

        # save only the region inside bbox
        fig.savefig(fname, dpi=300, bbox_inches=bbox)
        print(f"Saved {fname}")

    # plt.close(fig)  # optional: free memory


    plt.show()


root_folder = "/media/huifang/data/registration/result/center_align/DLPFC/attention_fusion/"
gene_folder = "/home/huifang/workspace/code/registration/data/DLPFC"
sampleids=[['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]
for i in range(2,3):

    fix_adata_path = f"{gene_folder}/{sampleids[i][0]}_preprocessed.h5"
    moving_adata_paths = [
        f"{gene_folder}/{sampleids[i][1]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][2]}_preprocessed.h5",
        f"{gene_folder}/{sampleids[i][3]}_preprocessed.h5"
    ]
    registration_paths = [
        f"{root_folder}/{i}_0_result.npz",
        f"{root_folder}/{i}_1_result.npz",
        f"{root_folder}/{i}_2_result.npz"
    ]
    # visualize_overlay_image(registration_paths)
    # visualize_overlay_coords_with_labels(registration_paths)
    # test = input()
    # coords_fix, _, _,label,_ = load_registration_result(registration_paths[0])
    # coords_moving_list = []
    # coords_warped_list = []
    #
    # for reg_path in registration_paths:
    #     _, coords_warp, coords_moving,_,_ = load_registration_result(reg_path)
    #     coords_moving_list.append(coords_moving)
    #     coords_warped_list.append(coords_warp)
    #
    # # Visualize
    # visualize_coordinate_overlay(coords_fix, coords_moving_list, coords_warped_list)

    visualize_gene_overlay(fix_adata_path, moving_adata_paths, registration_paths, gene='MOBP')
    # visualize_gene_overlay_no_alpha(fix_adata_path, moving_adata_paths, registration_paths, gene='MFGE8')
    # visualize_gene_overlay_with_colors(fix_adata_path, moving_adata_paths, registration_paths, gene='AGRN')
    #
    # for j in range(3):
    #     data_path = root_folder + str(i) + "_" + str(j) + "_result.npz"
    #     coords_fix, coords_warp, coords_moving, labels = load_data_from_folder(data_path)
    #
    #     fix_adata = sc.read_h5ad(gene_folder+str(sampleids[i][0])+'_preprocessed.h5')
    #
    #     gene = 'MFGE8'
    #     gene_expr_fix = fix_adata[:, gene].X.toarray().flatten() if hasattr(fix_adata[:, gene].X, 'toarray') else fix_adata[:,
    #                                                                                                       gene].X.flatten()
    #     norm_fix = (gene_expr_fix - gene_expr_fix.min()) / (gene_expr_fix.max() -gene_expr_fix.min())
    #     cmap_red = plt.colormaps['Reds']
    #     colors_xenium = cmap_red(norm_fix)
    #     colors_xenium[:, -1] = norm_fix  # set alpha channel
    #
    #
    #     plt.scatter(coords_fix[:, 0], coords_fix[:, 1],
    #                  color=colors_xenium, s=10, label='MFGE8')
    #     plt.axis("equal")
    #     plt.gca().invert_yaxis()  # optional
    #     plt.show()
    #
    #
    # gene_moving = sampleids[i][j + 1]

