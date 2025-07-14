import cv2
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
### ----------------------- FILE LOADING FUNCTION -----------------------
import matplotlib.cm as cm



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


def load_gene_expression(adata_path, gene):
    adata = sc.read_h5ad(adata_path)
    gene_expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
    gene_expr = np.log1p(gene_expr)
    gene_expr = np.log1p(gene_expr)
    norm_expr = (gene_expr - gene_expr.min()) / (gene_expr.max() - gene_expr.min() + 1e-6)
    return norm_expr


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

def visualize_gene_overlay(fix_adata_path, moving_adata_paths, registration_paths, gene='MFGE8'):
    # Load fixed slice
    norm_fix = load_gene_expression(fix_adata_path, gene)
    coords_fix, _, _ = load_registration_result(registration_paths[0])
    cmap = plt.cm.Reds
    colors_fix = cmap(norm_fix)
    colors_fix[:, -1] = norm_fix

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Fixed slice gene expression
    axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18)
    axes[0].set_title(f'Fixed Slice: {gene} Expression')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Panel 2: Overlay of raw moving slices
    for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
        norm_expr = load_gene_expression(adata_path, gene)
        _, _, coords_moving = load_registration_result(reg_path)
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr
        axes[1].scatter(coords_moving[:, 0], coords_moving[:, 1], color=colors, s=18)
    axes[1].set_title('Overlay of Raw Moving Slices')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()

    # Panel 3: Overlay of warped slices
    for adata_path, reg_path in zip(moving_adata_paths, registration_paths):
        norm_expr = load_gene_expression(adata_path, gene)
        _, coords_warp, _ = load_registration_result(reg_path)
        colors = cmap(norm_expr)
        colors[:, -1] = norm_expr
        axes[2].scatter(coords_warp[:, 0], coords_warp[:, 1], color=colors, s=18)
    axes[2].set_title('Overlay of Warped Slices')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.show()
    # Prepare 5 subplots
    # Prepare 5 subplots
    # fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    #
    # # Panel 1: Fixed slice gene expression
    # axes[0].scatter(coords_fix[:, 0], coords_fix[:, 1], color=colors_fix, s=18)
    # axes[0].set_title(f'Fixed Slice\n{gene}')
    # axes[0].set_aspect('equal')
    # axes[0].axis('off')
    # axes[0].invert_yaxis()
    #
    # # Panels 2–4: Individual moving slices
    # for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
    #     norm_expr = load_gene_expression(adata_path, gene)
    #     _, _, coords_moving = load_registration_result(reg_path)
    #     colors = cmap(norm_expr)
    #     colors[:, -1] = norm_expr  # transparency reflects expression level
    #     axes[idx + 1].scatter(coords_moving[:, 0], coords_moving[:, 1], color=colors, s=18)
    #     axes[idx + 1].set_title(f'Moving Slice {idx + 1}')
    #     axes[idx + 1].set_aspect('equal')
    #     axes[idx + 1].axis('off')
    #     axes[idx + 1].invert_yaxis()
    #
    # # Panel 5: Overlay of warped slices
    # for idx, (adata_path, reg_path) in enumerate(zip(moving_adata_paths, registration_paths)):
    #     norm_expr = load_gene_expression(adata_path, gene)
    #     _, coords_warp, _ = load_registration_result(reg_path)
    #     colors = cmap(norm_expr)
    #     colors[:, -1] = norm_expr
    #     axes[4].scatter(coords_warp[:, 0], coords_warp[:, 1], color=colors, s=18)
    # axes[4].set_title('Overlay of Warped Slices')
    # axes[4].set_aspect('equal')
    # axes[4].axis('off')
    # axes[4].invert_yaxis()
    #
    # plt.tight_layout()
    # plt.show()

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



def visualize_coordinate_overlay(registration_paths):
    fix_coords, _, _,label,_= load_registration_result(registration_paths[0])
    moving_coords_list = []
    warped_coords_list = []

    for reg_path in registration_paths:
        _, coords_warp, coords_moving,_,_ = load_registration_result(reg_path)
        moving_coords_list.append(coords_moving)
        warped_coords_list.append(coords_warp)


    fig, axes = plt.subplots(1, 2, figsize=(8,8))

    color_list =["#1e6cb3",'#e55709','#501d8a','#1c8041']

    # Panel 1: Fixed + raw moving slices
    axes[0].scatter(fix_coords[:, 0], fix_coords[:, 1], color=color_list[0], s=18, label='Fixed')
    for idx, coords in enumerate(moving_coords_list):
        # axes[0].scatter(coords[:, 0], coords[:, 1], color=color_list[idx + 1], s=18, label=f'Moving {idx + 1}')
        axes[0].scatter(coords[:, 0], coords[:, 1], color=color_list[idx + 1], s=18, label='Moving')
    axes[0].set_title('Unregistered',fontsize=14)
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=12)

    # Panel 2: Fixed + warped slices
    axes[1].scatter(fix_coords[:, 0], fix_coords[:, 1], color=color_list[0], s=18, label='Fixed')
    for idx, coords in enumerate(warped_coords_list):
        axes[1].scatter(coords[:, 0], coords[:, 1], color=color_list[idx + 1], s=18, label='Warped')
    axes[1].set_title('Registered',fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('/home/huifang/workspace/grant/k99/resubmission/figures/2.png', dpi=300)
    plt.show()

def visualize_overlay_coords_with_labels(registration_paths):
    # Prepare figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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

    cmap = cm.get_cmap('tab20', len(np.unique(labels_raw_concat)))
    colors_raw = cmap(labels_raw_concat)

    colors_warp = cmap(labels_raw_concat)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # ── left panel ─────────────────────────────────────────
    axes[0].scatter(coords_raw_concat[:, 0], coords_raw_concat[:, 1],
                    c=colors_raw, s=10, alpha=0.5, label='Fixed')
    axes[0].scatter(coords_raw_concat[:, 0], coords_raw_concat[:, 1],
                    c=colors_raw, s=10, alpha=0.5, label='Moving 1')
    axes[0].set_title('Overlay of Raw Moving Slices')

    # ── right panel ────────────────────────────────────────
    axes[1].scatter(coords_warp_concat[:, 0], coords_warp_concat[:, 1],
                    c=colors_warp, s=10, alpha=0.5, label='Fixed')
    axes[1].scatter(coords_warp_concat[:, 0], coords_warp_concat[:, 1],
                    c=colors_warp, s=10, alpha=0.5, label='Warped 1')
    axes[1].set_title('Overlay of Warped Moving Slices')

    # ── common limits (if sharex/sharey is problematic) ────
    all_x = np.concatenate([coords_raw_concat[:, 0],  coords_warp_concat[:, 0]])
    all_y = np.concatenate([coords_raw_concat[:, 1],  coords_warp_concat[:, 1]])
    axes[0].set_xlim(all_x.min(), all_x.max())
    axes[0].set_ylim(all_y.min(), all_y.max())
    for ax in axes[1:]:
        ax.set_xlim(axes[0].get_xlim())
        ax.set_ylim(axes[0].get_ylim())

    # ── equal aspect WITHOUT changing the limits ───────────
    for ax in axes:
        ax.set_aspect('equal', adjustable='datalim')

    # invert once (shared limits mean both will flip)
    axes[0].invert_yaxis()

    plt.tight_layout()
    plt.show()


root_folder = "/media/huifang/data/registration/result/center_align/scc/"
for i in [5,9,10]:
    registration_paths = [
        f"{root_folder}/{i}_0_result.npz",
        # f"{root_folder}/{i}_1_result.npz"
    ]
    visualize_coordinate_overlay(registration_paths)
    # coords_fix, _, _,label = load_registration_result(registration_paths[0])
    # coords_moving_list = []
    # coords_warped_list = []
    #
    # for reg_path in registration_paths:
    #     _, coords_warp, coords_moving,_ = load_registration_result(reg_path)
    #     coords_moving_list.append(coords_moving)
    #     coords_warped_list.append(coords_warp)
    #
    # # Visualize
    # visualize_coordinate_overlay(coords_fix, coords_moving_list, coords_warped_list)

    # visualize_gene_overlay(fix_adata_path, moving_adata_paths, registration_paths, gene='AGRN')
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

