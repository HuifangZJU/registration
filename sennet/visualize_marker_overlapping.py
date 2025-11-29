import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_marker_overlay():
    f, a = plt.subplots(1, 2, figsize=(16, 8))
    # Normalize gene expression to [0, 1]
    norm_xenium = (xenium_gene_expr - xenium_gene_expr.min()) / (xenium_gene_expr.max() - xenium_gene_expr.min())
    norm_codex = (codex_gene_expr - codex_gene_expr.min()) / (codex_gene_expr.max() - codex_gene_expr.min())

    # Convert cmap to RGBA with alpha = normalized expression
    cmap_red = cm.get_cmap('Reds')
    # cmap_red = plt.colormaps['Reds']
    colors_xenium = cmap_red(norm_xenium)
    colors_xenium[:, -1] = norm_xenium  # set alpha channel

    cmap_blue = cm.get_cmap('Blues')
    # cmap_blue = plt.colormaps['Blues']
    colors_codex = cmap_blue(norm_codex)
    colors_codex[:, -1] = norm_codex  # set alpha channel

    a[0].scatter(xenium_coordinate[:, 0], xenium_coordinate[:, 1],
                 color=colors_xenium, s=10, label='Xenium CDKN1A')
    a[0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1],
                 color=colors_codex, s=10, label='CODEX p16')

    a[0].set_title("Unregistered Marker Levels", fontsize=22)

    # Scatter with RGBA color array
    a[1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                 color=colors_xenium, s=10, label='Xenium CDKN1A')

    a[1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                 color=colors_codex, s=10, label='CODEX p16')

    a[1].set_title("Registered CODEX to Xenium with Marker Levels", fontsize=22)

    # Create normalization
    norm_xenium = mcolors.Normalize(vmin=xenium_gene_expr.min(), vmax=xenium_gene_expr.max())
    norm_codex = mcolors.Normalize(vmin=codex_gene_expr.min(), vmax=codex_gene_expr.max())

    # Create ScalarMappables (used for colorbars)
    sm_xenium = cm.ScalarMappable(cmap='Reds', norm=norm_xenium)
    sm_xenium.set_array([])  # required for colorbar

    sm_codex = cm.ScalarMappable(cmap='Blues', norm=norm_codex)
    sm_codex.set_array([])

    # Add colorbars to the subplot
    # Colorbar 1 (e.g., CDKN1A)
    cbar_ax1 = f.add_axes([0.91, 0.30, 0.01, 0.3])  # [left, bottom, width, height]
    cbar = f.colorbar(sm_xenium, cax=cbar_ax1)
    cbar.set_label('CDKN1A', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    # Colorbar 2 (e.g., p16) placed lower
    cbar_ax2 = f.add_axes([0.95, 0.30, 0.01, 0.3])
    cbar = f.colorbar(sm_codex, cax=cbar_ax2)
    cbar.set_label('p16', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    for ax in a.flat:
        ax.set_aspect('equal')
    plt.show()


file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = []
p_values = []


for i in range(0,len(sennet_pairs)):
    # bad ccases
    if i == 8 or i == 25 or i == 26:
        continue
    print(i)

    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/combined_registered_data_1024/xenium" + f"_{xenium_sampleid}_{xenium_regionid}.h5ad")

    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/combined_registered_data_1024/codex" + f"_{codex_sampleid}_{codex_regionid}.h5ad")


    gene_key = 'CDKN1A'
    protein_key = 'p16'
    coor_key = 'aligned'
    # coor_key = "aligned_combined"

    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_trans'].values,
        xenium_gene_data.obs['y_trans'].values
    ], axis=1)
    codex_coordinate = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    warped_xenium_coor = np.stack([
        xenium_gene_data.obs[f'x_{coor_key}'].values,
        xenium_gene_data.obs[f'y_{coor_key}'].values
    ], axis=1)

    warped_codex_coor= np.stack([
        codex_gene_data.obs[f'x_{coor_key}'].values,
        codex_gene_data.obs[f'y_{coor_key}'].values
    ], axis=1)

    xenium_gene_expr = xenium_gene_data[:, gene_key].to_df()[gene_key]

    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs[protein_key].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()

    xenium_gene_expr = np.log1p(xenium_gene_expr)
    codex_gene_expr = np.log1p(codex_gene_expr)

    visualize_marker_overlay()