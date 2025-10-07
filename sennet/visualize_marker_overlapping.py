import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.stats import binned_statistic_2d
import seaborn as sns
import pandas as pd

def compute_and_plot_spatial_correlation(coord1, expr1, coord2, expr2, bins=50, gene1='Gene1', gene2='Gene2'):
    # Determine shared spatial bounds
    y_min = min(coord1[:, 0].min(), coord2[:, 0].min())
    y_max = max(coord1[:, 0].max(), coord2[:, 0].max())
    x_min = min(coord1[:, 1].min(), coord2[:, 1].min())
    x_max = max(coord1[:, 1].max(), coord2[:, 1].max())
    range_xy = [[x_min, x_max], [y_min, y_max]]

    # Compute spatial density maps
    stat1, _, _, _ = binned_statistic_2d(
        coord1[:, 1], coord1[:, 0], values=expr1,
        statistic='mean', bins=bins, range=range_xy
    )
    stat2, _, _, _ = binned_statistic_2d(
        coord2[:, 1], coord2[:, 0], values=expr2,
        statistic='mean', bins=bins, range=range_xy
    )

    # Compute Pearson correlation
    mask = ~np.isnan(stat1) & ~np.isnan(stat2)
    if np.any(mask):
        r, p = pearsonr(stat1[mask].flatten(), stat2[mask].flatten())
    else:
        r, p = np.nan, np.nan

    # Plot the spatial maps
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    for ax, data, title in zip(
            axs,
            [stat1, stat2, stat1 - stat2],
            [f'{gene1} Spatial Density',
             f'{gene2} Spatial Density',
             f'Difference Map\nPearson r = {r:.2f}, p = {p:.2g}']
    ):
        sns.heatmap(data, ax=ax, cmap='viridis' if 'Difference' not in title else 'coolwarm',
                    center=0 if 'Difference' in title else None, cbar=True,
                    square=False)
        ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    return r, p

# Define the plotting function
def plot_gridwise_scatter(coord1, expr1, coord2, expr2, bins=50, gene1='CDKN1A', gene2='p16'):
    # Determine shared spatial bounds
    x_min = min(coord1[:, 0].min(), coord2[:, 0].min())
    x_max = max(coord1[:, 0].max(), coord2[:, 0].max())
    y_min = min(coord1[:, 1].min(), coord2[:, 1].min())
    y_max = max(coord1[:, 1].max(), coord2[:, 1].max())
    range_xy = [[x_min, x_max], [y_min, y_max]]

    # Compute spatial density maps
    stat1, _, _, _ = binned_statistic_2d(
        coord1[:, 0], coord1[:, 1], values=expr1,
        statistic='mean', bins=bins, range=range_xy
    )
    stat2, _, _, _ = binned_statistic_2d(
        coord2[:, 0], coord2[:, 1], values=expr2,
        statistic='mean', bins=bins, range=range_xy
    )

    # Mask valid grid points
    mask = ~np.isnan(stat1) & ~np.isnan(stat2)
    x_vals = stat1[mask].flatten()
    y_vals = stat2[mask].flatten()

    # Scatter plot

    r, p = pearsonr(x_vals, y_vals)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7)
    plt.xlabel(f'{gene1} (Xenium)')
    plt.ylabel(f'{gene2} (CODEX)')
    plt.title(f'Grid-wise Correlation\nPearson r = {r:.2f}, p = {p:.2g}')
    plt.grid(True)

    # Add diagonal reference line
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    return r,p
def visualize_marker_overlay():
    f, a = plt.subplots(1, 2, figsize=(16, 8))
    # # # Plot original coordinates with gene marker color
    # a[0,0].scatter(xenium_coordinate[:,0],xenium_coordinate[:,1], s=5)
    # a[0,0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1], s=5)
    # a[0, 0].set_title("Aligned cell distributions",fontsize=22)
    # a[0,1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1], s=5)
    # a[0,1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1], s=5)
    # a[0, 1].set_title("Registered cell distributions",fontsize=22)

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


file_path = '/media/huifang/data1/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = []
p_values = []


for i in range(0,len(sennet_pairs)):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')



    # Xenium: use CDKN1A expression from .X

    # xenium_image = plt.imread(
    #     '/media/huifang/data/sennet/hf_aligned_data/xenium' + f"_{xenium_sampleid}_{xenium_regionid}.png")
    # codex_image = plt.imread(
    #     '/media/huifang/data/sennet/hf_aligned_data/codex' + f"_{codex_sampleid}_{codex_regionid}.png")
    #
    # warped_xenium_image = plt.imread(
    #     '/media/huifang/data/sennet/registered_data/xenium' + f"_{xenium_sampleid}_{xenium_regionid}_registered.png")
    # warped_codex_image = plt.imread(
    #     '/media/huifang/data/sennet/registered_data/codex' + f"_{codex_sampleid}_{codex_regionid}_registered.png")
    #
    # plt.imshow(create_overlay(warped_xenium_image,warped_codex_image))
    # plt.gca().invert_yaxis()
    # plt.show()

    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data1/sennet/registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_registered.h5ad")

    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data1/sennet/registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}_registered.h5ad")


    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_trans'].values,
        xenium_gene_data.obs['y_trans'].values
    ], axis=1)
    codex_coordinate = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    warped_xenium_coor = np.stack([
        xenium_gene_data.obs['x_aligned'].values,
        xenium_gene_data.obs['y_aligned'].values
    ], axis=1)

    warped_codex_coor= np.stack([
        codex_gene_data.obs['x_aligned'].values,
        codex_gene_data.obs['y_aligned'].values
    ], axis=1)

    xenium_gene_expr = xenium_gene_data[:, 'CDKN1A'].to_df()['CDKN1A']

    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs['p16'].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()

    xenium_gene_expr = np.log1p(xenium_gene_expr)
    codex_gene_expr = np.log1p(codex_gene_expr)

    visualize_marker_overlay()

    # Call the function on the mock data
    r,p = plot_gridwise_scatter(
        warped_xenium_coor, xenium_gene_expr,
        warped_codex_coor, codex_gene_expr,
        bins=20, gene1='CDKN1A', gene2='p16'
    )
    r_values.append(r)
    p_values.append(p)
    # r,p = compute_and_plot_spatial_correlation(
    # warped_xenium_coor, xenium_gene_expr,
    # warped_codex_coor, codex_gene_expr,
    # bins=50, gene1='CDKN1A (Xenium)', gene2='p16 (CODEX)'
    # )
# Put data into a DataFrame
df = pd.DataFrame({'Pearson_r': r_values, 'p_value': p_values})
df['log10_p'] = np.log10(df['p_value'] + 1e-10)  # Avoid log(0)
# Add index to the DataFrame
df['index'] = df.index
# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Then plot using 'index' column
sns.regplot(data=df, x='index', y='Pearson_r', ax=axs[0],
            scatter_kws={'s': 40}, line_kws={'color': 'red'})
axs[0].set_title("Pearson r across sennet_pairs")
axs[0].set_xlabel("Pair Index")
axs[0].set_ylabel("Pearson r")
axs[0].grid(True)


sns.regplot(data=df, x='index', y='log10_p', ax=axs[1],
            scatter_kws={'s': 40}, line_kws={'color': 'blue'})
axs[1].set_title("log10(p-value) across sennet_pairs")
axs[1].set_xlabel("Pair Index")
axs[1].set_ylabel("log10(p-value)")
axs[1].grid(True)

plt.tight_layout()
plt.show()