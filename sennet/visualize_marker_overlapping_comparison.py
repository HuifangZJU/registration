import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.stats import binned_statistic_2d
import seaborn as sns
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D

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
def plot_gridwise_scatter(coord1, expr1, coord2, expr2, bins=50, gene1='CDKN1A', gene2='p16',save_root=None,vis=True):
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
    if vis:
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
        if save_root:
            plt.savefig(save_root, dpi=300)
            plt.close()
        else:
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

def plot_pvalues_by_group(p_values, use_neglog10=True, jitter=0.15):
    p1 = np.asarray(p_values['aligned'], dtype=float)
    p2 = np.asarray(p_values['aligned_combined'], dtype=float)

    x1 = np.arange(len(p1))
    x2 = np.arange(len(p2)) + jitter  # small shift so the groups don't overlap perfectly

    if use_neglog10:
        y1 = -np.log10(p1 + 1e-300)   # safe for zeros
        y2 = -np.log10(p2 + 1e-300)
        y_label = "-log10(p-value)"
    else:
        y1, y2 = p1, p2
        y_label = "p-value"

    plt.figure(figsize=(10, 5))
    plt.scatter(x1, y1, s=42, alpha=0.9, color='#1f77b4', label='aligned')
    plt.scatter(x2, y2, s=42, alpha=0.9, color='#ff7f0e', label='aligned_combined')

    # optional connecting lines (comment out if not desired)
    # plt.plot(x1, y1, color='#1f77b4', alpha=0.6, linewidth=1)
    # plt.plot(x2, y2, color='#ff7f0e', alpha=0.6, linewidth=1)

    plt.xlabel("Pair Index")
    plt.ylabel(y_label)
    plt.title(f"{y_label} across sennet_pairs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = defaultdict(list)
p_values = defaultdict(list)

for key in ['aligned_combined','aligned']:
    for i in range(0,len(sennet_pairs)):
        print(i)
        if i==8 or i==25 or i==26:
            continue
        line = sennet_pairs[i]
        xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
        xenium_gene_data = sc.read_h5ad(
            "/media/huifang/data/sennet/combined_registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}.h5ad")

        codex_gene_data = sc.read_h5ad(
            "/media/huifang/data/sennet/combined_registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}.h5ad")


        xenium_coordinate = np.stack([
            xenium_gene_data.obs['x_trans'].values,
            xenium_gene_data.obs['y_trans'].values
        ], axis=1)
        codex_coordinate = np.stack([
            codex_gene_data.obs['x_trans'].values,
            codex_gene_data.obs['y_trans'].values
        ], axis=1)

        warped_xenium_coor = np.stack([
            xenium_gene_data.obs[f'x_{key}'].values,
            xenium_gene_data.obs[f'y_{key}'].values
        ], axis=1)

        warped_codex_coor = np.stack([
            codex_gene_data.obs[f'x_{key}'].values,
            codex_gene_data.obs[f'y_{key}'].values
        ], axis=1)

        xenium_gene_expr = xenium_gene_data[:, 'CDKN1A'].to_df()['CDKN1A']
        # CODEX: use p16 intensity from .obs
        codex_gene_expr = codex_gene_data.obs['p16'].values
        codex_gene_expr = codex_gene_expr - codex_gene_expr.min()

        xenium_gene_expr = np.log1p(xenium_gene_expr)
        codex_gene_expr = np.log1p(codex_gene_expr)

        # visualize_marker_overlay()

        # Call the function on the mock data
        r,p = plot_gridwise_scatter(
            warped_xenium_coor, xenium_gene_expr,
            warped_codex_coor, codex_gene_expr,
            bins=20, gene1='CDKN1A', gene2='p16',save_root=None,vis=False
        )
        r_values[key].append(r)
        p_values[key].append(p)

        # r,p = compute_and_plot_spatial_correlation(
        # warped_xenium_coor, xenium_gene_expr,
        # warped_codex_coor, codex_gene_expr,
        # bins=50, gene1='CDKN1A (Xenium)', gene2='p16 (CODEX)'
        # )
# Put data into a DataFrame

# plot_pvalues_by_group(p_values)

# def _build_df(r_list, p_list):
#     df = pd.DataFrame({'Pearson_r': r_list, 'p_value': p_list})
#     df['log10_p'] = np.log10(df['p_value'] + 1e-10)  # Avoid log(0)
#     df['index'] = np.arange(len(df))
#     return df
#
# df_aligned = _build_df(r_values['aligned'],           p_values['aligned'])
# df_comb   = _build_df(r_values['aligned_combined'],   p_values['aligned_combined'])
#
# # Colors (one per group)
# color_aligned = '#1f77b4'         # blue
# color_comb    = '#ff7f0e'         # orange
#
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# plt.rcParams['font.size'] = 16
# # --- Pearson r ---
# sns.regplot(data=df_aligned, x='index', y='Pearson_r', ax=axs[0],
#             scatter_kws={'s': 40, 'alpha': 0.9, 'color': color_aligned},
#             line_kws={'color': color_aligned})
# sns.regplot(data=df_comb, x='index', y='Pearson_r', ax=axs[0],
#             scatter_kws={'s': 40, 'alpha': 0.9, 'color': color_comb},
#             line_kws={'color': color_comb})
#
# axs[0].set_title("Pearson r across sennet pairs",fontsize=14)
# axs[0].set_xlabel("Pair Index")
# axs[0].set_ylabel("Pearson r")
# axs[0].grid(True)
#
# # --- log10(p) ---
# sns.regplot(data=df_aligned, x='index', y='log10_p', ax=axs[1],
#             scatter_kws={'s': 40, 'alpha': 0.9, 'color': color_aligned},
#             line_kws={'color': color_aligned})
# sns.regplot(data=df_comb, x='index', y='log10_p', ax=axs[1],
#             scatter_kws={'s': 40, 'alpha': 0.9, 'color': color_comb},
#             line_kws={'color': color_comb})
#
# axs[1].set_title("log10(p-value) across sennet pairs",fontsize=14)
# axs[1].set_xlabel("Pair Index")
# axs[1].set_ylabel("log10(p-value)")
# axs[1].grid(True)
#
# # Single shared legend
# legend_lines = [
#     Line2D([0], [0], color=color_aligned, marker='o', linestyle='-', label='DAPI only'),
#     Line2D([0], [0], color=color_comb,    marker='o', linestyle='-', label='DAPI + Gene & Protein'),
# ]
# fig.legend(handles=legend_lines, loc='upper center', ncol=2)
#
# plt.tight_layout(rect=[0, 0, 1, 0.94])
# plt.show()

def _build_df(r_list, p_list):
    df = pd.DataFrame({'Pearson_r': r_list, 'p_value': p_list})
    # df['log10_p'] = np.log10(df['p_value'] + 1e-10)  # avoid log(0)
    eps = 1e-300  # near float min; avoids underflow to -inf
    df['log10_p'] = -np.log10(np.clip(df['p_value'].astype(float), eps, 1.0))
    return df

# Build per-group DataFrames
df_aligned = _build_df(r_values['aligned'],           p_values['aligned'])
df_comb    = _build_df(r_values['aligned_combined'],  p_values['aligned_combined'])

# Long-form for seaborn
def to_long(df, group_name, metric):
    return pd.DataFrame({
        'Group': group_name,
        'Value': df[metric].values
    })

df_r_long = pd.concat([
    to_long(df_aligned, 'DAPI only', 'Pearson_r'),
    to_long(df_comb,    'DAPI + Gene & Protein', 'Pearson_r')
], ignore_index=True)

df_p_long = pd.concat([
    to_long(df_aligned, 'DAPI only', 'log10_p'),
    to_long(df_comb,    'DAPI + Gene & Protein', 'log10_p')
], ignore_index=True)

# Colors (one per group)
color_aligned = '#1f77b4'         # blue
color_comb    = '#ff7f0e'         # orange
palette = {'DAPI only': color_aligned, 'DAPI + Gene & Protein': color_comb}

# Plot
plt.rcParams['font.size'] = 16
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# --- Pearson r ---
sns.boxplot(data=df_r_long, x='Group', y='Value', ax=axs[0], palette=palette, width=0.6)
sns.stripplot(data=df_r_long, x='Group', y='Value', ax=axs[0], color='k', alpha=0.4, jitter=0.15, size=6)
axs[0].set_title("Pearson r across sennet pairs", fontsize=14)
axs[0].set_xlabel("")
axs[0].set_ylabel("Pearson r")
axs[0].grid(True, axis='y')

# --- log10(p) ---
sns.boxplot(data=df_p_long, x='Group', y='Value', ax=axs[1], palette=palette, width=0.6)
sns.stripplot(data=df_p_long, x='Group', y='Value', ax=axs[1], color='k', alpha=0.4, jitter=0.15, size=6)
axs[1].set_title("-log10(p-value) across sennet pairs", fontsize=14)
axs[1].set_xlabel("")
axs[1].set_ylabel("log10(p-value)")
axs[1].grid(True, axis='y')

plt.tight_layout()
plt.show()