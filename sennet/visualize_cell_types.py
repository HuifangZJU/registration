import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
# STEP 1: Define your mapping
codex_to_xenium_map = {
    'B': 'Memory B cell',
    'CD4+ T': 'CD4+ T cell',
    'CD8+ T': 'CD8+ T cell',
    'Mast': 'Mast cell',
    'TA': 'TA cell',
    'Goblet': 'Goblet cell',
    'Stromal': 'Stromal cell',
    'Plasma': 'IgA plasma cell',
    'Epithelial': 'Colonocyte',
    'CD66+ epithelial': 'Colonocyte',
    'Neuroendocrine': 'EEC',
    'M2 macrophage': 'Macrophage',
    'Fibroblast': 'Stromal cell',
    'Cycling TA': 'TA cell',  # or 'Stem cell'
    'HLADR+ myeloid': 'Macrophage',
    'CD11b+ myeloid': 'Macrophage',
    'Treg': 'CD4+ T cell',
    'Smooth muscle': 'Smooth muscle'
}

def visualize_cell_types():
    # Load CODEX and Xenium types
    xenium_all = pd.read_csv('/media/huifang/data/sennet/xenium/ct.csv')
    xenium_types = set(xenium_all['ct'].dropna().unique())

    codex_all = sc.read_h5ad('/media/huifang/data/sennet/codex/20250603_sennet_annotated_updated_donormeta.h5ad')
    codex_types = set(codex_all.obs['cell_type_update'].dropna().unique())

    # Paired entries: only if both sides exist in the data
    paired = []
    for idx, (codex, xenium) in enumerate(codex_to_xenium_map.items(), start=1):
        if codex in codex_types and xenium in xenium_types:
            paired.append((idx, codex, xenium))

    # Build dataframe
    paired_rows = []
    for idx, codex, xenium in paired:
        label = f"{idx}. {codex} ↔ {xenium}"
        paired_rows.append({
            'Label': label,
            'In CODEX': 1,
            'In Xenium': 1
        })

    # Unpaired
    paired_codex = set([c for _, c, _ in paired])
    paired_xenium = set([x for _, _, x in paired])

    unpaired_codex = codex_types - paired_codex
    unpaired_xenium = xenium_types - paired_xenium

    # Add unpaired CODEX
    for c in sorted(unpaired_codex):
        paired_rows.append({
            'Label': f"– {c} (CODEX only)",
            'In CODEX': 1,
            'In Xenium': 0
        })

    # Add unpaired Xenium
    for x in sorted(unpaired_xenium):
        paired_rows.append({
            'Label': f"– {x} (Xenium only)",
            'In CODEX': 0,
            'In Xenium': 1
        })

    # Create DataFrame
    df = pd.DataFrame(paired_rows)
    df.set_index('Label', inplace=True)

    # Plot
    plt.figure(figsize=(10, 0.4 * len(df)))
    sns.heatmap(df[['In CODEX', 'In Xenium']], annot=True, cbar=False, cmap='Blues', linewidths=0.5, linecolor='gray')
    plt.title('CODEX ↔ Xenium Cell Type Mapping (with Paired Index)')
    plt.xticks(rotation=45)
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

def show_cell_distribution_all():
    xenium_to_canonical = {v: f"{i + 1}. {v}" for i, v in enumerate(sorted(set(codex_to_xenium_map.values())))}
    codex_to_canonical = {k: xenium_to_canonical[v] for k, v in codex_to_xenium_map.items() if v in xenium_to_canonical}

    # === Assign canonical labels to both datasets ===
    codex_gene_data.obs['type_label'] = codex_gene_data.obs['cell_type_update'].map(codex_to_canonical)
    xenium_gene_data.obs['type_label'] = xenium_gene_data.obs['cell_type'].map(xenium_to_canonical)

    # Fallback to original type if no match
    codex_gene_data.obs['type_label'] = codex_gene_data.obs['type_label'].fillna(
        codex_gene_data.obs['cell_type_update'].astype(str) + " (CODEX-only)"
    )
    xenium_gene_data.obs['type_label'] = xenium_gene_data.obs['type_label'].fillna(
        xenium_gene_data.obs['cell_type'].astype(str) + " (Xenium-only)"
    )

    # === Combine coordinates and labels ===
    codex_df = codex_gene_data.obs[['x_aligned', 'y_aligned', 'type_label']].copy()
    xenium_df = xenium_gene_data.obs[['x_aligned', 'y_aligned', 'type_label']].copy()
    codex_df['modality'] = 'CODEX'
    xenium_df['modality'] = 'Xenium'

    combined_df = pd.concat([codex_df, xenium_df], ignore_index=True)

    # === Color palette for all unique labels ===
    unique_types = sorted(combined_df['type_label'].unique())
    palette = sns.color_palette("hls", len(unique_types))
    color_map = dict(zip(unique_types, palette))

    # Enable constrained layout
    fig, axs = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)

    for i, modality in enumerate(['CODEX', 'Xenium']):
        subset = combined_df[combined_df['modality'] == modality]
        axs[i].scatter(subset['x_aligned'], subset['y_aligned'],
                       c=subset['type_label'].map(color_map), s=8, alpha=0.7)
        axs[i].invert_yaxis()
        axs[i].set_title(f'{modality} Cell Type Distribution')
        axs[i].set_aspect('equal', adjustable='box')

    # Create legend handles
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=color_map[label], markersize=6)
        for label in unique_types
    ]

    # Add legend below
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, title='Cell Types', frameon=False)

    plt.show()

def visualize_paired_celltype_distribution():
    xenium_to_canonical = {v: f"{i + 1}. {v}" for i, v in enumerate(sorted(set(codex_to_xenium_map.values())))}
    codex_to_canonical = {k: xenium_to_canonical[v] for k, v in codex_to_xenium_map.items() if v in xenium_to_canonical}

    # Annotate both datasets
    codex_gene_data.obs['type_label'] = codex_gene_data.obs['cell_type_update'].map(codex_to_canonical)
    xenium_gene_data.obs['type_label'] = xenium_gene_data.obs['cell_type'].map(xenium_to_canonical)

    # Filter: only keep paired types
    codex_paired = codex_gene_data.obs.dropna(subset=['type_label'])
    xenium_paired = xenium_gene_data.obs.dropna(subset=['type_label'])

    # Merge coordinates and modality tag
    codex_df = codex_paired[['x_aligned', 'y_aligned', 'type_label']].copy()
    xenium_df = xenium_paired[['x_aligned', 'y_aligned', 'type_label']].copy()
    codex_df['modality'] = 'CODEX'
    xenium_df['modality'] = 'Xenium'
    combined_df = pd.concat([codex_df, xenium_df], ignore_index=True)

    # Create a consistent color map for paired types
    paired_types = sorted(combined_df['type_label'].unique())
    palette = sns.color_palette("hls", len(paired_types))
    color_map = dict(zip(paired_types, palette))

    # Create three subplots: CODEX, Xenium, Overlay
    fig, axs = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=True)

    for i, modality in enumerate(['CODEX', 'Xenium']):
        subset = combined_df[combined_df['modality'] == modality]
        axs[i].scatter(subset['x_aligned'], subset['y_aligned'],
                       c=subset['type_label'].map(color_map), s=8, alpha=0.7)
        axs[i].invert_yaxis()
        axs[i].set_title(f'{modality} Cell Type Distribution')
        axs[i].set_aspect('equal', adjustable='box')

    # === Overlay plot ===
    codex_subset = combined_df[combined_df['modality'] == 'CODEX']
    xenium_subset = combined_df[combined_df['modality'] == 'Xenium']

    # CODEX: filled circle, Xenium: triangle
    axs[2].scatter(codex_subset['x_aligned'], codex_subset['y_aligned'],
                   c=codex_subset['type_label'].map(color_map), s=8, alpha=0.6, marker='o', label='CODEX')
    axs[2].scatter(xenium_subset['x_aligned'], xenium_subset['y_aligned'],
                   c=xenium_subset['type_label'].map(color_map), s=8, alpha=0.6, marker='^', label='Xenium')

    axs[2].invert_yaxis()
    axs[2].set_title('Overlay of CODEX & Xenium')
    axs[2].set_aspect('equal', adjustable='box')


    # Add legend below
    handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color_map[label], markersize=12)  # ← control size here
        for label in paired_types
    ]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.1),
               ncol=4, title='Paired Cell Types', frameon=False, fontsize=16)

    plt.show()


# Heatmap-based correlation function
def spatial_density(df, cell_type, x_col='x_aligned', y_col='y_aligned', bins=50, range_xy=None):
    subset = df[df['type_label'] == cell_type]
    if range_xy is None:
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        range_xy = [[x_min, x_max], [y_min, y_max]]
    heatmap, xedges, yedges = np.histogram2d(subset[x_col], subset[y_col], bins=bins, range=range_xy)
    return heatmap

def show_spatial_pearson():


    # Plot heatmaps side by side with correlation values
    fig, axes = plt.subplots(len(paired_labels), 2, figsize=(10, 2 * len(paired_labels)))
    for i, (label, codex_heat, xenium_heat) in enumerate(heatmaps):
        sns.heatmap(codex_heat, ax=axes[i, 0], cbar=False)
        axes[i, 0].set_title(f'{label} - CODEX')

        sns.heatmap(xenium_heat, ax=axes[i, 1], cbar=False)
        r = results_df.iloc[i]["pearson_r"]
        axes[i, 1].set_title(f'{label} - Xenium\nr={r:.2f}' if not np.isnan(r) else f'{label} - Xenium\nr=N/A')

    plt.tight_layout()
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Spatial Correlation Results", dataframe=results_df)
    plt.show()


# visualize_cell_types()
file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()

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
        "/media/huifang/data/sennet/registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_registered_with_ct.h5ad")
    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}_registered.h5ad")
    # visualize_paired_celltype_distribution()
    # Drop NaNs and convert to sets
    # codex_types = set(codex_gene_data.obs['cell_type_update'].dropna().unique())
    # xenium_types = set(xenium_gene_data.obs['cell_type'].dropna().unique())
    # === Invert mapping: xenium_type → canonical_label ===
    xenium_to_canonical = {v: f"{i + 1}. {v}" for i, v in enumerate(sorted(set(codex_to_xenium_map.values())))}
    codex_to_canonical = {k: xenium_to_canonical[v] for k, v in codex_to_xenium_map.items() if v in xenium_to_canonical}

    # Annotate both datasets
    codex_gene_data.obs['type_label'] = codex_gene_data.obs['cell_type_update'].map(codex_to_canonical)
    xenium_gene_data.obs['type_label'] = xenium_gene_data.obs['cell_type'].map(xenium_to_canonical)
    # visualize_paired_celltype_distribution()
    # Filter: only keep paired types
    codex_paired = codex_gene_data.obs.dropna(subset=['type_label'])
    xenium_paired = xenium_gene_data.obs.dropna(subset=['type_label'])
    # View the first few rows of .var

    # Merge coordinates and modality tag
    codex_df = codex_paired[['x_aligned', 'y_aligned', 'type_label']].copy()
    xenium_df = xenium_paired[['x_aligned', 'y_aligned', 'type_label']].copy()

    # Get paired labels from the current data
    paired_labels = sorted(set(codex_df['type_label']) & set(xenium_df['type_label']))
    # Establish shared spatial range
    x_range = [min(codex_df['x_aligned'].min(), xenium_df['x_aligned'].min()),
               max(codex_df['x_aligned'].max(), xenium_df['x_aligned'].max())]
    y_range = [min(codex_df['y_aligned'].min(), xenium_df['y_aligned'].min()),
               max(codex_df['y_aligned'].max(), xenium_df['y_aligned'].max())]
    range_xy = [x_range, y_range]

    # Compute correlations
    results = []
    heatmaps = []
    bin_value=50
    for label in paired_labels:
        codex_heat = spatial_density(codex_df, label, bins=bin_value, range_xy=range_xy)
        xenium_heat = spatial_density(xenium_df, label, bins=bin_value, range_xy=range_xy)

        mask = (codex_heat + xenium_heat) > 0
        if np.sum(mask) == 0:
            r = np.nan
            p = np.nan
        else:
            r, p = pearsonr(codex_heat[mask].flatten(), xenium_heat[mask].flatten())

        results.append({
            'type_label': label,
            'pearson_r': r,
            'p_value': p
        })
        heatmaps.append((label, codex_heat, xenium_heat))

    results_df = pd.DataFrame(results)

    results_sorted = results_df.sort_values(by='pearson_r', ascending=False).reset_index(drop=True)
    # # Plot barplot
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=results_sorted, x='pearson_r', y='type_label', palette='viridis')
    # plt.xlabel('Spatial Correlation (Pearson r)')
    # plt.ylabel('Cell Type')
    # plt.title('Spatial Correlation of Paired Cell Types (CODEX vs Xenium)')
    # plt.xlim(-1, 1)
    # plt.grid(axis='x', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    # Use existing overlay data
    codex_overlay = codex_df[['x_aligned', 'y_aligned', 'type_label']].copy()
    xenium_overlay = xenium_df[['x_aligned', 'y_aligned', 'type_label']].copy()
    codex_overlay['modality'] = 'CODEX'
    xenium_overlay['modality'] = 'Xenium'
    overlay_df = pd.concat([codex_overlay, xenium_overlay], ignore_index=True)

    # Determine shared spatial bounds
    x_min, x_max = overlay_df['x_aligned'].min(), overlay_df['x_aligned'].max()
    y_min, y_max = overlay_df['y_aligned'].min(), overlay_df['y_aligned'].max()

    # Grid setup (bins = 10 for this example)
    bins = bin_value
    x_grid = np.linspace(x_min, x_max, bins + 1)
    y_grid = np.linspace(y_min, y_max, bins + 1)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})

    # --- Overlay with grid ---
    ax0 = axes[0]
    for modality, marker in zip(['CODEX', 'Xenium'], ['o', '^']):
        subset = overlay_df[overlay_df['modality'] == modality]
        ax0.scatter(subset['x_aligned'], subset['y_aligned'],
                    c='gray', s=5, alpha=0.5, marker=marker, label=modality)

    # Draw grid
    for x in x_grid:
        ax0.axvline(x, color='red', linestyle='--', linewidth=0.5)
    for y in y_grid:
        ax0.axhline(y, color='red', linestyle='--', linewidth=0.5)

    ax0.set_aspect('equal', adjustable='box')
    ax0.invert_yaxis()
    ax0.set_title(f'Overlay of CODEX and Xenium\nGrid bins = {bins}x{bins}')
    ax0.legend()

    # --- Correlation bar plot ---
    results_sorted = results_df.sort_values(by='pearson_r', ascending=False).reset_index(drop=True)
    sns.barplot(data=results_sorted, x='pearson_r', y='type_label', palette='viridis', ax=axes[1])
    axes[1].set_xlabel('Spatial Correlation (Pearson r)')
    axes[1].set_ylabel('Cell Type',fontsize=14)
    axes[1].set_title('Spatial Correlation of Paired Cell Types')
    axes[1].set_xlim(-1, 1)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.show()