import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.colors as mcolors

def load_image_gene_coords(base_path, gene_name, datatype):
    img = cv2.imread(base_path + ".png", cv2.IMREAD_GRAYSCALE)
    adata = sc.read_h5ad(base_path + ".h5ad")
    coords = adata.obs[['x_trans', 'y_trans']].values

    if datatype == 'xenium':
        if gene_name not in adata.var_names:
            raise ValueError(f"Gene {gene_name} not found in {base_path}.h5ad")
        gene_idx = adata.var_names.get_loc(gene_name)
        expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx].flatten()
    elif datatype == 'codex':
        if gene_name not in adata.obs.columns:
            raise ValueError(f"Protein marker {gene_name} not found in .obs of {base_path}.h5ad")
        expr = adata.obs[gene_name].values
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    return img, coords, expr


def plot_image_with_gene_overlay(ax, img, coords, expr, title, cmap='magma', log_scale=False):
    """
    Plot grayscale image with gene/protein expression overlay.
    Low-expression cells show with brighter colors by percentile clipping.
    """
    ax.imshow(img, cmap='gray')

    if log_scale:
        expr = np.log1p(expr)

    # Compute normalization range (exclude extreme lows and highs)
    vmin = np.percentile(expr, 1)
    vmax = np.percentile(expr, 99)

    # Prevent vmin==vmax issues
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-3

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=expr, cmap=cmap, norm=norm,
        s=10, edgecolor='black', linewidth=0.1, alpha=0.9
    )
    ax.set_title(title)
    ax.axis('off')
    return sc

def visualize_gene_expression_pairs(pair_file_path, output_dir, start_line=0):
    df = pd.read_csv(pair_file_path, sep=None, engine='python', header=None)

    for i, row in enumerate(df.iloc[start_line:].itertuples(index=False, name=None)):
        xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = row

        xenium_base = f"xenium_{xenium_sampleid}_{str(xenium_regionid)}"
        codex_base = f"codex_{codex_sampleid}_{str(codex_regionid)}"

        xenium_path = os.path.join(output_dir, xenium_base)
        codex_path = os.path.join(output_dir, codex_base)

        try:
            xenium_img, xenium_coords, xenium_expr_raw = load_image_gene_coords(xenium_path, "CDKN1A", "xenium")
            codex_img, codex_coords, codex_expr = load_image_gene_coords(codex_path, "p16", "codex")
        except Exception as e:
            print(f"Skipping pair {xenium_base} <-> {codex_base} due to error: {e}")
            continue
        xenium_expr = np.log1p(xenium_expr_raw+1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sc0 = plot_image_with_gene_overlay(
            axes[0], xenium_img, xenium_coords, xenium_expr_raw,
            title=f"Xenium - CDKN1A (log1p)\n{xenium_sampleid} Region {xenium_regionid}",
            cmap='plasma', log_scale=True
        )

        sc1 = plot_image_with_gene_overlay(
            axes[1], codex_img, codex_coords, codex_expr,
            title=f"Codex - p16\n{codex_sampleid} Region {codex_regionid}",
            cmap='plasma', log_scale=False
        )

        fig.colorbar(sc0, ax=axes[0], label='log1p(Gene Expression)')
        fig.colorbar(sc1, ax=axes[1], label='Protein Signal')
        plt.tight_layout()
        plt.show()

# === Example usage ===
visualize_gene_expression_pairs(
    pair_file_path='/media/huifang/data/sennet/xenium_codex_pairs.txt',
    output_dir="/media/huifang/data/sennet/hf_aligned_data",
    start_line=0
)
