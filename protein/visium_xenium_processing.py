import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
import scanpy as sc
import json
from scipy.sparse import issparse
from imageio.v2 import imwrite
from scipy import sparse
from skimage.color import rgb2gray
from imageio.v2 import imwrite
import cv2
import re
import os
from anndata import AnnData
# -----------------------------
# Helpers
# -----------------------------
# -----------------------------
# 2) Protein -> Gene mapping (curated for your list)
#    None = ignore (stains or ambiguous "pan" targets)
# -----------------------------
PROTEIN_TO_GENE = {
    'DAPI': None,                       # DNA dye, not a gene
    'CD44': 'CD44',
    'CD4': 'CD4',
    'CD31': 'PECAM1',
    'E-Cadherin': 'CDH1',
    'LIF': 'LIF',
    'a-SMA': 'ACTA2',
    'CD45RO': 'PTPRC',                  # CD45 isoform -> gene PTPRC
    'CD68': 'CD68',
    'CD20': 'MS4A1',
    'b-Catenin1': 'CTNNB1',             # β-catenin
    'LAG3': 'LAG3',
    'b-Actin': 'ACTB',
    'Podoplanin': 'PDPN',
    'CD11c': 'ITGAX',
    'Collagen-IV': 'COL4A1',            # Often targets COL4A1/2; adjust to chain used by your antibody
    'CD8': 'CD8A',                      # Most panels use CD8A; use CD8B if appropriate
    'IDO1': 'IDO1',
    'HLA-A': 'HLA-A',
    'DC-LAMP': 'LAMP3',
    'CXCL13': 'CXCL13',
    'Pan-Cytokeratin': None,            # Mix of KRTs (e.g., KRT8/18/19); no 1:1 gene
    'Galectin3': 'LGALS3',
    'CTLA4': 'CTLA4',
    'HLA-DPB1': 'HLA-DPB1',
    'PD-L1': 'CD274',
    'NKX2-1': 'NKX2-1',
    'TP53': 'TP53',
    'Ki67': 'MKI67',
    'CD3e': 'CD3E',
    'CD163': 'CD163',
    'FOXP3': 'FOXP3',
    'PD-1': 'PDCD1',
    'ICOS': 'ICOS',
    'CD56': 'NCAM1',
    'TIGIT': 'TIGIT',
    'CD19': 'CD19',
    'Caveolin': 'CAV1',                 # Typically Caveolin-1; change to CAV2/CAV3 if that’s your reagent
}

# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    s = str(s).strip().upper()
    return re.sub(r'[\s\-]+', '', s)

def _genes_from_adata(adata):
    raw = pd.Index(adata.var_names.astype(str))
    norm = pd.Index([_norm(x) for x in raw])
    return raw, norm

def _adata_norm_name_set(adata):
    _, norm = _genes_from_adata(adata)
    return set(norm)

def _build_indexer_by_normnames(adata, wanted_genes_raw_like):
    raw, norm = _genes_from_adata(adata)
    first_idx = {}
    for i, g in enumerate(norm):
        if g not in first_idx:
            first_idx[g] = i
    idxs, missing = [], []
    for g in wanted_genes_raw_like:
        i = first_idx.get(_norm(g))
        if i is None:
            missing.append(g)
        else:
            idxs.append(i)
    return idxs, missing

def _detect_protein_label_column(protein_adata):
    """
    Try to find the human-readable protein label column in protein_adata.var.
    If none found, fall back to var_names.
    """
    candidates = ["target", "Target", "marker", "Marker", "protein", "Protein",
                  "antigen", "Antigen", "name", "Name"]
    for col in candidates:
        if col in protein_adata.var.columns:
            vals = pd.Index([_norm(v) for v in protein_adata.var[col].tolist()])
            if (vals.str.len() > 0).sum() > 0:
                return col
    return "__var_names__"

def _build_indexer_from_protein_column(protein_adata, wanted_labels_raw_like, var_col_name):
    """Map wanted protein labels (as strings) to protein_adata columns using var_col_name (or var_names)."""
    if var_col_name == "__var_names__":
        col_vals = pd.Index([_norm(v) for v in protein_adata.var_names])
    else:
        col_vals = pd.Index([_norm(v) for v in protein_adata.var[var_col_name].tolist()])
    first_idx = {}
    for i, s in enumerate(col_vals):
        if s and s not in first_idx:
            first_idx[s] = i
    idxs, missing = [], []
    for w in wanted_labels_raw_like:
        i = first_idx.get(_norm(w))
        if i is None:
            missing.append(w)
        else:
            idxs.append(i)
    return idxs, missing

# -----------------------------
# Main
# -----------------------------
def filter_all_three_modalities(protein_adata, visium_adata, xenium_adata,
                                proteins, map_dict=PROTEIN_TO_GENE,selected_genes=None):
    # 1) Protein -> gene mapping from your panel (drop None)
    mapped = [(p, map_dict.get(p, None)) for p in proteins]
    protein_to_gene = {p: str(g) for p, g in mapped if g is not None}
    gene_panel = list(dict.fromkeys(protein_to_gene.values()))  # unique, keep panel order

    if selected_genes:
        shared_genes = selected_genes
    else:
        # 2) Presence by name (normalized)
        vis_set = _adata_norm_name_set(visium_adata)
        xeu_set = _adata_norm_name_set(xenium_adata)
        # 3) Shared genes in both (preserve original panel order)
        shared_genes = [g for g in gene_panel if (_norm(g) in vis_set) and (_norm(g) in xeu_set)]
        # print(f"Shared genes (Visium ∩ Xenium) [{len(shared_genes)}]: {shared_genes}")
        if not shared_genes:
            print("WARNING: No shared genes between Visium and Xenium for this panel.")
            return protein_adata, visium_adata, xenium_adata, [], protein_to_gene


    # 4) Subset Visium/Xenium to shared order

    vis_idx, vis_missing = _build_indexer_by_normnames(visium_adata, shared_genes)
    xeu_idx, xeu_missing = _build_indexer_by_normnames(xenium_adata, shared_genes)
    if vis_missing or xeu_missing:
        print("[WARN] Unexpected missing after intersection.",
              f"Visium missing: {vis_missing}", f"Xenium missing: {xeu_missing}")

    visium_f = visium_adata[:, vis_idx]
    xenium_f = xenium_adata[:, xeu_idx]

    # 5) Subset Protein to channels whose mapped gene ∈ shared_genes (1-to-1; if duplicates, keep first by panel order)
    prot_label_col = _detect_protein_label_column(protein_adata)
    # Build gene -> first protein (respect PROTEINS order)
    gene2prot = {}
    for p in proteins:
        g = protein_to_gene.get(p, None)
        if g is not None and g in shared_genes and g not in gene2prot:
            gene2prot[g] = p
    # Order proteins to follow shared_genes
    ordered_proteins = [gene2prot[g] for g in shared_genes if g in gene2prot]

    # Index protein_adata by those protein labels
    prot_idx, prot_missing = _build_indexer_from_protein_column(protein_adata, ordered_proteins, prot_label_col)
    # if prot_missing:
    #     print(f"[protein] Warning: channels not found in '{prot_label_col}': {prot_missing}")
    # if not prot_idx:
    #     raise RuntimeError("No protein channels matched the shared gene panel. "
    #                        "Check the protein label column and mapping.")

    protein_f = protein_adata[:, prot_idx]

    return protein_f, visium_f, xenium_f, shared_genes, protein_to_gene

def downsample_adata(adata, dsrate):
    n_sub = max(1, int(adata.n_obs / dsrate))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # deterministic: evenly spaced indices
    step = max(1, int(np.floor(adata.n_obs / n_sub)))
    subset_idx = np.arange(0, adata.n_obs, step)[:n_sub]

    adata_sub = adata[subset_idx].copy()
    return adata_sub


def read_protein():
    out_h5ad = datadir + f"{data}_protein_trans.h5ad"
    adata = sc.read_h5ad(out_h5ad)
    stain_image = plt.imread(datadir + f"{data}_protein_DAPI_trans_enhanced.png")
    return stain_image, adata
def read_xenium():
    out_h5ad = datadir + f"{data}_xenium_trans.h5ad"
    adata = sc.read_h5ad(out_h5ad)

    dapi_img = plt.imread(datadir + f"{data}_xenium_DAPI_trans_enhanced.png")
    return dapi_img, adata
def read_visium():
    out_h5ad = datadir + f"{data}_visium_trans.h5ad"
    adata = sc.read_h5ad(out_h5ad)
    visium_gray=plt.imread(datadir + f"{data}_visium_GRAY_trans_enhanced.png")
    return visium_gray,adata




# -----------------------------
def _to_1d(arr):
    """Return a dense 1D float array from AnnData .X column (handles sparse)."""
    if sparse.issparse(arr):
        arr = arr.toarray()
    return np.asarray(arr, dtype=float).reshape(-1)

def _get_col(adata, j):
    """Get column j from adata.X as 1D array."""
    return _to_1d(adata.X[:, j])

def _robust_limits(arrs, low=5, high=95):
    """Robust vmin/vmax across multiple arrays."""
    cat = np.concatenate([a[np.isfinite(a)] for a in arrs if a.size > 0])
    if cat.size == 0:
        return (0.0, 1.0)
    vmin = np.percentile(cat, low)
    vmax = np.percentile(cat, high)
    if vmin == vmax:
        vmax = vmin + (1e-6 if vmin == 0 else 1e-3 * abs(vmin))
    return vmin, vmax

def _autosize(n_obs):
    """Auto marker size by dataset density (tweak if needed)."""
    base = 18000.0
    return float(np.clip(base / max(n_obs, 1), 1.5, 18.0))

def _scatter_one(ax, coords, values, title, vmin, vmax, s):
    m = np.isfinite(values)
    ax.scatter(coords[m, 0], coords[m, 1],
               c=values[m], s=s, cmap=CMAP, vmin=vmin, vmax=vmax, alpha=POINT_ALPHA, linewidths=0)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image-like orientation; comment out if undesired
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")

POINT_ALPHA = 0.85          # scatter point transparency
CMAP = "viridis"
# -----------------------------
# Main plotting
# -----------------------------
def plot_spatial_triplets(protein_f, visium_f, xenium_f,save=False):
    # Sanity: require same number/order of variables
    n_vars = visium_f.n_vars
    assert xenium_f.n_vars == n_vars == protein_f.n_vars, "All three AnnDatas must share the same n_vars in the same order."

    # Coordinates
    P_sp = np.asarray(protein_f.obsm["spatial"])
    V_sp = np.asarray(visium_f.obsm["spatial"])
    X_sp = np.asarray(xenium_f.obsm["spatial"])

    # Autosizes
    sP = _autosize(protein_f.n_obs)
    sV = _autosize(visium_f.n_obs)
    sX = _autosize(xenium_f.n_obs)

    # Iterate over panels
    for j in range(n_vars):
        # Names for titles / filenames
        gene_name = str(visium_f.var_names[j])  # use visium_f/xenium_f gene symbol
        # protein_f.var_names may be protein labels or genes; show both when helpful
        prot_label = str(protein_f.var_names[j])
        title_prot = f"Protein: {prot_label}"
        title_vis  = f"Visium:  {gene_name}"
        title_xen  = f"Xenium:  {gene_name}"

        # Values
        p_vals = _get_col(protein_f, j)
        v_vals = _get_col(visium_f, j)
        x_vals = _get_col(xenium_f, j)

        # Common color scale per gene
        vminp, vmaxp = _robust_limits([p_vals], low=2, high=98)
        vminv, vmaxv = _robust_limits([v_vals], low=2, high=98)
        vminx, vmaxx = _robust_limits([x_vals], low=2, high=98)



        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        _scatter_one(axes[0], P_sp, p_vals, title_prot, vminp, vmaxp, sP)
        _scatter_one(axes[1], V_sp, v_vals, title_vis,  vminv, vmaxv, sV)
        _scatter_one(axes[2], X_sp, x_vals, title_xen,  vminx, vmaxx, sX)

        # Single shared colorbar
        im = axes[2].collections[0]
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.ax.set_ylabel("Expression / Intensity (scaled)", rotation=90)

        if save:
            outdir = f"/media/huifang/data/registration/phenocycler/H5ADs/feature_preview/{data}/"
            os.makedirs(outdir,exist_ok=True)
            plt.savefig(outdir+gene_name+".png", dpi=300)
        else:
            plt.show()



LOW_PCT = 2
HIGH_PCT = 98

def _to_dense_col(x_col):
    """x_col: shape (n_obs, 1) or (n_obs,), possibly sparse -> (n_obs,) dense float."""
    if sparse.issparse(x_col):
        x_col = x_col.toarray()
    return np.asarray(x_col, dtype=float).reshape(-1)

def _colwise_clip_log1p(adata, low=LOW_PCT, high=HIGH_PCT, out_dtype=np.float32):
    """
    For each column j in adata.X:
      - compute low/high percentiles on finite values only
      - clamp to [low, high]
      - apply np.log1p
    Returns a dense matrix (n_obs, n_vars) of dtype `out_dtype`.
    """
    n_obs, n_vars = adata.shape
    out = np.empty((n_obs, n_vars), dtype=out_dtype)

    for j in range(n_vars):
        col = _to_dense_col(adata.X[:, j])
        finite_mask = np.isfinite(col)
        if finite_mask.any():
            vmin = np.percentile(col[finite_mask], low)
            vmax = np.percentile(col[finite_mask], high)
            if vmin == vmax:  # avoid zero range
                vmax = vmin + (1e-6 if vmin == 0 else 1e-3 * abs(vmin))
        else:
            vmin, vmax = 0.0, 1.0  # fallback if everything is non-finite

        col = np.clip(col, vmin, vmax)
        col = np.log1p(col)
        out[:, j] = col.astype(out_dtype, copy=False)

    return out


def channelwise_min_max_normalize(data):
    mins = data.min(axis=0)   # shape (C,)
    maxs = data.max(axis=0)   # shape (C,)
    ranges = maxs - mins

    # Avoid division by zero (when all values in a channel are the same)
    ranges[ranges == 0] = 1e-8

    normalized_data = (data - mins) / ranges
    return normalized_data

from sklearn.decomposition import PCA
def reduce_gene_reads(gene_reads: np.ndarray, method: str = 'pca', n_components: int = 10) -> np.ndarray:

    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")

    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(gene_reads)

    elif method.lower() == 'umap':
        # UMAP
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(gene_reads)
    else:
        raise ValueError("method must be one of ['pca', 'umap'].")

    return reduced_data

def get_gene_feature_matrix(coords: np.ndarray,
                            reduced_data: np.ndarray,
                            image_size=(2016, 2016),
                            patch_size=32):            # how many smoothing passes if local
    """
    Build a (H/ps, W/ps, D) patch feature matrix. Empty patches can be filled to soften borders.

    fill_mode:
      - "global_mean": fill empties with per-feature global mean across valid patches.
      - "local_mean" : iteratively fill empties with mean of 8-neighbors (per feature).
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    n_spots, n_dims = reduced_data.shape
    height, width = image_size
    out_height = height // patch_size
    out_width  = width  // patch_size

    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) bin spots to patches
    for i in range(n_spots):
        x, y = coords[i]  # expected pixel coords
        px = int(x) // patch_size
        py = int(y) // patch_size
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]
            count_array[py, px] += 1

    # 2) per-patch averages where we have data
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = sum_array[valid_mask, :] / count_array[valid_mask, None]
    return patch_matrix, valid_mask

def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray,ncols=5,):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]

    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")


    plt.show()



datasets=["LUAD_2_A", "LUAD_3_A", "TSU_20_1",
              "TSU_24", "TSU_33", "TSU_35"]

selected_gene_panels=[['ACTA2','CD44','CD68','CDH1','CTNNB1','HLA-A','HLA-DPB1','IDO1','MKI67','NKX2-1'],
                      ['ACTA2','CD44','CDH1','CTNNB1','CXCL13','HLA-A','HLA-DPB1','NKX2-1','PTPRC'],
                      ['ACTA2','CD68','CTNNB1','CXCL13','ITGAX','NKX2-1'],
                      [],
                      [],
                      ['ACTA2','NKX2-1','PDPN','PECAM1','PTPRC']]



datadir = "/media/huifang/data/registration/phenocycler/H5ADs/huifang/"

startidx=2
for data,genes in zip(datasets[startidx:],selected_gene_panels[startidx:]):
    print(data)
    protein_dapi,protein_adata= read_protein()
    visium_gray,visium_adata = read_visium()
    xenium_dapi,xenium_adata = read_xenium()

    proteins = protein_adata.var_names.tolist()

    protein_f, visium_f, xenium_f, shared_genes, protein_to_gene = filter_all_three_modalities(
        protein_adata, visium_adata, xenium_adata,proteins)

    # protein_f = downsample_adata(protein_f,100)
    # xenium_f = downsample_adata(xenium_f,100)

    # plot_spatial_triplets(protein_f, visium_f, xenium_f)




    if data == 'TSU_20_1':
        protein_mat = _colwise_clip_log1p(protein_f, low=0, high=100)
        visium_mat = _colwise_clip_log1p(visium_f, low=0, high=100)
        xenium_mat = _colwise_clip_log1p(xenium_f, low=0, high=100)
    else:
        protein_mat = _colwise_clip_log1p(protein_f, low=1, high=99)
        visium_mat  = _colwise_clip_log1p(visium_f,  low=1, high=99)
        xenium_mat  = _colwise_clip_log1p(xenium_f,  low=5, high=95)


    gene_data_list = [protein_mat, visium_mat,xenium_mat]
    coords_list = [protein_f.obsm['spatial'],visium_f.obsm['spatial'],xenium_f.obsm['spatial']]
    image_list=[protein_dapi,visium_gray,xenium_dapi]

    # gene_data_list = [protein_mat,  xenium_mat]
    # coords_list = [protein_f.obsm['spatial'],  xenium_f.obsm['spatial']]
    # image_list = [protein_dapi, xenium_dapi]


    combined_data = np.vstack(gene_data_list)
    # combined_data = combined_data[:,[7,13,17,19,21]]
    # combined_data = combined_data[:, [0, 4, 7, 13,17]]
    combined_data = combined_data[:, [3, 4, 5, 15, 17,18]]

    reduced_data = combined_data
    # reduced_data = reduce_gene_reads(
    #     combined_data,
    #     method='pca',
    #     n_components=5
    # )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    # reduced_data = combined_data
    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        reduced_data = channelwise_min_max_normalize(reduced_data)
        index_start = index_end

        coords = coords_list[i]
        image = image_list[i]

        if i==1:
            ps=16
        else:
            ps = 8

        if data == 'TSU_20_1':
            if i==1:
                ps = 32
            if i==2:
                ps = 32


        feature_matrix, gene_mask = get_gene_feature_matrix(coords, reduced_slice_data,
                                                            (image.shape[0], image.shape[1]),
                                                            patch_size=ps)

        if i ==1:
            feature_matrix = np.stack([
                cv2.resize(feature_matrix[:, :, i], (128, 128), interpolation=cv2.INTER_NEAREST)
                for i in range(feature_matrix.shape[2])
            ], axis=-1)
            valid_mask = (feature_matrix > 0)
            gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)



        # feature_matrix = preprocess_feature_matrix(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)
        # valid_mask = (feature_matrix > 0.01)
        # gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)

        # prefix=f"{k}_{i}"
        plot_dimensional_images_side_by_side(feature_matrix, ncols=5)
