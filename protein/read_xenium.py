#!/usr/bin/env python3

from scipy.sparse import issparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread
from skimage.transform import resize
import json


def save_top_genes(DATA_DIR,adata):
    OUT_DIR = DATA_DIR / "xenium_previews"
    OUT_DIR.mkdir(exist_ok=True)
    # same protein→gene mapping as before
    protein_to_gene = {
        "CD44": "CD44", "CD4": "CD4", "CD31": "PECAM1", "E-Cadherin": "CDH1",
        "LIF": "LIF", "a-SMA/ACTA2": "ACTA2", "CD45RO": "PTPRC", "CD68": "CD68",
        "CD20": "MS4A1", "b-Catenin1": "CTNNB1", "CD223/LAG3": "LAG3",
        "Beta-actin": "ACTB", "Podoplanin": "PDPN", "CD11c": "ITGAX",
        "Collagen-IV": "COL4A1", "CD8": "CD8A", "IDO1": "IDO1",
        "HLA-A": "HLA-A", "DC-LAMP": "CD208", "BCA1/CXCL13": "CXCL13",
        "Pan-Cytokeratin": "KRT19", "Mac2/Galectin3": "LGALS3", "CTLA4": "CTLA4",
        "HLA-DPB1": "HLA-DPB1", "CD274/PD-L1": "CD274", "TTF-1/NKX2-1": "NKX2-1",
        "TP53": "TP53", "Ki67": "MKI67", "CD3e": "CD3E", "CD163": "CD163",
        "FOXP3": "FOXP3", "CD279//PD-1": "PDCD1", "CD278/ICOS": "ICOS",
        "CD56": "NCAM1", "TIGIT": "TIGIT"
    }
    TARGET_GENES = sorted(set(protein_to_gene.values()))
    # ----------------------------------------
    # === 5. Plot the marker genes ===
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    for gene in TARGET_GENES:
        if gene not in adata.var_names:
            print(f"[Skip] {gene} not in dataset")
            continue

        # Extract values safely as dense
        vals = adata[:, gene].X
        if issparse(vals):
            vals = vals.toarray().flatten()
        else:
            vals = np.asarray(vals).flatten()

        # Skip genes with all zeros or NaNs
        if np.allclose(vals, 0) or np.isnan(vals).all():
            print(f"[Skip] {gene} all zeros or NaN")
            continue

        # Robust intensity normalization
        lo, hi = np.nanpercentile(vals, [2, 98])
        if hi <= lo:  # prevent divide-by-zero
            continue
        vals = np.clip((vals - lo) / (hi - lo), 0, 1)

        # Plot
        plt.figure(figsize=(16, 16))
        plt.scatter(x, y, c=vals, cmap="magma", s=1, alpha=0.8)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.title(f"{gene} spatial map (Xenium per-cell)")
        fn = OUT_DIR / f"gene_spatial_{gene}.png"
        plt.savefig(fn, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fn.name}")

def read_he(data_path):
    he_img = plt.imread(next(data_path.glob("XeniumHE*_down5x.png")))
    # --- Rotate 90 degrees counterclockwise ---
    he_rot = np.rot90(he_img, k=1)


    # --- Flip left-right (horizontal mirror) ---
    he_rot_flipped = np.flipud(he_rot)
    return he_rot_flipped

def save_h5ad(DATA_DIR):
    H5_PATH = DATA_DIR / "cell_feature_matrix.h5"
    CELLS_PATH = DATA_DIR / "cells.parquet"

    # === 1. Read the Xenium H5 matrix ===
    print(f"Reading Xenium cell_feature_matrix.h5 ...")
    adata = sc.read_10x_h5(H5_PATH)
    adata.var_names_make_unique()
    print(adata)

    # === 2. Read centroid coordinates from cells.parquet ===
    cells = pd.read_parquet(CELLS_PATH)
    if "x_centroid" in cells.columns and "y_centroid" in cells.columns:
        coords = cells[["x_centroid", "y_centroid"]].to_numpy()
    else:
        coords = cells[[c for c in cells.columns if 'x' in c.lower() or 'y' in c.lower()]].iloc[:, :2].to_numpy()
    adata.obsm["spatial"] = coords
    print(f"Added spatial coordinates: {adata.obsm['spatial'].shape}")

    # === 4. Save adata ===
    adata.write(DATA_DIR / "adata_xenium_cell_level.h5ad")
    print("Saved AnnData:", DATA_DIR / "adata_xenium_cell_level.h5ad")
    return adata




root = Path("/media/huifang/data/registration/phenocycler/")
h5ad_path = "/media/huifang/data/registration/phenocycler/H5ADs/"
for data in ["LUAD_2_A", "TSU_20_1", "TSU_23", "TSU_28", "TSU_33",
             "LUAD_3_A", "TSU_21", "TSU_24", "TSU_30", "TSU_35"]:

    print(data)
    data_path = root / data / "xenium"
    para_path = data_path /"experiment.xenium"


    dapi_img = plt.imread(data_path / "morphology_focus.ome_down10x.png")
    he_img = read_he(data_path)

    adata = sc.read_h5ad(data_path / "adata_xenium_cell_level.h5ad")

    with open(para_path, "r") as file:
        experiment_data = json.load(file)
    pixel_size = experiment_data.get("pixel_size")


    # === 5. Plot the marker genes ===
    x = adata.obsm["spatial"][:, 0]/pixel_size
    y = adata.obsm["spatial"][:, 1]/pixel_size


    # Extract values safely as dense
    gene = "ACTA2"
    vals = adata[:, gene].X
    if issparse(vals):
        vals = vals.toarray().flatten()
    else:
        vals = np.asarray(vals).flatten()

    x = x[::100]
    y = y[::100]
    vals = vals[::100]

    # Robust intensity normalization
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo), 0, 1)

    f,a = plt.subplots(1,2,figsize=(16, 8))
    a[0].imshow(dapi_img)
    a[0].scatter(x/10, y/10, c=vals, cmap="magma", s=1)

    a[1].imshow(he_img)
    # a[1].scatter(x/17.1, y/17.1, c=vals, cmap="magma", s=1)

    plt.title(f"{gene} spatial map (Xenium per-cell)")
    plt.show()
