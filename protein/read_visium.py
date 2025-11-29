import scanpy as sc
import pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from tifffile import imread
from skimage.transform import resize
from pathlib import Path



# ----------------------------------------

def save_top_genes(DATA_DIR, adata):
    OUT_DIR = DATA_DIR / "visium_previews"
    OUT_DIR.mkdir(exist_ok=True)
    # --- protein → gene name mapping (manually curated) ---
    protein_to_gene = {
        "DAPI": "DAPI",  # nucleus dye (no transcript)
        "CD44": "CD44",
        "CD4": "CD4",
        "CD31": "PECAM1",
        "E-Cadherin": "CDH1",
        "LIF": "LIF",
        "a-SMA/ACTA2": "ACTA2",
        "CD45RO": "PTPRC",     # CD45 isoform; CD45RO/CD45RA both -> PTPRC
        "CD68": "CD68",
        "CD20": "MS4A1",
        "b-Catenin1": "CTNNB1",
        "CD223/LAG3": "LAG3",
        "Beta-actin": "ACTB",
        "Podoplanin": "PDPN",
        "CD11c": "ITGAX",
        "Collagen-IV": "COL4A1",
        "CD8": "CD8A",
        "IDO1": "IDO1",
        "HLA-A": "HLA-A",
        "DC-LAMP": "CD208",
        "BCA1/CXCL13": "CXCL13",
        "Pan-Cytokeratin": "KRT19",
        "Mac2/Galectin3": "LGALS3",
        "CTLA4": "CTLA4",
        "HLA-DPB1": "HLA-DPB1",
        "CD274/PD-L1": "CD274",
        "TTF-1/NKX2-1": "NKX2-1",
        "TP53": "TP53",
        "Ki67": "MKI67",
        "CD3e": "CD3E",
        "CD163": "CD163",
        "FOXP3": "FOXP3",
        "CD279//PD-1": "PDCD1",
        "CD278/ICOS": "ICOS",
        "CD56": "NCAM1",
        "TIGIT": "TIGIT"
    }
    TOP_GENES = sorted(set([g for g in protein_to_gene.values() if g != "DAPI"]))
    # === 5. Visualize matched genes ===
    X = adata.obsm["spatial"][:, 1]   # col
    Y = adata.obsm["spatial"][:, 0]   # row

    for gene in TOP_GENES:
        if gene not in adata.var_names:
            print(f"[Skip] {gene} not in dataset")
            continue
        vals = adata[:, gene].X.toarray().flatten()
        lo, hi = np.nanpercentile(vals, [2, 98])
        vals = np.clip((vals - lo) / (hi - lo + 1e-6), 0, 1)

        plt.figure(figsize=(8,8))
        plt.scatter(X, Y, c=vals, cmap="magma", s=12, alpha=0.8)
        plt.gca().invert_yaxis()
        plt.axis("off")
        plt.title(f"{gene} spatial map (protein-correlated)")
        plt.savefig(OUT_DIR / f"gene_spatial_{gene}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved gene_spatial_{gene}.png")



def save_h5ad(DATA_DIR):
    # ---------------- CONFIG ----------------

    H5_PATH = DATA_DIR / "filtered_feature_bc_matrix.h5"
    SPATIAL_DIR = DATA_DIR / "spatial"
    # === 1. Load gene–spot matrix ===
    print(f"Reading 10x matrix: {H5_PATH}")
    adata = sc.read_10x_h5(H5_PATH)
    adata.var_names_make_unique()

    # === 2. Load spot coordinates and scalefactors ===
    pos_file = SPATIAL_DIR / "tissue_positions_list.csv"
    sf_file = SPATIAL_DIR / "scalefactors_json.json"

    df_pos = pd.read_csv(pos_file, header=None)
    df_pos.columns = [
        "barcode", "in_tissue", "array_row", "array_col",
        "pxl_col_in_fullres", "pxl_row_in_fullres"
    ]
    df_pos.set_index("barcode", inplace=True)
    df_pos = df_pos.loc[adata.obs_names]
    adata.obs = adata.obs.join(df_pos)
    adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    print(f"Added spatial coordinates: {adata.obsm['spatial'].shape}")
    if sf_file.exists():
        with open(sf_file) as f:
            sf = json.load(f)
        adata.uns["spatial_scalefactors"] = sf
        print("Loaded scale factors:", sf.keys())
    adata.write_h5ad(DATA_DIR / "adata_visium.h5ad")
    print("Saved AnnData:", DATA_DIR / "adata_visium.h5ad")
    return adata
# ------------------------------------------------------



root = Path("/media/huifang/data/registration/phenocycler/")
h5ad_path = "/media/huifang/data/registration/phenocycler/H5ADs/"
for data in ["LUAD_2_A", "TSU_20_1", "TSU_23", "TSU_28", "TSU_33",
             "LUAD_3_A", "TSU_21", "TSU_24", "TSU_30", "TSU_35"]:

    print(data)
    data_path = root / data / "visium"
    # save_h5ad(data_path)

    spatial_dir = data_path / "spatial"
    gene_name = "ACTA2"  # choose your gene of interest

    adata = sc.read_h5ad(data_path / "adata_visium.h5ad")
    scales = adata.uns['spatial_scalefactors']
    # Usually named "tissue_hires_image.png" or ".tif"
    img_path = next(spatial_dir.glob("tissue_hires_image_1.png"))
    he = plt.imread(img_path)

    # --- Compute scaled spot coordinates ---
    scale = scales["tissue_hires_scalef"]
    xy = adata.obsm["spatial"] * scale  # scale Visium spot positions to match hires image

    # --- Get gene expression values ---
    vals = adata[:, gene_name].X.toarray().flatten()
    lo, hi = np.nanpercentile(vals, [2, 98])
    vals = np.clip((vals - lo) / (hi - lo + 1e-6), 0, 1)

    # --- Plot overlay ---
    plt.figure(figsize=(8, 8))
    plt.imshow(he)
    plt.scatter(xy[:, 0], xy[:, 1], c=vals, s=8, cmap="magma", alpha=0.7)
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title(f"{gene_name} expression over H&E")
    plt.show()

