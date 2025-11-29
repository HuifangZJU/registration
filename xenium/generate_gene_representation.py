import scanpy as sc
import anndata as ad
import scvi
import os
import torch
torch.set_float32_matmul_precision("high")
from scvi import settings
settings.dl_num_workers = 15         # dataloader workers
settings.batch_size = 128            # global batch size if you want to change it
settings.dl_pin_memory = True    # good when using CUDA
def get_adata(dataset):
    adata = sc.read_h5ad(f"/media/huifang/data/Xenium/xenium_data/{dataset}/xenium_annodata_with_morphology.h5ad")
    return adata


def run_mouse():
    datasets = ['Xenium_V1_FFPE_TgCRND8_2_5_months', 'Xenium_V1_FFPE_TgCRND8_5_7_months',
                'Xenium_V1_FFPE_TgCRND8_17_9_months',
                'Xenium_V1_FFPE_wildtype_2_5_months', 'Xenium_V1_FFPE_wildtype_5_7_months',
                'Xenium_V1_FFPE_wildtype_13_4_months']
    adatas = []
    for dataset in datasets:
        adatas.append(get_adata(dataset))

    for a, name in zip(adatas, [
        "Tg_2.5m", "Tg_5.7m", "Tg_17.9m", "WT_2.5m", "WT_5.7m", "WT_13.4m"
    ]):
        a.obs["sample_id"] = name
        a.obs["genotype"] = "TgCRND8" if name.startswith("Tg") else "WT"
        a.obs["age_months"] = float(name.split("_")[1].replace("m", ""))
        a.obs_names_make_unique()

    adata = ad.concat(adatas, label="dataset_id", merge="same")
    adata.layers["counts"] = adata.X.copy()
    # ---------- 2) Normalization & batch integration (scVI) ----------
    # 2) scVI with batch + covariates

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",  # use raw counts if available
        batch_key="sample_id",
        categorical_covariate_keys=["genotype"],
        continuous_covariate_keys=["age_months"],
    )

    vae = scvi.model.SCVI(adata, n_latent=30)

    vae.train(
        max_epochs=100, precision="32-true", accelerator="gpu", devices=1
    )

    adata.obsm["X_scvi"] = vae.get_latent_representation()

    for sid in adata.obs["sample_id"].unique():
        sub = adata[adata.obs["sample_id"] == sid].copy()
        fn = os.path.join("/media/huifang/data/registration/xenium/cell_typing", f"xenium_{sid}_scvi.h5ad")
        sub.write_h5ad(fn)
        print(f"Saved: {fn}")



datasets = [ 'Xenium_V1_FFPE_Human_Breast_ILC','Xenium_V1_FFPE_Human_Breast_ILC_With_Addon']
adatas = []
for dataset in datasets:
    adatas.append(get_adata(dataset))

for a, name in zip(adatas, [
    "ILC", "ILC_addon"
]):
    a.obs["sample_id"] = name
    a.obs_names_make_unique()

adata = ad.concat(adatas, label="dataset_id", merge="same")
adata.layers["counts"] = adata.X.copy()
# ---------- 2) Normalization & batch integration (scVI) ----------
# 2) scVI with batch + covariates

scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",  # use raw counts if available
    batch_key="sample_id",
)

vae = scvi.model.SCVI(adata, n_latent=30)

vae.train(
    max_epochs=100, precision="32-true", accelerator="gpu", devices=1
)

adata.obsm["X_scvi"] = vae.get_latent_representation()

for sid in adata.obs["sample_id"].unique():
    sub = adata[adata.obs["sample_id"] == sid].copy()
    fn = os.path.join("/media/huifang/data/registration/xenium/cell_typing", f"xenium_{sid}_scvi.h5ad")
    sub.write_h5ad(fn)
    print(f"Saved: {fn}")