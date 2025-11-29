
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.cm import get_cmap
import os
from scipy.spatial import cKDTree
sc.settings.n_jobs = 16
os.environ["OMP_NUM_THREADS"]  = "16"
os.environ["MKL_NUM_THREADS"]  = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
sc.settings.n_jobs = 16

def donw_sample_adata(adata,dsrate):
    # assume your object is named `adata`
    frac = 1 / dsrate  # downsample ratio
    n_sub = max(1, int(adata.n_obs * frac))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # reproducible sampling
    rng = np.random.default_rng(seed=0)
    subset_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)

    adata_sub = adata[subset_idx].copy()
    return adata_sub

def generate_spatial_domain(adata):
    # --- transcriptomic neighbors on use_rep ---
    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15)
    A_trx = adata.obsp["connectivities"].tocsr()
    # --- spatial neighbors ---
    coords = adata.obsm['spatial']
    k = min(8, max(1, coords.shape[0]))
    nn = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(return_distance=False)
    rows = np.repeat(np.arange(nn.shape[0])[:, None], nn.shape[1], axis=1).ravel()
    cols = nn.ravel()
    data = np.ones_like(rows, dtype=float)
    A_sp = sp.csr_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    # binarize spatial (already 0/1); ensure symmetry
    A_sp = ((A_sp + A_sp.T) > 0).astype(float)
    # --- mix graphs ---
    # both are (n_sub, n_sub) CSR matrices
    A_mix = (0.6 * A_trx) + (0.4 * A_sp)

    # --- Leiden on mixed adjacency (per sample only) ---
    print('leiden clustering')
    sc.tl.leiden(
        adata,
        adjacency=A_mix,
        resolution=1.5,
        key_added="domain_id",
        random_state=0,
    )
    adata.obs["domain_id"] = adata.obs["domain_id"].astype("category")
    return adata


from matplotlib.patches import Patch

def visualize_domains(save_dir=None):
    leiden_key = "domain_id_shared"  # or your key
    adata.obs[leiden_key] = adata.obs[leiden_key].astype("category")
    domains = adata.obs[leiden_key].cat.categories.tolist()
    palette = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cm = get_cmap(cmap_name, 20)
        palette.extend([to_hex(cm(i)) for i in range(cm.N)])
    if len(domains) > len(palette):
        extra = len(domains) - len(palette)
        for i in range(extra):
            h = (i / max(1, extra)) % 1.0
            palette.append(to_hex(plt.cm.hsv(h)))
    palette = palette[:len(domains)]
    # store for reproducibility
    adata.uns[f"{leiden_key}_colors"] = palette


    # The mapping you asked about:
    dom2color = dict(zip(domains, palette))


    # global domains + colors (dom2color must already exist)
    domains = list(adata.obs[leiden_key].cat.categories)
    handles = [Patch(color=dom2color[d], label=str(d)) for d in domains]

    sids = sorted(adata.obs["sample_id"].astype(str).unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # if fewer than 6 samples, hide unused axes
    for ax in axes[len(sids):]:
        ax.axis("off")

    for i, sid in enumerate(sids):
        ax = axes[i]
        mask = adata.obs["sample_id"].astype(str).values == sid
        coords = adata.obsm["spatial"][mask, :2]
        labs = adata.obs.loc[adata.obs.index[mask], leiden_key].astype(str).values

        # draw each domain as one scatter for speed
        counts = pd.Series(labs).value_counts()
        for d in counts.index:
            m = (labs == d)
            ax.scatter(coords[m, 0], coords[m, 1], s=5, c=dom2color[d], marker='.', linewidths=0)

        ax.set_title(f"{sid} — {leiden_key}", fontsize=11)
        ax.set_aspect('equal');
        ax.invert_yaxis();
        ax.axis('off')

    # make room on the right for legend
    plt.subplots_adjust(right=0.72)

    # one shared legend on the right
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=True,
        title="Spatial domains",
        fontsize=9,
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir,'spatial_domain.png'), dpi=600)
    else:
        plt.show()

# -----------------------------
# Config
# -----------------------------
data_dir = "/media/huifang/data/registration/xenium/cell_typing"
# sample_ids = ["Tg_2.5m", "Tg_5.7m", "Tg_17.9m", "WT_2.5m", "WT_5.7m", "WT_13.4m"]
sample_ids =["ILC", "ILC_addon"]

use_rep = "X_scvi"          # global gene-expression representation
n_neighbors_scvi = 30       # gene neighbors (global)
n_neighbors_spatial = 8     # spatial neighbors (per sample)
resolution = 1.0 # mouse setting
alpha = 0.5                 # mix weight: alpha*gene + (1-alpha)*spatial
leiden_key = "domain_id_shared"
random_state = 0

# -----------------------------
# Load & concat
# -----------------------------
adatas = []
for sid in sample_ids:
    fn = os.path.join(data_dir, f"xenium_{sid}_scvi.h5ad")
    a = sc.read_h5ad(fn)
    a = donw_sample_adata(a,10)

    # ensure sample_id exists and obs_names are globally unique
    if "sample_id" not in a.obs:
        a.obs["sample_id"] = sid
    a.obs_names_make_unique()
    # (optional) prefix with sample to guarantee global uniqueness
    a.obs_names = [f"{sid}:{x}" for x in a.obs_names]
    adatas.append(a)

adata = ad.concat(adatas, merge="same")   # keep your own sample_id strings
adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)

# Ensure latent is float32 (faster, same neighbors)
adata.obsm[use_rep] = np.asarray(adata.obsm[use_rep], dtype=np.float32)

# Build global gene graph from X_scvi
sc.pp.neighbors(
    adata,
    use_rep=use_rep,                 # "X_scvi"
    n_neighbors=n_neighbors_scvi,    # e.g., 15
    method="umap",                   # default; explicit for clarity
    random_state=0,
)


A_trx = adata.obsp["connectivities"].tocsr().astype(np.float32)

# 3) Block-diagonal spatial kNN per-sample (cKDTree is very fast in 2D and multi-threaded)
N = adata.n_obs
rows_all, cols_all = [], []
sample_col = adata.obs["sample_id"].values.astype(str)

for sid in sorted(pd.unique(sample_col)):
    idx = np.where(sample_col == sid)[0]
    if idx.size == 0:
        continue
    coords = adata.obsm["spatial"][idx, :2].astype(np.float32)

    k = min(n_neighbors_spatial, max(1, coords.shape[0]))
    tree = cKDTree(coords)
    # cKDTree is multi-threaded with workers=-1
    _, nbrs = tree.query(coords, k=k, workers=-1)
    if k == 1:
        nbrs = nbrs[:, None]

    r = np.repeat(idx[:, None], k, axis=1).ravel()
    c = idx[nbrs.ravel()]
    rows_all.append(r)
    cols_all.append(c)

rows = np.concatenate(rows_all) if rows_all else np.empty(0, dtype=int)
cols = np.concatenate(cols_all) if cols_all else np.empty(0, dtype=int)
data = np.ones(rows.size, dtype=np.float32)

A_sp = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
# Fast symmetric binarize
A_sp = A_sp.maximum(A_sp.T)       # <- faster than (A + A.T) > 0 + astype

# 4) Mix graphs
A_mix = (alpha * A_trx) + ((1.0 - alpha) * A_sp)
A_mix = A_mix.tocsr()

# 5) Leiden (igraph is fast; no accuracy change)
sc.tl.leiden(
    adata,
    adjacency=A_mix,
    resolution=resolution,
    key_added=leiden_key,
    flavor="igraph",
    n_iterations=10,
    random_state=random_state,
)
adata.obs[leiden_key] = adata.obs[leiden_key].astype("category")

out_dir = os.path.join(data_dir, "joint_domains")
os.makedirs(out_dir, exist_ok=True)
# visualize_domains(save_dir=out_dir)
visualize_domains()

# -----------------------------
# Save per-sample (optional)
# -----------------------------


for sid in sample_ids:
    sub = adata[adata.obs["sample_id"] == sid].copy()
    sub.write_h5ad(os.path.join(out_dir, f"xenium_{sid}_domains_shared.h5ad"))
print("Saved per-sample files with shared domain IDs.")
