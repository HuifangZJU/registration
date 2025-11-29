
import scanpy as sc
import matplotlib.pyplot as plt
import os
from matplotlib import patches as mpatches


sc.settings.n_jobs = 16
os.environ["OMP_NUM_THREADS"]  = "16"
os.environ["MKL_NUM_THREADS"]  = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
sc.settings.n_jobs = 16

import json
import numpy as np
import pandas as pd
from matplotlib.path import Path
from anndata import AnnData

# ----------------------------
# Helpers
# ----------------------------
def _build_polygons_from_labelme(json_path):
    """
    Parse a LabelMe JSON into polygon Paths grouped by clusters and markers.
    Expected labels: cluster1..cluster4, marker1..marker8 (case-insensitive).
    Returns:
        {
          'cluster': [{'name':'cluster1','id':1,'path':Path}, ...],
          'marker' : [{'name':'marker1', 'id':1,'path':Path}, ...]
        }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    polygons = {"cluster": [], "marker": []}

    for sh in shapes:
        label = sh.get("label", "").strip()
        pts = sh.get("points", None)
        if not label or not pts:
            continue

        lname = label.lower()
        if lname.startswith("cluster"):
            try:
                idx = int(lname.replace("cluster", ""))
            except ValueError:
                continue  # skip non-standard labels
            # if not (1 <= idx <= 4):
            #     continue
            path = Path(np.asarray(pts, dtype=float))
            polygons["cluster"].append({"name": f"cluster{idx}", "id": idx, "path": path})

        elif lname.startswith("marker"):
            try:
                idx = int(lname.replace("marker", ""))
            except ValueError:
                continue
            # if not (1 <= idx <= 8):
            #     continue
            path = Path(np.asarray(pts, dtype=float))
            polygons["marker"].append({"name": f"marker{idx}", "id": idx, "path": path})

    # Sort by numeric id to make assignment order deterministic
    polygons["cluster"].sort(key=lambda d: d["id"])
    polygons["marker"].sort(key=lambda d: d["id"])
    return polygons


def _get_cell_points(adata: AnnData, x_key="centroid_x", y_key="centroid_y"):
    """
    Return Nx2 array of (x,y) for cells.
    Prefers obs[x_key], obs[y_key]. Falls back to obsm['spatial'] if needed.
    """
    if x_key in adata.obs and y_key in adata.obs:
        xs = pd.to_numeric(adata.obs[x_key], errors="coerce").to_numpy()
        ys = pd.to_numeric(adata.obs[y_key], errors="coerce").to_numpy()
        pts = np.column_stack([xs, ys]).astype(float)
    elif "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        pts = np.asarray(adata.obsm["spatial"][:, :2], dtype=float)
    else:
        raise ValueError(
            f"Could not find '{x_key}/{y_key}' in .obs or a 2D 'spatial' array in .obsm."
        )

    # Optional: sanity check for NaNs
    if np.isnan(pts).any():
        raise ValueError("NaNs found in cell coordinates; please clean centroids/spatial first.")
    return pts


def _assign_by_polygons(points_xy: np.ndarray, polygons_list):
    """
    Assign a single integer ID and label from a list of polygons (all same type),
    returning (ids, labels). Background=0 / 'none' if not in any polygon.

    If a point falls in multiple polygons, the first matching polygon in
    polygons_list order wins (which is sorted by id).
    """
    n = points_xy.shape[0]
    ids = np.zeros(n, dtype=np.int16)              # 0 = background
    labels = np.array(["none"] * n, dtype=object)  # 'none' for background

    # Vectorized containment per polygon; first match wins
    assigned = np.zeros(n, dtype=bool)
    for poly in polygons_list:
        inside = poly["path"].contains_points(points_xy)
        # Only assign those not already assigned
        mask = inside & (~assigned)
        if np.any(mask):
            ids[mask] = poly["id"]
            labels[mask] = poly["name"]
            assigned[mask] = True
    return ids, labels


# ----------------------------
# Main entry point
# ----------------------------
def annotate_from_labelme(
    adata: AnnData,
    labelme_json_path: str,
    x_key: str = "centroid_x",
    y_key: str = "centroid_y",
    cluster_obs_prefix: str = "cluster",
    marker_obs_prefix: str = "marker",
):
    """
    Reads LabelMe polygons and annotates AnnData.obs with:
        - f"{cluster_obs_prefix}_id" (int16, 0=none, else 1..4)
        - f"{cluster_obs_prefix}_label" ('none' or 'cluster#')
        - f"{marker_obs_prefix}_id"  (int16, 0=none, else 1..8)
        - f"{marker_obs_prefix}_label" ('none' or 'marker#')
    Also stores provenance in adata.uns['labelme_annotation_meta'].
    """
    polygons = _build_polygons_from_labelme(labelme_json_path)
    points_xy = _get_cell_points(adata, x_key=x_key, y_key=y_key)

    # Assign clusters
    cluster_ids, cluster_labels = _assign_by_polygons(points_xy, polygons["cluster"])
    # Assign markers
    marker_ids, marker_labels = _assign_by_polygons(points_xy, polygons["marker"])

    # Add to obs
    adata.obs[f"{cluster_obs_prefix}_id"] = pd.Categorical(cluster_ids)
    adata.obs[f"{cluster_obs_prefix}_label"] = pd.Categorical(cluster_labels)
    adata.obs[f"{marker_obs_prefix}_id"] = pd.Categorical(marker_ids)
    adata.obs[f"{marker_obs_prefix}_label"] = pd.Categorical(marker_labels)

    # Provenance
    adata.uns.setdefault("labelme_annotation_meta", {})
    adata.uns["labelme_annotation_meta"]["source_json"] = labelme_json_path
    adata.uns["labelme_annotation_meta"]["cluster_map"] = {
        p["name"]: p["id"] for p in polygons["cluster"]
    }
    adata.uns["labelme_annotation_meta"]["marker_map"] = {
        p["name"]: p["id"] for p in polygons["marker"]
    }
    adata.uns["labelme_annotation_meta"]["background_id"] = 0

    return adata


def donwsample_adata(adata,dsrate):
    # assume your object is named `adata`
    frac = 1 / dsrate  # downsample ratio
    n_sub = max(1, int(adata.n_obs * frac))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # reproducible sampling
    rng = np.random.default_rng(seed=0)
    subset_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)

    adata_sub = adata[subset_idx].copy()
    return adata_sub


def _xy_from_adata(adata, x_key="centroid_x", y_key="centroid_y"):
    if x_key in adata.obs and y_key in adata.obs:
        xs = pd.to_numeric(adata.obs[x_key], errors="coerce").to_numpy()
        ys = pd.to_numeric(adata.obs[y_key], errors="coerce").to_numpy()
        xy = np.column_stack([xs, ys]).astype(float)
    elif "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        xy = np.asarray(adata.obsm["spatial"][:, :2], dtype=float)
    else:
        raise ValueError("Need obs['centroid_x/y'] or obsm['spatial'] with 2 columns.")
    if np.isnan(xy).any():
        raise ValueError("NaNs in coordinates. Clean them before plotting.")
    return xy

def _ensure_int_labels(arr):
    """Convert obs column (possibly categorical/str) to int np.ndarray."""
    if pd.api.types.is_categorical_dtype(arr):
        arr = arr.astype(str)
    if isinstance(arr, pd.Series):
        arr = arr.values
    # Try direct int conversion; if fails (e.g., 'cluster1'), extract trailing digits
    try:
        return pd.to_numeric(arr, errors="raise").astype(int)
    except Exception:
        ids = []
        for a in arr:
            if a is None or a == "none":
                ids.append(0)
            else:
                s = str(a)
                digits = "".join([c for c in s if c.isdigit()])
                ids.append(int(digits) if digits else 0)
        return np.asarray(ids, dtype=int)

def _build_palette(id_values, kind="cluster"):
    """
    Map integer IDs -> colors with 0 as light gray.
    kind in {'cluster','marker'} just to pick different distinct palettes if desired.
    """
    uniq = sorted(set(int(i) for i in id_values if i is not None))
    max_id = max([u for u in uniq if u >= 0], default=0)

    # Choose base colors (avoid too-similar hues)
    if kind == "cluster":
        base = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    else:  # marker: more slots
        base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Background 0
    palette = {0: "#D3D3D3"}  # light gray
    color_idx = 0
    for k in range(1, max_id + 1):
        palette[k] = base[color_idx % len(base)]
        color_idx += 1
    return palette

def _make_legend_handles(palette, prefix):
    handles = []
    # Always put background first
    if 0 in palette:
        handles.append(mpatches.Patch(color=palette[0], label=f"{prefix} 0 (none)"))
    for k in sorted(k for k in palette.keys() if k != 0):
        handles.append(mpatches.Patch(color=palette[k], label=f"{prefix} {k}"))
    return handles

def _plot_one(ax, xy, ids, palette, title=None, s_bg=1.0, s_fg=2.0, invert_y=True):
    # background first
    bg = (ids == 0)
    if np.any(bg):
        ax.scatter(xy[bg, 0], xy[bg, 1], s=s_bg, c=palette[0], lw=0, alpha=0.35)
    # then each positive id
    for k in sorted(set(ids) - {0}):
        m = (ids == k)
        if np.any(m):
            ax.scatter(xy[m, 0], xy[m, 1], s=s_fg, c=palette.get(k, "#000000"), lw=0)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)

# ----------------------------
# Main comparison plot
# ----------------------------
def compare_two_adatas_spatial(
    adatas,
    names=("A", "B"),
    cluster_key="cluster_id",
    marker_key="marker_id",
    x_key="centroid_x",
    y_key="centroid_y",
    point_sizes=(0.8, 1.6),  # (bg, fg)
    invert_y=True,
    figsize=(12, 10),
):
    assert len(adatas) == 2 and len(names) == 2, "Provide exactly two adatas and two names."

    # Prepare data
    xy_list = [_xy_from_adata(ad, x_key=x_key, y_key=y_key) for ad in adatas]
    cid_list = [_ensure_int_labels(ad.obs[cluster_key]) for ad in adatas]
    mid_list = [_ensure_int_labels(ad.obs[marker_key]) for ad in adatas]

    # Shared palettes across both samples
    cluster_palette = _build_palette(np.concatenate(cid_list), kind="cluster")
    marker_palette  = _build_palette(np.concatenate(mid_list), kind="marker")

    # Figure layout: 2 rows (cluster, marker) × 2 cols (sample1, sample2)
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    # Top row: clusters
    _plot_one(axs[0, 0], xy_list[0], cid_list[0], cluster_palette, title=f"{names[0]} — {cluster_key}", s_bg=point_sizes[0], s_fg=point_sizes[1], invert_y=invert_y)
    _plot_one(axs[0, 1], xy_list[1], cid_list[1], cluster_palette, title=f"{names[1]} — {cluster_key}", s_bg=point_sizes[0], s_fg=point_sizes[1], invert_y=invert_y)

    # Bottom row: markers
    _plot_one(axs[1, 0], xy_list[0], mid_list[0], marker_palette, title=f"{names[0]} — {marker_key}", s_bg=point_sizes[0], s_fg=point_sizes[1], invert_y=invert_y)
    _plot_one(axs[1, 1], xy_list[1], mid_list[1], marker_palette, title=f"{names[1]} — {marker_key}", s_bg=point_sizes[0], s_fg=point_sizes[1], invert_y=invert_y)

    # Legends on the right
    cluster_legend = _make_legend_handles(cluster_palette, prefix="cluster")
    marker_legend  = _make_legend_handles(marker_palette,  prefix="marker")

    # Place legends outside axes (right side)
    axs[0, 1].legend(handles=cluster_legend, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Clusters")
    axs[1, 1].legend(handles=marker_legend,  loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Markers")

    fig.suptitle("Spatial comparison of clusters and markers", fontsize=13)
    return fig, axs
# -----------------------------
# Config
# -----------------------------
data_dir = "/media/huifang/data/registration/xenium/cell_typing"
sample_ids =["ILC", "ILC_addon"]
folder_names=["Xenium_V1_FFPE_Human_Breast_ILC","Xenium_V1_FFPE_Human_Breast_ILC_With_Addon"]
# -----------------------------
# Load & concat
# -----------------------------
adatas=[]
out_dir = os.path.join(data_dir, "joint_domains")
os.makedirs(out_dir, exist_ok=True)
for sid,fid in zip(sample_ids,folder_names):
    fn = os.path.join(data_dir, f"xenium_{sid}_scvi.h5ad")
    adata = sc.read_h5ad(fn)
    adata.obs['centroid_x'] = adata.obs['centroid_x']/10
    adata.obs['centroid_y'] = adata.obs['centroid_y'] / 10
    adata.obsm['spatial'] = adata.obsm['spatial'] / 10

    # adata = donwsample_adata(adata)
    json_path = f"/media/huifang/data/Xenium/xenium_data/{fid}/morphology_down10x.json"
    adata = annotate_from_labelme(adata, json_path)
    adatas.append(adata)
    adata.write_h5ad(os.path.join(out_dir, f"xenium_{sid}_domains_shared.h5ad"))

fig, axs = compare_two_adatas_spatial(
    adatas,
    names=sample_ids,              # ("ILC", "ILC_addon")
    cluster_key="cluster_id",
    marker_key="marker_id",
    x_key="centroid_x",            # or set to 'spatial' via the helper if needed
    y_key="centroid_y",
    point_sizes=(0.6, 1.2),        # tweak for your downsampled density
    invert_y=True,                 # LabelMe/imagery is top-left origin
    figsize=(12, 9),
)
plt.show()




