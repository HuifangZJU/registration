
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from scipy.ndimage import rotate, shift, map_coordinates
#
#
# # ---------- helpers ----------
#
#
# def to_dark_half(values):
#     # normalize to [0, 1]
#     v = (values - values.min()) / (values.max() - values.min() + 1e-8)
#     # push into [0.4, 1.0] range => darker colors
#     return 0.8 + 0.2 * v
#
# def sample_spots_from_alpha(alpha, step=40):
#     """
#     Generate regular grid spots and keep only those with centers inside alpha>0.
#     Returns spot_x, spot_y and fake read counts.
#     """
#     H, W = alpha.shape
#     yy, xx = np.mgrid[0:H:step, 0:W:step]  # yy=row, xx=col
#     coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2) -> (x, y)
#
#     alpha_vals = alpha[coords[:, 1], coords[:, 0]]
#     inside = alpha_vals > 0
#
#     spot_x = coords[inside, 0]
#     spot_y = coords[inside, 1]
#
#     reads = np.random.gamma(shape=2.0, scale=5.0, size=spot_x.shape[0])
#     return spot_x, spot_y, reads
#
#
# def deform_alpha(alpha, angle=10, shift_xy=(10, -15),
#                  deform_amplitude=8, deform_period=220):
#     """
#     Apply rotation, translation, and a simple sinusoidal (non-rigid) warp
#     to the alpha mask to mimic slice/sample differences.
#     """
#     # 1) R + T (rotate / shift)
#     alpha_rt = rotate(alpha, angle=angle, reshape=False, order=1,
#                       mode="constant", cval=0.0)
#     alpha_rt = shift(alpha_rt, shift=shift_xy, order=1,
#                      mode="constant", cval=0.0)
#
#     # 2) Simple smooth deformable warp (horizontal sinusoidal displacement)
#     H, W = alpha_rt.shape
#     yy, xx = np.indices((H, W))
#     disp_x = deform_amplitude * np.sin(2 * np.pi * yy / deform_period)
#     xx_warp = np.clip(xx + disp_x, 0, W - 1)
#
#     alpha_def = map_coordinates(
#         alpha_rt,
#         [yy, xx_warp],
#         order=1,
#         mode="constant",
#         cval=0.0
#     )
#
#     return alpha_def
#
#
# # ---------- main ----------
#
# # 0) Load mask and extract alpha channel as liver mask
# img = Image.open("mask.png").convert("RGBA")
# img = img.crop([80,80,950,950])
# rgba = np.array(img)
# alpha = rgba[..., 3].astype(float)  # (H, W)
# H, W = alpha.shape
#
# step = 40  # grid spacing for spots
#
# # 1) Slice 1 (original mask)
# spot_x1, spot_y1, reads1 = sample_spots_from_alpha(alpha, step=step)
#
# # 2) Slice 2 (R/T + deformable mask)
# alpha2 = deform_alpha(alpha, angle=12, shift_xy=(15, -20),
#                       deform_amplitude=10, deform_period=240)
# spot_x2, spot_y2, reads2 = sample_spots_from_alpha(alpha2, step=step)
#
# reads1 = to_dark_half(reads1)
# reads2 = to_dark_half(reads2)
#
# # 3) Griddized common space on original mask
# #    Make grid centers and keep those inside the original mask
# gy, gx = np.mgrid[step//2:H:step, step//2:W:step]  # centers
# mask_grid = alpha[gy, gx] > 0
# grid_x = gx[mask_grid]
# grid_y = gy[mask_grid]
#
# # Fake "aligned" readouts for both slices at each grid cell
# reads1_grid = np.random.gamma(shape=2.0, scale=1.0, size=grid_x.shape[0])
# reads2_grid = np.random.gamma(shape=2.0, scale=1.0, size=grid_x.shape[0])
#
# # Small offset so two dots are visible inside each grid cell
# dx = step * 0.2
#
# # ---------- plotting ----------
# c_fixed ='Oranges'
# c_moving='Blues'
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# for ax in axes:
#     ax.set_aspect("equal")
#     ax.axis("off")
#
# # --- Panel 1: Slice 1 ---
# axes[0].imshow(alpha > 0, cmap="gray", alpha=0.15, origin="upper")
# axes[0].scatter(
#     spot_x1,
#     spot_y1,
#     c=reads1,
#     s=70,
#     cmap=c_fixed,        # colormap for slice 1
#     edgecolor="black",
#     linewidth=0.4,
#     alpha=0.95,
# )
# axes[0].set_title("Slice 1 (original liver mask)")
#
# # --- Panel 2: Slice 2 ---
# axes[1].imshow(alpha2 > 0, cmap="gray", alpha=0.15, origin="upper")
# axes[1].scatter(
#     spot_x2,
#     spot_y2,
#     c=reads2,
#     s=70,
#     cmap=c_moving,       # different colormap for slice 2
#     edgecolor="black",
#     linewidth=0.4,
#     alpha=0.95,
# )
# axes[1].set_title("Slice 2 (R/T + deformable liver)")
#
# # --- Panel 3: Griddized common space on original mask ---
# ax3 = axes[2]
# # light liver background
# ax3.imshow(alpha > 0, cmap="gray", alpha=0.1, origin="upper")
#
# # draw more obvious grid lines
# for x in np.arange(0, W + 1, step):
#     ax3.axvline(x, linestyle="-", linewidth=0.8, alpha=0.5, color="black")
# for y in np.arange(0, H + 1, step):
#     ax3.axhline(y, linestyle="-", linewidth=0.8, alpha=0.5, color="black")
#
# # diagonal offsets so dots are not crowded
# dx = step * 0.2
# dy = step * 0.2
#
# # points for slice 1 (upper-left in each cell)
# ax3.scatter(
#     grid_x - dx,
#     grid_y - dy,
#     c=reads1_grid,
#     s=40,
#     cmap=c_fixed,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.9,
#     label="Slice 1",
# )
#
# # points for slice 2 (lower-right in each cell)
# ax3.scatter(
#     grid_x + dx,
#     grid_y + dy,
#     c=reads2_grid,
#     s=40,
#     cmap=c_moving,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.9,
#     label="Slice 2",
# )
#
# ax3.set_title("Griddized common space (two diagonal dots per cell)")
# ax3.legend(loc="lower right", fontsize=8, frameon=False)
#
# plt.tight_layout()
# plt.show()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, map_coordinates, gaussian_filter

# ---------- helpers ----------

def sample_spots_from_alpha(alpha, step=40):
    """
    Generate regular grid spots and keep only those with centers inside alpha>0.
    Returns spot_x, spot_y and fake read counts.
    """
    H, W = alpha.shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]  # yy=row, xx=col
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2) -> (x, y)

    alpha_vals = alpha[coords[:, 1], coords[:, 0]]
    inside = alpha_vals > 0

    spot_x = coords[inside, 0]
    spot_y = coords[inside, 1]

    reads = np.random.gamma(shape=2.0, scale=1.0, size=spot_x.shape[0])
    return spot_x, spot_y, reads


def deform_alpha(alpha, angle=10, shift_xy=(10, -15),
                 deform_amplitude=8, deform_period=220):
    """
    Apply rotation, translation, and a simple sinusoidal (non-rigid) warp
    to the alpha mask to mimic slice/sample differences.
    """
    # 1) R + T (rotate / shift)
    alpha_rt = rotate(alpha, angle=angle, reshape=False, order=1,
                      mode="constant", cval=0.0)
    alpha_rt = shift(alpha_rt, shift=shift_xy, order=1,
                     mode="constant", cval=0.0)

    # 2) Simple smooth deformable warp (horizontal sinusoidal displacement)
    H, W = alpha_rt.shape
    yy, xx = np.indices((H, W))
    disp_x = deform_amplitude * np.sin(2 * np.pi * yy / deform_period)
    xx_warp = np.clip(xx + disp_x, 0, W - 1)

    alpha_def = map_coordinates(
        alpha_rt,
        [yy, xx_warp],
        order=1,
        mode="constant",
        cval=0.0
    )

    return alpha_def


def to_dark_half(values, low=0.4):
    """
    Normalize to [0,1], then push into [low, 1] to avoid very pale colors.
    """
    v = (values - values.min()) / (values.max() - values.min() + 1e-8)
    return low + (1.0 - low) * v


# ---------- main ----------

# 0) Load mask and extract alpha channel as liver mask
img = Image.open("mask.png").convert("RGBA")
img = img.crop([80,100,910,900])
rgba = np.array(img)
alpha = rgba[..., 3].astype(float)  # (H, W)
H, W = alpha.shape

step = 40  # grid spacing for spots

# 1) Slice 1 (original mask)
spot_x1, spot_y1, reads1 = sample_spots_from_alpha(alpha, step=step)
reads1_dark = to_dark_half(reads1, low=0.5)

# 2) Slice 2 (R/T + deformable mask)
alpha2 = deform_alpha(alpha, angle=12, shift_xy=(15, -20),
                      deform_amplitude=10, deform_period=240)
spot_x2, spot_y2, reads2 = sample_spots_from_alpha(alpha2, step=step)
reads2_dark = to_dark_half(reads2, low=0.5)

# 3) Griddized common space on original mask
gy, gx = np.mgrid[step//2:H:step, step//2:W:step]  # grid cell centers
mask_grid = alpha[gy, gx] > 0
grid_x = gx[mask_grid]
grid_y = gy[mask_grid]

reads1_grid = np.random.gamma(shape=2.0, scale=1.0, size=grid_x.shape[0])
reads2_grid = np.random.gamma(shape=2.0, scale=1.0, size=grid_x.shape[0])
reads1_grid_dark = to_dark_half(reads1_grid, low=0.5)
reads2_grid_dark = to_dark_half(reads2_grid, low=0.5)

# small diagonal offset so two dots are not crowded
dx = step * 0.2
dy = step * 0.2

# 4) Imputed continuous map based on the grid values
#    Put mean of the two slices at each grid cell into full-res array, then smooth.
imputed_full = np.zeros((H, W), dtype=float)
imputed_vals = 0.5 * (reads1_grid_dark + reads2_grid_dark)
imputed_full[grid_y, grid_x] = imputed_vals

# Smooth to get a continuous map (imputation cartoon)
sigma = step / 2.0  # controls smoothness
imputed_smooth = gaussian_filter(imputed_full, sigma=sigma)

# ---------- plotting (4 panels) ----------

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

# --- Panel 1: Slice 1 ---
axes[0].imshow(alpha > 0, cmap="gray", alpha=0.08, origin="upper")
axes[0].scatter(
    spot_x1,
    spot_y1,
    c=reads1_dark,
    s=70,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    edgecolor="black",
    linewidth=0.4,
    alpha=1.0,
)
axes[0].set_title("Slice 1 (original liver mask)")

# --- Panel 2: Slice 2 ---
axes[1].imshow(alpha2 > 0, cmap="gray", alpha=0.08, origin="upper")
axes[1].scatter(
    spot_x2,
    spot_y2,
    c=reads2_dark,
    s=70,
    cmap="Oranges",
    vmin=0.0,
    vmax=1.0,
    edgecolor="black",
    linewidth=0.4,
    alpha=1.0,
)
axes[1].set_title("Slice 2 (R/T + deformable liver)")

# # --- Panel 3: Griddized common space (two diagonal dots per cell) ---
# ax3 = axes[2]
# ax3.imshow(alpha > 0, cmap="gray", alpha=0.1, origin="upper")
#
# # more obvious grid lines
# for x in np.arange(0, W + 1, step):
#     ax3.axvline(x, linestyle="-", linewidth=0.8, alpha=0.5, color="black")
# for y in np.arange(0, H + 1, step):
#     ax3.axhline(y, linestyle="-", linewidth=0.8, alpha=0.5, color="black")
#
# # dots for slice 1 (upper-left)
# ax3.scatter(
#     grid_x - dx,
#     grid_y - dy,
#     c=reads1_grid_dark,
#     s=40,
#     cmap="Blues",
#     vmin=0.0,
#     vmax=1.0,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.9,
#     label="Slice 1",
# )
#
# # dots for slice 2 (lower-right)
# ax3.scatter(
#     grid_x + dx,
#     grid_y + dy,
#     c=reads2_grid_dark,
#     s=40,
#     cmap="Oranges",
#     vmin=0.0,
#     vmax=1.0,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.9,
#     label="Slice 2",
# )
#
# ax3.set_title("Griddized common space\n(two diagonal dots per cell)")
# ax3.legend(loc="lower right", fontsize=8, frameon=False)













# --- Panel 3: Griddized common space with aggregated values from slice 1 & 2 ---
ax3 = axes[2]
ax3.imshow(alpha > 0, cmap="gray", alpha=0.1, origin="upper")

# draw more obvious grid lines
x_edges = np.arange(0, W + step, step)
y_edges = np.arange(0, H + step, step)
for x in x_edges:
    ax3.axvline(x, linestyle="-", linewidth=0.8, alpha=0.5, color="black")
for y in y_edges:
    ax3.axhline(y, linestyle="-", linewidth=0.8, alpha=0.5, color="black")

# grid cell centers
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

# diagonal offsets so two dots are not crowded
dx = step * 0.25
dy = step * 0.25

# containers for aggregated dots
grid_x1, grid_y1, val1 = [], [], []
grid_x2, grid_y2, val2 = [], [], []

for iy, cy in enumerate(y_centers):
    for ix, cx in enumerate(x_centers):
        # skip cells clearly outside liver (use center point)
        cy_i = int(np.clip(cy, 0, H - 1))
        cx_i = int(np.clip(cx, 0, W - 1))
        if alpha[cy_i, cx_i] == 0:
            continue

        x0, x1_edge = x_edges[ix], x_edges[ix + 1]
        y0, y1_edge = y_edges[iy], y_edges[iy + 1]

        # slice 1 spots in this cell
        mask1 = (
            (spot_x1 >= x0) & (spot_x1 < x1_edge) &
            (spot_y1 >= y0) & (spot_y1 < y1_edge)
        )
        if mask1.any():
            grid_x1.append(cx - dx)
            grid_y1.append(cy - dy)
            val1.append(reads1_dark[mask1].mean())

        # slice 2 spots in this cell
        mask2 = (
            (spot_x2 >= x0) & (spot_x2 < x1_edge) &
            (spot_y2 >= y0) & (spot_y2 < y1_edge)
        )
        if mask2.any():
            grid_x2.append(cx + dx)
            grid_y2.append(cy + dy)
            val2.append(reads2_dark[mask2].mean())

grid_x1 = np.array(grid_x1)
grid_y1 = np.array(grid_y1)
val1 = np.array(val1)

grid_x2 = np.array(grid_x2)
grid_y2 = np.array(grid_y2)
val2 = np.array(val2)

# slice 1 dots (upper-left of cell)
ax3.scatter(
    grid_x1,
    grid_y1,
    c=val1,
    s=55,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    label="Slice 1",
)

# slice 2 dots (lower-right of cell)
ax3.scatter(
    grid_x2,
    grid_y2,
    c=val2,
    s=55,
    cmap="Oranges",
    vmin=0.0,
    vmax=1.0,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    label="Slice 2",
)

ax3.set_title("Griddized common space\n(aggregated from warped slices)")
ax3.legend(loc="lower right", fontsize=8, frameon=False)





















#
# # --- Panel 4: Imputed continuous map ---
# ax4 = axes[3]
# im = ax4.imshow(
#     imputed_smooth,
#     cmap="magma",
#     origin="upper"
# )
# # overlay liver boundary lightly
# ax4.imshow(alpha > 0, cmap="gray", alpha=0.12, origin="upper")
# ax4.set_title("Imputed continuous map")
#
# # optional colorbar for the imputed intensity
# cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
# cbar.ax.set_ylabel("Imputed value", fontsize=8)

# --- build max field & dominance at grid cells ---
# max value at each grid cell (slice1 or slice2)
max_vals = np.maximum(reads1_grid_dark, reads2_grid_dark)

# which slice is larger at each grid cell: 1 if slice1 >= slice2 else 0
dom1 = (reads1_grid_dark >= reads2_grid_dark).astype(float)

imputed_max_full = np.zeros((H, W), dtype=float)
dom1_full = np.zeros((H, W), dtype=float)

imputed_max_full[grid_y, grid_x] = max_vals
dom1_full[grid_y, grid_x] = dom1

# --- smooth to get dense / imputed fields ---
sigma = step / 2.0  # same as before, controls smoothness
imputed_smooth = gaussian_filter(imputed_max_full, sigma=sigma)
dom1_smooth = gaussian_filter(dom1_full, sigma=sigma)

# --- normalize values within the liver only ---
mask_liver = alpha > 0
vals = imputed_smooth.copy()
inside = mask_liver

if inside.sum() > 0:
    mn = vals[inside].min()
    mx = vals[inside].max()
    vals[inside] = (vals[inside] - mn) / (mx - mn + 1e-8)
    vals[~inside] = 0.0
else:
    vals[:] = 0.0

# --- convert to a color image using Blues / Oranges, weighted by dominance ---
cmapB = plt.get_cmap("Blues")    # slice 1
cmapO = plt.get_cmap("Oranges")  # slice 2

rgbaB = cmapB(vals)  # (H, W, 4)
rgbaO = cmapO(vals)

# dom1_smooth ~ dominance of slice 1
dom = np.clip(dom1_smooth, 0.0, 1.0)

# BIAS TOWARD BLUE:
# even when dom ~ 0, we still keep some blue; when dom ~ 1, it's almost pure blue
blue_bias = 0.4   # increase this (e.g. 0.4) for more overall blue
w1 = blue_bias + (1.0 - blue_bias) * dom   # weight for blue (slice 1)
w1 = np.clip(w1, 0.0, 1.0)
w2 = 1.0 - w1                                # weight for orange (slice 2)

w1 = w1[..., None]  # broadcast over RGBA channels
w2 = w2[..., None]

# combined = w1 * rgbaB + w2 * rgbaO
combined = w1 * rgbaB
# make outside-liver transparent
combined[..., 3] = np.where(mask_liver, 1.0, 0.0)

ax4 = axes[3]
ax4.imshow(combined, origin="upper")
ax4.set_title("Imputed continuous map\n(blue-biased max per grid)")
ax4.axis("off")

plt.tight_layout()
plt.show()







# --- define grid (same as in plot 3) ---
# step = int(step/2)
x_edges = np.arange(0, W + step, step)
y_edges = np.arange(0, H + step, step)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

n_y = len(y_centers)
n_x = len(x_centers)

# pixelized value arrays (grid space)
grid1 = np.full((n_y, n_x), np.nan)  # slice 1
grid2 = np.full((n_y, n_x), np.nan)  # slice 2

# --- fill grid with mean value of spots in each cell ---
for iy, (y0, y1_edge) in enumerate(zip(y_edges[:-1], y_edges[1:])):
    for ix, (x0, x1_edge) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        cy = int(np.clip(y_centers[iy], 0, H - 1))
        cx = int(np.clip(x_centers[ix], 0, W - 1))

        # skip cells clearly outside liver (optional)
        if alpha[cy, cx] == 0:
            continue

        # slice 1 spots in this cell
        mask1 = (
            (spot_x1 >= x0) & (spot_x1 < x1_edge) &
            (spot_y1 >= y0) & (spot_y1 < y1_edge)
        )
        if mask1.any():
            grid1[iy, ix] = reads1_dark[mask1].mean()

        # slice 2 spots in this cell
        mask2 = (
            (spot_x2 >= x0) & (spot_x2 < x1_edge) &
            (spot_y2 >= y0) & (spot_y2 < y1_edge)
        )
        if mask2.any():
            grid2[iy, ix] = reads2_dark[mask2].mean()

# --- set outside-liver cells to NaN so they plot as empty ---
for iy in range(n_y):
    for ix in range(n_x):
        cy = int(np.clip(y_centers[iy], 0, H - 1))
        cx = int(np.clip(x_centers[ix], 0, W - 1))
        if alpha[cy, cx] == 0:
            grid1[iy, ix] = np.nan
            grid2[iy, ix] = np.nan

# --- make colormaps show NaN as white ---
cmap_blues = plt.get_cmap("Blues").copy()
cmap_oranges = plt.get_cmap("Oranges").copy()
cmap_blues.set_bad(color="white")
cmap_oranges.set_bad(color="white")

# normalize both grids to [0,1] for nicer contrast (optional)
def norm_grid(g):
    g2 = g.copy()
    valid = ~np.isnan(g2)
    if valid.sum() > 0:
        mn = g2[valid].min()
        mx = g2[valid].max()
        g2[valid] = (g2[valid] - mn) / (mx - mn + 1e-8)
    return g2

grid1_n = norm_grid(grid1)
grid2_n = norm_grid(grid2)

## --- plot pixelized maps (slice 1 & warped slice 2) ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

n_y, n_x = grid1_n.shape

# These edges are used BOTH by imshow (via extent) and for grid lines,
# so the colored blocks and lines align perfectly.
x_edges_plot = np.linspace(0, W, n_x + 1)
y_edges_plot = np.linspace(0, H, n_y + 1)

# ----- Slice 1 pixelized -----
im1 = axes[0].imshow(
    grid1_n,
    origin="upper",
    cmap=cmap_blues,
    interpolation="nearest",
    extent=(0, W, H, 0),  # full image coordinates
)

# grid lines aligned with imshow cells
for x in x_edges_plot:
    axes[0].axvline(x, color="black", linewidth=0.5, alpha=0.6)
for y in y_edges_plot:
    axes[0].axhline(y, color="black", linewidth=0.5, alpha=0.6)

axes[0].set_title("Slice 1 (pixelized grid)")

# ----- Warped slice 2 pixelized -----
im2 = axes[1].imshow(
    grid2_n,
    origin="upper",
    cmap=cmap_oranges,
    interpolation="nearest",
    extent=(0, W, H, 0),
)

for x in x_edges_plot:
    axes[1].axvline(x, color="black", linewidth=0.5, alpha=0.6)
for y in y_edges_plot:
    axes[1].axhline(y, color="black", linewidth=0.5, alpha=0.6)

axes[1].set_title("Warped slice 2 (pixelized grid)")

plt.tight_layout()
plt.show()
