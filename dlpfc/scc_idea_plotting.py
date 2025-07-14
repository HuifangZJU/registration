import cv2
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
### ----------------------- FILE LOADING FUNCTION -----------------------
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pathlib
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib import cm, colors, patches
def pad_maps_to_same_shape(maps, fill=0):
    """
    Parameters
    ----------
    maps : list of 2-D ndarrays
        Your grids (may have different shapes).
    fill : scalar
        Value used to pad new cells (e.g., 0 or np.nan).

    Returns
    -------
    padded : list of 2-D ndarrays
        Same number/order as `maps`, but all with identical shape.
    target_shape : tuple
        (max_rows, max_cols) that was applied.
    """
    # 1️⃣  discover target shape (largest dims across all arrays)
    max_rows = max(m.shape[0] for m in maps)
    max_cols = max(m.shape[1] for m in maps)

    padded = []
    for m in maps:
        r, c = m.shape
        # how much free space on each side?
        top = (max_rows - r) // 2
        bottom = max_rows - r - top  # put any extra row on bottom
        left = (max_cols - c) // 2
        right = max_cols - c - left  # extra col on right

        out = np.full((max_rows, max_cols), fill, dtype=m.dtype)
        out[top: top + r, left: left + c] = m
        padded.append(out)

    return padded, max_rows, max_cols

def plot_3d_maps(feature_maps):
    rows, cols = feature_maps[0].shape
    n_layers = len(feature_maps)

    # ─── colour map that shows NaNs / 0-cells as white ───────────────
    cmap = cm.Blues.copy()
    cmap.set_bad('white')  # NaNs will appear white
    norm = colors.Normalize(vmin=0, vmax=1)

    # ─── 3-D figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    layer_gap = 10.0  # distance between successive slabs
    slab_thk = 0.01  # almost 2-D (thin in X)

    for li, fm in enumerate(feature_maps):
        # mask zeros so they plot as white
        masked = np.ma.masked_equal(fm, 0.0)
        rgba = cmap(norm(masked))  # (rows, cols, 4)

        # pick an alpha per layer ────────────────
        if li == 0:  # first feature map (solid)
            slab_alpha = 0.95
        elif li == 1:  # second feature map (make it faint)
            slab_alpha = 0.75  # ← adjust until arrows stand out
        else:  # any further layers
            slab_alpha = 0.85

        x0 = li * layer_gap  # X-offset for this slab
        for r in range(rows):
            for c in range(cols):
                # centre each square on the slab
                verts = [
                    (x0, c, r),
                    (x0, c + 1, r),
                    (x0, c + 1, r + 1),
                    (x0, c, r + 1),
                ]

                # overwrite just the alpha channel of this cell’s colour
                col = list(rgba[r, c])  # RGBA → list so we can edit
                col[3] = slab_alpha  # new transparency
                poly = Poly3DCollection([verts],
                                        facecolors=[col],
                                        edgecolors='k', linewidths=0.3)
                ax.add_collection3d(poly)

    # thickness of the slab in X (used only for centering the arrow start)
    slab_thickness = 0.01  # whatever you used for verts[0][0]

    # -------------------------------------------------------------
    # --- pick the first two layers ----------------------------------
    fm0, fm1 = feature_maps[0], feature_maps[1]  # map-0 → map-1

    # centres of every non-zero cell in map-0
    src_idx = np.argwhere(fm0 > 0)  # (Ns, 2) [row,col]
    # centres of every non-zero cell in map-1
    tgt_idx = np.argwhere(fm1 > 0)  # (Nt, 2)

    # helper for 3-D coordinates (x,y,z) of a cell centre
    def centre(layer, r, c):
        x = layer * layer_gap + slab_thickness / 2.0
        y = c + 0.5
        z = r + 0.5
        return x, y, z

    # pre-compute 2-D grid positions for distance search
    src_rc = src_idx.astype(float)  # (row,col)
    tgt_rc = tgt_idx.astype(float)

    # draw one arrow for each target cell (map-1)
    for trow, tcol in tgt_rc:
        # Euclidean distances to every source cell
        dists = np.linalg.norm(src_rc - [trow, tcol], axis=1)
        srow, scol = src_rc[dists.argmin()]  # closest source

        # start & end coordinates in 3-D
        sx, sy, sz = centre(0, srow, scol)
        ex, ey, ez = centre(1, trow, tcol)

        # arrow vector
        ax.quiver(
            sx, sy, sz,
            ex - sx, ey - sy, ez - sz,
            arrow_length_ratio=0.08,
            color='red', linewidth=0.8, alpha=0.6
        )

    # ─── axis / view tweaks ──────────────────────────────────────────
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Column')
    ax.set_zlabel('Row')
    ax.set_xticks([i * layer_gap for i in range(n_layers)])
    ax.set_yticks(np.arange(cols + 1))
    ax.set_zticks(np.arange(rows + 1))
    ax.view_init(elev=5, azim=-25)
    ax.set_box_aspect([layer_gap * n_layers, cols, rows])
    ax.set_xlim(-0.5, layer_gap * (n_layers - 1) + 0.5)
    ax.set_ylim(-0.5, cols)
    ax.set_zlim(-0.5, rows)
    ax.axis('off')

    # colour-bar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate(feature_maps))

    plt.tight_layout()
    plt.show()


def plot_maps(feature_maps, layer_gap=100, grid_color='grey', grid_alpha=0.3):
    rows, cols = feature_maps[0].shape
    n_layers   = len(feature_maps)

    rows, cols = feature_maps[0].shape
    n_layers = len(feature_maps)

    # --------- colour map (Blues, 0→white) -----------------------
    cmap = cm.Blues.copy()
    cmap.set_bad('white')
    norm = colors.Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))


    # # --------- draw correspondences (layer-0 ➜ layer-1) ---------
    # fm0, fm1 = feature_maps[0], feature_maps[1]
    # src_idx = np.argwhere(fm0 > 0)  # (row,col)
    # tgt_idx = np.argwhere(fm1 > 0)
    #
    # src_rc = src_idx.astype(float)
    # tgt_rc = tgt_idx.astype(float)
    #
    # for trow, tcol in tgt_rc:
    #     # nearest non-zero cell in layer-0
    #     dists = np.linalg.norm(src_rc - [trow, tcol], axis=1)
    #     srow, scol = src_rc[dists.argmin()]
    #
    #     # start / end (centre of cell)
    #     sx, sy = scol + 0.5, srow + 0.5
    #     ex, ey = layer_gap + tcol + 0.5, trow + 0.5
    #
    #     arrow = patches.FancyArrowPatch(
    #         (sx, sy), (ex, ey),
    #         connectionstyle='arc3,rad=-0.3',
    #         arrowstyle='-|>',  # or 'Simple,tail_width=0.4,head_width=4,head_length=6'
    #         mutation_scale=15,  # << enlarge the head (default is 10)
    #         color='red',
    #         linewidth=0.9,
    #         alpha=0.5,
    #         zorder=5  # keep it in front of the heat-map
    #     )
    #     ax.add_patch(arrow)

    # --------- draw each map side-by-side ------------------------
    for li, fm in enumerate(feature_maps):
        masked = np.ma.masked_equal(fm, 0)
        x0 = li * layer_gap  # left edge
        extent = [x0, x0 + cols, 0, rows]  # [x0,x1,y0,y1]
        ax.imshow(masked, cmap=cmap, norm=norm,
                  origin='lower', extent=extent)
        # ---- bold black frame ---------------------------------------
        rect = patches.Rectangle(
            (x0, 0),  # lower-left corner of slab
            width=cols,  # slab width  (x-direction)
            height=rows,  # slab height (y-direction)
            fill=False,  # outline only
            edgecolor='black',
            linewidth=1.0  # “boldness”; raise for thicker lines
        )
        ax.add_patch(rect)

        # --- grid lines over this slab --------------------------
        for c in range(cols + 1):
            ax.vlines(x0 + c, 0, rows,
                      colors=grid_color, linewidth=0.4, alpha=grid_alpha)
        for r in range(rows + 1):
            ax.hlines(r, x0, x0 + cols,
                      colors=grid_color, linewidth=0.4, alpha=grid_alpha)



    # --------- tidy up axes / colour-bar -------------------------
    ax.set_xlim(-0.5, layer_gap * (n_layers - 1) + cols + 0.5)
    ax.set_ylim(-0.5, rows + 1.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.concatenate(feature_maps))
    # fig.colorbar(sm, ax=ax, pad=0.02, label='Activation value')

    plt.tight_layout();
    plt.show()


def load_registration_result(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)
    image = data["img_fix"]
    if image.shape[0]>512 and image.shape[0]<1500:
        pts1 = data["pts_fix"]/2
        pts2 = data["pts_warp"]/2
        pts3 = data["pts_moving"] / 2
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    elif image.shape[0] >1500 and image.shape[0]<2500:
        pts1 = data["pts_fix"]/4
        pts2 = data["pts_warp"] / 4
        pts3 = data["pts_moving"] / 4
        return pts1,pts2,pts3,data["fixed_label"],data['moving_label']
    else:
        return data["pts_fix"], data["pts_warp"], data["pts_moving"],data["fixed_label"],data['moving_label']


def load_registration_image(folder_path):
    """Load image and spot data from a given folder."""
    data = np.load(folder_path)

    return data["img_fix"], data["img_warp"], data["img_moving"]


def get_grid_backup(coords,lower=0.2,up=0.8):
    # Choose a grid step (bin size) in the same units as coordinates
    step = 80  # e.g. 100‑unit squares

    # ------------------------------------------------------------------
    # Bin extents
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xmin -= xmin % step
    ymin -= ymin % step
    xmax += step - (xmax % step)
    ymax += step - (ymax % step)
    nx = int((xmax - xmin) / step)
    ny = int((ymax - ymin) / step)

    # Empty grid
    grid = np.zeros((ny, nx), dtype=float)

    # Occupied bins get a random value (0.3–1.0) for colour only
    rng = np.random.default_rng(0)
    for x, y in coords:
        gx = int((x - xmin) / step)
        gy = int((y - ymin) / step)

        if grid[gy, gx] == 0:
            grid[gy, gx] = rng.uniform(lower, up)
    return nx,ny,step,xmin,ymin,grid

def get_grid(coords, step=4, lower=0.3, upper=1.0):
    # ── 1️⃣  original grid ─────────────────────────────────────────
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    xmin -= xmin % step
    ymin -= ymin % step
    xmax += step - xmax % step
    ymax += step - ymax % step

    nx = int((xmax - xmin) // step)  # width  (original grid)
    ny = int((ymax - ymin) // step)  # height

    grid = np.zeros((ny, nx), dtype=float)
    rng = np.random.default_rng(0)
    for x, y in coords:
        gx = int((x - xmin) // step)
        gy = int((y - ymin) // step)
        if grid[gy, gx] == 0:
            grid[gy, gx] = rng.uniform(lower, upper)

    # # ── 2️⃣  insert gaps ───────────────────────────────────────────
    # ny, nx = 2 * ny - 1, 2 * nx - 1  # new size
    # grid_gap = np.zeros((ny, nx), dtype=float)
    # grid_gap[0::2, 0::2] = grid  # originals in even slots
    #
    # # new (smaller) bin size: each cell now covers half the old width/height
    # step = step / 2.0

    return nx, ny, step, xmin, ymin, grid

def plot_grids(nx,ny,step,xmin,ymin,grid):
    # Visualise
    fig, ax = plt.subplots(figsize=(6, 6))

    # Make NaNs for empty cells so they render as white
    grid_masked = np.ma.masked_equal(grid, 0.0)
    cmap = cm.viridis.copy()
    cmap.set_bad(color='white')

    im = ax.imshow(grid_masked, origin='lower', cmap=cm.Blues, vmin=0.3, vmax=1.0)

    # Grid lines
    for gx in range(nx + 1):
        ax.axvline(gx - 0.5, color='grey', linewidth=0.5, alpha=0.3)
    for gy in range(ny + 1):
        ax.axhline(gy - 0.5, color='grey', linewidth=0.5, alpha=0.3)

    # Real‑world tick labels
    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels((np.arange(nx) * step + xmin).astype(int), rotation=90)
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels((np.arange(ny) * step + ymin).astype(int))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    plt.tight_layout()
    plt.show()

def plot_coords(coords):
    nx,ny,step,xmin,ymin,grid = get_grid(coords)
    plot_grids(nx,ny,step,xmin,ymin,grid)
    # ------------------------------------------------------------------

def plot_imputation_coords(coords):
    nx, ny, _, _, _, grid = get_grid(coords)
    plot_imputation_grids(nx, ny, grid)

def plot_imputation_grids(nx, ny, grid):

    # ---------------------------------------------------------------
    # 2) FIND **IMPUTED** EMPTY CELLS  (≥ 1 non-empty 8-neighbour)
    neigh = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    imputed = np.zeros_like(grid, bool)

    for gy in range(ny):
        for gx in range(nx):
            if grid[gy, gx] == 0:  # empty so far
                for dx, dy in neigh:
                    nyg, nxg = gy + dy, gx + dx
                    if 0 <= nyg < ny and 0 <= nxg < nx and grid[nyg, nxg] > 0:
                        imputed[gy, gx] = True  # mark as “to colour”
                        break
    # assign random values to imputed bins
    imputed_values = np.zeros_like(grid)
    imputed_values[imputed] = np.random.uniform(0.2, 0.6, imputed.sum())

    # -------------------  plotting ------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    mask = np.ma.masked_equal(grid, 0)
    cmap = cm.viridis.copy()
    cmap.set_bad('white')
    ax.imshow(mask, origin='lower', cmap=cm.Blues, vmin=0.3, vmax=1.0)

    #   3-b  imputed empty cells (transparent overlay)
    # imputed cells with random intensity
    ax.imshow(np.ma.masked_equal(imputed_values, 0),
              origin='lower', cmap=cm.Reds, vmin=0.3, vmax=1.0, alpha=0.2)

    # light grid lines
    for g in range(nx + 1): ax.axvline(g - 0.5, color='grey', lw=.4, alpha=.3)
    for g in range(ny + 1): ax.axhline(g - 0.5, color='grey', lw=.4, alpha=.3)

    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    # def centre(ix, iy):
    #     return ix , iy   # cell-centre coords
    #
    # head_gap = 0.12  # leave 12 % of the cell size before empty-centre
    # for gy in range(ny):
    #     for gx in range(nx):
    #         if grid[gy, gx] == 0:  # empty bin
    #             cx, cy = centre(gx, gy)
    #             for dx, dy in dirs:
    #                 ngx, ngy = gx + dx, gy + dy
    #                 if 0 <= ngx < nx and 0 <= ngy < ny and grid[ngy, ngx] > 0:
    #                     sx, sy = centre(ngx, ngy)  # start at neighbour centre
    #                     dx_vec, dy_vec = cx - sx, cy - sy
    #                     ex = sx + dx_vec * (1 - head_gap)
    #                     ey = sy + dy_vec * (1 - head_gap)
    #                     ax.arrow(sx, sy, ex - sx, ey - sy,
    #                              head_width=.15, head_length=.11,
    #                              fc='red', ec='red', lw=.7, alpha=.8,
    #                              length_includes_head=True)
    #
    #                     # helper dot showing the true neighbour centre
    #                     ax.plot(sx, sy, 'k.', ms=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-.5, nx - .5)
    ax.set_ylim(-.5, ny - .5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

root_folder = "/media/huifang/data/registration/result/center_align/scc/"
for i in [9,10]:
    registration_paths = [
        f"{root_folder}/{i}_0_result.npz",
        f"{root_folder}/{i}_1_result.npz"
    ]
    coor_fixed, coor_warpped0, coor_moving0, _, _ = load_registration_result(registration_paths[0])
    _, coor_warpped1, coor_moving1, _, _ = load_registration_result(registration_paths[1])

    # coor_moving = coor_moving+50


    plot_imputation_coords(coor_fixed)
    _, _, _, _, _, grid = get_grid(coor_fixed)
    _, _, _, _, _, grid0 = get_grid(coor_warpped0)
    _, _, _, _, _, grid1 = get_grid(coor_warpped1)
    a,b, c, d,e,  grid_plus = get_grid(coor_fixed,lower=0.7,upper=1.0)
    #
    maps = [grid,grid0,grid1,grid_plus]
    padded_maps,ny,nx = pad_maps_to_same_shape(maps)
    plot_maps(padded_maps)

    # padded_maps = [padded_maps[0], padded_maps[1],padded_maps[2], 1.3*np.maximum.reduce([padded_maps[0], padded_maps[1], padded_maps[2]])]
    padded_maps = [padded_maps[0], padded_maps[1], padded_maps[2],
                   padded_maps[0]+padded_maps[1]+padded_maps[2]]
    plot_imputation_grids(nx,ny,padded_maps[0]+padded_maps[1]+padded_maps[2])
    # plot_grids(a,b,c,d,e,padded_maps[2])

    plot_maps(padded_maps)

