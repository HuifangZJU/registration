# Generate a family of **coarse blue-grid** transcriptional pseudo-image assets.
# Spec: 8x8 tiles, Blues colormap, visible grid lines, transparent background, PNG + SVG.

# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams["figure.dpi"] = 200
#
# def save_grid_image(data, path_png, path_svg, draw_lines=True, line_alpha=0.45, line_color="#003a8c", lw=0.9):
#     fig = plt.figure(figsize=(3,3))
#     ax = fig.add_axes([0,0,1,1])
#     ax.axis("off")
#     # ax.imshow(data, cmap="Blues", interpolation="nearest")
#     ax.imshow(data, cmap="Oranges", interpolation="nearest")
#     # ax.imshow(data, cmap="Purples", interpolation="nearest")
#     if draw_lines:
#         h, w = data.shape
#         for i in range(h+1):
#             ax.plot([0, w-1], [i-0.5, i-0.5], linewidth=lw, alpha=line_alpha, color=line_color)
#         for j in range(w+1):
#             ax.plot([j-0.5, j-0.5], [0, h-1], linewidth=lw, alpha=line_alpha, color=line_color)
#     plt.savefig(path_png, bbox_inches="tight", pad_inches=0, transparent=True)
#     plt.savefig(path_svg, bbox_inches="tight", pad_inches=0, transparent=True)
#     plt.close(fig)
#
# def smooth_grid(A, rounds=1):
#     A = A.copy()
#     for _ in range(rounds):
#         A = (np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1)+A)/5
#     return A
#
# paths = []
# root = '/media/huifang/data/registration/paper_figure'
# # 1–6: Random seeds (coarse variants)
# for seed in [10, 20, 30, 40, 50, 60]:
#     rng = np.random.default_rng(seed)
#     base = rng.random((8,8))**1.2
#     png = f"{root}/bluegrid_coarse_seed{seed}_o.png"
#     svg = f"{root}/bluegrid_coarse_seed{seed}_o.svg"
#     save_grid_image(base, png, svg)
#     paths += [png, svg]

# # 7: Diagonal gradient + light noise
# rng = np.random.default_rng(101)
# x = np.linspace(0,1,8); y = np.linspace(0,1,8)
# X, Y = np.meshgrid(x, y)
# G = 0.6*X + 0.4*Y + 0.12*rng.random((8,8))
# save_grid_image(G, f"{root}/bluegrid_coarse_diaggrad.png", f"{root}/bluegrid_coarse_diaggrad.svg")
#
# # 8: Center hotspot (Gaussian) + noise
# rng = np.random.default_rng(102)
# Xc, Yc = np.meshgrid(np.linspace(-1,1,8), np.linspace(-1,1,8))
# R2 = Xc**2 + Yc**2
# H = np.exp(-3*R2) + 0.25*rng.random((8,8))
# save_grid_image(smooth_grid(H,1), f"{root}/bluegrid_coarse_centerhot.png", f"{root}/bluegrid_coarse_centerhot.svg")
#
# # 9: Checker blocks (2x2 super-cells) + jitter
# rng = np.random.default_rng(103)
# blocks = np.zeros((8,8))
# for i in range(8):
#     for j in range(8):
#         blocks[i,j] = ((i//2 + j//2) % 2) * 0.8 + 0.15*rng.random()
# save_grid_image(blocks, f"{root}/bluegrid_coarse_blocks.png", f"{root}/bluegrid_coarse_blocks.svg")
#
# # 10: Row bands with jitter
# rng = np.random.default_rng(104)
# bands = np.zeros((8,8))
# for i in range(8):
#     bands[i,:] = (i%2)*0.6 + 0.25*rng.random(8)
# save_grid_image(smooth_grid(bands,1), f"{root}/bluegrid_coarse_bands.png", f"{root}/bluegrid_coarse_bands.svg")
#
# # 11: Ring annulus + noise
# rng = np.random.default_rng(105)
# cx = cy = 3.5
# ring = np.zeros((8,8))
# for i in range(8):
#     for j in range(8):
#         r = np.sqrt((i-cy)**2 + (j-cx)**2)
#         ring[i,j] = np.exp(-((r-2.0)**2)/0.6) + 0.2*rng.random()
# save_grid_image(smooth_grid(ring,1), f"{root}/bluegrid_coarse_ring.png", f"{root}/bluegrid_coarse_ring.svg")
#
# # 12: Column anisotropy (columns vary more) + jitter
# rng = np.random.default_rng(106)
# col = rng.random(8)
# A = np.tile(col, (8,1)) * 0.7 + 0.3*rng.random((8,8))
# save_grid_image(smooth_grid(A,1), f"{root}/bluegrid_coarse_aniso.png", f"{root}/bluegrid_coarse_aniso.svg")


# Fix cluster variant indexing bug and regenerate Visium & Xenium assets.

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import RegularPolygon
# from matplotlib.collections import PatchCollection
#
# plt.rcParams["figure.dpi"] = 200
# root = '/media/huifang/data/registration/paper_figure'
# def save_fig(fig, png_path, svg_path):
#     fig.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
#     fig.savefig(svg_path, bbox_inches="tight", pad_inches=0, transparent=True)
#     plt.close(fig)
#
# # VISIUM hex-grid
# def visium_hex_image(seed=0, n_cols=14, n_rows=12, jitter=0.0, missing_ratio=0.0, cmap="Blues"):
#     rng = np.random.default_rng(seed)
#     dx = 1.0
#     dy = np.sqrt(3)/2 * dx
#     patches, colors = [], []
#     missing = rng.random((n_rows, n_cols)) < missing_ratio
#
#     base = rng.random((n_rows, n_cols))**1.1
#     gx = np.linspace(0,1,n_cols); gy = np.linspace(0,1,n_rows)
#     Gx, Gy = np.meshgrid(gx, gy); trend = 0.4*Gx + 0.6*Gy
#     vals = 0.6*base + 0.4*trend
#     vals = (vals - vals.min())/(vals.max()-vals.min()+1e-8)
#
#     for r in range(n_rows):
#         for c in range(n_cols):
#             if missing[r,c]: continue
#             x = c*dx + (r%2)*dx/2.0; y = r*dy
#             if jitter>0:
#                 x += rng.uniform(-jitter, jitter)*dx*0.2
#                 y += rng.uniform(-jitter, jitter)*dy*0.2
#             hexagon = RegularPolygon((x,y), numVertices=6, radius=0.48, orientation=np.pi/6, ec="#003a8c", lw=0.6)
#             patches.append(hexagon); colors.append(vals[r,c])
#
#     fig = plt.figure(figsize=(3.4,3.0))
#     ax = fig.add_axes([0,0,1,1]); ax.set_aspect('equal'); ax.axis("off")
#     coll = PatchCollection(patches, cmap=cmap, edgecolor="#003a8c", linewidth=0.6)
#     coll.set_array(np.array(colors)); ax.add_collection(coll)
#     ax.set_xlim(-0.5, n_cols*dx + 0.5); ax.set_ylim(-0.5, n_rows*dy + 0.5)
#     return fig
#
# # XENIUM point cloud
# def xenium_points_image(seed=10, n_points=1200, pattern="gradient"):
#     rng = np.random.default_rng(seed)
#     x = rng.random(n_points); y = rng.random(n_points)
#     if pattern == "gradient":
#         vals = 0.6*x + 0.4*y + 0.2*rng.random(n_points)
#     elif pattern == "cluster":
#         cx = np.array([0.3, 0.7]); cy = np.array([0.35, 0.65])
#         z = rng.choice([0,1,2], size=n_points, p=[0.4,0.4,0.2])
#         # build arrays safely to avoid out-of-bounds when z==2 (background)
#         x2 = np.empty_like(x); y2 = np.empty_like(y)
#         mask_bg = (z==2)
#         mask_c0 = (z==0); mask_c1 = (z==1)
#         x2[mask_bg] = x[mask_bg];           y2[mask_bg] = y[mask_bg]
#         x2[mask_c0] = cx[0] + 0.09*rng.standard_normal(mask_c0.sum())
#         y2[mask_c0] = cy[0] + 0.09*rng.standard_normal(mask_c0.sum())
#         x2[mask_c1] = cx[1] + 0.09*rng.standard_normal(mask_c1.sum())
#         y2[mask_c1] = cy[1] + 0.09*rng.standard_normal(mask_c1.sum())
#         x = np.clip(x2,0,1); y = np.clip(y2,0,1)
#         vals = 0.5 + 0.5*rng.random(n_points)
#     elif pattern == "ring":
#         theta = 2*np.pi*rng.random(n_points); r = 0.5 + 0.15*rng.standard_normal(n_points)
#         x = 0.5 + r*np.cos(theta); y = 0.5 + r*np.sin(theta)
#         x = np.clip(x,0,1); y = np.clip(y,0,1)
#         vals = 0.5 + 0.5*rng.random(n_points)
#     else:
#         vals = rng.random(n_points)
#     vals = (vals - vals.min())/(vals.max() - vals.min() + 1e-8)
#
#     fig = plt.figure(figsize=(3.2,3.2))
#     ax = fig.add_axes([0,0,1,1]); ax.axis("off")
#     sizes = 6 + 20*vals
#     ax.scatter(x, y, c=vals, s=sizes, cmap="Blues", marker="o", linewidths=0)
#     ax.set_xlim(0,1); ax.set_ylim(0,1)
#     return fig
#
# # Generate VISIUM variants
# visium_cfgs = [
#     {"seed": 1, "missing_ratio": 0.00, "jitter": 0.00, "name":"visium_hex_clean"},
#     {"seed": 2, "missing_ratio": 0.10, "jitter": 0.00, "name":"visium_hex_missing"},
#     {"seed": 3, "missing_ratio": 0.00, "jitter": 0.15, "name":"visium_hex_jitter"},
#     {"seed": 4, "missing_ratio": 0.15, "jitter": 0.12, "name":"visium_hex_miss_jitter"},
# ]
# paths = []
# for cfg in visium_cfgs:
#     fig = visium_hex_image(seed=cfg["seed"], missing_ratio=cfg["missing_ratio"], jitter=cfg["jitter"])
#     png = f"{root}/{cfg['name']}.png"; svg = f"{root}/{cfg['name']}.svg"
#     save_fig(fig, png, svg); paths += [png, svg]
#
# # Generate XENIUM variants
# xenium_cfgs = [
#     {"seed": 21, "pattern":"gradient", "name":"xenium_pts_gradient"},
#     {"seed": 31, "pattern":"cluster",  "name":"xenium_pts_cluster"},
#     {"seed": 41, "pattern":"ring",     "name":"xenium_pts_ring"},
# ]
# for cfg in xenium_cfgs:
#     fig = xenium_points_image(seed=cfg["seed"], pattern=cfg["pattern"])
#     png = f"{root}/{cfg['name']}.png"; svg = f"{root}/{cfg['name']}.svg"
#     save_fig(fig, png, svg); paths += [png, svg]
#
# print("\n".join(paths))







# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
#
# plt.rcParams["figure.dpi"] = 200
# root = '/media/huifang/data/registration/paper_figure'
# def save_fig(fig, png_path, svg_path):
#     fig.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
#     fig.savefig(svg_path, bbox_inches="tight", pad_inches=0, transparent=True)
#     plt.close(fig)
#
# ############################
# # Square grid renderer (Visium-like squares)
# ############################
# def visium_square_image(seed=0, n_cols=12, n_rows=12, jitter=0.0, missing_ratio=0.0, cmap="Blues"):
#     rng = np.random.default_rng(seed)
#     patches, colors = [], []
#     # base intensity with mild spatial trend
#     base = rng.random((n_rows, n_cols))**1.1
#     gx = np.linspace(0,1,n_cols); gy = np.linspace(0,1,n_rows)
#     Gx, Gy = np.meshgrid(gx, gy); trend = 0.35*Gx + 0.65*Gy
#     vals = 0.6*base + 0.4*trend
#     vals = (vals - vals.min())/(vals.max() - vals.min() + 1e-8)
#
#     missing = rng.random((n_rows, n_cols)) < missing_ratio
#
#     fig = plt.figure(figsize=(3.2,3.2))
#     ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_aspect('equal')
#
#     for r in range(n_rows):
#         for c in range(n_cols):
#             if missing[r,c]: continue
#             x = c + 0.05; y = r + 0.05
#             size = 0.9  # square side length
#             if jitter>0:
#                 x += rng.uniform(-jitter, jitter)*0.15
#                 y += rng.uniform(-jitter, jitter)*0.15
#             sq = Rectangle((x, y), size, size, ec="#003a8c", lw=0.6)
#             patches.append(sq); colors.append(vals[r,c])
#
#     coll = PatchCollection(patches, cmap=cmap, edgecolor="#003a8c", linewidth=0.6)
#     coll.set_array(np.array(colors)); ax.add_collection(coll)
#     ax.set_xlim(0, n_cols+0.2); ax.set_ylim(0, n_rows+0.2)
#     return fig
#
# def visium_square_gradient(n_cols=12, n_rows=12, kind="diag", cmap="Oranges"):
#     gx = np.linspace(0,1,n_cols); gy = np.linspace(0,1,n_rows)
#     Gx, Gy = np.meshgrid(gx, gy)
#     if kind == "diag":
#         vals = 0.6*Gx + 0.4*Gy
#     elif kind == "row":
#         vals = Gy
#     elif kind == "col":
#         vals = Gx
#     elif kind == "center":
#         Xc, Yc = np.meshgrid(np.linspace(-1,1,n_cols), np.linspace(-1,1,n_rows))
#         vals = np.exp(-3*(Xc**2 + Yc**2))
#     else:
#         vals = 0.6*Gx + 0.4*Gy
#     vals = (vals - vals.min())/(vals.max()-vals.min()+1e-8)
#
#     fig = plt.figure(figsize=(3.2,3.2))
#     ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_aspect('equal')
#
#     patches, colors = [], []
#     for r in range(n_rows):
#         for c in range(n_cols):
#             x = c + 0.05; y = r + 0.05
#             sq = Rectangle((x, y), 0.9, 0.9, ec="#003a8c", lw=0.6)
#             patches.append(sq); colors.append(vals[r,c])
#
#     coll = PatchCollection(patches, cmap=cmap, edgecolor="#003a8c", linewidth=0.6)
#     coll.set_array(np.array(colors)); ax.add_collection(coll)
#     ax.set_xlim(0, n_cols+0.2); ax.set_ylim(0, n_rows+0.2)
#     return fig
#
# paths = []
#
# # Square grid patterns (Visium-like squares)
# cfgs_sq = [
#     {"seed": 1, "missing_ratio": 0.0,  "jitter": 0.0,  "name": "visium_sq_clean"},
#     {"seed": 2, "missing_ratio": 0.10, "jitter": 0.0,  "name": "visium_sq_missing"},
#     {"seed": 3, "missing_ratio": 0.0,  "jitter": 0.15, "name": "visium_sq_jitter"},
#     {"seed": 4, "missing_ratio": 0.15, "jitter": 0.12, "name": "visium_sq_miss_jitter"},
# ]
# for cfg in cfgs_sq:
#     fig = visium_square_image(seed=cfg["seed"], missing_ratio=cfg["missing_ratio"], jitter=cfg["jitter"])
#     png = f"{root}/{cfg['name']}.png"; svg = f"{root}/{cfg['name']}.svg"
#     save_fig(fig, png, svg); paths += [png, svg]
#
# # Visium gradient maps (square grid)
# for kind in ["diag", "row", "col", "center"]:
#     fig = visium_square_gradient(kind=kind)
#     png = f"{root}/visium_sq_gradient_{kind}.png"; svg = f"{root}/visium_sq_gradient_{kind}.svg"
#     save_fig(fig, png, svg); paths += [png, svg]
#
#
#
# print("\n".join(paths))
#













# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Circle
#
# def generate_visium_like_spots(
#     side_length=1.0,
#     n_rows=5,
#     n_cols=5,
#     spot_diameter_frac=0.015,
# ):
#     """
#     Generate a hexagonal grid of spot centers inside a square [0, side_length]^2.
#     Returns arrays of x, y coordinates.
#     """
#     sx = side_length / n_cols
#     sy = np.sqrt(3) / 2 * sx  # hex vertical spacing
#
#     xs, ys = [], []
#     for j in range(n_rows):
#         y = (j + 0.5) * sy
#         if y > side_length:
#             break
#
#         # shift every other row by half a column (hex packing)
#         x_offset = (sx / 2) if (j % 2 == 1) else 0.0
#
#         for i in range(n_cols):
#             x = (i + 0.5) * sx + x_offset
#             if 0 < x < side_length:
#                 xs.append(x)
#                 ys.append(y)
#
#     return np.array(xs), np.array(ys)
#
#
# def plot_visium_capture_area(xs, ys, side_length=1.0, spot_diameter_frac=0.15):
#     """
#     Plot a square capture area with circular spots at (xs, ys).
#     """
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     # Draw the square capture area
#     ax.add_patch(
#         Rectangle(
#             (0, 0),
#             side_length,
#             side_length,
#             fill=False,
#             linewidth=5,
#         )
#     )
#
#     # Draw each spot as a small circle
#     radius = (spot_diameter_frac * side_length) / 2.0
#     for x, y in zip(xs, ys):
#         ax.add_patch(
#             Circle(
#                 (x, y+0.05),
#                 radius=radius,
#                 linewidth=0.5,
#                 color='orange'
#             )
#         )
#
#     ax.set_aspect("equal")
#     ax.set_xlim(-0.05 * side_length, 1.2 * side_length)
#     ax.set_ylim(-0.05 * side_length, 1.2 * side_length)
#     ax.axis("off")
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 1) Generate synthetic Visium-like spots
#     xs, ys = generate_visium_like_spots()
#
#     # 2) Plot them on a square capture area
#     plot_visium_capture_area(xs, ys)






import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D

def draw_transformer_icon(ax, x=0.0, y=0.0, width=1.6, height=1.8,
                          n_blocks=3):
    """
    Draw a small transformer-style icon on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Bottom-left corner of the icon (in data coordinates).
    width, height : float
        Size of the whole icon.
    n_blocks : int
        Number of stacked encoder blocks.
    """
    # Outer rounded box: the transformer module
    outer = FancyBboxPatch(
        (x, y),
        width, height,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(outer)

    # Margin inside the outer box
    pad_x = 0.12 * width
    pad_y = 0.12 * height

    inner_x0 = x + pad_x
    inner_x1 = x + width - pad_x
    inner_y0 = y + pad_y
    inner_y1 = y + height - pad_y

    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0

    # Height of each encoder block
    block_h = inner_h / n_blocks
    attn_frac = 0.5  # part for MHSA vs FFN

    # Draw stacked encoder blocks
    for i in range(n_blocks):
        by0 = inner_y0 + i * block_h
        by1 = by0 + block_h

        # Attention sub-block (top strip)
        attn_h = block_h * attn_frac
        attn = Rectangle(
            (inner_x0, by1 - attn_h),
            inner_w, attn_h,
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(attn)

        # FFN sub-block (bottom strip)
        ffn_h = block_h * (1 - attn_frac)
        ffn = Rectangle(
            (inner_x0, by0),
            inner_w, ffn_h,
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(ffn)

    # Draw small "multi-head" circles above the top block
    # so it's obvious this is an attention-based transformer
    heads_y = inner_y1 + 0.08 * height
    heads_xs = [
        x + width * 0.35,
        x + width * 0.50,
        x + width * 0.65,
    ]
    head_r = 0.06 * width

    for hx in heads_xs:
        c = Circle(
            (hx, heads_y),
            radius=head_r,
            edgecolor="black",
            facecolor="white",
            linewidth=1.0,
        )
        ax.add_patch(c)

        # Connect each head to the top encoder block (attention strip)
        ax.add_line(
            Line2D(
                [hx, hx],
                [heads_y - head_r, inner_y1],
                linewidth=0.8,
                color="black",
            )
        )

    # Optional: input/output stubs on left/right
    stub_len = 0.25 * width
    mid_y = y + height * 0.5
    ax.add_line(
        Line2D(
            [x - stub_len, x],
            [mid_y, mid_y],
            linewidth=1.3,
            color="black",
        )
    )
    ax.add_line(
        Line2D(
            [x + width, x + width + stub_len],
            [mid_y, mid_y],
            linewidth=1.3,
            color="black",
        )
    )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw icon at (0,0) with width 1.6, height 1.8
    draw_transformer_icon(ax, x=0.0, y=0.0, width=1.6, height=1.8, n_blocks=3)

    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-0.5, 3.0)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
