# Regenerate AXIS deformation graphic with margin between axis frame and grids.
# We'll map both reference and warped grids into an inner square [m, 1-m] to create padding.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# plt.rcParams["figure.dpi"] = 200
#
# # Domain + grid
# n = 31
# x = np.linspace(0, 1, n)
# y = np.linspace(0, 1, n)
# X, Y = np.meshgrid(x, y)
#
# # Deformation ϕ on the unit square
# amp = 0.12
# Xw = X + amp * np.sin(2*np.pi * Y)
# Yw = Y + amp * np.sin(2*np.pi * X)
#
# # Map to inner square to create margin
# m = 0.17  # margin fraction
# def to_inner(Z):  # affine map [0,1] -> [m, 1-m]
#     return m + (1-2*m)*Z
#
# Xm, Ym = to_inner(X), to_inner(Y)
# Xwm, Ywm = to_inner(Xw), to_inner(Yw)
#
# fig = plt.figure(figsize=(4.8, 4.4))
# ax = fig.add_axes([0.12, 0.12, 0.75, 0.75])
# ax.set_aspect('equal')
#
# # Reference grid (light), now inside margins
# for i in range(n):
#     ax.plot(xm := to_inner(x), np.full_like(xm, Ym[i,0]), linewidth=0.5, alpha=0.45, color="gray")
#     ax.plot(np.full_like(y, Xm[0,i]), ym := to_inner(y), linewidth=0.5, alpha=0.45, color="gray")
#
# # Warped grid (inside margins)
# for i in range(n):
#     ax.plot(Xwm[i,:], Ywm[i,:], linewidth=0.5)
#     ax.plot(Xwm[:,i], Ywm[:,i], linewidth=0.5)
#
# # Sample displacement vectors (also inside margins)
# # idx = [2, 5, 8]
# # for i in idx:
# #     for j in idx:
# #         dx = Xwm[i,j] - Xm[i,j]
# #         dy = Ywm[i,j] - Ym[i,j]
# #         ax.arrow(Xm[i,j], Ym[i,j], dx, dy,
# #                  length_includes_head=True, head_width=0.015, head_length=0.02, linewidth=0.9, color="black")
#
# # Limits and custom axis arrows
# # ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 1.08)
# for spine in ax.spines.values(): spine.set_visible(False)
# ax.set_xticks([]); ax.set_yticks([])
#
# arrowprops = dict(arrowstyle="-|>", mutation_scale=12, lw=2, color="black")
# ax.add_patch(FancyArrowPatch((-0.06, 0.0), (1.05, 0.0), **arrowprops))  # X
# ax.add_patch(FancyArrowPatch((0.0, -0.06), (0.0, 1.05), **arrowprops))  # Y
#
# ax.text(1.055, -0.02, "AXIS atlas X →", ha="left", va="top", fontsize=11)
# ax.text(-0.02, 1.055, "AXIS atlas Y ↑", ha="right", va="bottom", fontsize=11)
#
# ax.set_title("Deformation field ϕ with inner-margin grid in the AXIS atlas frame", pad=10)
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_axis_frame():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- frame limits ----
    lim = 1.0
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_box_aspect([1, 1, 1])

    # ---- make panes and grids transparent / light ----
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['linewidth'] = 0.3
        axis._axinfo["grid"]['linestyle'] = '--'
        axis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.4)

    # Make background panes transparent
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    # ---- draw a semi-transparent cube for context ----
    # cube corners
    r = [0, lim]
    vertices = np.array([
        [r[0], r[0], r[0]],
        [r[1], r[0], r[0]],
        [r[1], r[1], r[0]],
        [r[0], r[1], r[0]],
        [r[0], r[0], r[1]],
        [r[1], r[0], r[1]],
        [r[1], r[1], r[1]],
        [r[0], r[1], r[1]],
    ])

    # faces of the cube (as lists of vertex indices)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4],  # left
    ]

    cube_faces = [[vertices[idx] for idx in face] for face in faces]

    cube = Poly3DCollection(
        cube_faces,
        facecolors=(0.9, 0.9, 0.9, 0.1),  # light gray, very transparent
        edgecolors=(0.5, 0.5, 0.5, 0.4),
        linewidths=0.6
    )
    ax.add_collection3d(cube)

    # ---- draw main axes (bold) ----
    # origin
    o = np.array([0., 0., 0.])

    # X axis (red)
    ax.plot(
        [o[0], lim], [o[1], o[1]], [o[2], o[2]],
        color='red', linewidth=3
    )
    # Y axis (green)
    ax.plot(
        [o[0], o[0]], [o[1], lim], [o[2], o[2]],
        color='green', linewidth=3
    )
    # Z axis (blue)
    ax.plot(
        [o[0], o[0]], [o[1], o[1]], [o[2], lim],
        color='blue', linewidth=3
    )

    # optional arrowheads via small quivers
    head_len = 0.06
    ax.quiver( lim-head_len, 0,         0, head_len, 0,        0, color='red')
    ax.quiver( 0,         lim-head_len, 0, 0,        head_len, 0, color='green')
    ax.quiver( 0,         0,         lim-head_len, 0,        0, head_len, color='blue')

    # ---- axis labels ----
    ax.text(lim*1.05, 0, 0, 'X', color='red',  fontsize=12)
    ax.text(0, lim*1.05, 0, 'Y', color='green', fontsize=12)
    ax.text(0, 0, lim*1.05, 'Z', color='blue', fontsize=12)

    # turn off default tick labels to keep it clean
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_axis_frame()
