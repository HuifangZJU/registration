import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ─── sample 3 layers (4 × 4 each) ────────────────────────────────
maps = [np.array([[0.8,0.2,0.5,0.1],[0.3,0.9,0.4,0.6],[0.7,0.2,0.8,0.3],[0.1,0.5,0.6,0.9]]),
        np.array([[0.2,0.7,0.4,0.8],[0.5,0.3,0.9,0.1],[0.6,0.8,0.2,0.5],[0.9,0.4,0.7,0.3]]),
        np.array([[0.7,0.1,0.9,0.4],[0.2,0.8,0.3,0.7],[0.5,0.6,0.1,0.8],[0.3,0.9,0.5,0.2]])]

rows, cols, layers = maps[0].shape[0], maps[0].shape[1], len(maps)

# ───  parameters  ────────────────────────────────────────────────
layer_gap = 20.0          # distance between slabs (along X)
dx = 0.01                # slab thickness (X-direction)  «thin»
dy = dz = 1.0            # full size in Y & Z
cmap = plt.cm.viridis; norm = mcolors.Normalize(0, 1)

# ───  plotting  ─────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 5))
ax  = fig.add_subplot(111, projection='3d')

for li, fm in enumerate(maps):
    x0 = li * (dx + layer_gap)               # slab’s X-corner
    for r in range(rows):
        for c in range(cols):
            x = x0
            y = c
            z = r
            colour = cmap(norm(fm[r, c]))
            ax.bar3d(x, y, z, dx, dy, dz,
                     color=colour, edgecolor='k', linewidth=.35, alpha=.92)

# axes & labels
ax.set_xlabel('Layer'); ax.set_ylabel('Column'); ax.set_zlabel('Row')
ax.set_xticks([li * (dx + layer_gap)         for li in range(layers)])
ax.set_yticks(range(cols));  ax.set_zticks(range(rows))
ax.view_init(elev=25, azim=-45)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array(np.concatenate(maps))
fig.colorbar(sm, ax=ax, pad=.05, label='Activation')

plt.tight_layout(); plt.show()
