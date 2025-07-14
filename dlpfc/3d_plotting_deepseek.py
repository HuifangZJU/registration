import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import matplotlib.cm as cm

# Sample feature maps (4 layers of 4x4 arrays)
feature_maps = [
    np.array([[0.8, 0.2, 0.5, 0.1],
              [0.3, 0.9, 0.4, 0.6],
              [0.7, 0.2, 0.8, 0.3],
              [0.1, 0.5, 0.6, 0.9]]),

    np.array([[0.2, 0.7, 0.4, 0.8],
              [0.5, 0.3, 0.9, 0.1],
              [0.6, 0.8, 0.2, 0.5],
              [0.9, 0.4, 0.7, 0.3]]),

    np.array([[0.7, 0.1, 0.9, 0.4],
              [0.2, 0.8, 0.3, 0.7],
              [0.5, 0.6, 0.1, 0.8],
              [0.3, 0.9, 0.5, 0.2]]),

    np.array([[0.4, 0.6, 0.3, 0.9],
              [0.8, 0.1, 0.5, 0.2],
              [0.3, 0.7, 0.8, 0.4],
              [0.6, 0.2, 0.1, 0.7]])
]

# Create 3D visualization
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Get dimensions
num_layers = len(feature_maps)
rows, cols = feature_maps[0].shape

# Create colormap
cmap = plt.cm.viridis
norm = colors.Normalize(vmin=0, vmax=1)

# Adjust these parameters to control spacing
layer_thickness = 0.1  # Make feature maps thinner
layer_gap = 20.0  # Increase distance between feature maps

# Plot each feature map as a solid block
for layer_idx, fm in enumerate(feature_maps):
    # Calculate x-position for this layer with increased gap
    x_offset = layer_idx * (layer_thickness + layer_gap)

    # Create a grid for this layer
    x = np.full((rows, cols), x_offset)
    y, z = np.meshgrid(np.arange(cols), np.arange(rows))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = fm.flatten()
    colors_array = cmap(norm(values))

    # Plot the layer with thin feature maps
    ax.bar3d(x, y, z,
             layer_thickness, 1.0, 1.0,  # Thin in x-direction
             color=colors_array,
             alpha=0.8,
             edgecolor='k',
             linewidth=0.3)

# Set labels and title
ax.set_xlabel('Layer Index', labelpad=15)
ax.set_ylabel('Column Index', labelpad=15)
ax.set_zlabel('Row Index', labelpad=15)
ax.set_title('3D Feature Maps with Thin Layers and Increased Spacing', pad=25)

# Set axis limits with extra space
x_max = num_layers * (layer_thickness + layer_gap) - layer_gap
ax.set_xlim(-0.5, x_max + 0.5)
ax.set_ylim(-0.5, cols - 0.5)
ax.set_zlim(-0.5, rows - 0.5)

# Set custom x-ticks at the center of each feature map
ax.set_xticks([i * (layer_thickness + layer_gap) + layer_thickness / 2 for i in range(num_layers)])
ax.set_xticklabels(np.arange(num_layers))

# Set y and z ticks
ax.set_yticks(np.arange(cols))
ax.set_zticks(np.arange(rows))

# Create colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(np.concatenate(feature_maps))
fig.colorbar(mappable, ax=ax, pad=0.1, label='Activation Value')

# Adjust viewing angle
ax.view_init(elev=25, azim=-45)
plt.tight_layout()
plt.show()