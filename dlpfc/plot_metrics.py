import matplotlib.pyplot as plt
import numpy as np

# Metrics and methods
metrics_spot_label_1 = ['Dice', 'Spatial Correlation']
metrics_spot_label_2 = ['Mean Centroid Shift']
metrics_image = ['MI', 'SSIM', 'NCC']
methods = ['unregistered', 'SimpleITK', 'PASTE', 'Ours']

# Data (extracted manually from the image)
data_spot_label_1 = np.array([
    [0.55, 0.61],  # unaligned
    [0.67, 0.69],  # SimpleITK
    [0.69, 0.70],  # PASTE
    [0.75, 0.77],  # Ours
])

data_spot_label_2 = np.array([
    [95.0],  # unaligned
    [65.0],  # SimpleITK
    [55.0],  # PASTE
    [35.0],  # Ours
])

data_image = np.array([
    [0.55, 0.61, 0.70],  # unaligned
    [0.63, 0.66, 0.75],  # SimpleITK
    [0.66, 0.69, 0.77],  # PASTE
    [0.73, 0.74, 0.81],  # Ours
])

# Color palette
colors = ['#e0c7e3', '#eae0e9', '#ae98b6', '#c6d182']

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 0.4, 1.5]})
plt.rcParams.update({'font.size': 14})
# First subplot: Dice and Spatial Correlation
x = np.arange(len(metrics_spot_label_1))
width = 0.2
for i in range(len(methods)):
    axes[0].bar(x + i * width, data_spot_label_1[i], width, label=methods[i], color=colors[i])
axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels(metrics_spot_label_1,fontsize=14)
axes[0].set_ylim(0.5,0.8)
txt = axes[0].set_title('Spatial domain metrics', loc='left')  # create title
txt.set_x(0.65)        # 0 = left edge, 0.5 = centre, 1 = right edge


# Second subplot: Mean Centroid Shift
x = np.arange(len(metrics_spot_label_2))
for i in range(len(methods)):
    axes[1].bar(x + i * width, data_spot_label_2[i], width, label=methods[i], color=colors[i])
axes[1].set_xticks(x + 1.5 * width)
axes[1].set_xticklabels(metrics_spot_label_2,fontsize=14)
axes[1].set_ylim(15,100)
# axes[1].set_title('Spatial domain centroid shift')

# Third subplot: Image Metrics
x = np.arange(len(metrics_image))
for i in range(len(methods)):
    axes[2].bar(x + i * width, data_image[i], width, label=methods[i], color=colors[i])
axes[2].set_xticks(x + 1.5 * width)
axes[2].set_xticklabels(metrics_image,fontsize=14)
axes[2].set_title('Histological alignment')
axes[2].legend(frameon=True, bbox_to_anchor=(1, 0.7))
axes[2].set_ylim(0.5,0.85)
plt.tight_layout()
plt.savefig('/home/huifang/workspace/grant/k99/resubmission/figures/1.png', dpi=300)
plt.show()
