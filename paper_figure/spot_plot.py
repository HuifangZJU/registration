import matplotlib.pyplot as plt
import numpy as np

# Metrics and methods
metrics_spot_label_1 = ['Dice', 'Spatial Correlation']
metrics_spot_label_2 = ['Mean Centroid Shift (↓)']
metrics_image = ['MI', 'SSIM', 'NCC']
methods = ['unaligned', 'SimpleITK',  'PASTE', 'NiceTrans','Ours']

# Data from your table
data_spot_label_1 = np.array([
    [0.6067, 0.6862],  # unaligned
    [0.6399, 0.7068],  # SimpleITK
    [0.6421, 0.7192],  # PASTE
    [0.6483, 0.7198],  # NiceTrans
    [0.7517, 0.7959],  # Ours
])

data_spot_label_2 = np.array([
    [20.01],  # unaligned
    [19.53],  # SimpleITK
    [17.59],  # PASTE
    [18.70],  # NiceTrans
    [13.60],  # Ours
])

data_image = np.array([
    [0.8264, 0.5372, 0.7118],  # unaligned
    [1.0566, 0.5478, 0.2656],  # SimpleITK
    [1.3130, 0.5737, 0.4806],  # PASTE
    [1.7155, 0.7356, 0.9231],  # NiceTrans
    [1.7314, 0.7355, 0.9303],  # Ours
])

# Original palette you had
base_colors = ['#e0c7e3', '#eae0e9', '#ae98b6', '#7e5a9b', '#c6d182']

# Re-map so that "Ours" uses green (#4daf4a)
colors = base_colors

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 0.5, 1.5]})
plt.rcParams.update({'font.size': 14})

# First subplot: Dice and Spatial Correlation
x = np.arange(len(metrics_spot_label_1))
width = 0.15
for i in range(len(methods)):
    axes[0].bar(x + i * width, data_spot_label_1[i], width, label=methods[i], color=colors[i])
axes[0].set_xticks(x + 2 * width)
axes[0].set_xticklabels(metrics_spot_label_1, fontsize=14)
axes[0].set_ylim(0.5, 0.85)
txt = axes[0].set_title('Spatial domain metrics', loc='left')
txt.set_x(0.65)

# Second subplot: Mean Centroid Shift
x = np.arange(len(metrics_spot_label_2))
for i in range(len(methods)):
    axes[1].bar(x + i * width, data_spot_label_2[i], width, label=methods[i], color=colors[i])
axes[1].set_xticks(x + 2 * width)
axes[1].set_xticklabels(metrics_spot_label_2, fontsize=14)
axes[1].set_ylim(10, 25)
# axes[1].set_title('Centroid shift (↓)', loc='center')

# Third subplot: Image Metrics
x = np.arange(len(metrics_image))
for i in range(len(methods)):
    axes[2].bar(x + i * width, data_image[i], width, label=methods[i], color=colors[i])
axes[2].set_xticks(x + 2 * width)
axes[2].set_xticklabels(metrics_image, fontsize=14)
axes[2].set_title('Histological alignment')
axes[2].legend(frameon=True, bbox_to_anchor=(1.05, 0.7))
axes[2].set_ylim(0.2, 1.9)

plt.tight_layout()
plt.show()
