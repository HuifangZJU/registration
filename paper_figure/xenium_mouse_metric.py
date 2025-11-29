import numpy as np
import matplotlib.pyplot as plt

# Metrics and methods
metrics_spot_label_1 = ['Dice', 'Spatial Correlation']
metrics_spot_label_2 = ['Mean Centroid Shift (↓)']
metrics_image = ['MI', 'SSIM', 'NCC']

methods = ['unaligned', 'SimpleITK', 'VoxelMorph', 'PASTE', 'GPSA', 'SANTO', 'Ours']

# spatial domain alignment
data_spot_label_1 = np.array([
    [0.4262, 0.5281],  # unaligned
    [0.6935, 0.7557],  # SimpleITK
    [0.4196, 0.5247],  # VoxelMorph
    [0.6241, 0.6783],  # PASTE
    [0.5820, 0.6762],  # GPSA
    [0.6424, 0.7242],  # SANTO
    [0.8096, 0.8024],  # Ours
])

data_spot_label_2 = np.array([
    [8.021],   # unaligned
    [5.169],  # SimpleITK
    [8.303],  # VoxelMorph
    [5.517],  # PASTE
    [7.602],  # GPSA
    [4.807],  # SANTO
    [2.212],  # Ours
])


# palette
colors = ['#dcd2da','#376795', '#528fad',  '#ffe6b7', '#ffd06f', '#ef8a47', '#e76254']

# plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 0.5]},constrained_layout=True)
plt.rcParams.update({'font.size': 14})
fig.suptitle("Xenium Mouse Brain, 2 groups, 4 pairs", fontsize=16, fontweight='bold')

width = 0.12

# --- First subplot: Dice + Spatial correlation ---
x = np.arange(len(metrics_spot_label_1))*1.2
for i in range(len(methods)):
    if i == 0:  # first bar style: transparent fill + black dashed edge + hatch
        axes[0].bar(
            x + i * width, data_spot_label_1[i], width,
            label=methods[i],
            facecolor='none',
            edgecolor='black',
            hatch='//',
            linewidth=1,
            linestyle='--'  # <-- dashed border
        )
    else:
        vals = data_spot_label_1[i]
        xpos = x + i * width
        if np.isnan(vals).all():
            # plot × markers
            for xi in xpos:
                axes[0].text(xi, 0.35, '×', ha='center', va='bottom',
                             fontsize=14,label=methods[i], color=colors[i])
        else:
            axes[0].bar(
                xpos, vals, width,
                label=methods[i], color=colors[i]
            )
axes[0].set_xticks(x + (len(methods)//2) * width)
axes[0].set_xticklabels(metrics_spot_label_1, fontsize=14)
axes[0].set_ylim(0.3, 0.9)


# --- Second subplot: Mean centroid shift ---
x = np.arange(len(metrics_spot_label_2))
for i in range(len(methods)):
    if i == 0:
        axes[1].bar(
            x + i * width, data_spot_label_2[i], width,
            label=methods[i],
            facecolor='none', edgecolor='black', hatch='//', linewidth=1,
        linestyle='--'
        )
    else:
        if np.isnan(data_spot_label_2[i]):
            axes[1].text(x + i * width, 3, '×', ha='center', va='bottom',
                             fontsize=14,label=methods[i], color=colors[i])
        else:
            axes[1].bar(
                x + i * width, data_spot_label_2[i], width,
                label=methods[i], color=colors[i]
            )
axes[1].set_xticks(x + (len(methods)//2) * width)
axes[1].set_xticklabels(metrics_spot_label_2, fontsize=14)
axes[1].set_ylim(1, 9)   # tighter around observed values


handles, labels = axes[1].get_legend_handles_labels()
# Put the legend at the left-most side of the figure
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), frameon=True)
plt.show()
