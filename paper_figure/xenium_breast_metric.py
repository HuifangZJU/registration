import numpy as np
import matplotlib.pyplot as plt

# Metrics and methods
metrics_spot_label_1 = ['Dice', 'Spatial Correlation']
metrics_spot_label_2 = ['Mean Centroid Shift (↓)']
metrics_image = ['MI', 'SSIM', 'NCC']

methods = ['unaligned', 'SimpleITK', 'VoxelMorph', 'NiceTrans', 'PASTE', 'GPSA', 'SANTO', 'Ours']

# spatial domain alignment
data_spot_label_1 = np.array([
    [0.9216, 0.9608],  # unaligned
    [0.9720, 0.9898],  # SimpleITK
    [0.9176, 0.9603],  # VoxelMorph
    [0.9752, 0.9870],  # NiceTrans
    [0.9065, 0.9707],  # PASTE
    [0.9716, 0.9893],  # GPSA
    [0.8251, 0.9131],  # SANTO
    [0.9794, 0.9901],  # Ours
])

data_spot_label_2 = np.array([
    [2.042],   # unaligned
    [0.344],  # SimpleITK
    [2.053],  # VoxelMorph
    [0.334],  # NiceTrans
    [1.409],  # PASTE
    [0.287],  # GPSA
    [3.488],  # SANTO
    [0.163],  # Ours
])


# palette
colors = ['#dcd2da','#376795', '#528fad', '#aadce0', '#ffe6b7', '#ffd06f', '#ef8a47', '#e76254']

# plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 0.5]},constrained_layout=True)
plt.rcParams.update({'font.size': 14})
fig.suptitle("Xenium Breast cancer, 1 pair, broader domain", fontsize=16, fontweight='bold')
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
axes[0].set_ylim(0.8, 1)


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
axes[1].set_ylim(0.1, 4)   # tighter around observed values

# axes[1].legend(frameon=True)

handles, labels = axes[1].get_legend_handles_labels()
# Put the legend at the left-most side of the figure
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), frameon=True)
plt.show()
