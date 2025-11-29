import numpy as np
import matplotlib.pyplot as plt

# Metrics and methods
metrics_spot_label_1 = ['Dice']
metrics_spot_label_2 = ['Mean Centroid Shift (↓)']


# methods = ['unaligned', 'SimpleITK', 'VoxelMorph', 'NiceTrans', 'PASTE', 'GPSA', 'SANTO', 'Ours']
#
# colors = ['#dcd2da','#376795', '#528fad', '#aadce0', '#ffe6b7', '#ffd06f', '#ef8a47', '#e76254']

# spatial domain alignment
# data_spot_label_1 = np.array([
#     # [0.1743, 0.6637],  # unaligned
#     [0.7887, 0.9835],  # SimpleITK
#     # [0.1749, 0.6676],  # VoxelMorph
#     [0.7069, 0.9187],  # NiceTrans
#     # [0.4225, 0.8808],  # PASTE
#     [0.6319, 0.9748],  # GPSA
#     # [0.1191, 0.6354],  # SANTO
#     [0.8397, 0.9870],  # Ours
# ])
data_spot_label_1 = np.array([
    [0.7887],  # SimpleITK
    [0.7069],  # NiceTrans
    [0.6319],  # GPSA
    [0.8397],  # Ours
])

data_spot_label_2 = np.array([
    # [30.993],   # unaligned
    [3.803],  # SimpleITK
    # [31.37],  # VoxelMorph
    [8.023],  # NiceTrans
    # [15.792],  # PASTE
    [5.179],  # GPSA
    # [33.502],  # SANTO
    [2.5185],  # Ours
])
methods = [ 'SimpleITK',  'NiceTrans','GPSA', 'Ours']
# palette
colors = ['#376795',  '#aadce0',  '#ffd06f',  '#e76254']

# plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [1, 1]},constrained_layout=True)
plt.rcParams.update({'font.size': 14})
fig.suptitle("Xenium Breast cancer, 1 pair, small markers", fontsize=16, fontweight='bold')
width = 0.8

# --- First subplot: Dice + Spatial correlation ---
x = np.arange(len(metrics_spot_label_1))
for i in range(len(methods)):
    if i < 0:  # first bar style: transparent fill + black dashed edge + hatch
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
axes[0].set_ylim(0.6, 0.9)


# --- Second subplot: Mean centroid shift ---
x = np.arange(len(metrics_spot_label_2))
for i in range(len(methods)):
    if i< 0:
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
axes[1].set_ylim(0.1, 9.5)   # tighter around observed values

# axes[1].legend(frameon=True)

handles, labels = axes[1].get_legend_handles_labels()
# Put the legend at the left-most side of the figure
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), frameon=True)
plt.show()
