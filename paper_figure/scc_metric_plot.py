import numpy as np
import matplotlib.pyplot as plt

# Metrics and methods
metrics_spot_label_1 = ['spot label consistency']

methods = ['unaligned', 'SimpleITK', 'VoxelMorph', 'NiceTrans', 'PASTE', 'GPSA', 'SANTO', 'Ours']

# spatial domain alignment
data_spot_label_1 = np.array([
    [0.4406],  # unaligned
    [0.4835],  # SimpleITK
    [0.4526],  # VoxelMorph
    [0.5003],  # NiceTrans
    [0.4638],  # PASTE
    [0.4641],  # GPSA
    [0.4521],  # SANTO
    [0.5024],  # Ours
])

# palette
colors = ['#dcd2da','#376795', '#528fad', '#aadce0', '#ffe6b7', '#ffd06f', '#ef8a47', '#e76254']

# plotting
# fig, axes = plt.subplots(1, 3, figsize=(24, 5), gridspec_kw={'width_ratios': [1, 0.5, 1.5]},constrained_layout=True)
fig,axes = plt.subplots(1,2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]},constrained_layout=True)
plt.rcParams.update({'font.size': 14})

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
        axes[0].bar(
            x + i * width, data_spot_label_1[i], width,
            label=methods[i], color=colors[i]
        )
axes[0].set_xticks(x + (len(methods)//2) * width)
axes[0].set_xticklabels(metrics_spot_label_1, fontsize=14)
axes[0].set_ylim(0.4, 0.55)
axes[0].set_title('Spot label consistency')

# axes[1].legend(frameon=True, bbox_to_anchor=(1.05, 0.7))
handles, labels = axes[0].get_legend_handles_labels()
# Put the legend at the left-most side of the figure
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), frameon=True,fontsize=12)
plt.show()
