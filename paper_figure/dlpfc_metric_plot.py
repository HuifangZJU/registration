import numpy as np
import matplotlib.pyplot as plt

# Metrics and methods
metrics_spot_label_1 = ['Dice', 'Spatial Correlation']
metrics_spot_label_2 = ['Mean Centroid Shift (↓)']
metrics_image = ['MI', 'SSIM', 'NCC']

methods = ['unaligned', 'SimpleITK', 'VoxelMorph', 'NiceTrans', 'PASTE', 'GPSA', 'SANTO', 'Ours']

# spatial domain alignment
data_spot_label_1 = np.array([
    [0.6156, 0.7386],  # unaligned
    [0.6467, 0.7593],  # SimpleITK
    [0.6355, 0.7505],  # VoxelMorph
    [0.6569, 0.7750],  # NiceTrans
    [0.6509, 0.7681],  # PASTE
    [0.5820, 0.6054],  # GPSA
    [0.5452, 0.6924],  # SANTO
    [0.7558, 0.8570],  # Ours
])

data_spot_label_2 = np.array([
    [6.27],   # unaligned
    [5.681],  # SimpleITK
    [6.172],  # VoxelMorph
    [5.387],  # NiceTrans
    [5.646],  # PASTE
    [7.237],  # GPSA
    [8.524],  # SANTO
    [3.927],  # Ours
])

# histological structure alignment (GPSA excluded here)
data_image = np.array([
    [0.8264, 0.5372, 0.7118],  # unaligned
    [1.0566, 0.5478, 0.2656],  # SimpleITK
    [1.5172, 0.6193, 0.8646],  # VoxelMorph
    [1.7155, 0.7356, 0.9231],  # NiceTrans
    [np.nan, np.nan, np.nan],  # PASTE
    [np.nan, np.nan, np.nan],  # GPSA
    [np.nan, np.nan, np.nan],  # SANTO
    [1.7314, 0.7355, 0.9303],  # Ours
])

# palette
colors = ['#dcd2da','#376795', '#528fad', '#aadce0', '#ffe6b7', '#ffd06f', '#ef8a47', '#e76254']

# plotting
fig, axes = plt.subplots(1, 3, figsize=(24, 5), gridspec_kw={'width_ratios': [1, 0.5, 1.5]},constrained_layout=True)
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
axes[0].set_ylim(0.5, 0.9)
axes[0].set_title('Spatial domain metrics')

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
        axes[1].bar(
            x + i * width, data_spot_label_2[i], width,
            label=methods[i], color=colors[i]
        )
axes[1].set_xticks(x + (len(methods)//2) * width)
axes[1].set_xticklabels(metrics_spot_label_2, fontsize=14)
axes[1].set_ylim(3, 10)   # tighter around observed values
axes[1].set_title('Centroid shift (↓)')
# axes[1].legend(frameon=True)


# --- Third subplot: Image metrics (exclude GPSA) ---
x = np.arange(len(metrics_image)) * 1.2
for i, method in enumerate(methods):
    xpos = x + i * width
    vals = data_image[i]
    if i == 0:
        axes[2].bar(x + i * width, data_image[i], width, label=method, facecolor='none', edgecolor='black',
                           hatch='//', linewidth=1, linestyle='--')
    else:
        if np.isnan(vals).all():
            # plot × markers
            for xi in xpos:
                axes[2].text(xi, 0.25, '×', ha='center', va='bottom',
                             fontsize=14, color=colors[i])
        else:
            axes[2].bar(xpos, vals, width, label=method, color=colors[i])


axes[2].set_xticks(x + (len(methods) // 2) * width)
axes[2].set_xticklabels(metrics_image, fontsize=14)
axes[2].set_ylim(0.2, 1.9)
axes[2].set_title('Histological alignment')
# axes[1].legend(frameon=True, bbox_to_anchor=(1.05, 0.7))
handles, labels = axes[1].get_legend_handles_labels()




# Put the legend at the left-most side of the figure
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=True)
plt.show()
