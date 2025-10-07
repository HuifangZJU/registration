import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Data
# -------------------------------
levels = ["Original", "Degrade1", "Degrade2", "Degrade3", "Degrade4", "Degrade5"]
x = np.arange(len(levels))

data = {
    "Unaligned": [0.4406, 0.4406, 0.4406, 0.4406, 0.4406, 0.4406],
    "SimpleITK": [0.4835, 0.4878, 0.4818, 0.3856, 0.3394, 0.3806],
    "VoxelMorph": [0.4737, 0.4518, 0.4578, 0.4340, 0.4296, 0.3965],
    "NiceTrans": [0.5003, 0.4860, 0.4757, 0.4392, 0.3927, 0.3656],
    "PASTE": [0.4638, 0.4638, 0.4638, 0.4638, 0.4638, 0.4638],
    "GPSA": [0.4641, 0.4641, 0.4641, 0.4641, 0.4641, 0.4641],
    "SANTO": [0.4521, 0.4521, 0.4521, 0.4521, 0.4521, 0.4521],
    "Ours": [0.5025, 0.4843, 0.4826, 0.4804, 0.4873, 0.4729],
}

# -------------------------------
# Plot configuration
# -------------------------------
plt.figure(figsize=(9, 6))

colors = {
    "Unaligned": "#aaaaaa",
    "SimpleITK": "#c6a5cc",
    "VoxelMorph": "#8da0cb",
    "NiceTrans": "#66c2a5",
    "PASTE": "#ffd92f",
    "GPSA": "#fc8d62",
    "SANTO": "#e78ac3",
    "Ours": "#4daf4a",  # highlight in green
}

for method, vals in data.items():
    lw = 2.8 if method == "Ours" else 1.8
    alpha = 1.0 if method == "Ours" else 0.8
    plt.plot(x, vals, '-o', label=method, linewidth=lw, alpha=alpha, color=colors.get(method, None))

# -------------------------------
# Axis, legend, and aesthetics
# -------------------------------
plt.xticks(x, levels, rotation=25, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Degradation Level", fontsize=13)
plt.ylabel("Performance (Score)", fontsize=13)
plt.title("Performance Drop under Image Quality Degradation", fontsize=14, pad=10)
plt.grid(alpha=0.3)
plt.ylim(0.3, 0.55)
plt.legend(fontsize=10, loc="lower left", frameon=False)
plt.tight_layout()
plt.show()
