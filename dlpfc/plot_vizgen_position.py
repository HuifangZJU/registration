import matplotlib.pyplot as plt

# New data from the second figure
noise_levels_swap = [0, 2, 4, 6, 8, 10, 15, 20]
position_only = [0.8204, 0.8211, 0.8383, 0.8351, 0.8422, 0.8377, 0.8045, 0.8134]
image_position = [0.8722, 0.8722, 0.8907, 0.8957, 0.9042, 0.8907, 0.8869, 0.8875]

# Define colors
position_color = '#5293c9'       # blue
image_position_color = '#df81a5' # pink
# fill_color = '#ecadc4'           # soft pink fill
fill_color = '#b2d0e8'           # soft pink fill
# Plot
plt.figure(figsize=(8, 5))
plt.plot(noise_levels_swap, position_only, marker='o', label='Spatial', linewidth=2, color=position_color)
plt.plot(noise_levels_swap, image_position, marker='s', label='Morphology + Spatial', linewidth=2, color=image_position_color)

# Annotate last values
plt.annotate(f"{position_only[-1]:.2f}", (noise_levels_swap[-1], position_only[-1]),
             textcoords="offset points", xytext=(-10, -15), ha='center', fontsize=11, color=position_color)
plt.annotate(f"{image_position[-1]:.2f}", (noise_levels_swap[-1]-0.2, image_position[-1]),
             textcoords="offset points", xytext=(10, 10), ha='center', fontsize=11, color=image_position_color)

# Fill between the two curves
plt.fill_between(noise_levels_swap, position_only, image_position, color=fill_color, alpha=0.4, label='Performance Gap')

# Aesthetic settings
# plt.title('Effect of Cell Swap on Performance', fontsize=15, fontweight='bold')
plt.xlabel('Gaussian Noise Level', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
# plt.ylim(0.7, 1)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='lower center',bbox_to_anchor=(0.45, 0))
plt.tight_layout()
plt.savefig('/home/huifang/workspace/grant/k99/resubmission/figures/6-2.png', dpi=300)
# Show plot
plt.show()
