import matplotlib.pyplot as plt

# Define data
noise_levels = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
image_only = [0.8409, 0.8473, 0.8236, 0.8185, 0.7949, 0.7732, 0.6863, 0.6166]
image_position = [0.8722, 0.8728, 0.8735, 0.8773, 0.8658, 0.8364, 0.8313, 0.8192]

# Define custom colors
image_only_color = '#5293c9'     # Blue
image_position_color = '#df81a5' # Pink
# fill_color = '#ecadc4'           # Light pink fill
fill_color = '#b2d0e8'
# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, image_only, marker='o', label='Morphology', linewidth=2, color=image_only_color)
plt.plot(noise_levels, image_position, marker='s', label='Morphology + Spatial', linewidth=2, color=image_position_color)

# Annotate last values
plt.annotate(f"{image_only[-1]:.2f}", (noise_levels[-1]+0.003, image_only[-1]+0.02),
             textcoords="offset points", xytext=(-7, -12), ha='center', fontsize=11, color=image_only_color)
plt.annotate(f"{image_position[-1]:.2f}", (noise_levels[-1]-0.003, image_position[-1]),
             textcoords="offset points", xytext=(10, 10), ha='center', fontsize=11, color=image_position_color)

# Add fill between curves
plt.fill_between(noise_levels, image_only, image_position, color=fill_color, alpha=0.4, label='Performance Gap')

# Aesthetics
# plt.title('Performance Under Varying Noise Levels', fontsize=15, fontweight='bold')
plt.xlabel('Gaussian Noise Level', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='lower left')
plt.tight_layout()
plt.savefig('/home/huifang/workspace/grant/k99/resubmission/figures/6-1.png', dpi=300)
# Show plot
plt.show()
