import numpy as np
import matplotlib.pyplot as plt

# Parameters
center_spacing = 100  # µm, center-to-center distance
spot_diameter = 55    # µm
spot_radius = spot_diameter / 2

# Hexagonal (triangular) lattice coordinates for a small patch
rows, cols = 3, 3  # you can change these to show more/less spots
coords = []
for i in range(rows):
    for j in range(cols):
        x = j * center_spacing + (center_spacing / 2 if i % 2 else 0)
        y = i * (center_spacing * np.sqrt(3) / 2)
        coords.append((x, y))
coords = np.array(coords)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
for (x, y) in coords:
    circle = plt.Circle((x, y), spot_radius, fill=False)
    ax.add_patch(circle)

# Annotate one center-to-center distance
x0, y0 = coords[4]
x1, y1 = coords[5]
# --- scale bar: 100 µm ------------------------------------------
ax.annotate(                       # double-headed arrow
    '', xy=(x1, y0-4), xytext=(x0, y0-5),
    arrowprops=dict(arrowstyle='<->', lw=1.4, color='k')
)
ax.text((x0 + x1) / 2+10, y0 -24, '100 µm',
        ha='center', va='bottom',fontsize=25)

# --- spot diameter: 55 µm ---------------------------------------
ax.annotate(
    '', xy=(x0 + spot_radius, y0), xytext=(x0 - spot_radius, y0),
    arrowprops=dict(arrowstyle='<->', lw=1.4, color='r')
)
ax.text(x0, y0 + 10, '55 µm',
        ha='center', va='bottom',fontsize=25)


# Formatting
ax.set_aspect('equal')
ax.set_xlim(-0, cols * center_spacing + center_spacing-20)
ax.set_ylim(-30, rows * (center_spacing * np.sqrt(3) / 2))
ax.axis('off')

# Calculate and display coverage fraction
spot_area = np.pi * spot_radius**2
cell_area = (np.sqrt(3) / 2) * center_spacing**2  # area per spot in a hex lattice
coverage_fraction = spot_area / cell_area * 100

plt.title(f'10X Visium Spot coverage ≈ {coverage_fraction:.1f}% ',fontsize=26)

plt.show()
