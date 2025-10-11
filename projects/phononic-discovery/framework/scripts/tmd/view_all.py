#!/usr/bin/env python3
"""
Quick viewer to display all generated visualizations
"""
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

viz_dir = Path("/workspaces/algo/qcmd_hybrid_framework/discovery_visualization")

# List all PNG files
png_files = sorted(viz_dir.glob("*.png"))

print(f"Found {len(png_files)} visualizations:")
for f in png_files:
    print(f"  - {f.name}")

# Create a figure showing all visualizations
n_images = len(png_files)
fig = plt.figure(figsize=(20, 25))

for i, png_file in enumerate(png_files, 1):
    ax = fig.add_subplot(n_images, 1, i)
    img = imread(str(png_file))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(png_file.name, fontsize=16, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(viz_dir / "ALL_VISUALIZATIONS.png", dpi=150, bbox_inches='tight')
print(f"\n✅ Combined visualization saved: {viz_dir}/ALL_VISUALIZATIONS.png")
print(f"\n🌐 INTERACTIVE 3D VIEWER: {viz_dir}/1_interactive_3d_viewer.html")
print("   Open this HTML file in your browser to see the rotating molecule!")
