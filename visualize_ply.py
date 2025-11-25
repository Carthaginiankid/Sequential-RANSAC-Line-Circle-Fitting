import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys

def load_ply_file(filename):
    """Load PLY file and extract points and colors."""
    points = []
    colors = []
    
    with open(filename, 'r') as f:
        # Skip header until we find vertex count
        vertex_count = 0
        for line in f:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line.strip() == 'end_header':
                break
        
        # Read vertex data
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                points.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def visualize_ply(filename):
    """Visualize PLY file with colors."""
    print(f"Loading {filename}...")
    points, colors = load_ply_file(filename)
    print(f"Loaded {len(points)} points")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # Normalize colors to [0, 1] range
    colors_normalized = colors / 255.0
    
    # Plot points with their colors
    ax.scatter(points[:, 0], points[:, 1], 
              c=colors_normalized, s=20, alpha=0.7, 
              edgecolors='none', zorder=2)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_title(f'PLY Visualization: {filename}\n'
                f'Total points: {len(points)}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Count points by color
    red_points = np.sum(np.all(colors == [255, 0, 0], axis=1))
    blue_points = np.sum(np.all(colors == [0, 0, 255], axis=1))
    white_points = np.sum(np.all(colors == [255, 255, 255], axis=1))
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = []
    if red_points > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label=f'Circle inliers ({red_points} points)')
        )
    if blue_points > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, label=f'Line inliers ({blue_points} points)')
        )
    if white_points > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=8, alpha=0.6, label=f'Remaining ({white_points} points)')
        )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    filename = "7.ply"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    visualize_ply(filename)

