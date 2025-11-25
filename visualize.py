import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sol import load_lidar_file, sequential_ransac

def visualize_results(filename, circles, lines, remaining_points, save_path=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    if len(remaining_points) > 0:
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1], 
                  c='lightgray', s=15, alpha=0.6, edgecolors='none', 
                  label=f'Remaining points ({len(remaining_points)})', zorder=1)
    
    for circle, inliers in circles:
        x0, y0, r = circle
        
        ax.scatter(inliers[:, 0], inliers[:, 1], 
                  c='red', s=20, alpha=0.7, edgecolors='darkred', linewidths=0.5,
                  zorder=3)
        
        circle_patch = Circle((x0, y0), r, fill=False, 
                             edgecolor='red', linewidth=2, linestyle='-',
                             alpha=0.8, zorder=2)
        ax.add_patch(circle_patch)
        
        ax.plot(x0, y0, 'ro', markersize=6, markeredgecolor='darkred', 
               markeredgewidth=1, zorder=4)
    
    for line, inliers in lines:
        a, b, c = line
        
        ax.scatter(inliers[:, 0], inliers[:, 1], 
                  c='blue', s=20, alpha=0.7, edgecolors='darkblue', linewidths=0.5,
                  zorder=3)
        
        
        direction = np.array([-b, a])  # Direction vector perpendicular to normal
        direction = direction / np.linalg.norm(direction)
        
        # Project inliers onto line direction
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        projections = centered @ direction
        
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        p1 = centroid + min_proj * direction
        p2 = centroid + max_proj * direction
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
               color='blue', linewidth=2, alpha=0.8, zorder=2)
    
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_title(f'RANSAC Results: {filename}\n'
                f'Circles: {len(circles)}, Lines: {len(lines)}, '
                f'Remaining: {len(remaining_points)}', 
                fontsize=14, fontweight='bold', pad=20)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, markeredgecolor='darkred', markeredgewidth=1,
               label=f'Circle inliers ({sum(len(inliers) for _, inliers in circles)} points)'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Circles ({len(circles)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, markeredgecolor='darkblue', markeredgewidth=1,
               label=f'Line inliers ({sum(len(inliers) for _, inliers in lines)} points)'),
        Line2D([0], [0], color='blue', linewidth=2, label=f'Lines ({len(lines)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
               markersize=6, alpha=0.6, label=f'Remaining ({len(remaining_points)} points)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    
    stats_text = f'Statistics:\n'
    stats_text += f'Total circles: {len(circles)}\n'
    stats_text += f'Total lines: {len(lines)}\n'
    if circles:
        circle_radii = [r for (x0, y0, r), _ in circles]
        stats_text += f'Circle radii: {[f"{r:.3f}" for r in circle_radii]} m\n'
    if lines:
        line_lengths = []
        for (a, b, c), inliers in lines:
            direction = np.array([-b, a])
            direction = direction / np.linalg.norm(direction)
            centroid = np.mean(inliers, axis=0)
            centered = inliers - centroid
            projections = centered @ direction
            length = np.max(projections) - np.min(projections)
            line_lengths.append(length)
        stats_text += f'Line lengths: {[f"{l:.3f}" for l in line_lengths[:5]]} m\n'
        if len(line_lengths) > 5:
            stats_text += f'  ... and {len(line_lengths) - 5} more'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    return fig, ax


def visualize_all_files(file_numbers=[7, 17, 27], show_plots=True, save_plots=True):
    figs = []
    
    for num in file_numbers:
        filename = f"{num}"
        print(f"\nVisualizing {filename}...")
        
        points = load_lidar_file(filename)
        circles, lines, remaining = sequential_ransac(points)
        
        print(f"  Found {len(circles)} circles and {len(lines)} lines")
        print(f"  Remaining points: {len(remaining)}")
        
        save_path = f"{num}_visualization.png" if save_plots else None
        fig, ax = visualize_results(filename, circles, lines, remaining, save_path)
        figs.append(fig)
        
        if show_plots:
            plt.show(block=False)
    
    if show_plots:
        plt.show()
    
    return figs


if __name__ == "__main__":
    visualize_all_files(show_plots=True, save_plots=True)

