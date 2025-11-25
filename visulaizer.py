import pyvista as pv
import numpy as np
from load import load_lidar_file, sequential_ransac

# Load data and run RANSAC
filename = "7"  # change this to 7, 17, or 27
points = load_lidar_file(filename)
circles, lines, remaining = sequential_ransac(points)

print(f"Found {len(circles)} circles and {len(lines)} lines")

# Create plotter
plotter = pv.Plotter(window_size=[1400, 1000])
plotter.set_background('white')
plotter.show_grid()

# Plot remaining points (white/gray) - background
if len(remaining) > 0:
    remaining_3d = np.column_stack([remaining, np.zeros(len(remaining))])
    remaining_cloud = pv.PolyData(remaining_3d)
    plotter.add_mesh(remaining_cloud, color='lightgray', point_size=8,
                    render_points_as_spheres=True, opacity=0.5)

# Draw circle arcs (only the visible portion due to self-occlusion)
for circle, inliers in circles:
    x0, y0, r = circle
    
    # Plot circle inlier points (red)
    if len(inliers) > 0:
        circle_3d = np.column_stack([inliers, np.zeros(len(inliers))])
        circle_cloud = pv.PolyData(circle_3d)
        plotter.add_mesh(circle_cloud, color='red', point_size=12,
                        render_points_as_spheres=True)
    
    # Compute and draw the arc (only visible portion due to self-occlusion)
    # The arc represents the portion of the circle that was actually detected
    if len(inliers) > 0:
        # Convert inlier points to angles relative to circle center
        vectors = inliers - np.array([x0, y0])
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # Angles in [-pi, pi]
        
        # Find the angular span of the detected arc
        # For arcs (not full circles), we find the range that covers most points
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        angle_span = max_angle - min_angle
        
        # If span is small (< pi), it's a simple arc
        # If span is large (> pi), the arc might wrap around, but for self-occlusion
        # arcs are typically < 180 degrees, so we handle the simple case
        if angle_span < np.pi:
            # Simple arc: draw from min to max with small padding
            padding = angle_span * 0.05  # 5% padding
            theta = np.linspace(min_angle - padding, max_angle + padding, 100)
        else:
            # Arc might wrap or span more than 180 degrees
            # Use the points directly to create a smooth arc
            # Sort points by angle and draw arc through them
            sorted_indices = np.argsort(angles)
            # Create arc that goes through all points in order
            theta_start = angles[sorted_indices[0]]
            theta_end = angles[sorted_indices[-1]]
            # If wrapping, adjust
            if theta_end < theta_start:
                theta_end += 2*np.pi
            theta = np.linspace(theta_start, theta_end, 100)
        
        # Draw the arc
        arc_points = np.column_stack([
            x0 + r * np.cos(theta),
            y0 + r * np.sin(theta),
            np.zeros(len(theta))
        ])
        arc_poly = pv.PolyData(arc_points)
        arc_tube = arc_poly.tube(radius=0.03)
        plotter.add_mesh(arc_tube, color='red', opacity=0.7)
    
    # Mark center
    center_point = pv.PolyData(np.array([[x0, y0, 0]]))
    plotter.add_mesh(center_point, color='darkred', point_size=15,
                    render_points_as_spheres=True)

# Draw line segments
for line, inliers in lines:
    a, b, c = line
    
    # Plot line inlier points (blue)
    if len(inliers) > 0:
        line_3d = np.column_stack([inliers, np.zeros(len(inliers))])
        line_cloud = pv.PolyData(line_3d)
        plotter.add_mesh(line_cloud, color='blue', point_size=12,
                        render_points_as_spheres=True)
    
    # Compute line segment endpoints
    direction = np.array([-b, a])
    direction = direction / np.linalg.norm(direction)
    centroid = np.mean(inliers, axis=0)
    centered = inliers - centroid
    projections = centered @ direction
    p1 = centroid + np.min(projections) * direction
    p2 = centroid + np.max(projections) * direction
    
    # Draw the line segment
    line_points = np.array([[p1[0], p1[1], 0], [p2[0], p2[1], 0]])
    line_poly = pv.PolyData(line_points)
    line_tube = line_poly.tube(radius=0.03)
    plotter.add_mesh(line_tube, color='blue', opacity=0.7)

# Set camera to top view
plotter.camera_position = 'xy'
plotter.camera.elevation = 90

# Add title
plotter.add_text(f'RANSAC Results: {filename}\n'
                f'{len(circles)} circles, {len(lines)} lines',
                font_size=14, position='upper_left')

# Show the plot
plotter.show()
