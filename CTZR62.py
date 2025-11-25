import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Data Loading
# -------------------------
def load_polar_data(filename):
    data = np.loadtxt(filename)
    angles = np.deg2rad(data[:, 0])
    radii = data[:, 1] / 1000.0  # mm → meters
    x, y = radii * np.cos(angles), radii * np.sin(angles)
    return np.column_stack((x, y))

# -------------------------
# Geometry Fitting
# -------------------------
def fit_circle(points):
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([np.ones(len(points)), -2*x, -2*y])
    b = -(x**2 + y**2)
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    x0, y0 = params[1], params[2]
    r = np.sqrt(x0**2 + y0**2 - params[0])
    return np.array([x0, y0]), r

def fit_line(points):
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    return centroid, Vt[0]

# -------------------------
# RANSAC for Circle & Line
# -------------------------
def ransac_circle(points, threshold=0.035, min_inliers=8, min_r=0.2, max_r=0.5, iterations=3000):
    best_center, best_radius, best_inliers = None, None, None
    max_count = 0

    for _ in range(iterations):
        if len(points) < 3:
            break
        sample = points[np.random.choice(len(points), 3, replace=False)]
        try:
            center, radius = fit_circle(sample)
        except np.linalg.LinAlgError:
            continue
        if not (min_r <= radius <= max_r) or np.isnan(radius):
            continue
        inliers = np.abs(np.linalg.norm(points - center, axis=1) - radius) < threshold
        if inliers.sum() > max_count:
            max_count = inliers.sum()
            best_center, best_radius, best_inliers = center, radius, inliers

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None, None, None

    center, radius = fit_circle(points[best_inliers])
    inliers = np.abs(np.linalg.norm(points - center, axis=1) - radius) < threshold
    if inliers.sum() < min_inliers:
        return None, None, None
    return center, radius, inliers

def ransac_line(points, threshold=0.025, min_inliers=10, min_length=0.4, iterations=2000):
    best_centroid, best_direction, best_inliers = None, None, None
    max_count = 0

    for _ in range(iterations):
        if len(points) < 2:
            break
        sample = points[np.random.choice(len(points), 2, replace=False)]
        direction = sample[1] - sample[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction /= norm
        v = points - sample[0]
        perp = v - np.outer(np.dot(v, direction), direction)
        inliers = np.linalg.norm(perp, axis=1) < threshold
        if inliers.sum() > max_count:
            max_count = inliers.sum()
            best_centroid, best_direction, best_inliers = sample[0], direction, inliers

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None, None, None

    inlier_pts = points[best_inliers]
    centroid, direction = fit_line(inlier_pts)
    v = points - centroid
    perp = v - np.outer(np.dot(v, direction), direction)
    inliers = np.linalg.norm(perp, axis=1) < threshold

    if inliers.sum() < min_inliers:
        return None, None, None

    projections = np.dot(points[inliers] - centroid, direction)
    if projections.max() - projections.min() < min_length:
        return None, None, None

    return centroid, direction, inliers

# -------------------------
# Sequential RANSAC
# -------------------------
def sequential_ransac(points, circle_threshold=0.035, line_threshold=0.025):
    remaining = np.ones(len(points), dtype=bool)
    circles, lines = [], []

    while remaining.sum() >= 8:
        pts = points[remaining]
        c_center, c_radius, c_inliers = ransac_circle(pts, threshold=circle_threshold)
        l_centroid, l_dir, l_inliers = ransac_line(pts, threshold=line_threshold)

        c_count = 0 if c_inliers is None else c_inliers.sum()
        l_count = 0 if l_inliers is None else l_inliers.sum()
        if c_count == 0 and l_count == 0:
            break

        if c_count >= l_count:
            idx = np.where(remaining)[0][c_inliers]
            circles.append({'center': c_center, 'radius': c_radius, 'points': points[idx], 'indices': idx})
            remaining[idx] = False
        else:
            idx = np.where(remaining)[0][l_inliers]
            inlier_pts = points[idx]
            lines.append({'centroid': l_centroid, 'direction': l_dir, 'points': inlier_pts, 'indices': idx})
            remaining[idx] = False

    return circles, lines, remaining

# -------------------------
# Save PLY
# -------------------------
def save_ply(points, circles, lines, unused_mask, filename="output.ply"):
    colors = np.full((len(points), 3), 128, dtype=np.uint8)  # grey default
    for c in circles:
        colors[c['indices']] = [255, 0, 0]  # red
    for l in lines:
        colors[l['indices']] = [0, 0, 255]  # blue

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} 0.0 {r} {g} {b}\n")
    print(f"\n✓ Saved PLY file: {filename}")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)
    CIRCLE_THRESHOLD = 0.03
    LINE_THRESHOLD = 0.025
    files = ["5"]

    print("\nSequential RANSAC: Circle and Line Detection")
    print(f"Circle threshold: {CIRCLE_THRESHOLD}m, Line threshold: {LINE_THRESHOLD}m\n")

    for fname in files:
        print(f"# Processing: {fname}")
        try:
            points = load_polar_data(fname)
            print(f"Loaded {len(points)} points: X [{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
                  f"Y [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
            
            circles, lines, unused = sequential_ransac(
                points, 
                circle_threshold=CIRCLE_THRESHOLD, 
                line_threshold=LINE_THRESHOLD
            )
            
            # Save PLY with file-specific name
            out_name = fname + "_output.ply"
            save_ply(points, circles, lines, unused, filename=out_name)

        except FileNotFoundError:
            print(f"✗ File '{fname}' not found")
        except Exception as e:
            print(f"✗ Error processing '{fname}': {e}")
            import traceback
            traceback.print_exc()

    print("\nProcessing Complete!")
