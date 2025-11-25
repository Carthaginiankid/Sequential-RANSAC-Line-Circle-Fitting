import numpy as np

# ---------------------- PARAMETERS ----------------------

T_LINE = 0.02
T_CIRCLE = 0.03
MAX_ITERS = 3000

MIN_INLIERS_LINE = 10
MIN_INLIERS_CIRCLE = 8

MIN_LINE_LENGTH = 0.4
R_MIN, R_MAX = 0.2, 0.5

LINE_SAMPLE = 8     # instead of 2 → more stable, less noisy
CIRCLE_SAMPLE = 3


# ---------------------- FITTING MODELS ----------------------

def fit_line_svd(points):
    """Fit line normal vector using SVD."""
    if len(points) < 2:
        return None, None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]            # normal = eigenvector with smallest variance
    normal /= np.linalg.norm(normal)
    c = -np.dot(normal, centroid)
    return normal, c


def line_length(points, normal):
    """Length = projection onto tangent direction."""
    tangent = np.array([-normal[1], normal[0]])
    proj = points @ tangent
    return proj.max() - proj.min()


def fit_circle(points):
    """Least-squares circle: linear formulation."""
    if len(points) < 3:
        return None, None
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2*x, 2*y, np.ones(len(points))]
    b = x*x + y*y
    try:
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        return None, None
    R = np.sqrt(cx*cx + cy*cy + c)
    return np.array([cx, cy]), R


# ---------------------- DISTANCES ----------------------

def dist_line(p, n, c):
    return abs(n[0]*p[0] + n[1]*p[1] + c)

def dist_circle(p, center, R):
    return abs(np.linalg.norm(p - center) - R)


# ---------------------- RANSAC LINE ----------------------

def ransac_line(points):
    best_inliers = []

    for _ in range(MAX_ITERS):
        if len(points) < LINE_SAMPLE:
            break

        idx = np.random.choice(len(points), LINE_SAMPLE, replace=False)
        sample = points[idx]

        n, c = fit_line_svd(sample)
        if n is None:
            continue

        d = np.abs(points @ n + c)
        inliers = np.where(d < T_LINE)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < MIN_INLIERS_LINE:
        return None, None

    final_pts = points[best_inliers]
    n, c = fit_line_svd(final_pts)
    length = line_length(final_pts, n)

    if length < MIN_LINE_LENGTH:
        return None, None

    return {'type': 'line', 'normal': n, 'c': c, 'length': length}, best_inliers


# ---------------------- RANSAC CIRCLE ----------------------

def ransac_circle(points):
    best_inliers = []

    for _ in range(MAX_ITERS):
        if len(points) < 3:
            break

        idx = np.random.choice(len(points), CIRCLE_SAMPLE, replace=False)
        p1, p2, p3 = points[idx]

        # Solve circle through 3 points
        A = np.array([
            [p2[0]-p1[0], p2[1]-p1[1]],
            [p3[0]-p1[0], p3[1]-p1[1]]
        ])
        b = np.array([
            (p2@p2 - p1@p1) / 2,
            (p3@p3 - p1@p1) / 2
        ])
        try:
            center = np.linalg.solve(A, b)
        except:
            continue

        R = np.linalg.norm(center - p1)

        d = np.abs(np.linalg.norm(points - center, axis=1) - R)
        inliers = np.where(d < T_CIRCLE)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < MIN_INLIERS_CIRCLE:
        return None, None

    final_pts = points[best_inliers]
    center, R = fit_circle(final_pts)

    if center is None or not (R_MIN <= R <= R_MAX):
        return None, None

    return {'type': 'circle', 'center': center, 'radius': R}, best_inliers


# ---------------------- SEQUENTIAL RANSAC ----------------------

def sequential_ransac(points):
    remaining = np.arange(len(points))
    models = []

    while len(remaining) > MIN_INLIERS_CIRCLE:
        cur = points[remaining]

        line_model, line_inl = ransac_line(cur)
        circle_model, circ_inl = ransac_circle(cur)

        # Choose best model
        candidates = []
        if line_model: candidates.append((line_model, line_inl, len(line_inl)))
        if circle_model: candidates.append((circle_model, circ_inl, len(circ_inl)))

        if not candidates:
            break

        best = max(candidates, key=lambda x: x[2])
        model, inlier_idx, _ = best

        # Map to original indices
        model['indices'] = remaining[inlier_idx]
        models.append(model)

        remaining = np.delete(remaining, inlier_idx)

    return models, remaining
