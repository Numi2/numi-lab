def project_ball(v, c, r):
    v = np.asarray(v, float)
    c = np.asarray(c, float)
    w = v - c
    norm = np.linalg.norm(w, axis=1)
    return c + w * np.minimum(1.0, float(r) / np.maximum(norm, 1e-300))[:, None]

def project_node_sets(v, source_center, source_cap, plane_rows, plane_normals, plane_lower, anchors, trust_center, trust_radius):
    """Exact per-node active-set projection onto source/trust balls and exit plane."""
    v = np.asarray(v, float)
    source_center = np.asarray(source_center, float)
    trust_center = np.asarray(trust_center, float)
    out = project_ball(v, trust_center, trust_radius)
    plane_by_node = {int(i): (np.asarray(n, float), float(h)) for i, n, h in zip(plane_rows, plane_normals, plane_lower)}
    anchor_set = set(map(int, anchors))
    overlap_count = 0
    full_enumeration_count = 0
    for row in range(len(v)):
        if row in anchor_set:
            out[row] = 0.0
            continue
        pair = plane_by_node.get(row)
        source_contains_trust = np.linalg.norm(trust_center[row] - source_center[row]) + trust_radius <= source_cap + 1e-10
        if pair is not None:
            n, h = pair
            if source_contains_trust:
                out[row] = project_ball_halfspace(v[row:row + 1], trust_center[row:row + 1], trust_radius, n[None, :], np.array([h]), np.array([0], np.int64))[0]
            else:
                overlap_count += 1
                full_enumeration_count += 1
                out[row] = project_two_balls_halfspace(v[row], source_center[row], source_cap, trust_center[row], trust_radius, n, h)
        elif np.linalg.norm(out[row] - source_center[row]) > source_cap + 1e-10:
            overlap_count += 1
            full_enumeration_count += 1
            out[row] = project_two_balls_halfspace(v[row], source_center[row], source_cap, trust_center[row], trust_radius)
    for row, (n, h) in plane_by_node.items():
        if row in anchor_set and h > 1e-10:
            raise InfeasibleProjection(row, h, source_cap)
    ball_violation = np.maximum(0.0, np.linalg.norm(out - source_center, axis=1) - source_cap)
    trust_violation = np.maximum(0.0, np.linalg.norm(out - trust_center, axis=1) - trust_radius)
    plane_violation = np.array([max(0.0, h - float(np.dot(n, out[row]))) for row, (n, h) in plane_by_node.items()])
    err = max(float(ball_violation.max(initial=0.0)), float(trust_violation.max(initial=0.0)), float(plane_violation.max(initial=0.0)))
    return (out, {'method': 'exact_active_set_candidates', 'intersection_residual_mm': err, 'overlap_nodes': overlap_count, 'full_active_set_enumerations': full_enumeration_count})
