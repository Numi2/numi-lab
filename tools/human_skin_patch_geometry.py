"""Ordered annular triangulation for local Human surface repair.

This constructs a candidate strip only. Callers must independently check
closure, intersections, layer cells, source ancestry and mass before admission.
"""
import numpy as np


def zipper_annulus(vertices, outer_loop, inner_loop, *, quality_floor=1e-5,
                   max_edge=None, phase_band=0.35):
    """Return (triangles, report) for disjoint, consistently ordered loops.

    The returned outer boundary follows outer_loop; its inner boundary runs
    opposite inner_loop. Reverse the complete result if the existing boundary
    requires opposite winding. Every cyclic inner phase is considered.
    """
    x = np.asarray(vertices, dtype=np.float64)
    raw_a, raw_b = np.asarray(outer_loop), np.asarray(inner_loop)
    if raw_a.ndim != 1 or raw_b.ndim != 1 or raw_a.dtype.kind not in "iu" or raw_b.dtype.kind not in "iu":
        raise ValueError("one-dimensional integer index loops required")
    a = raw_a.astype(np.int64)
    b0 = raw_b.astype(np.int64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite Nx3 vertices required")
    if min(len(a), len(b0)) < 3:
        raise ValueError("loops need at least three vertices")
    if (len(np.unique(a)) != len(a) or len(np.unique(b0)) != len(b0)
            or np.intersect1d(a, b0).size or min(a.min(), b0.min()) < 0
            or max(a.max(), b0.max()) >= len(x)):
        raise ValueError("valid disjoint simple index loops required")
    if not 0 < quality_floor < 1 or not 0 < phase_band <= 1:
        raise ValueError("quality floor and phase band must be positive fractions")
    n, m = len(a), len(b0)
    scale = max(np.ptp(x[np.r_[a, b0]], axis=0).max(), 1e-12)
    if max_edge is not None and (not np.isfinite(max_edge) or max_edge <= 0):
        raise ValueError("maximum edge must be finite and positive")
    edge_limit = np.inf if max_edge is None else float(max_edge)
    best = None

    def metric(p, q, r):
        e0 = q - p
        e1 = r - p
        e2 = r - q
        areas2 = np.linalg.norm(np.cross(e0, e1), axis=-1)
        lengths2 = np.stack((np.sum(e0*e0, axis=-1),
                             np.sum(e1*e1, axis=-1),
                             np.sum(e2*e2, axis=-1)), axis=-1)
        quality = 2*np.sqrt(3)*areas2 / np.maximum(lengths2.sum(-1), 1e-300)
        longest2 = lengths2.max(-1)
        cost = longest2/(scale*scale) + 0.01/np.maximum(quality, 1e-300)
        return np.where((quality >= quality_floor) &
                        (longest2 <= edge_limit*edge_limit), cost, np.inf)

    aa = np.r_[a, a[:1]]
    for phase in range(m):
        b = np.roll(b0, -phase)
        bb = np.r_[b, b[:1]]
        # Advancing A: (A_i,A_i+1,B_j).
        ca = metric(np.broadcast_to(x[aa[:-1], None], (n, m+1, 3)),
                    np.broadcast_to(x[aa[1:], None], (n, m+1, 3)),
                    np.broadcast_to(x[bb][None], (n, m+1, 3)))
        # Advancing B: (A_i,B_j+1,B_j).
        cb = metric(np.broadcast_to(x[aa, None], (n+1, m, 3)),
                    np.broadcast_to(x[bb[1:]][None], (n+1, m, 3)),
                    np.broadcast_to(x[bb[:-1]][None], (n+1, m, 3)))
        d = np.full((n+1, m+1), np.inf)
        prev = np.zeros((n+1, m+1), dtype=np.uint8)
        d[0, 0] = 0.0
        for i in range(n+1):
            lo = max(0, int(np.ceil((i/n-phase_band)*m)))
            hi = min(m, int(np.floor((i/n+phase_band)*m)))
            for j in range(lo, hi+1):
                if i == j == 0:
                    continue
                da = d[i-1, j]+ca[i-1, j] if i else np.inf
                db = d[i, j-1]+cb[i, j-1] if j else np.inf
                if da <= db:
                    d[i, j], prev[i, j] = da, 1
                else:
                    d[i, j], prev[i, j] = db, 2
        if not np.isfinite(d[n, m]) or (best is not None and d[n, m] >= best[0]):
            continue
        i, j = n, m
        faces = []
        while i or j:
            if prev[i, j] == 1:
                faces.append((aa[i-1], aa[i], bb[j]))
                i -= 1
            elif prev[i, j] == 2:
                faces.append((aa[i], bb[j], bb[j-1]))
                j -= 1
            else:
                raise RuntimeError("incomplete dynamic-programming path")
        best = (float(d[n, m]), phase, np.asarray(faces[::-1], dtype=np.int64))
    if best is None:
        raise ValueError("no strip satisfies quality/edge/progress constraints")
    score, phase, faces = best
    edges = {}
    for face in faces:
        for i, j in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(i), int(j))))
            edges.setdefault(key, []).append((int(i), int(j)))
    boundary = {v[0] for v in edges.values() if len(v) == 1}
    expected = {(int(i), int(j)) for i, j in zip(a, np.roll(a, -1))}
    expected |= {(int(j), int(i)) for i, j in zip(b0, np.roll(b0, -1))}
    if boundary != expected or any(len(v) > 2 or
            (len(v) == 2 and v[0] != v[1][::-1]) for v in edges.values()):
        raise RuntimeError("constructed strip violates oriented boundary contract")
    pts = x[faces]
    area = np.linalg.norm(np.cross(pts[:,1]-pts[:,0], pts[:,2]-pts[:,0]), axis=1)/2
    edge = np.linalg.norm(pts-np.roll(pts, -1, axis=1), axis=2)
    quality = 4*np.sqrt(3)*area/np.sum(edge*edge, axis=1)
    return faces, dict(score=score, inner_phase=phase, triangles=len(faces),
                       minimum_area_m2=float(area.min()),
                       minimum_quality=float(quality.min()),
                       maximum_edge_m=float(edge.max()),
                       oriented_boundary_exact=True,
                       physical_admission=False)

def ordered_layer_prism_tets(faces):
    """Return node IDs, O/I/D layer IDs and signs for native three-tet prisms.

    Axes are (face, region [skin, fat], local tet, corner). Both regions use
    identical sorted-global-ID connectivity; only the layer pair changes.
    Signs follow the source face winding, using integer inversion counts.
    No material, thickness or admission criterion is changed here.
    """
    raw = np.asarray(faces)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu":
        raise ValueError("integer Fx3 faces required")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError("nonnegative int64 source IDs required")
    f = raw.astype(np.int64)
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2]) | (f[:, 1] == f[:, 2])):
        raise ValueError("three distinct source IDs required per face")
    v = np.sort(f, axis=1)
    rows = v[:, np.array([[0, 1, 2, 2], [0, 1, 1, 2], [0, 0, 1, 2]])]
    nodes = np.repeat(rows[:, None], 2, axis=1)
    local_layers = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int8)
    layers = np.broadcast_to(local_layers[None, None] +
                             np.arange(2, dtype=np.int8)[None, :, None, None],
                             nodes.shape).copy()
    # Do not sum NumPy bool scalars: their addition is logical OR.
    inversions = ((f[:, 0] > f[:, 1]).astype(np.int8) +
                  (f[:, 0] > f[:, 2]).astype(np.int8) +
                  (f[:, 1] > f[:, 2]).astype(np.int8))
    sign = np.where(inversions % 2, 1., -1.)
    signs = np.broadcast_to(sign[:, None, None] * np.array([1., -1., 1.]),
                            (len(f), 2, 3)).copy()
    return nodes, layers, signs


def tetra_determinants_and_gradients(points):
    """Batched signed six-volumes and exact coordinate gradients.

    The final axes of points are (4, 3); preceding batch axes are retained.
    Gradients have the same shape as points, in physical coordinate units.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim < 2 or p.shape[-2:] != (4, 3) or not np.isfinite(p).all():
        raise ValueError("finite (...,4,3) tetrahedron points required")
    a, b, c = p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :], p[..., 3, :] - p[..., 0, :]
    g1, g2, g3 = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    gradient = np.stack((-(g1 + g2 + g3), g1, g2, g3), axis=-2)
    return np.sum(a * g1, axis=-1), gradient

def split_surface_edges(vertices, faces, edge_parameters):
    """Conformingly split selected edges in every incident source triangle.

    edge_parameters maps oriented (i,j) to 0<t<1 from i toward j. Return
    (vertices, faces, parent_face_rows, new_vertex_edges, new_vertex_weights).
    The last two arrays give affine ancestry in the input vertex array.
    Untouched triangles retain their exact indices and orientation.
    This is source preparation, not physical or collision admission.
    """
    x = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(faces)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite Nx3 vertices required")
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu":
        raise ValueError("integer Fx3 faces required")
    if raw.size and (np.any(raw < 0) or np.any(raw >= len(x))):
        raise ValueError("source face index outside vertex range")
    f = raw.astype(np.int64)
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2]) | (f[:, 1] == f[:, 2])):
        raise ValueError("three distinct source IDs required per face")
    cuts = {}
    for edge, value in edge_parameters.items():
        e = np.asarray(edge)
        if e.shape != (2,) or e.dtype.kind not in "iu":
            raise ValueError("integer source edge pairs required")
        i, j = map(int, e)
        t = float(value)
        if not 0 <= i < len(x) or not 0 <= j < len(x) or i == j or not np.isfinite(t) or not 0 < t < 1:
            raise ValueError("valid source edge and interior finite parameter required")
        if i > j:
            i, j, t = j, i, 1-t
        key = (i, j)
        if key in cuts and abs(cuts[key]-t) > 1e-14:
            raise ValueError("conflicting oriented edge parameters")
        cuts[key] = t
    edges = np.array(sorted(cuts), dtype=np.int64).reshape(-1, 2)
    weights = np.array([[1-cuts[tuple(e)], cuts[tuple(e)]] for e in edges],
                       dtype=np.float64).reshape(-1, 2)
    new_points = np.sum(x[edges]*weights[:, :, None], axis=1)
    result_x = np.concatenate((x, new_points))
    ids = {tuple(e): len(x)+k for k, e in enumerate(edges)}
    used = set()
    result_f, parents = [], []
    for fi, face in enumerate(f):
        polygon = []
        touched = False
        for i, j in zip(face, np.roll(face, -1)):
            polygon.append(int(i))
            key = tuple(sorted((int(i), int(j))))
            if key in ids:
                polygon.append(ids[key])
                used.add(key)
                touched = True
        if not touched:
            result_f.append(face.tolist())
            parents.append(fi)
            continue
        p = x[face]
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        norm = np.linalg.norm(normal)
        if norm <= 0:
            raise ValueError("cannot split a degenerate source triangle")
        normal /= norm
        floor = norm*1e-14
        # The polygon is a convex source triangle with collinear edge points.
        # Remove positive-area ears, never a collinear edge-point triple.
        while len(polygon) > 3:
            candidates = []
            for k in range(len(polygon)):
                a, b, c = polygon[k-1], polygon[k], polygon[(k+1) % len(polygon)]
                u, v = result_x[b]-result_x[a], result_x[c]-result_x[a]
                area2 = float(np.dot(np.cross(u, v), normal))
                if area2 > floor:
                    # A source-edge point on the candidate diagonal must not
                    # be skipped: doing so leaves a degenerate remainder.
                    other = [j for j in polygon if j not in (a,b,c)]
                    contains = False
                    for j in other:
                        point = result_x[j]
                        h = [np.dot(np.cross(result_x[t]-result_x[s],
                                             point-result_x[s]),normal)
                             for s,t in ((a,b),(b,c),(c,a))]
                        if min(h) >= -floor:
                            contains = True
                            break
                    if contains:
                        continue
                    lengths2 = np.dot(u,u)+np.dot(v,v)+np.sum((result_x[c]-result_x[b])**2)
                    candidates.append((area2/max(lengths2, 1e-300), -k, (a,b,c)))
            if not candidates:
                raise ValueError("source edge split has no nondegenerate triangulation")
            _, negative_k, tri = max(candidates)
            result_f.append(tri); parents.append(fi)
            polygon.pop(-negative_k)
        a,b,c = polygon
        if np.dot(np.cross(result_x[b]-result_x[a], result_x[c]-result_x[a]), normal) <= floor:
            raise ValueError("degenerate final source-edge triangle")
        result_f.append(polygon); parents.append(fi)
    if used != set(cuts):
        raise ValueError("selected split edge is absent from source faces")
    return (result_x, np.asarray(result_f, dtype=np.int64).reshape(-1,3),
            np.asarray(parents,dtype=np.int64), edges, weights)


def transverse_surface_section(vertices, faces, value, *, axis=2):
    """Audit a transverse plane section using mesh-edge identity.

    No coordinate rounding or welding is used: coincident but disconnected
    sheets remain disconnected. A plane through a referenced vertex is
    rejected so a caller must choose and record an unambiguous nearby plane.
    This reports topology only, not collision clearance or anatomical validity.
    """
    x = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(faces)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite Nx3 vertices required")
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu":
        raise ValueError("integer Mx3 faces required")
    f = raw.astype(np.int64)
    if axis not in (0, 1, 2) or not np.isfinite(value):
        raise ValueError("finite plane and axis 0, 1 or 2 required")
    if len(f) and (f.min() < 0 or f.max() >= len(x)
                   or np.any(np.sort(f, axis=1)[:, 1:] ==
                             np.sort(f, axis=1)[:, :-1])):
        raise ValueError("valid nonrepeating triangle indices required")
    h = x[:, axis] - value
    if len(f) and np.any(h[np.unique(f)] == 0):
        raise ValueError("section plane passes through a referenced vertex")
    active = np.flatnonzero((h[f].min(1) < 0) & (h[f].max(1) > 0))
    edge_to_node, edges, xyz, segments = {}, [], [], []
    for fi in active:
        tri = f[fi]
        nodes = []
        for a, b in zip(tri, np.roll(tri, -1)):
            if (h[a] < 0) == (h[b] < 0):
                continue
            edge = (int(min(a, b)), int(max(a, b)))
            if edge not in edge_to_node:
                u, v = edge
                edge_to_node[edge] = len(edges)
                edges.append(edge)
                t = -h[u] / (h[v]-h[u])
                xyz.append(x[u]+t*(x[v]-x[u]))
            nodes.append(edge_to_node[edge])
        if len(nodes) != 2 or nodes[0] == nodes[1]:
            raise RuntimeError("invalid transverse triangle section")
        segments.append(nodes)
    adj = [[] for _ in edges]
    for a, b in segments:
        adj[a].append(b)
        adj[b].append(a)
    visited, components = set(), []
    for v in range(len(edges)):
        if v in visited:
            continue
        todo, component = [v], []
        visited.add(v)
        while todo:
            u = todo.pop()
            component.append(u)
            for n in adj[u]:
                if n not in visited:
                    visited.add(n)
                    todo.append(n)
        components.append(component)
    degrees = np.array([len(a) for a in adj], dtype=np.int64)
    return {
        "plane_axis": int(axis), "plane_value": float(value),
        "edge_nodes": np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        "points": np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        "segments": np.asarray(segments, dtype=np.int64).reshape(-1, 2),
        "source_face_rows": active,
        "components": components,
        "closed_loops": sum(all(len(adj[v]) == 2 for v in c) for c in components),
        "degree_histogram": {int(k): int(np.count_nonzero(degrees == k))
                             for k in np.unique(degrees)},
    }
