"""Ordered annular triangulation for local Human surface repair.

This constructs a candidate strip only. Callers must independently check
closure, intersections, layer cells, source ancestry and mass before admission.
"""
import numpy as np


def triangle_deformation_metrics(reference_triangles, candidate_triangles):
    """Return per-face area ratio, normal cosine, and oriented-area ratio.

    Inputs are already gathered Nx3x3 triangle coordinates: the second axis
    always selects corners, the third coordinates. This explicit contract
    avoids confusing a vertex-index array with a triangle/corner array.
    Collapsed candidate faces return zero in all three metrics. Degenerate
    reference faces cannot define a comparison and are rejected.
    """
    reference = np.asarray(reference_triangles, dtype=np.float64)
    candidate = np.asarray(candidate_triangles, dtype=np.float64)
    if (reference.ndim != 3 or reference.shape[1:] != (3, 3)
            or reference.shape != candidate.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        raise ValueError("matching finite Nx3x3 triangle coordinates required")
    nr = np.cross(reference[:, 1]-reference[:, 0], reference[:, 2]-reference[:, 0])
    nc = np.cross(candidate[:, 1]-candidate[:, 0], candidate[:, 2]-candidate[:, 0])
    ar = np.linalg.norm(nr, axis=1); ac = np.linalg.norm(nc, axis=1)
    if np.any(ar == 0):
        raise ValueError("nondegenerate reference triangles required")
    dot = np.einsum("ij,ij->i", nr, nc)
    cosine = np.divide(dot, ar*ac, out=np.zeros(len(ar)), where=ac != 0)
    return ac/ar, np.clip(cosine, -1, 1), dot/(ar*ar)


def p1_tetrahedral_sublevel_fractions(nodal_values, tetrahedra, level):
    """Exact volume fractions of a linear nodal field <= level in each tet.

    This is a scalar-field integration primitive, independent of orientation
    and physical cell volume. Repeated nodal values are supported without a
    divided-difference denominator across equal knots. A constant cell at the
    threshold has fraction one; interface builders must separately reject a
    positive-volume plateau rather than interpreting it as a unique surface.
    No surface closure or anatomical/material admission is implied.
    """
    values = np.asarray(nodal_values, dtype=np.float64)
    raw = np.asarray(tetrahedra)
    threshold = np.asarray(level, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional nodal values required")
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.dtype.kind not in "iu":
        raise ValueError("Nx4 integer tetrahedral indices required")
    if threshold.ndim != 0 or not np.isfinite(threshold):
        raise ValueError("finite scalar level required")
    if raw.size and (raw.min() < 0 or raw.max() >= len(values)):
        raise ValueError("tetrahedral index outside nodal values")
    if raw.size and np.any(np.diff(np.sort(raw, axis=1), axis=1) == 0):
        raise ValueError("four distinct node indices per tetrahedron required")
    knots = np.sort(values[raw], axis=1)
    below = np.sum(knots <= threshold, axis=1)
    result = (below == 4).astype(np.float64)
    for count in (1, 2, 3):
        rows = np.flatnonzero(below == count)
        if not len(rows):
            continue
        a, b, c, d = knots[rows].T
        if count == 1:
            fraction = ((threshold-a)/(b-a) * (threshold-a)/(c-a)
                        * (threshold-a)/(d-a))
        elif count == 3:
            fraction = 1 - ((d-threshold)/(d-a) * (d-threshold)/(d-b)
                            * (d-threshold)/(d-c))
        else:
            ac = (threshold-a)/(c-a)
            ad = (threshold-a)/(d-a)
            bc = (threshold-b)/(c-b)
            bd = (threshold-b)/(d-b)
            # Three tetrahedra partition the clipped six-vertex polyhedron.
            fraction = ac*ad + ad*bc*(1-ac) + bc*bd*(1-ad)
        result[rows] = fraction
    if not np.isfinite(result).all() or np.any((result < -1e-14) | (result > 1+1e-14)):
        raise ValueError("invalid linear-field fraction; rescale extreme nodal values")
    return np.clip(result, 0, 1)


def extract_p1_tetrahedral_interface(vertices, tetrahedra, nodal_values, level):
    """Extract the oriented boundary of a P1 sublevel inside tetrahedra.

    Exact threshold vertices retain their coordinates and identity. Shared
    edges share one interpolated vertex; no coordinate welding, perturbation,
    or degenerate-face removal is performed. Returns Float32 positions,
    triangles, source edge endpoints, and endpoint-one interpolation weights.
    Construction-domain exterior faces are not automatically physical skin.
    """
    x = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(tetrahedra)
    phi = np.asarray(nodal_values, dtype=np.float64)
    # Reuse strict field and index validation, including repeated-node rejection.
    p1_tetrahedral_sublevel_fractions(phi, raw, level)
    if x.shape != (len(phi), 3) or not np.isfinite(x).all():
        raise ValueError("finite Nx3 vertices matching nodal values required")
    t = raw.astype(np.int64)
    threshold = float(level)
    values = phi[t]
    if np.any(np.all(values == threshold, axis=1)):
        raise ValueError("constant threshold cell has no unique interface")
    active = np.flatnonzero((np.min(values, axis=1) < threshold)
                            & (np.max(values, axis=1) >= threshold))
    keys, points, weights, index = [], [], [], {}
    triangles = []
    facet_owners = {}
    facet_counts = {}
    for row in active:
        nodes = t[row]
        local = values[row]
        edges = []
        for node in nodes[local == threshold]:
            edges.append((int(node), int(node)))
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            if (local[i] < threshold < local[j]
                    or local[j] < threshold < local[i]):
                edges.append(tuple(sorted((int(nodes[i]), int(nodes[j])))))
        if len(edges) < 3:
            continue
        if len(edges) > 4:
            raise ValueError("nonregular tetrahedral interface")
        ids = []
        for key in edges:
            if key not in index:
                a, b = key
                alpha = 0. if a == b else (threshold-phi[a])/(phi[b]-phi[a])
                point = x[a] if a == b else (1-alpha)*x[a]+alpha*x[b]
                index[key] = len(points)
                keys.append(key); points.append(point); weights.append(alpha)
            ids.append(index[key])
        p = np.asarray([points[i] for i in ids])
        try:
            gradient = np.linalg.solve(x[nodes[1:]]-x[nodes[0]], local[1:]-local[0])
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular tetrahedral geometry") from exc
        normal = gradient/np.linalg.norm(gradient)
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(axis, normal); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        center = p.mean(axis=0)
        order = np.argsort(np.arctan2((p-center)@v, (p-center)@u))
        ids = [ids[i] for i in order]
        if len(ids) == 3:
            children = [ids]
        else:
            diagonal0 = tuple(sorted((keys[ids[0]], keys[ids[2]])))
            diagonal1 = tuple(sorted((keys[ids[1]], keys[ids[3]])))
            if diagonal0 <= diagonal1:
                children = [[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]]
            else:
                children = [[ids[1], ids[2], ids[3]], [ids[1], ids[3], ids[0]]]
        for face in children:
            key = tuple(sorted(face))
            facet_counts[key] = facet_counts.get(key, 0)+1
            if facet_counts[key] > 2:
                raise ValueError("nonmanifold interface facet")
            if key in facet_owners:
                # Two sublevel cells sharing an all-threshold facet cancel.
                previous = facet_owners.pop(key)
                old = triangles[previous]
                same_orientation = any(tuple(face) == tuple(old[k:]+old[:k])
                                       for k in range(3))
                if same_orientation:
                    raise ValueError("overlapping oriented interface facet")
                triangles[previous] = None
            else:
                facet_owners[key] = len(triangles)
                triangles.append(face)
    faces = np.asarray([f for f in triangles if f is not None], dtype=np.int64).reshape(-1, 3)
    emitted = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(faces):
        p = emitted.astype(np.float64)[faces]
        cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        before = np.asarray(points)[faces]
        reference = np.cross(before[:, 1]-before[:, 0], before[:, 2]-before[:, 0])
        if np.any(np.einsum("ij,ij->i", cross, reference) <= 0):
            raise ValueError("interface face degenerates or reverses after Float32 emission")
    return (emitted, faces, np.asarray(keys, dtype=np.int64).reshape(-1, 2),
            np.asarray(weights, dtype=np.float64))


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


def select_tetrahedra_by_surface_parent(face_parents, selected_face_rows, tet_parents):
    """Select complete parent columns, never treating current face rows as IDs.

    A refined face may share its parent identity with other faces. Selecting it
    intentionally includes that complete parent column for a conforming cavity.
    """
    fp = np.asarray(face_parents)
    rows = np.asarray(selected_face_rows)
    tp = np.asarray(tet_parents)
    for name, a in (("face parents", fp), ("face rows", rows), ("tet parents", tp)):
        if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError(name + " must be nonnegative integer identities")
    if len(rows) and np.max(rows) >= len(fp):
        raise ValueError("selected surface face row out of range")
    parents = np.unique(fp[rows])
    if not np.all(np.isin(parents, tp)):
        raise ValueError("selected surface parent has no tetrahedral descendants")
    return np.flatnonzero(np.isin(tp, parents))


def tetrahedral_region_boundary(tetrahedra, materials):
    """Extract oriented exterior and material interfaces from a tet region.

    Assumes positive geometric orientation; callers separately check signed
    volumes in both source precision and emitted Float32 coordinates.
    Rejects nonmanifold or inconsistently oriented faces and pinched boundaries.
    """
    t = np.asarray(tetrahedra)
    m = np.asarray(materials)
    if (t.ndim != 2 or t.shape[1] != 4 or not np.issubdtype(t.dtype, np.integer)
            or np.any(t < 0) or m.shape != (len(t),)
            or not np.issubdtype(m.dtype, np.integer) or np.any(m < 0)):
        raise ValueError("invalid tetrahedron or material identities")
    if len(t) and np.any(np.diff(np.sort(t, axis=1), axis=1) == 0):
        raise ValueError("tetrahedron has repeated nodes")
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("duplicate tetrahedron")
    uses = {}
    template = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for i, tet in enumerate(t):
        for ids in template:
            f = tuple(int(tet[j]) for j in ids)
            sign = 1 if sum(f[a] > f[b] for a in range(3) for b in range(a+1, 3)) % 2 == 0 else -1
            uses.setdefault(tuple(sorted(f)), []).append((f, i, sign))
    exterior, owners, interfaces, pairs = [], [], [], []
    for key in sorted(uses):
        u = uses[key]
        if len(u) == 1:
            exterior.append(u[0][0])
            owners.append(u[0][1])
        elif len(u) == 2:
            if u[0][2] + u[1][2] != 0:
                raise ValueError("shared tetrahedron face has inconsistent orientation")
            a, b = u
            if m[a[1]] != m[b[1]]:
                if m[a[1]] > m[b[1]]:
                    a, b = b, a
                interfaces.append(a[0])
                pairs.append((a[1], b[1]))
        else:
            raise ValueError("nonmanifold tetrahedron face")
    edges = {}
    for f in exterior:
        for a, b in zip(f, f[1:]+f[:1]):
            e = tuple(sorted((a, b)))
            count, sign = edges.get(e, (0, 0))
            edges[e] = (count+1, sign+(1 if a < b else -1))
    if any(v != (2, 0) for v in edges.values()):
        raise ValueError("region boundary is pinched, open or inconsistently oriented")
    # Edge-manifold tests do not detect two closed shells touching at one
    # vertex. Each exterior vertex must have exactly one cyclic triangle link.
    links = {}
    for f in exterior:
        for j, v in enumerate(f):
            a, b = f[(j+1) % 3], f[(j+2) % 3]
            adj = links.setdefault(v, {})
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    for adj in links.values():
        if any(len(neighbors) != 2 for neighbors in adj.values()):
            raise ValueError("boundary vertex link is not a cycle")
        seen, todo = set(), [next(iter(adj))]
        while todo:
            v = todo.pop()
            if v not in seen:
                seen.add(v)
                todo.extend(adj[v])
        if len(seen) != len(adj):
            raise ValueError("boundary vertex joins disconnected shells")
    return {
        "exterior_faces": np.asarray(exterior, dtype=np.int64).reshape(-1, 3),
        "exterior_tet_rows": np.asarray(owners, dtype=np.int64),
        "interface_faces": np.asarray(interfaces, dtype=np.int64).reshape(-1, 3),
        "interface_tet_pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
    }


def partition_tetrahedral_repair_batch(tetrahedra, selected_tet_sets, movable_node_sets):
    """Group a repair batch by actual tetrahedral ownership, not fixed stitches.

    Every movable node's complete incident tetrahedron star is included before
    merging. Components may share *fixed* node IDs; these are exact retained
    stitches, not independent copies to move. Only overlapping element sets
    merge. Returned rows refer to the original input arrays and are sorted
    deterministically. This selects construction domains; it does not establish
    geometric closure, collision freedom, source ancestry, mass or admission.
    """
    def ids(value, name, *, upper=None, empty_ok=False):
        a = np.asarray(value)
        if a.ndim == 1 and a.size == 0 and empty_ok:
            return set()
        if a.ndim != 1 or a.dtype.kind not in "iu":
            raise ValueError(name + " must contain one-dimensional integer IDs")
        if not empty_ok and not a.size:
            raise ValueError(name + " must not be empty")
        if a.size and (np.any(a > np.iinfo(np.int64).max) or
                       np.any(a < 0) or (upper is not None and np.any(a >= upper))):
            raise ValueError(name + " contains an out-of-range ID")
        return set(map(int, a))
    raw = np.asarray(tetrahedra)
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.dtype.kind not in "iu":
        raise ValueError("integer Tx4 tetrahedra required")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError("tetrahedral node IDs must be nonnegative int64")
    t = raw.astype(np.int64)
    if any(len(set(map(int, row))) != 4 for row in t):
        raise ValueError("tetrahedra need four distinct node IDs")
    if len(selected_tet_sets) != len(movable_node_sets):
        raise ValueError("one movable-node set is required per selected-tet set")
    if not len(selected_tet_sets):
        return [], []
    incidence = {}
    for ti, row in enumerate(t):
        for node in row:
            incidence.setdefault(int(node), set()).add(ti)
    parts = []
    for ci, (selected, movable) in enumerate(zip(selected_tet_sets, movable_node_sets)):
        cells = ids(selected, "selected tetrahedra", upper=len(t))
        moved = ids(movable, "movable nodes", empty_ok=True)
        original_nodes = set(map(int, t[sorted(cells)].ravel()))
        if not moved <= original_nodes:
            raise ValueError("a movable node is absent from its selected component")
        for node in moved:
            cells.update(incidence[node])
        parts.append({"seed_components": {ci}, "selected_global_tet_ids": cells,
                      "movable_global_node_ids": moved})
    # Union on tetrahedral IDs. Shared fixed vertices do not create ownership
    # conflicts, including contact between distinct volume components.
    parent = list(range(len(parts)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for ci, part in enumerate(parts):
        for ti in sorted(part["selected_global_tet_ids"]):
            if ti in owner:
                a, b = root(ci), root(owner[ti])
                parent[max(a, b)] = min(a, b)
            else:
                owner[ti] = ci
    merged = {}
    for ci, part in enumerate(parts):
        g = merged.setdefault(root(ci), {k: set() for k in part})
        for key, value in part.items():
            g[key].update(value)
    result = sorted(merged.values(),
                    key=lambda g: (min(g["selected_global_tet_ids"]),
                                   min(g["seed_components"])))
    node_sets = []
    for g in result:
        cells, moved = g["selected_global_tet_ids"], g["movable_global_node_ids"]
        for node in moved:
            if not incidence[node] <= cells:
                raise RuntimeError("movable node has an unowned incident tetrahedron")
        node_sets.append(set(map(int, t[sorted(cells)].ravel())))
    shared = []
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            common = node_sets[i] & node_sets[j]
            if common:
                if common & (result[i]["movable_global_node_ids"] |
                             result[j]["movable_global_node_ids"]):
                    raise RuntimeError("independent repair components share a movable node")
                shared.append({"component_a": i, "component_b": j,
                               "fixed_global_node_ids": np.array(sorted(common), dtype=np.int64)})
    return [{k: np.array(sorted(v), dtype=np.int64) for k, v in g.items()}
            for g in result], shared


def boundary_tetrahedron_quality_ceiling(vertices, faces):
    """Upper-bound any incident tet's abs(det)/longest_edge**3, per face.

    For a face with double area a and longest edge l, an incident tet with
    longest edge L >= l obeys abs(det) <= a*L. Thus its normalized determinant
    is at most a/l**2. A ceiling below a required quality proves fixed-face
    meshing impossible; a higher ceiling does NOT prove a valid tet exists.
    Pass the reference coordinates actually used by the volume compiler.
    This scale-invariant local metric is not area/max_area over a mesh.
    """
    x = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(faces)
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu"):
        raise ValueError("finite Nx3 vertices and integer Fx3 faces required")
    f = raw.astype(np.int64)
    if f.size and (f.min() < 0 or f.max() >= len(x)):
        raise ValueError("face vertex is out of range")
    t = x[f]
    a, b, c = t[:,1]-t[:,0], t[:,2]-t[:,0], t[:,2]-t[:,1]
    longest2 = np.maximum.reduce((np.einsum("ij,ij->i",a,a),
                                 np.einsum("ij,ij->i",b,b),
                                 np.einsum("ij,ij->i",c,c)))
    area2 = np.linalg.norm(np.cross(a,b),axis=1)
    result = np.zeros(len(f),dtype=np.float64)
    np.divide(area2,longest2,out=result,where=longest2>0)
    return result
