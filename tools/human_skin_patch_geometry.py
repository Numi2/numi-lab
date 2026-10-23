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
