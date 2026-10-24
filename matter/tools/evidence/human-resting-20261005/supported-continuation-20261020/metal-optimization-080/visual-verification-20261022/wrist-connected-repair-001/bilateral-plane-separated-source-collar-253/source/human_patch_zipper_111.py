"""Ordered annular triangulation for local Human surface repair.

This constructs a candidate strip only. Callers must independently check
closure, intersections, layer cells, source ancestry and mass before admission.
"""
import numpy as np


def zipper_annulus(vertices, outer_loop, inner_loop, *, quality_floor=1e-5,
                   max_edge=None, phase_band=0.35, max_crossring_edge=None):
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
    best_key = None

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
        if not np.isfinite(d[n, m]):
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
        faces = np.asarray(faces[::-1], dtype=np.int64)
        aset=set(map(int,a)); bset=set(map(int,b0))
        cross_lengths=[]
        for face in faces:
            for i,j in zip(face,np.roll(face,-1)):
                i=int(i);j=int(j)
                if (i in aset and j in bset) or (j in aset and i in bset):
                    cross_lengths.append(float(np.linalg.norm(x[i]-x[j])))
        cross_max=max(cross_lengths, default=0.0)
        feasible = max_crossring_edge is None or cross_max <= float(max_crossring_edge)
        key=(0 if feasible else 1, float(d[n,m]), cross_max)
        if best_key is None or key < best_key:
            best_key=key; best = (float(d[n, m]), phase, faces, cross_max)
    if best is None:
        raise ValueError("no strip satisfies quality/edge/progress constraints")
    score, phase, faces, cross_max = best
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
                       maximum_edge_m=float(edge.max()), maximum_crossring_edge_m=float(cross_max),
                       oriented_boundary_exact=True,
                       physical_admission=False)
