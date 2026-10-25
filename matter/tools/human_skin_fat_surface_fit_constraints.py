"""Small numerical helpers for a connected skin area-constrained SCP fit.

Coordinates passed to the area routines are in millimetres. This module is
construction-only; it does not certify emitted geometry.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

class InfeasibleProjection(ValueError):
    """A requested halfspace does not intersect the source-relative ball."""
    def __init__(self, node: int, signed_plane_offset: float, radius: float):
        self.node = int(node)
        self.signed_plane_offset = float(signed_plane_offset)
        self.radius = float(radius)
        super().__init__(f"node {node}: halfspace is {signed_plane_offset:.9g} mm from ball center, outside radius {radius:.9g} mm")


def project_ball_halfspace(v, c, radius_mm: float, normals, lower, hard_indices=None, anchors=(), tol=1e-10):
    """Project each 3-D node onto its source ball and optional hard halfspace.

    The halfspace for hard_indices[i] is dot(normals[i], x[node]) >= lower[i].
    The projection is the exact Euclidean projection onto ball ∩ halfspace;
    infeasible intersections raise InfeasibleProjection instead of returning an
    unconverged Dykstra iterate. Anchor rows are fixed to zero displacement.
    """
    v = np.asarray(v, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or c.shape != v.shape:
        raise ValueError("v and c must have shape (N,3)")
    r = float(radius_mm)
    if not np.isfinite(r) or r <= 0:
        raise ValueError("radius_mm must be finite and positive")
    if not np.isfinite(v).all() or not np.isfinite(c).all():
        raise ValueError("projection input contains nonfinite values")
    raw_idx = np.asarray([] if hard_indices is None else hard_indices)
    if raw_idx.ndim != 1 or (raw_idx.size and (not np.issubdtype(raw_idx.dtype, np.integer) and (not np.isfinite(raw_idx).all() or not np.equal(raw_idx, np.floor(raw_idx)).all()))):
        raise ValueError("hard indices must be integer-valued rows")
    idx = raw_idx.astype(np.int64, copy=False)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    h = np.asarray(lower, dtype=np.float64).reshape(-1)
    if len(idx) != len(n) or len(idx) != len(h):
        raise ValueError("hard indices, normals, and lower bounds must have equal length")
    if len(idx) and (np.any(idx < 0) or np.any(idx >= len(v)) or len(np.unique(idx)) != len(idx)):
        raise ValueError("hard indices must be unique valid node rows")
    if not np.isfinite(n).all() or not np.isfinite(h).all():
        raise ValueError("halfspace input contains nonfinite values")
    nn = np.linalg.norm(n, axis=1)
    if np.any(nn <= 1e-15):
        raise ValueError("halfspace normals must be nonzero")
    n = n / nn[:, None]
    h = h / nn
    raw_anchors = np.asarray(tuple(anchors))
    if raw_anchors.ndim != 1 or (raw_anchors.size and (not np.issubdtype(raw_anchors.dtype, np.integer) and (not np.isfinite(raw_anchors).all() or not np.equal(raw_anchors, np.floor(raw_anchors)).all()))):
        raise ValueError("anchor indices must be integer-valued rows")
    anchors = raw_anchors.astype(np.int64, copy=False)
    out = np.empty_like(v)
    delta = v - c
    lengths = np.linalg.norm(delta, axis=1)
    out[:] = c + delta * np.minimum(1.0, r / np.maximum(lengths, 1e-300))[:, None]
    if len(anchors) and (np.any(anchors < 0) or np.any(anchors >= len(v)) or len(np.unique(anchors)) != len(anchors)):
        raise ValueError("anchor indices must be unique valid rows")
    if len(anchors) and np.any(np.linalg.norm(c[anchors], axis=1) > r + tol):
        bad_anchor = int(anchors[np.flatnonzero(np.linalg.norm(c[anchors], axis=1) > r + tol)[0]])
        raise InfeasibleProjection(bad_anchor, float(np.linalg.norm(c[bad_anchor])), r)
    if len(idx):
        ci = c[idx]
        vi = v[idx]
        ni = n
        # Feasibility of n·x >= h on the closed source ball.
        support = np.einsum("ij,ij->i", ni, ci) + r
        bad = np.flatnonzero(h > support + tol)
        if len(bad):
            k = int(bad[0]); raise InfeasibleProjection(int(idx[k]), float(h[k] - np.dot(ni[k], ci[k])), r)
        # If the ordinary ball projection is already feasible, it is optimal.
        bi = out[idx]
        bdot = np.einsum("ij,ij->i", ni, bi)
        need = bdot < h - tol
        if np.any(need):
            vv = vi[need]; cc = ci[need]; nneed = ni[need]; hneed = h[need]
            # Halfspace projection may be inside the ball; if so it is the answer.
            vdot = np.einsum("ij,ij->i", nneed, vv)
            hp = vv + (hneed-vdot)[:, None]*nneed
            hp_inside = np.linalg.norm(hp-cc, axis=1) <= r + tol
            ans = hp.copy()
            # Otherwise project onto the sphere/plane intersection circle.
            for j in np.flatnonzero(~hp_inside):
                nj = nneed[j]; cj = cc[j]; hj = hneed[j]
                signed = float(hj - np.dot(nj, cj))
                if abs(signed) > r + tol:
                    node = int(idx[np.flatnonzero(need)[j]])
                    raise InfeasibleProjection(node, signed, r)
                signed = min(r, max(-r, signed))
                cp = cj + signed*nj
                rho = float(np.sqrt(max(0.0, r*r-signed*signed)))
                tangent = vv[j]-cp
                tangent -= np.dot(tangent, nj)*nj
                tn = float(np.linalg.norm(tangent))
                if tn <= 1e-14:
                    # Deterministic tangent basis least aligned with n.
                    axis = np.eye(3)[int(np.argmin(np.abs(nj)))]
                    tangent = axis - np.dot(axis,nj)*nj
                    tn = float(np.linalg.norm(tangent))
                ans[j] = cp + rho*tangent/tn
            selected = np.flatnonzero(need)
            bi[selected] = ans
            out[idx] = bi
    # Fixed registered anchors are zero displacement and must satisfy any plane.
    if len(anchors):
        anchor_set = set(int(x) for x in anchors)
        for j, node in enumerate(idx):
            if int(node) in anchor_set and h[j] > tol:
                raise InfeasibleProjection(int(node), float(h[j]), r)
        out[anchors] = 0.0
    return out


def area_linearization(points_mm, faces, face_rows, reference_normals, reference_double_area, minimum_ratio=0.1):
    """Return sparse local rows G and rhs for n_ref·cross(e1,e2)>=ratio*Aref.

    `points_mm` is the current absolute geometry in mm. Returned rows are in
    displacement variables d_mm relative to `points_mm`: G d >= rhs. Each row
    has exactly nine nonzero coefficients across its triangle vertices.
    """
    from scipy import sparse
    p = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    rows = np.asarray(face_rows, dtype=np.int64)
    nr = np.asarray(reference_normals, dtype=np.float64)
    ar = np.asarray(reference_double_area, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("points and faces must be (N,3) and (M,3)")
    if np.any(rows < 0) or np.any(rows >= len(f)):
        raise ValueError("face row out of range")
    if nr.shape != (len(rows),3) or ar.shape != (len(rows),):
        raise ValueError("reference normal/area shapes do not match selected faces")
    if not (0 < minimum_ratio <= 1) or not np.isfinite(p).all() or not np.isfinite(nr).all() or not np.isfinite(ar).all():
        raise ValueError("invalid geometry or area threshold")
    if np.any(ar <= 0): raise ValueError("reference areas must be positive")
    nrm = np.linalg.norm(nr,axis=1)
    if np.any(nrm <= 1e-15): raise ValueError("reference normals must be nonzero")
    nr = nr/nrm[:,None]
    tri = f[rows]
    a,b,c = p[tri[:,0]],p[tri[:,1]],p[tri[:,2]]
    e1=b-a; e2=c-a
    # d/db n·(e1×e2)= e2×n; d/dc=n×e1; d/da is negative sum.
    gb=np.cross(e2,nr); gc=np.cross(nr,e1); ga=-gb-gc
    cur=np.einsum('ij,ij->i',nr,np.cross(e1,e2))
    rhs=minimum_ratio*ar-cur
    # Flatten coordinates in node-major xyz order.
    rr=np.repeat(np.arange(len(rows)),9)
    cc=(tri[:,:,None]*3 + np.arange(3)[None,None,:]).reshape(-1)
    vv=np.stack((ga,gb,gc),axis=1).reshape(-1)
    G=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rows),p.size)).tocsr()
    return G,rhs,cur,{'face_rows':rows.copy(),'reference_normals':nr,'reference_double_area':ar.copy(),'minimum_ratio':float(minimum_ratio)}


def face_area_gradient(points_mm, face, reference_normal):
    """Analytic 9-vector gradient for one oriented area dot value."""
    p=np.asarray(points_mm,dtype=np.float64); tri=np.asarray(face,dtype=np.int64); n=np.asarray(reference_normal,dtype=np.float64)
    a,b,c=p[tri]; e1=b-a; e2=c-a
    gb=np.cross(e2,n); gc=np.cross(n,e1); ga=-gb-gc
    return np.concatenate((ga,gb,gc))


def ball_intersection_support(c1, radius1_mm, c2, radius2_mm, normal, tol=1e-12):
    """Maximum dot(normal, x) over the intersection of two 3-D closed balls."""
    c1=np.asarray(c1,dtype=np.float64); c2=np.asarray(c2,dtype=np.float64); n=np.asarray(normal,dtype=np.float64)
    if c1.shape!=(3,) or c2.shape!=(3,) or n.shape!=(3,): raise ValueError('centres and normal must be 3-vectors')
    r1,r2=float(radius1_mm),float(radius2_mm)
    if not np.isfinite(c1).all() or not np.isfinite(c2).all() or not np.isfinite(n).all(): raise ValueError('ball support inputs must be finite')
    if not np.isfinite([r1,r2]).all() or r1<=0 or r2<=0: raise ValueError('ball radii must be finite and positive')
    nn=float(np.linalg.norm(n))
    if nn<=1e-15: raise ValueError('support normal must be nonzero')
    n=n/nn; delta=c2-c1; distance=float(np.linalg.norm(delta))
    if distance>r1+r2+tol: raise ValueError('the two balls do not intersect')
    if distance<=tol: return float(np.dot(n,c1)+min(r1,r2))
    p1=c1+r1*n
    if np.linalg.norm(p1-c2)<=r2+tol: return float(np.dot(n,p1))
    p2=c2+r2*n
    if np.linalg.norm(p2-c1)<=r1+tol: return float(np.dot(n,p2))
    u=delta/distance; a=(r1*r1-r2*r2+distance*distance)/(2.0*distance); rho2=r1*r1-a*a
    if rho2 < -tol*max(1.0,r1*r1): raise ValueError('numerically inconsistent ball intersection')
    rho=float(np.sqrt(max(0.0,rho2))); cp=c1+a*u; tangent=n-float(np.dot(n,u))*u; tn=float(np.linalg.norm(tangent))
    point=cp if tn<=1e-15 else cp+rho*tangent/tn
    if np.linalg.norm(point-c1)>r1+1e-8 or np.linalg.norm(point-c2)>r2+1e-8: raise ArithmeticError('computed support point falls outside ball intersection')
    return float(np.dot(n,point))


def progressive_plane_lower(current_projection_mm, target_mm, support_mm, fraction=0.95):
    """Advance toward target without reaching the support boundary of a ball lens."""
    cur=np.asarray(current_projection_mm,dtype=np.float64); target=np.asarray(target_mm,dtype=np.float64); support=np.asarray(support_mm,dtype=np.float64)
    if cur.shape!=target.shape or cur.shape!=support.shape: raise ValueError('projection, target, and support arrays must have matching shapes')
    f=float(fraction)
    if not np.isfinite(cur).all() or not np.isfinite(target).all() or not np.isfinite(support).all(): raise ValueError('plane progress inputs must be finite')
    if not 0<f<1: raise ValueError('fraction must lie strictly between zero and one')
    if np.any(support<cur-2e-8): raise ValueError('current point is outside the supplied feasible support')
    return np.maximum(cur,np.minimum(target,cur+f*np.maximum(0.0,support-cur)))


def _circle_nearest_point(center, axis_normal, radius, target):
    """Closest point on a circle, with deterministic choice for axial targets."""
    center=np.asarray(center,dtype=np.float64); axis=np.asarray(axis_normal,dtype=np.float64); target=np.asarray(target,dtype=np.float64)
    radius=float(radius)
    if radius<=1e-14: return center.copy()
    axis=axis/np.linalg.norm(axis)
    tangent=target-center; tangent=tangent-np.dot(tangent,axis)*axis
    length=float(np.linalg.norm(tangent))
    if length<=1e-14:
        basis=np.eye(3)[int(np.argmin(np.abs(axis)))]
        tangent=basis-np.dot(basis,axis)*axis; length=float(np.linalg.norm(tangent))
    return center+radius*tangent/length


def project_two_balls_halfspace(v, c1, radius1, c2, radius2, normal=None, lower=None, tol=1e-9):
    """Exact 3-D projection candidate enumeration onto two balls and one halfspace.

    Enumerates KKT active sets: the point, single active boundaries, pairwise
    boundary intersections, and the three-way sphere/sphere/plane intersection.
    Raises when no numerically feasible candidate is found; it never returns an
    unconverged alternating-projection iterate.
    """
    v=np.asarray(v,dtype=np.float64); c1=np.asarray(c1,dtype=np.float64); c2=np.asarray(c2,dtype=np.float64)
    r1=float(radius1); r2=float(radius2)
    if v.shape!=(3,) or c1.shape!=(3,) or c2.shape!=(3,): raise ValueError('point and ball centres must be 3-vectors')
    if not np.isfinite(v).all() or not np.isfinite(c1).all() or not np.isfinite(c2).all(): raise ValueError('projection coordinates must be finite')
    if not np.isfinite([r1,r2]).all() or r1<=0 or r2<=0: raise ValueError('radii must be finite and positive')
    if normal is None:
        if lower is not None: raise ValueError('lower bound requires a plane normal')
        n=None; h=None
    else:
        n=np.asarray(normal,dtype=np.float64)
        if n.shape!=(3,) or not np.isfinite(n).all(): raise ValueError('plane normal must be finite 3-vector')
        nn=float(np.linalg.norm(n))
        if nn<=1e-15: raise ValueError('plane normal must be nonzero')
        n=n/nn; h=float(lower)/nn
        if not np.isfinite(h): raise ValueError('plane lower bound must be finite')
    delta=c2-c1; D=float(np.linalg.norm(delta))
    if D>r1+r2+tol: raise ValueError('two balls are disjoint')

    def feasible(p):
        if not np.isfinite(p).all(): return False
        if np.linalg.norm(p-c1)>r1+tol or np.linalg.norm(p-c2)>r2+tol: return False
        return n is None or float(np.dot(n,p))>=h-tol
    candidates=[]
    def add(p):
        p=np.asarray(p,dtype=np.float64)
        if feasible(p): candidates.append(p.copy())
    def ball_projection(c,r):
        d=v-c; length=float(np.linalg.norm(d))
        return v.copy() if length<=r else c+d*(r/length)

    add(v)
    add(ball_projection(c1,r1)); add(ball_projection(c2,r2))
    if n is not None:
        add(v+max(0.0,h-float(np.dot(n,v)))*n)
        for c,r in ((c1,r1),(c2,r2)):
            signed=h-float(np.dot(n,c))
            if abs(signed)<=r+tol:
                cp=c+signed*n; rho2=r*r-signed*signed
                if rho2>=-tol*max(1.0,r*r): add(_circle_nearest_point(cp,n,np.sqrt(max(0.0,rho2)),v))

    if D<=tol:
        # Concentric balls reduce to their smaller ball and its possible plane cut.
        c,r=(c1,r1) if r1<=r2 else (c2,r2)
        if n is not None:
            signed=h-float(np.dot(n,c))
            if abs(signed)<=r+tol:
                cp=c+signed*n; rho2=r*r-signed*signed
                if rho2>=-tol*max(1.0,r*r): add(_circle_nearest_point(cp,n,np.sqrt(max(0.0,rho2)),v))
    elif D<=abs(r1-r2)+tol:
        # One ball is contained in the other; its own boundary determines the lens.
        c,r=(c1,r1) if r1<=r2 else (c2,r2)
        if n is not None:
            signed=h-float(np.dot(n,c))
            if abs(signed)<=r+tol:
                cp=c+signed*n; rho2=r*r-signed*signed
                if rho2>=-tol*max(1.0,r*r): add(_circle_nearest_point(cp,n,np.sqrt(max(0.0,rho2)),v))
    else:
        u=delta/D; a=(r1*r1-r2*r2+D*D)/(2.0*D); rho2=r1*r1-a*a
        if rho2>=-tol*max(1.0,r1*r1):
            rho=float(np.sqrt(max(0.0,rho2))); cp=c1+a*u
            add(_circle_nearest_point(cp,u,rho,v))
            if n is not None:
                q=n-float(np.dot(n,u))*u; q2=float(np.dot(q,q)); plane_delta=h-float(np.dot(n,cp))
                if q2>1e-24:
                    chord_r2=rho*rho-plane_delta*plane_delta/q2
                    if chord_r2>=-tol*max(1.0,rho*rho):
                        chord_center=cp+(plane_delta/q2)*q
                        chord_dir=np.cross(u,q); chord_dir/=np.linalg.norm(chord_dir)
                        chord_r=float(np.sqrt(max(0.0,chord_r2)))
                        add(chord_center+chord_r*chord_dir); add(chord_center-chord_r*chord_dir)
                elif abs(plane_delta)<=tol:
                    add(_circle_nearest_point(cp,u,rho,v))
    if not candidates:
        raise ValueError('no feasible active-set projection candidate')
    return min(candidates,key=lambda p:float(np.dot(p-v,p-v)))

def project_node_sets_batched(v, source_center, source_cap, plane_rows, plane_normals,
                             plane_lower, anchors, trust_center, trust_radius):
    """Exact construction projection; batch contained-ball rows, enumerate overlap rows.

    Eliminates the Python scan across every node. The same analytic projectors
    remain responsible for each active constraint; exact geometry gates are
    still required on the emitted surface.
    """
    v = np.asarray(v, dtype=np.float64)
    source_center = np.asarray(source_center, dtype=np.float64)
    trust_center = np.asarray(trust_center, dtype=np.float64)
    rows = np.asarray(plane_rows, dtype=np.int64)
    normals = np.asarray(plane_normals, dtype=np.float64).reshape(-1, 3)
    lower = np.asarray(plane_lower, dtype=np.float64).reshape(-1)
    anchors = np.asarray(anchors, dtype=np.int64)
    if (v.ndim != 2 or v.shape[1] != 3 or source_center.shape != v.shape
        or trust_center.shape != v.shape or rows.ndim != 1 or anchors.ndim != 1
        or len(rows) != len(normals) or len(rows) != len(lower)):
        raise ValueError("malformed node projection arrays")
    if (not np.isfinite(source_center).all() or not np.isfinite(normals).all()
        or not np.isfinite(lower).all() or not np.isfinite(source_cap) or source_cap <= 0):
        raise ValueError("nonfinite node constraints")
    if (len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(v))
        or len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(v))
        or np.any(np.linalg.norm(normals, axis=1) <= 1e-15)):
        raise ValueError("invalid constraint identities or normals")
    fixed = np.zeros(len(v), dtype=bool)
    fixed[anchors] = True
    contained = np.linalg.norm(trust_center - source_center, axis=1) + trust_radius <= source_cap + 1e-10
    batched = contained[rows] & ~fixed[rows]
    out = project_ball_halfspace(v, trust_center, trust_radius,
                                normals[batched], lower[batched], rows[batched])
    plane_lookup = {int(row): i for i, row in enumerate(rows)}
    overlap_rows = rows[~contained[rows] & ~fixed[rows]]
    unconstrained_source_outside = np.linalg.norm(out-source_center,axis=1) > source_cap + 1e-10
    unconstrained_source_outside[rows] = False
    unconstrained_source_outside[anchors] = False
    plain_rows = np.flatnonzero(unconstrained_source_outside)
    for row in overlap_rows:
        i = plane_lookup[int(row)]
        out[row] = project_two_balls_halfspace(v[row],source_center[row],source_cap,
                                               trust_center[row],trust_radius,normals[i],lower[i])
    for row in plain_rows:
        out[row] = project_two_balls_halfspace(v[row],source_center[row],source_cap,
                                               trust_center[row],trust_radius)
    if len(anchors):
        if (np.any(np.linalg.norm(source_center[anchors],axis=1) > source_cap + 1e-10)
            or np.any(np.linalg.norm(trust_center[anchors],axis=1) > trust_radius + 1e-10)):
            raise ValueError("anchor outside source or trust ball")
        anchored_planes = fixed[rows]
        if np.any(lower[anchored_planes] > 1e-10):
            i = np.flatnonzero(anchored_planes & (lower > 1e-10))[0]
            raise InfeasibleProjection(int(rows[i]),lower[i],source_cap)
        out[anchors] = 0.0
    errors = [
        np.maximum(0,np.linalg.norm(out-source_center,axis=1)-source_cap).max(initial=0.0),
        np.maximum(0,np.linalg.norm(out-trust_center,axis=1)-trust_radius).max(initial=0.0),
        np.maximum(0,lower-np.einsum("ij,ij->i",normals,out[rows])).max(initial=0.0),
    ]
    overlap_count = len(overlap_rows) + len(plain_rows)
    return out, dict(method="exact_active_set_candidates_batched",
                     intersection_residual_mm=float(max(errors)),
                     overlap_nodes=overlap_count, full_active_set_enumerations=overlap_count)


# Witness-local backtracking helper, staged for canonical integration (183).
# This only scales a proposal; exact geometry gates remain the caller's job.
def _local_witness_indices_183(value, name, ndim):
    a = np.asarray(value)
    if a.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if a.size == 0:
        return np.empty(a.shape, dtype=np.int64)
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    return a.astype(np.int64, copy=False)


def reduce_witnessed_surface_step_scales_183(
    faces,
    step_scales,
    bad_face_rows,
    self_pair_face_rows=None,
    anchor_rows=(),
    halving_counts=None,
    *,
    factor=0.5,
    max_halvings=6,
):
    """Reduce proposal multipliers only at nodes in exact-gate witnesses.

    ``step_scales[i]`` multiplies the caller's proposal displacement for node i.
    Bad face rows and both faces of each exact self-pair mark affected nodes.
    Anchors are forced to zero and no neighboring-node smoothing is applied.
    Returns ``(scales, halving_counts, report)``. This is not a geometry gate.
    """
    f = _local_witness_indices_183(faces, "faces", 2)
    if f.shape[1:] != (3,):
        raise ValueError("faces must have shape (M,3)")
    s = np.asarray(step_scales)
    if s.ndim != 1 or s.dtype.kind not in "fiu" or s.dtype.kind == "b":
        raise ValueError("step_scales must be a numeric vector")
    s = s.astype(np.float64, copy=False)
    if len(s) == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("step_scales must be nonempty, finite, and within [0,1]")
    if f.size and (int(f.min()) < 0 or int(f.max()) >= len(s)):
        raise ValueError("face node index out of range")
    if len(f) == 0 or any(len(set(map(int, row))) != 3 for row in f):
        raise ValueError("faces must be nonempty with three distinct vertices each")

    bad = _local_witness_indices_183(bad_face_rows, "bad_face_rows", 1)
    if len(set(map(int, bad))) != len(bad):
        raise ValueError("bad_face_rows must be unique")
    if np.any(bad < 0) or np.any(bad >= len(f)):
        raise ValueError("bad face row out of range")

    if self_pair_face_rows is None:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        raw_pairs = np.asarray(self_pair_face_rows)
        # A plain empty sequence (e.g. []) means no pairs. Preserve explicit
        # two-dimensional shapes so malformed (0,3) input fails below.
        if raw_pairs.ndim == 1 and raw_pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            pairs = _local_witness_indices_183(raw_pairs, "self_pair_face_rows", 2)
        if pairs.shape[1:] != (2,):
            raise ValueError("self_pair_face_rows must have shape (K,2)")
        if len(pairs):
            if np.any(pairs < 0) or np.any(pairs >= len(f)) or np.any(pairs[:, 0] == pairs[:, 1]):
                raise ValueError("invalid self-pair face rows")
            canonical = [tuple(sorted(map(int, row))) for row in pairs]
            if len(set(canonical)) != len(canonical):
                raise ValueError("self_pair_face_rows contain duplicate pairs")

    if len(bad) == 0 and len(pairs) == 0:
        raise ValueError("at least one bad face or self-pair witness is required")

    anchors = np.asarray(anchor_rows)
    if anchors.size == 0:
        anchors = np.empty(0, dtype=np.int64)
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu" or anchors.dtype.kind == "b":
        raise ValueError("anchor_rows must be a 1D integer index vector")
    anchors = anchors.astype(np.int64, copy=False)
    if len(set(map(int, anchors))) != len(anchors):
        raise ValueError("anchor_rows contain duplicates")
    if np.any(anchors < 0) or np.any(anchors >= len(s)):
        raise ValueError("anchor row out of range")

    if not np.isfinite(float(factor)) or not (0.0 < float(factor) < 1.0):
        raise ValueError("factor must be finite and in (0,1)")
    if isinstance(max_halvings, (bool, np.bool_)) or int(max_halvings) != max_halvings or int(max_halvings) < 1:
        raise ValueError("max_halvings must be a positive integer")
    max_halvings = int(max_halvings)

    affected_faces = np.unique(np.r_[bad, pairs.ravel()]).astype(np.int64)
    affected_nodes = np.unique(f[affected_faces].ravel()).astype(np.int64)
    if halving_counts is None:
        counts = np.zeros(len(s), dtype=np.int64)
    else:
        counts_in = np.asarray(halving_counts)
        if counts_in.shape != s.shape or counts_in.dtype.kind not in "iu" or counts_in.dtype.kind == "b":
            raise ValueError("halving_counts must be an integer vector matching step_scales")
        counts = counts_in.astype(np.int64, copy=True)
        if np.any(counts < 0):
            raise ValueError("halving_counts cannot be negative")

    out = s.copy()
    zero_fallback = []
    halved = []
    for node in affected_nodes:
        i = int(node)
        if counts[i] >= max_halvings:
            out[i] = 0.0
            zero_fallback.append(i)
        elif out[i] != 0.0:
            out[i] *= float(factor)
            counts[i] += 1
            halved.append(i)
    out[anchors] = 0.0
    untouched = np.ones(len(s), dtype=bool)
    untouched[affected_nodes] = False
    untouched[anchors] = False
    unchanged = bool(np.array_equal(out[untouched], s[untouched]))
    if not unchanged:
        raise AssertionError("unaffected scales changed")
    if np.any(out[anchors] != 0.0):
        raise AssertionError("anchors are not fixed")

    report = {
        "status": "PROPOSAL_SCALES_REDUCED_ONLY",
        "construction_only": True,
        "geometry_certified": False,
        "bad_face_rows": bad.tolist(),
        "self_pair_face_rows": pairs.tolist(),
        "affected_faces": affected_faces.tolist(),
        "affected_node_rows": affected_nodes.tolist(),
        "halved_node_rows": halved,
        "exact_zero_fallback_node_rows": zero_fallback,
        "anchor_rows_forced_zero": anchors.tolist(),
        "unaffected_scales_bitwise_unchanged": unchanged,
        "max_halvings": max_halvings,
        "factor": float(factor),
    }
    return out, counts, report


def reduce_surface_witness_node_scales_183(
    faces, step_scales, bad_face_rows, self_pair_face_rows, *, anchors=(), zero_below=1.0 / 128.0
):
    """Runner-compatible witness-local scale reduction with zero fallback."""
    threshold = float(zero_below)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
        raise ValueError("zero_below must be finite and in [0,1)")
    out, counts, report = reduce_witnessed_surface_step_scales_183(
        faces, step_scales, bad_face_rows, self_pair_face_rows,
        anchor_rows=anchors, max_halvings=32,
    )
    affected = np.asarray(report["affected_node_rows"], dtype=np.int64)
    fallback = affected[(out[affected] > 0.0) & (out[affected] <= threshold)]
    out[fallback] = 0.0
    report["threshold_zero_fallback_node_rows"] = fallback.tolist()
    report["zero_below"] = threshold
    report["halving_counts"] = counts.tolist()
    report["geometry_certified"] = False
    return out, report
