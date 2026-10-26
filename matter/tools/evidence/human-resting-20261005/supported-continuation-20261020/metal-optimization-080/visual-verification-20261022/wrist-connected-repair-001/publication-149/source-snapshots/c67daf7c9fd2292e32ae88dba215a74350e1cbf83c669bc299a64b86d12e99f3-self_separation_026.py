#!/usr/bin/env python3
"""Linearized barriers for a small, explicit set of threatened surface-face pairs.

Rows act on global vertex displacement in millimetres. They preserve a separating
plane (or a common-vertex wedge) from the current safe geometry; exact surface
self-audits remain mandatory after every accepted trial.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog

# A moving separating midplane provides only half the pair gap to either side.
MAX_DISJOINT_MARGIN_FRACTION = 0.49


def _merge_coefficients(items):
    acc = {}
    for node, vec in items:
        node = int(node)
        acc[node] = acc.get(node, np.zeros(3, dtype=np.float64)) + np.asarray(vec, dtype=np.float64)
    out = [(i, v) for i, v in sorted(acc.items()) if float(np.dot(v, v)) > 1e-28]
    if not out:
        raise ValueError("empty separation row")
    # Every separation row is a difference of points/barycentres, so its
    # coefficients are translation invariant and must sum to zero. Correct
    # only accumulation roundoff at one stable largest coefficient.
    residual = np.asarray(
        [math.fsum(float(v[d]) for _, v in out) for d in range(3)],
        dtype=np.float64,
    )
    pivot = max(range(len(out)), key=lambda j: (float(np.dot(out[j][1], out[j][1])), -out[j][0]))
    node, vec = out[pivot]
    out[pivot] = (node, np.asarray(vec, dtype=np.float64) - residual)
    remaining = np.asarray(
        [math.fsum(float(v[d]) for _, v in out) for d in range(3)],
        dtype=np.float64,
    )
    scale = max(1.0, max(math.fsum(abs(float(v[d])) for _, v in out) for d in range(3)))
    if float(np.max(np.abs(remaining))) > 8.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("separation coefficients are not translation invariant")
    return out


def _row(items, lower, reference_mm, label):
    merged = _merge_coefficients(items)
    # Evaluate the zero-sum linear form relative to one row node. This avoids
    # subtracting millimetre-scale absolute coordinates to recover micrometre
    # or smaller pair gaps.
    anchor = np.asarray(reference_mm[merged[0][0]], dtype=np.float64)
    value = math.fsum(
        math.fsum(
            float(c[d]) * (float(reference_mm[i, d]) - float(anchor[d]))
            for d in range(3)
        )
        for i, c in merged
    )
    return {
        "nodes": np.asarray([i for i, _ in merged], dtype=np.int64),
        "coefficients": np.asarray([c for _, c in merged], dtype=np.float64),
        "lower": float(lower - value),
        "label": label,
    }


def _closest_point_triangle(p, tri):
    a, b, c = tri
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = float(ab@ap), float(ac@ap)
    if d1 <= 0 and d2 <= 0: return a.copy(), np.array([1.,0.,0.])
    bp=p-b; d3,d4=float(ab@bp),float(ac@bp)
    if d3 >= 0 and d4 <= d3: return b.copy(), np.array([0.,1.,0.])
    vc=d1*d4-d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v=d1/(d1-d3); return a+v*ab,np.array([1-v,v,0.])
    cp=p-c; d5,d6=float(ab@cp),float(ac@cp)
    if d6 >= 0 and d5 <= d6: return c.copy(),np.array([0.,0.,1.])
    vb=d5*d2-d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w=d2/(d2-d6); return a+w*ac,np.array([1-w,0.,w])
    va=d3*d6-d5*d4
    if va <= 0 and d4-d3 >= 0 and d5-d6 >= 0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); return b+w*(c-b),np.array([0.,1-w,w])
    den=1.0/(va+vb+vc); v=vb*den; w=vc*den
    return a+ab*v+ac*w,np.array([1-v-w,v,w])


def _closest_segments(p0,p1,q0,q1):
    u=p1-p0; v=q1-q0; w=p0-q0
    a=float(u@u); b=float(u@v); c=float(v@v); d=float(u@w); e=float(v@w)
    den=a*c-b*b
    s=0.0 if den <= 1e-28 else float(np.clip((b*e-c*d)/den,0,1))
    t=float(np.clip((b*s+e)/c,0,1)) if c > 1e-28 else 0.0
    if t <= 0:
        t=0.0; s=float(np.clip(-d/a,0,1)) if a > 1e-28 else 0.0
    elif t >= 1:
        t=1.0; s=float(np.clip((b-d)/a,0,1)) if a > 1e-28 else 0.0
    return p0+s*u, q0+t*v, s, t


def _closest_triangles(pa, pb):
    candidates=[]
    eye=np.eye(3)
    for i,p in enumerate(pa):
        q,b=_closest_point_triangle(p,pb)
        candidates.append((float(np.linalg.norm(p-q)),p.copy(),q.copy(),eye[i].copy(),b))
    for i,p in enumerate(pb):
        q,b=_closest_point_triangle(p,pa)
        candidates.append((float(np.linalg.norm(p-q)),q.copy(),p.copy(),b,eye[i].copy()))
    for i in range(3):
        for j in range(3):
            p,q,s,t=_closest_segments(pa[i],pa[(i+1)%3],pb[j],pb[(j+1)%3])
            ba=np.zeros(3); ba[i]=1-s; ba[(i+1)%3]=s
            bb=np.zeros(3); bb[j]=1-t; bb[(j+1)%3]=t
            candidates.append((float(np.linalg.norm(p-q)),p,q,ba,bb))
    return min(candidates,key=lambda x:x[0])


def _sat_axes(pa,pb):
    axes=[]
    for tri in (pa,pb):
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        if np.linalg.norm(n)>1e-24: axes.append(n/np.linalg.norm(n))
    ea=[pa[(i+1)%3]-pa[i] for i in range(3)]
    eb=[pb[(i+1)%3]-pb[i] for i in range(3)]
    for a in ea:
        for b in eb:
            n=np.cross(a,b); nn=np.linalg.norm(n)
            if nn>1e-24: axes.append(n/nn)
    unique=[]
    for n in axes:
        # Axis sign is selected later by the larger of the two directed gaps.
        if not any(abs(float(n@q))>1-1e-10 for q in unique): unique.append(n)
    return unique


def build_pair_rows(points_mm, faces, pair_ids, *, margin_fraction=0.25):
    """Build sparse linear rows from accepted F32 geometry and stable face-row pairs."""
    p=np.asarray(points_mm,dtype=np.float64); f=np.asarray(faces,dtype=np.int64)
    if p.ndim!=2 or p.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3: raise ValueError("bad mesh shape")
    if not (0 < margin_fraction < 1): raise ValueError("margin_fraction must be in (0,1)")
    rows=[]; records=[]
    for fi,fj in sorted({tuple(sorted(map(int,x))) for x in pair_ids}):
        if fi<0 or fj<0 or fi>=len(f) or fj>=len(f) or fi==fj: raise ValueError("invalid face pair")
        ia,ib=f[fi],f[fj]; common=sorted(set(map(int,ia))&set(map(int,ib)))
        pa,pb=p[ia],p[ib]
        if len(common)==1:
            v=common[0]; va=[int(x) for x in ia if int(x)!=v]; vb=[int(x) for x in ib if int(x)!=v]
            candidates=[]
            for sign in (1.0,-1.0):
                vecs=np.asarray([sign*(p[x]-p[v]) for x in va]+[-sign*(p[x]-p[v]) for x in vb])
                A=np.column_stack((-vecs,np.ones(len(vecs))))
                result=linprog([0.,0.,0.,-1.],A_ub=A,b_ub=np.zeros(len(vecs)),
                               bounds=[(-1,1)]*3+[(None,None)],method="highs")
                if result.success:
                    n=result.x[:3]; nn=float(np.linalg.norm(n))
                    if nn>1e-14:
                        n=n/nn; gaps=vecs@n
                        candidates.append((float(gaps.min()),sign,n,gaps))
            if not candidates: raise ValueError(f"no separating wedge axis for {fi}/{fj}")
            min_gap,sign,n,gaps=max(candidates,key=lambda x:x[0])
            if min_gap<=1e-12: raise ValueError(f"nonseparable common-vertex wedge {fi}/{fj}: {min_gap}")
            margin=margin_fraction*min_gap
            for k,x in enumerate(va):
                rows.append(_row([(x,sign*n),(v,-sign*n)],margin,p,f"{fi}:{fj}:A{x}"))
            for x in vb:
                rows.append(_row([(x,-sign*n),(v,sign*n)],margin,p,f"{fi}:{fj}:B{x}"))
            records.append({"faces":[fi,fj],"shared_vertex":v,"kind":"common_vertex_wedge",
                            "axis_world":n.tolist(),"side_sign":sign,"closest_distance_mm":0.0,
                            "minimum_vertex_side_gap_mm":min_gap,"preserved_margin_mm":margin,
                            "requested_margin_fraction":float(margin_fraction),
                            "effective_margin_fraction":float(margin_fraction),
                            "margin_cap_applied":False,
                            "witness_points_world_mm":[p[v].tolist()]})
            continue
        if common:
            raise ValueError(f"pair shares unsupported feature {fi}/{fj}: {common}")
        dist,qa,qb,ba,bb=_closest_triangles(pa,pb)
        if dist<=1e-12: raise ValueError(f"disjoint face pair is not separated at base {fi}/{fj}")
        n=(qb-qa)/dist
        gap=float(np.min(pb@n)-np.max(pa@n))
        if gap<=1e-12:
            opts=[]
            for axis in _sat_axes(pa,pb):
                ga=float(np.min(pb@axis)-np.max(pa@axis))
                gb=float(np.min(pa@axis)-np.max(pb@axis))
                if ga>1e-12: opts.append((ga,axis))
                if gb>1e-12: opts.append((gb,-axis))
            if not opts: raise ValueError(f"no SAT separating axis for {fi}/{fj}")
            gap,n=max(opts,key=lambda x:x[0])
        effective_margin_fraction=min(float(margin_fraction), MAX_DISJOINT_MARGIN_FRACTION)
        margin=effective_margin_fraction*min(gap,dist)
        proj_a=pa@n; proj_b=pb@n
        a_support=np.flatnonzero(np.abs(proj_a-proj_a.max())<=1e-10)
        b_support=np.flatnonzero(np.abs(proj_b-proj_b.min())<=1e-10)
        wa=np.zeros(3); wb=np.zeros(3); wa[a_support]=1.0/len(a_support); wb[b_support]=1.0/len(b_support)
        # Midplane follows the two support barycentres, so pair translation is free.
        for k,node in enumerate(ia):
            items=[]
            for l,q in enumerate(ia): items.append((q,0.5*wa[l]*n))
            for l,q in enumerate(ib): items.append((q,0.5*wb[l]*n))
            items.append((node,-n))
            rows.append(_row(items,margin,p,f"{fi}:{fj}:A{int(node)}"))
        for k,node in enumerate(ib):
            items=[]
            for l,q in enumerate(ia): items.append((q,-0.5*wa[l]*n))
            for l,q in enumerate(ib): items.append((q,-0.5*wb[l]*n))
            items.append((node,n))
            rows.append(_row(items,margin,p,f"{fi}:{fj}:B{int(node)}"))
        mid=0.5*(float(wa@proj_a)+float(wb@proj_b))
        records.append({"faces":[fi,fj],"shared_vertex":None,"kind":"moving_sat_midplane",
                        "axis_world":n.tolist(),"closest_distance_mm":float(dist),"support_gap_mm":float(gap),
                        "preserved_margin_mm":float(margin),
                        "requested_margin_fraction":float(margin_fraction),
                        "effective_margin_fraction":float(effective_margin_fraction),
                        "margin_cap_applied":bool(effective_margin_fraction < float(margin_fraction)),
                        "closest_a_barycentric":ba.tolist(),
                        "closest_b_barycentric":bb.tolist(),"closest_points_world_mm":[qa.tolist(),qb.tolist()],
                        "support_a_weights":wa.tolist(),"support_b_weights":wb.tolist(),
                        "midplane_projection_mm":mid})
    # Baseline must satisfy every generated row before solver use.
    violations=[max(0.0,float(r["lower"])) for r in rows]
    if max(violations,default=0.0)>2e-8: raise ValueError(f"self-separation baseline infeasible: {max(violations)} mm")
    return rows,records


def project_linear_rows(v, rows, *, max_sweeps=120, tolerance_mm=2e-4):
    """Dykstra projection onto small sparse linear halfspaces."""
    d=np.asarray(v,dtype=np.float64).copy()
    corr=np.zeros(len(rows),dtype=np.float64)
    norms=np.asarray([float(np.einsum('ij,ij->',r['coefficients'],r['coefficients'])) for r in rows])
    if np.any(norms<=1e-24): return d,{"converged":False,"reason":"zero_self_constraint_gradient"}
    maxv=float("inf")
    for sweep in range(max_sweeps):
        for k,r in enumerate(rows):
            ids=r['nodes']; c=r['coefficients']; y=d[ids]+corr[k]*c
            val=float(np.einsum('ij,ij->',y,c))
            step=max(0.0,(float(r['lower'])-val)/norms[k])
            d[ids]=y+step*c; corr[k]=-step
        vals=[float(np.einsum('ij,ij->',d[r['nodes']],r['coefficients'])) for r in rows]
        maxv=max((float(r['lower'])-v0 for r,v0 in zip(rows,vals)),default=0.0)
        maxv=max(0.0,maxv)
        if maxv<=tolerance_mm:
            return d,{"converged":True,"sweeps":sweep+1,"max_violation_mm":maxv}
    return d,{"converged":False,"sweeps":max_sweeps,"max_violation_mm":maxv}



def relax_pair_rows_for_projection(rows, records, allowance_mm):
    """Spend a declared positive numerical allowance while preserving a positive barrier margin.

    The exact emitted-F32 self-intersection audit remains the acceptance gate.
    This only prevents a tiny iterative projection residual from rejecting the
    linearized ADMM subproblem.
    """
    allowance_mm=float(allowance_mm)
    if not np.isfinite(allowance_mm) or allowance_mm <= 0:
        raise ValueError("projection allowance must be finite and positive")
    adjusted=[]
    for record in records:
        margin=float(record.get("preserved_margin_mm", float("nan")))
        remaining=margin-allowance_mm
        if not np.isfinite(margin) or remaining <= 0:
            raise ValueError("projection allowance consumes the pair separation margin")
        copy=dict(record)
        copy["projection_relaxation_mm"]=allowance_mm
        copy["effective_preserved_margin_mm"]=remaining
        adjusted.append(copy)
    if not adjusted and rows:
        raise ValueError("pair rows have no corresponding geometry records")
    for row in rows:
        row["lower"] = float(row["lower"])-allowance_mm
    return adjusted
