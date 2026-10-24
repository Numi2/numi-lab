"""Explicit mesh-domain adapter for the 402 absolute P1 field owner.

Construction/preflight only. The adapter does not promote or install a field.
"""
from __future__ import annotations
import numpy as np


def _int_vector(value, name):
    a=np.asarray(value)
    if a.ndim!=1 or (a.size and not np.issubdtype(a.dtype,np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return a.astype(np.int64,copy=False)


def map_output_tets_to_base_tets(output_parent_offsets, output_parent_rows,
                                 intermediate_parent_base_rows, output_tet_count,
                                 base_tet_count):
    """Compose output→intermediate and intermediate→base tet ancestry CSR."""
    off=_int_vector(output_parent_offsets,"output_parent_offsets")
    rows=_int_vector(output_parent_rows,"output_parent_rows")
    mid=_int_vector(intermediate_parent_base_rows,"intermediate_parent_base_rows")
    if output_tet_count<0 or base_tet_count<1:
        raise ValueError("invalid tetrahedron domain size")
    if off.shape!=(output_tet_count+1,) or off[0]!=0 or off[-1]!=len(rows) or np.any(np.diff(off)<0):
        raise ValueError("output tet-parent CSR does not match output mesh domain")
    if len(rows)==0 or np.any(rows<0) or np.any(rows>=len(mid)):
        raise ValueError("output tet-parent rows exceed intermediate tet domain")
    if np.any(mid<0) or np.any(mid>=base_tet_count):
        raise ValueError("intermediate-to-base tet map exceeds base mesh domain")
    boff=[0]; brows=[]
    for t in range(output_tet_count):
        parents=sorted(set(int(mid[r]) for r in rows[off[t]:off[t+1]]))
        if not parents:
            raise ValueError(f"output tet {t} has no containing base-tet ancestry")
        brows.extend(parents); boff.append(len(brows))
    return np.asarray(boff,dtype=np.int64),np.asarray(brows,dtype=np.int64)


def compose_parent_csrs(child_offsets, child_parent_rows, parent_offsets, parent_parent_rows, *, child_count, intermediate_count, base_count):
    """Compose child to intermediate and intermediate to base ancestry CSRs."""
    co=_int_vector(child_offsets,"child_offsets"); cr=_int_vector(child_parent_rows,"child_parent_rows")
    po=_int_vector(parent_offsets,"parent_offsets"); pr=_int_vector(parent_parent_rows,"parent_parent_rows")
    if co.shape!=(int(child_count)+1,) or co[0]!=0 or co[-1]!=len(cr) or np.any(np.diff(co)<0):
        raise ValueError("child ancestry CSR does not match child mesh domain")
    if po.shape!=(int(intermediate_count)+1,) or po[0]!=0 or po[-1]!=len(pr) or np.any(np.diff(po)<0):
        raise ValueError("parent ancestry CSR does not match intermediate mesh domain")
    if np.any(cr<0) or np.any(cr>=int(intermediate_count)):
        raise ValueError("child parent row exceeds intermediate tet domain")
    if np.any(pr<0) or np.any(pr>=int(base_count)):
        raise ValueError("intermediate parent row exceeds base tet domain")
    out_off=[0];out_rows=[]
    for child in range(int(child_count)):
        bases=set()
        for mid in cr[co[child]:co[child+1]]:
            bases.update(map(int,pr[po[int(mid)]:po[int(mid)+1]]))
        if not bases:
            raise ValueError(f"child tet {child} has no base ancestry")
        out_rows.extend(sorted(bases));out_off.append(len(out_rows))
    return np.asarray(out_off,dtype=np.int64),np.asarray(out_rows,dtype=np.int64)

def _barycentric(point, tet_points):
    x=np.asarray(tet_points,dtype=np.float64)
    p=np.asarray(point,dtype=np.float64)
    A=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]))
    if not np.isfinite(A).all() or abs(float(np.linalg.det(A)))<=np.finfo(np.float64).tiny:
        raise ValueError("degenerate base tetrahedron while mapping newpoint initializer")
    u=np.linalg.solve(A,p-x[0])
    b=np.array([1.0-u.sum(),u[0],u[1],u[2]],dtype=np.float64)
    return b


def initialize_field_on_mesh(mesh_points, mesh_tetrahedra,
                             base_points, base_tetrahedra, base_phi,
                             physical_outer_node_ids,
                             output_to_base_tet_offsets,
                             output_to_base_tet_rows,
                             *, containment_tolerance=2e-8):
    """Copy the unchanged 398 prefix and P1-interpolate every appended mesh node.

    Appended nodes use only geometrically containing base tetrahedra gathered
    from the current output tetrahedra's explicit parent ancestry. No nearest
    point, extrapolation, or guessed correspondence is allowed.
    """
    X=np.asarray(mesh_points)
    Traw=np.asarray(mesh_tetrahedra)
    if Traw.ndim!=2 or Traw.shape[1]!=4:
        raise ValueError("current mesh tetrahedra must have shape [T,4]")
    T=_int_vector(Traw.reshape(-1),"mesh_tetrahedra").reshape(-1,4)
    Xb=np.asarray(base_points)
    Tb=np.asarray(base_tetrahedra)
    phi=np.asarray(base_phi,dtype=np.float64)
    outer=_int_vector(physical_outer_node_ids,"physical_outer_node_ids")
    boff=_int_vector(output_to_base_tet_offsets,"output_to_base_tet_offsets")
    brows=_int_vector(output_to_base_tet_rows,"output_to_base_tet_rows")
    if X.ndim!=2 or X.shape[1]!=3 or X.dtype!=np.float32 or not np.isfinite(X).all():
        raise ValueError("current mesh points must be finite Float32 [N,3]")
    if T.ndim!=2 or T.shape[1]!=4 or np.any(T<0) or np.any(T>=len(X)):
        raise ValueError("invalid current mesh tetrahedron domain")
    if Xb.ndim!=2 or Xb.shape[1]!=3 or Xb.dtype!=np.float32 or not np.isfinite(Xb).all():
        raise ValueError("base points must be finite Float32 [N,3]")
    if Tb.ndim!=2 or Tb.shape[1]!=4 or not np.issubdtype(Tb.dtype,np.integer) or np.any(Tb<0) or np.any(Tb>=len(Xb)):
        raise ValueError("invalid base tetrahedron domain")
    if phi.shape!=(len(Xb),) or not np.isfinite(phi).all() or np.any(phi<0) or np.any(phi>1):
        raise ValueError("base initializer must match base node domain and lie in [0,1]")
    if len(X)<len(Xb) or not np.array_equal(X[:len(Xb)],Xb):
        raise ValueError("current mesh does not preserve the pinned base node prefix exactly")
    if boff.shape!=(len(T)+1,) or boff[0]!=0 or boff[-1]!=len(brows) or np.any(np.diff(boff)<0):
        raise ValueError("base-parent tet CSR does not match current tet domain")
    if np.any(brows<0) or np.any(brows>=len(Tb)):
        raise ValueError("base-parent tet row outside initializer mesh domain")
    if np.any(outer<0) or np.any(outer>=len(X)) or len(np.unique(outer))!=len(outer):
        raise ValueError("invalid current physical-O node map")
    if np.any(outer>=len(Xb)) or np.any(phi[outer]!=0):
        raise ValueError("physical O rows must remain in prefix and have exact zero initializer")
    if not np.isfinite(containment_tolerance) or containment_tolerance<0:
        raise ValueError("invalid barycentric containment tolerance")

    out=np.empty(len(X),dtype=np.float64);out[:len(Xb)]=phi
    chosen=np.full(len(X),-1,dtype=np.int64)
    chosen_bary=np.full((len(X),4),np.nan,dtype=np.float64)
    # Build current tet adjacency once. This avoids Nnew × Ntet scans.
    flat=T.reshape(-1)
    tet_rows=np.repeat(np.arange(len(T),dtype=np.int64),4)
    order=np.argsort(flat,kind="stable")
    sorted_nodes=flat[order]
    starts=np.searchsorted(sorted_nodes,np.arange(len(X)+1),side="left")
    for node in range(len(Xb),len(X)):
        a,b=int(starts[node]),int(starts[node+1])
        incident=np.unique(tet_rows[order[a:b]])
        if len(incident)==0:
            raise ValueError(f"appended mesh node {node} is unreferenced")
        candidate_rows=sorted(set(int(r) for ti in incident for r in brows[boff[ti]:boff[ti+1]]))
        matches=[]
        for br in candidate_rows:
            ids=Tb[br]
            try: bary=_barycentric(X[node],Xb[ids])
            except (ValueError,np.linalg.LinAlgError): continue
            if bary.min()>=-containment_tolerance and bary.max()<=1+containment_tolerance and abs(float(bary.sum())-1)<=2e-12:
                recon=bary@Xb[ids].astype(np.float64)
                if np.max(np.abs(recon-X[node].astype(np.float64)))<=5e-8:
                    matches.append((br,bary))
        if not matches:
            raise ValueError(f"appended node {node} has no containing base tet among its current tet ancestry ({len(candidate_rows)} candidates)")
        br,bary=min(matches,key=lambda x:x[0])
        out[node]=float(bary@phi[Tb[br]])
        chosen[node]=br;chosen_bary[node]=bary
    if not np.isfinite(out).all() or np.any(out<0) or np.any(out>1):
        bad=np.flatnonzero((out<0)|(out>1))
        witness=[{"node":int(n),"parent_398_tet":int(chosen[n]),"barycentric":chosen_bary[n].tolist(),"phi":float(out[n])} for n in bad[:12]]
        raise ValueError(f"mapped current-mesh initializer out of bounds [{float(np.min(out))},{float(np.max(out))}], low={int(np.count_nonzero(out<0))}, high={int(np.count_nonzero(out>1))}, witnesses={witness}")
    if not np.array_equal(out[outer],np.zeros(len(outer),dtype=np.float64)):
        raise ValueError("current physical O initializer lost exact zeros")
    return out,chosen,chosen_bary


def clip_typed_triangle_to_mesh(points, tetrahedra, triangle,
                                triangle_tet_barycentric_polygon):
    """Rebuild one triangle/tet constraint domain on exactly the supplied mesh."""
    X=np.asarray(points,dtype=np.float32)
    T=np.asarray(tetrahedra)
    P=np.asarray(triangle,dtype=np.float32)
    if X.ndim!=2 or X.shape[1]!=3 or T.ndim!=2 or T.shape[1]!=4 or P.shape!=(3,3):
        raise ValueError("invalid mesh or typed-triangle clipping domain")
    if not np.isfinite(X).all() or not np.isfinite(P).all() or not np.issubdtype(T.dtype,np.integer) or np.any(T<0) or np.any(T>=len(X)):
        raise ValueError("nonfinite points or invalid tetrahedron indices")
    tri_lo=P.min(axis=0).astype(np.float64);tri_hi=P.max(axis=0).astype(np.float64)
    tet_xyz=X[T].astype(np.float64)
    lo=tet_xyz.min(axis=1);hi=tet_xyz.max(axis=1)
    candidates=np.flatnonzero(np.all(lo<=tri_hi,axis=1)&np.all(hi>=tri_lo,axis=1))
    constraints=[];world_polygons=[];bad=[]
    for ti in candidates:
        try: B,C=triangle_tet_barycentric_polygon(tet_xyz[ti],P.astype(np.float64))
        except Exception as exc:
            bad.append({'tet_row':int(ti),'error':str(exc)})
            continue
        if len(B):
            if (not np.isfinite(B).all() or np.any(B<0) or np.max(np.abs(B.sum(axis=1)-1))>2e-12 or
                not np.isfinite(C).all() or np.any(C<0) or np.max(np.abs(C.sum(axis=1)-1))>2e-12):
                raise ValueError(f"invalid clipping barycentric certificate for current tet {ti}")
            constraints.append((int(ti),B.copy()))
            world_polygons.append({'tet_row':int(ti),'tet_barycentric':B,'triangle_barycentric':C})
    if bad:
        raise ValueError(f"clipper could not certify {len(bad)} current-mesh tet candidates; first={bad[0]}")
    return constraints,world_polygons,{'aabb_candidate_tet_count':int(len(candidates)),'intersected_tet_count':int(len(constraints))}


def fixed_zero_capacity_report(tetrahedra, constraints, fixed_node_ids, node_count, *, maximum_value=1.0):
    T=np.asarray(tetrahedra,dtype=np.int64)
    fixed=_int_vector(fixed_node_ids,"fixed_node_ids")
    if T.ndim!=2 or T.shape[1]!=4 or np.any(T<0) or np.any(T>=int(node_count)):
        raise ValueError("invalid tet domain for fixed-zero capacity calculation")
    is_fixed=np.zeros(int(node_count),dtype=bool)
    if np.any(fixed<0) or np.any(fixed>=len(is_fixed)):
        raise ValueError("fixed nodes exceed current mesh node domain")
    is_fixed[fixed]=True
    values=[]
    for ti,B in constraints:
        free=~is_fixed[T[int(ti)]]
        cap=np.sum(np.asarray(B,dtype=np.float64)[:,free],axis=1)*float(maximum_value)
        for j,v in enumerate(cap): values.append({'tet_row':int(ti),'polygon_vertex':int(j),'max_free_capacity':float(v)})
    return values
