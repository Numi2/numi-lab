"""Small tested support for a coupled O/I/D shared-column fit.

This module only defines geometry/derivative plumbing. It does not encode a
bone-entry policy or claim anatomical clearance.
"""
from __future__ import annotations
from collections import Counter
import numpy as np


def patch_columns_from_seed(seed_columns, oriented_outer_faces):
    """Return incident-face patch, its exact boundary vertices, and interior columns."""
    seeds=np.asarray(seed_columns)
    F=np.asarray(oriented_outer_faces)
    if seeds.ndim!=1 or seeds.dtype.kind not in "iu" or F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu":
        raise ValueError("seed columns/faces must be integer arrays [N] and [F,3]")
    seeds=np.unique(seeds.astype(np.int64));F=F.astype(np.int64)
    if np.any(seeds<0) or np.any(F<0):raise ValueError("negative column identity")
    patch_rows=np.flatnonzero(np.isin(F,seeds).any(axis=1))
    patch=F[patch_rows]
    if not len(patch):return patch_rows,np.empty(0,np.int64),np.empty(0,np.int64)
    edges=Counter()
    for a,b,c in patch:
        edges[tuple(sorted((int(a),int(b))))]+=1
        edges[tuple(sorted((int(b),int(c))))]+=1
        edges[tuple(sorted((int(c),int(a))))]+=1
    if any(v>2 for v in edges.values()):raise ValueError("patch edge incidence exceeds two")
    boundary=np.asarray(sorted({x for e,n in edges.items() if n==1 for x in e}),np.int64)
    columns=np.unique(patch)
    movable=np.setdiff1d(columns,boundary,assume_unique=True)
    return patch_rows,boundary,movable


def apply_shared_column_translations(points, columns, delta, *, column_count=2411, layer_count=3):
    """Translate O/I/D nodes by the same vector for each selected column.

    Points are [layer_count*column_count,3]. Delta is [len(columns),3].
    Output retains all unselected rows bitwise in the input dtype.
    """
    X=np.asarray(points)
    C=np.asarray(columns)
    D=np.asarray(delta,dtype=np.float64)
    if X.ndim!=2 or X.shape!=(layer_count*column_count,3) or not np.all(np.isfinite(X)):
        raise ValueError("points must be finite [layer_count*column_count,3]")
    if C.ndim!=1 or C.dtype.kind not in "iu" or D.shape!=(len(C),3) or not np.all(np.isfinite(D)):
        raise ValueError("columns/delta shape or values invalid")
    C=C.astype(np.int64)
    if len(np.unique(C))!=len(C) or np.any(C<0) or np.any(C>=column_count):
        raise ValueError("column IDs must be unique and in range")
    Y=np.array(X,copy=True)
    for layer in range(layer_count):
        Y[layer*column_count+C]=np.asarray(X[layer*column_count+C],np.float64)+D
    return Y


def normalized_tet_quality_and_gradient(vertices):
    """Return signed det/maxedge^3 and gradients wrt all four xyz vertices."""
    V=np.asarray(vertices,dtype=np.float64)
    if V.ndim!=3 or V.shape[1:]!=(4,3) or not np.all(np.isfinite(V)):
        raise ValueError("vertices must be finite [N,4,3]")
    u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    det=np.einsum("ij,ij->i",u,np.cross(v,w))
    gd1=np.cross(v,w);gd2=np.cross(w,u);gd3=np.cross(u,v);gd0=-(gd1+gd2+gd3)
    gd=np.stack((gd0,gd1,gd2,gd3),axis=1)
    pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    edge=np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in pairs],axis=1)
    em=np.argmax(edge,axis=1);L=np.sqrt(np.maximum(edge[np.arange(len(V)),em],1e-30))
    q=det/L**3;G=gd/L[:,None,None]**3
    ep=V[np.arange(len(V)),np.array(pairs)[em,0]]-V[np.arange(len(V)),np.array(pairs)[em,1]]
    cc=(-3.0*det/L**5)[:,None]
    np.add.at(G,(np.arange(len(V)),np.array(pairs)[em,0]),cc*ep)
    np.add.at(G,(np.arange(len(V)),np.array(pairs)[em,1]),-cc*ep)
    return q,G


def reference_J_and_gradient(current_vertices, reference_vertices):
    """Return det(F) and nodal gradients for P1 tetrahedra in matching row order."""
    X=np.asarray(current_vertices,dtype=np.float64);R=np.asarray(reference_vertices,dtype=np.float64)
    if X.shape!=R.shape or X.ndim!=3 or X.shape[1:]!=(4,3) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(R)):
        raise ValueError("current/reference vertices must be finite matching [N,4,3]")
    U=R[:,1]-R[:,0];V=R[:,2]-R[:,0];W=R[:,3]-R[:,0]
    den=np.einsum("ij,ij->i",U,np.cross(V,W))
    if np.any(~np.isfinite(den)) or np.any(np.abs(den)<1e-24):raise ValueError("degenerate reference tetrahedron")
    u=X[:,1]-X[:,0];v=X[:,2]-X[:,0];w=X[:,3]-X[:,0]
    det=np.einsum("ij,ij->i",u,np.cross(v,w))
    g1=np.cross(v,w)/den[:,None];g2=np.cross(w,u)/den[:,None];g3=np.cross(u,v)/den[:,None]
    g0=-(g1+g2+g3)
    return det/den,np.stack((g0,g1,g2,g3),axis=1)


def aggregate_vertex_gradient_to_columns(vertex_gradient, tetrahedron_nodes, movable_columns, *, column_count=2411):
    """Chain a [cells,4,3] vertex gradient to shared-column translation DOFs."""
    G=np.asarray(vertex_gradient,dtype=np.float64);T=np.asarray(tetrahedron_nodes)
    C=np.asarray(movable_columns)
    if G.ndim!=3 or G.shape[1:]!=(4,3) or T.shape!=(len(G),4) or T.dtype.kind not in "iu":
        raise ValueError("gradient/tetrahedra shapes invalid")
    if C.ndim!=1 or C.dtype.kind not in "iu" or len(np.unique(C))!=len(C):raise ValueError("movable column IDs invalid")
    C=C.astype(np.int64); lookup=np.full(column_count,-1,np.int64)
    if np.any(C<0) or np.any(C>=column_count):raise ValueError("movable column out of range")
    lookup[C]=np.arange(len(C))
    col=lookup[T.astype(np.int64)%column_count]
    J=np.zeros((len(G),3*len(C)),np.float64)
    rr=np.broadcast_to(np.arange(len(G))[:,None],col.shape)
    for k in range(4):
        ok=col[:,k]>=0
        if np.any(ok):
            for axis in range(3):np.add.at(J,(rr[ok,k],3*col[ok,k]+axis),G[ok,k,axis])
    return J


def quality_and_column_jacobian(points, tetrahedron_nodes, rows, movable_columns, *, column_count=2411):
    X=np.asarray(points,dtype=np.float64);T=np.asarray(tetrahedron_nodes,dtype=np.int64);R=np.asarray(rows,dtype=np.int64)
    q,G=normalized_tet_quality_and_gradient(X[T[R]])
    return q,aggregate_vertex_gradient_to_columns(G,T[R],movable_columns,column_count=column_count)


def J_and_column_jacobian(points, reference_points, tetrahedron_nodes, rows, movable_columns, *, column_count=2411):
    X=np.asarray(points,dtype=np.float64);Ref=np.asarray(reference_points,dtype=np.float64);T=np.asarray(tetrahedron_nodes,dtype=np.int64);R=np.asarray(rows,dtype=np.int64)
    J,G=reference_J_and_gradient(X[T[R]],Ref[T[R]])
    return J,aggregate_vertex_gradient_to_columns(G,T[R],movable_columns,column_count=column_count)


def expand_seed_patch(seed_columns, oriented_outer_faces, *, max_rings=6):
    """Grow a face patch until seed columns are interior or ring budget is spent.

    Each expansion adds every face incident to a current patch-boundary vertex.
    The returned trace makes the measured patch growth reviewable; no anatomical
    radius or fixed vertex count is assumed.
    """
    seeds=np.asarray(seed_columns)
    F=np.asarray(oriented_outer_faces)
    if seeds.ndim!=1 or seeds.dtype.kind not in "iu" or F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu":
        raise ValueError("seed columns/faces must be integer arrays [N] and [F,3]")
    seeds=np.unique(seeds.astype(np.int64));F=F.astype(np.int64)
    if len(seeds)==0 or np.any(seeds<0) or np.any(F<0) or np.any(seeds>=int(F.max())+1):
        raise ValueError("empty or out-of-domain seed columns")
    incident={}
    for fi,face in enumerate(F):
        for node in face: incident.setdefault(int(node),[]).append(fi)
    if any(int(n) not in incident for n in seeds): raise ValueError("seed column absent from outer faces")
    selected=set()
    for n in seeds:selected.update(incident[int(n)])
    trace=[]
    for ring in range(max_rings+1):
        face_rows=np.asarray(sorted(selected),np.int64);patch=F[face_rows]
        edge_count=Counter()
        for a,b,c in patch:
            edge_count[tuple(sorted((int(a),int(b))))]+=1
            edge_count[tuple(sorted((int(b),int(c))))]+=1
            edge_count[tuple(sorted((int(c),int(a))))]+=1
        if any(v>2 for v in edge_count.values()):raise ValueError("grown patch is nonmanifold")
        boundary_nodes=np.asarray(sorted({n for e,k in edge_count.items() if k==1 for n in e}),np.int64)
        boundary_set=set(boundary_nodes.tolist())
        stuck=np.asarray([n for n in seeds if int(n) in boundary_set],np.int64)
        trace.append({"ring":ring,"face_count":int(len(face_rows)),"column_count":int(len(np.unique(patch))),
                      "boundary_column_count":int(len(boundary_nodes)),"seed_columns_still_on_boundary":int(len(stuck))})
        if not len(stuck) or ring==max_rings:
            columns=np.unique(patch)
            movable=np.setdiff1d(columns,boundary_nodes,assume_unique=True)
            return {"face_rows":face_rows,"patch_columns":columns,"boundary_columns":boundary_nodes,
                    "movable_columns":movable,"trace":trace,"unanchored_seed_columns":stuck}
        add=set()
        for n in boundary_nodes:add.update(incident.get(int(n),()))
        if not add-selected:raise ValueError("patch cannot expand around a seed on the physical boundary")
        selected.update(add)
    raise AssertionError("unreachable")


def plane_constraint_slack_and_jacobian(points, movable_columns, hit_points, plane_normals,
                                          endpoint_node_rows, margin, *, column_count=2411):
    """Build linear outside-side constraints for route-entry witnesses.

    `endpoint_node_rows` is [hits,K], with -1 for unused slots. Every populated
    endpoint is constrained to n·(x-hit) >= margin. The normal must already be
    directed toward the proven outside side by the source-route auditor.
    """
    X=np.asarray(points,dtype=np.float64);C=np.asarray(movable_columns);H=np.asarray(hit_points,dtype=np.float64)
    N=np.asarray(plane_normals,dtype=np.float64);E=np.asarray(endpoint_node_rows)
    if X.ndim!=2 or X.shape[1]!=3 or not np.all(np.isfinite(X)):raise ValueError("points must be finite [N,3]")
    if C.ndim!=1 or C.dtype.kind not in "iu":raise ValueError("movable columns must be integer IDs")
    C=C.astype(np.int64)
    if len(np.unique(C))!=len(C) or np.any(C<0) or np.any(C>=column_count):raise ValueError("bad movable columns")
    if H.ndim!=2 or H.shape[1]!=3 or N.shape!=H.shape or not np.all(np.isfinite(H)) or not np.all(np.isfinite(N)):
        raise ValueError("hit points/normals must be finite [hits,3]")
    if E.ndim!=2 or E.shape[0]!=len(H) or E.dtype.kind not in "iu":raise ValueError("endpoint rows must be integer [hits,K]")
    E=E.astype(np.int64)
    if np.any(E < -1) or np.any(E >= len(X)):raise ValueError("endpoint node out of range")
    norms=np.linalg.norm(N,axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms<1-2e-8) or np.any(norms>1+2e-8):raise ValueError("plane normals must be unit vectors")
    Mr=np.asarray(margin,dtype=np.float64)
    if Mr.ndim==0:Mr=np.full(len(H),float(Mr),np.float64)
    if Mr.shape!=(len(H),) or not np.all(np.isfinite(Mr)) or np.any(Mr<0):raise ValueError("margin must be finite nonnegative scalar or [hits]")
    lookup=np.full(column_count,-1,np.int64);lookup[C]=np.arange(len(C))
    rows=[];slack=[];jac=[];source=[]
    for hi in range(len(H)):
        for node in E[hi]:
            if node<0:continue
            node=int(node);ci=int(lookup[node%column_count])
            rows.append(node);source.append(hi)
            slack.append(float(np.dot(N[hi],X[node]-H[hi])-Mr[hi]))
            g=np.zeros(3*len(C),np.float64)
            if ci>=0:g[3*ci:3*ci+3]=N[hi]
            jac.append(g)
    return {"endpoint_node_rows":np.asarray(rows,np.int64),"hit_rows":np.asarray(source,np.int64),
            "slack":np.asarray(slack,np.float64),"jacobian":np.asarray(jac,np.float64).reshape(-1,3*len(C)),
            "fixed_endpoint_rows":np.asarray([i for i,n in enumerate(rows) if lookup[int(n)%column_count]<0],np.int64)}


class _DSU:
    def __init__(self,n):self.p=np.arange(n,dtype=np.int64);self.r=np.zeros(n,np.int8)
    def find(self,x):
        x=int(x)
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]];x=int(self.p[x])
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]:a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]:self.r[a]+=1


def shared_column_components(tetrahedra, movable_columns, *, column_count=2411, coupling_edges=(), extra_row_groups=()):
    """Exact separability components induced by cells and declared edge terms."""
    T=np.asarray(tetrahedra);C=np.asarray(movable_columns)
    if T.ndim!=2 or T.shape[1]!=4 or T.dtype.kind not in "iu" or C.ndim!=1 or C.dtype.kind not in "iu":
        raise ValueError("tetrahedra/columns must be integer arrays")
    C=C.astype(np.int64);T=T.astype(np.int64)
    if len(np.unique(C))!=len(C) or np.any(C<0) or np.any(C>=column_count) or np.any(T<0):raise ValueError("invalid column/tet IDs")
    lookup=np.full(column_count,-1,np.int64);lookup[C]=np.arange(len(C));dsu=_DSU(len(C))
    for tet in T:
        ids=np.unique(lookup[tet%column_count]);ids=ids[ids>=0]
        for j in ids[1:]:dsu.union(ids[0],j)
    for a,b in coupling_edges:
        ia=int(lookup[int(a)]);ib=int(lookup[int(b)])
        if ia>=0 and ib>=0:dsu.union(ia,ib)
        elif ia>=0 or ib>=0:
            # A fixed endpoint contributes to an objective but does not couple DOFs.
            pass
    for group in extra_row_groups:
        ids=np.unique(lookup[np.asarray(group,dtype=np.int64)%column_count]);ids=ids[ids>=0]
        for j in ids[1:]:dsu.union(ids[0],j)
    roots=np.asarray([dsu.find(i) for i in range(len(C))],np.int64)
    ordered=sorted(set(roots.tolist()),key=lambda r:(-int(np.sum(roots==r)),int(r)))
    groups=[C[np.flatnonzero(roots==r)] for r in ordered]
    # Return only variable columns; each cell's variable support must be wholly in one group.
    gid=np.full(column_count,-1,np.int64)
    for k,g in enumerate(groups):gid[g]=k
    return groups,gid


def surface_edge_pairs(faces):
    """Unique sorted undirected O-column edges from oriented triangles."""
    F=np.asarray(faces)
    if F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu" or np.any(F<0):raise ValueError("faces must be nonnegative integer [F,3]")
    E=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
    E=np.sort(E.astype(np.int64),axis=1)
    return np.unique(E,axis=0)


def _string_array(values, name, n):
    a=np.asarray(values)
    if a.ndim!=1 or len(a)!=n or a.dtype.kind not in "USO":raise ValueError(f"{name} must be a length-N string array")
    return np.asarray([str(x).strip().lower() for x in a],dtype=object)


def route_entry_endpoint_constraints(column_rows, segment_codes, start_node_rows, end_node_rows,
                                     route_roles, start_classes, end_classes, hit_points_world_m,
                                     outward_normals_world, segment_start_world_m, *,
                                     column_count=2411, total_nodes=7233, side_tol_m=1e-8):
    """Select only proven first-entry rows and map downstream endpoints by O/I/D role.

    Code 0 is O->I and constrains I and D. Code 1 is I->D and constrains D.
    The route exporter must supply a unit normal toward the certified outside
    side. Other rows are retained in the returned audit counts as nonbinding.
    """
    C=np.asarray(column_rows);S=np.asarray(segment_codes);A=np.asarray(start_node_rows);B=np.asarray(end_node_rows)
    if C.ndim!=1 or C.dtype.kind not in "iu":raise ValueError("column rows must be integer [N]")
    n=len(C);C=C.astype(np.int64)
    if S.shape!=(n,) or S.dtype.kind not in "iu" or A.shape!=(n,) or A.dtype.kind not in "iu" or B.shape!=(n,) or B.dtype.kind not in "iu":
        raise ValueError("segment/start/end node arrays must be integer [N]")
    S=S.astype(np.int64);A=A.astype(np.int64);B=B.astype(np.int64)
    if np.any(C<0) or np.any(C>=column_count) or np.any(~np.isin(S,[0,1])):raise ValueError("invalid column or segment code")
    if np.any(A<0) or np.any(A>=total_nodes) or np.any(B<0) or np.any(B>=total_nodes):raise ValueError("route endpoint node out of range")
    roles=_string_array(route_roles,"route_roles",n);sc=_string_array(start_classes,"start_classes",n);ec=_string_array(end_classes,"end_classes",n)
    H=np.asarray(hit_points_world_m,dtype=np.float64);N=np.asarray(outward_normals_world,dtype=np.float64);P=np.asarray(segment_start_world_m,dtype=np.float64)
    if H.shape!=(n,3) or N.shape!=(n,3) or P.shape!=(n,3) or not np.all(np.isfinite(H)) or not np.all(np.isfinite(P)):
        raise ValueError("route hit/start points must be finite [N,3]")
    if not np.isfinite(side_tol_m) or side_tol_m<0:raise ValueError("outside-side tolerance invalid")
    expected_a=np.where(S==0,C,column_count+C);expected_b=np.where(S==0,column_count+C,2*column_count+C)
    if np.any(A!=expected_a) or np.any(B!=expected_b):raise ValueError("route endpoint global-node IDs disagree with O/I/D column roles")
    hit_ids=[];target=[];nonbinding={"not_entry":0,"start_not_proven_outside":0,"end_class_unknown":0,"invalid_normal":0,"start_not_on_outside_side":0}
    for i in range(n):
        if roles[i]!="entry":nonbinding["not_entry"]+=1;continue
        if sc[i] not in ("outside","outside_solid","outside_union"):nonbinding["start_not_proven_outside"]+=1;continue
        if ec[i] in ("unknown","ambiguous","boundary","coplanar",""):
            nonbinding["end_class_unknown"]+=1;continue
        ni=N[i];nn=float(np.linalg.norm(ni))
        if not np.all(np.isfinite(ni)) or abs(nn-1)>2e-7:nonbinding["invalid_normal"]+=1;continue
        side=float(np.dot(ni,P[i]-H[i]))
        if side < -side_tol_m:nonbinding["start_not_on_outside_side"]+=1;continue
        hit_ids.append(i)
        if S[i]==0:target.append((int(column_count+C[i]),int(2*column_count+C[i])))
        else:target.append((int(2*column_count+C[i]),-1))
    return {"source_hit_rows":np.asarray(hit_ids,np.int64),
            "endpoint_node_rows":np.asarray(target,np.int64).reshape(-1,2),
            "nonbinding_counts":nonbinding,
            "binding_segment_code":S[np.asarray(hit_ids,np.int64)] if hit_ids else np.empty(0,np.int64)}


def expand_construction_plane_targets(column_rows, target_layer_mask, plane_point_world_m,
                                      unit_exterior_normal_world, clearance_m, *,
                                      column_count=2411, allowed_masks=(4,6)):
    """Expand certified owner planes to the exact downstream global O/I/D node rows."""
    C=np.asarray(column_rows);M=np.asarray(target_layer_mask)
    if C.ndim!=1 or C.dtype.kind not in "iu" or M.shape!=C.shape or M.dtype.kind not in "iu":
        raise ValueError("column rows and target masks must be integer vectors")
    C=C.astype(np.int64);M=M.astype(np.int64);n=len(C)
    if np.any(C<0) or np.any(C>=column_count) or np.any(~np.isin(M,np.asarray(allowed_masks,np.int64))):
        raise ValueError("unsupported column or target layer mask")
    P=np.asarray(plane_point_world_m,dtype=np.float64);N=np.asarray(unit_exterior_normal_world,dtype=np.float64)
    G=np.asarray(clearance_m,dtype=np.float64)
    if P.shape!=(n,3) or N.shape!=(n,3) or G.shape!=(n,) or not np.all(np.isfinite(P)) or not np.all(np.isfinite(N)) or not np.all(np.isfinite(G)):
        raise ValueError("plane arrays must be finite and row-aligned")
    if np.any(G<0) or np.any(np.abs(np.linalg.norm(N,axis=1)-1)>2e-7):raise ValueError("clearance or unit plane normal invalid")
    E=np.full((n,2),-1,np.int64)
    for i,(col,mask) in enumerate(zip(C,M)):
        layers=[k for k in range(3) if int(mask)&(1<<k)]
        if not layers or 0 in layers:raise ValueError("bone-entry plane cannot target outer O layer")
        for slot,layer in enumerate(layers):E[i,slot]=layer*column_count+int(col)
    return {"endpoint_node_rows":E,"plane_point_world_m":P,"unit_exterior_normal_world":N,
            "clearance_m":G,"target_layer_mask":M}
