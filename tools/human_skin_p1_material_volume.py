"""Conforming P1 two-band material construction; no per-step CPU physics."""
from collections import defaultdict
import numpy as np

_FACES=((1,2,3),(0,3,2),(0,1,3),(0,2,1))


def split_material_bands(points, tetrahedra, phi, skin_level, deep_level):
    """Clip positive source tets into skin phi<=I and fat I<=phi<=D.

    Source-edge/level identities are shared globally. Convex-cell pulling
    triangulation uses the same total node order on every shared polygon.
    No proximity welding, removed slivers, invented center nodes or averaging.
    Source-node and edge barycentric ancestry plus source-tet parents are kept.
    Geometry, Float32 quality, mass and physical admission remain separate.
    """
    x=np.asarray(points,dtype=np.float32)
    t=np.asarray(tetrahedra);p=np.asarray(phi,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError("finite Float32 Nx3 points required")
    if t.ndim!=2 or t.shape[1]!=4 or t.dtype.kind not in "iu" or np.any(t<0) or np.any(t>=len(x)):
        raise ValueError("integer source tetrahedron identities required")
    if p.shape!=(len(x),) or not np.isfinite(p).all():
        raise ValueError("finite source P1 field required")
    if not np.isfinite([skin_level,deep_level]).all() or not skin_level<deep_level:
        raise ValueError("ordered finite material levels required")
    v=x[t].astype(float);e=v[:,1:]-v[:,:1]
    det=np.einsum("ij,ij->i",e[:,0],np.cross(e[:,1],e[:,2]))
    if np.any(det<=0) or not np.isfinite(det).all():
        raise ValueError("positive finite source tetrahedra required")
    nodes=list(x.astype(float));values=list(p)
    supports=[(i,i) for i in range(len(x))];alpha=[0.0]*len(x);cache={}
    children=[];materials=[];parents=[]
    def crossing(a,b,level):
        if values[a]==level:return a
        if values[b]==level:return b
        support=sorted(set(supports[a]+supports[b]))
        if len(support)!=2:raise ValueError("clip edge is not an original source edge")
        u,w=support;key=(u,w,float(level))
        if key not in cache:
            s=(level-p[u])/(p[w]-p[u])
            if not 0<s<1:raise ValueError("invalid edge crossing interpolation")
            cache[key]=len(nodes)
            nodes.append((1-s)*x[u].astype(float)+s*x[w].astype(float))
            values.append(float(level));supports.append((u,w));alpha.append(float(s))
        return cache[key]
    def clip(faces,level,keep_low,gradient):
        active=sorted({a for f in faces for a in f})
        q=np.asarray([values[a] for a in active])
        if keep_low:
            if q.max()<=level:return faces
            if q.min()>=level:return []
        else:
            if q.min()>=level:return faces
            if q.max()<=level:return []
        out=[];cut=set()
        inside=lambda a:values[a]<=level if keep_low else values[a]>=level
        for face in faces:
            poly=[]
            for a,b in zip(face,face[1:]+face[:1]):
                if inside(a):poly.append(a)
                if inside(a)!=inside(b):poly.append(crossing(a,b,level))
            poly=list(dict.fromkeys(poly))
            if len(poly)>=3:
                out.append(poly)
                cut.update(a for a in poly if values[a]==level)
        if len(cut)<3:raise ValueError("clipped cell has no material interface polygon")
        ids=np.asarray(sorted(cut),dtype=np.int64)
        normal=np.asarray(gradient if keep_low else -gradient,float)
        normal/=np.linalg.norm(normal)
        axis=np.eye(3)[int(np.argmin(np.abs(normal)))]
        u=np.cross(axis,normal);u/=np.linalg.norm(u);w=np.cross(normal,u)
        r=np.asarray([nodes[a] for a in ids]);r-=r.mean(axis=0)
        angles=np.arctan2(r@w,r@u)
        order=np.lexsort((ids,angles));cap=ids[order].tolist()
        out.append(cap)
        return out
    for parent,tet in enumerate(t):
        tv=p[tet];base=[[int(tet[j]) for j in face] for face in _FACES]
        gradient=np.linalg.solve(e[parent],tv[1:]-tv[0])
        for material,lo,hi in ((1,None,float(skin_level)),(2,float(skin_level),float(deep_level))):
            if tv.min()>hi or (lo is not None and tv.max()<=lo):continue
            faces=base
            if lo is not None:faces=clip(faces,lo,False,gradient)
            if not faces:continue
            faces=clip(faces,hi,True,gradient)
            if not faces:continue
            apex=min(a for f in faces for a in f)
            for face in faces:
                if apex in face:continue
                k=face.index(min(face));face=face[k:]+face[:k]
                for j in range(1,len(face)-1):
                    child=(apex,face[0],face[j],face[j+1])
                    a,b,c,d=np.asarray([nodes[i] for i in child])
                    signed=float(np.dot(b-a,np.cross(c-a,d-a)))
                    if not np.isfinite(signed) or signed<=0:
                        raise ValueError(f"nonpositive clipped tetrahedron in source row {parent}")
                    children.append(child);materials.append(material);parents.append(parent)
    if not children:raise ValueError("no positive material volume")
    used=np.unique(children);remap=np.full(len(nodes),-1,np.int64);remap[used]=np.arange(len(used))
    X=np.asarray(nodes,dtype=np.float32)[used];T=remap[np.asarray(children,np.int64)]
    support=np.asarray(supports,np.int64)[used];a=np.asarray(alpha,dtype=np.float64)[used]
    return dict(points_world_m=X,tetrahedra=T,material_ids=np.asarray(materials,np.uint32),
                source_tetrahedron_rows=np.asarray(parents,np.int64),
                source_node_ids=support,source_edge_alpha=a,
                source_field_values=np.asarray(values,dtype=np.float64)[used],
                material_levels=np.asarray([skin_level,deep_level],np.float64),
                native_admission=False)


def material_role_faces(tetrahedra, material_ids):
    """Derive actual O/I/D facet tables for a conforming clipped volume.

    Geometry quality and coverage are deliberately not certified here; use the
    independent material-boundary certifier before passing them to any owner.
    """
    inc=defaultdict(list)
    for tet,material in zip(tetrahedra,material_ids,strict=True):
        for loc in _FACES:
            f=tuple(int(tet[j]) for j in loc);inc[tuple(sorted(f))].append((int(material),f))
    result={k:[] for k in ("outer_faces","interface_faces","deep_faces")}
    for key,rows in sorted(inc.items()):
        if len(rows)==1:
            m,f=rows[0];result["outer_faces" if m==1 else "deep_faces"].append(f)
        elif len(rows)==2:
            if rows[0][0]!=rows[1][0]:
                result["interface_faces"].append(next(f for m,f in rows if m==1))
        else:raise ValueError(f"nonmanifold material facet {key}")
    return {k:np.asarray(v,np.int64).reshape(-1,3) for k,v in result.items()}


def exterior_source_face_rows(volume, role_faces, source_boundary_faces):
    """Map exterior child facets to source PLC triangles through exact ancestry.

    -1 marks an isosurface facet with no source-boundary parent. This is only
    source-node identity coverage, not an anatomical or geometric certificate.
    Ambiguous parents are rejected instead of selecting by nearest distance.
    """
    source=np.asarray(source_boundary_faces)
    if source.ndim!=2 or source.shape[1]!=3 or source.dtype.kind not in "iu":
        raise ValueError("integer source boundary faces required")
    bynode=defaultdict(set)
    for row,f in enumerate(source):
        if len(set(map(int,f)))!=3:raise ValueError("repeated source boundary node")
        for v in f:bynode[int(v)].add(row)
    supports=np.asarray(volume["source_node_ids"])
    result={}
    for role,faces in role_faces.items():
        out=[]
        for f in faces:
            ids=set(map(int,supports[f].reshape(-1)))
            possible=None
            for v in sorted(ids):
                possible=bynode[v].copy() if possible is None else possible & bynode[v]
            if possible and len(possible)>1:
                raise ValueError("ambiguous source boundary parent")
            out.append(next(iter(possible)) if possible else -1)
        result[role]=np.asarray(out,np.int64)
    return result



class InteriorSupportRefinementError(ValueError):
    """Failed construction with reproducible Float32 cell witnesses."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def refine_interior_support(points, tetrahedra, *, selected_edges=(),
                            selected_faces=(), selected_tetrahedra=()):
    """Add conforming free construction coordinates on selected interior supports.

    Boundary facets/edges and original Float32 source rows are immutable. Shared
    interior edge/face centers use exact source identities, never coordinate
    welding. Every touched source cell is coned to one interior center. Faces
    touched by an edge split get one shared face center and a deterministic fan.
    Returns existing array/CSR ancestry conventions, not an admitted tissue asset.
    New coordinates can improve scalar-field support; physical/mass/contact and
    Float32 cell-quality qualification remain mandatory.
    """
    from itertools import combinations
    x=np.asarray(points,dtype=np.float32);raw=np.asarray(tetrahedra)
    if (x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or
        raw.ndim!=2 or raw.shape[1]!=4 or raw.dtype.kind not in "iu" or
        np.any(raw<0) or np.any(raw>=len(x))):
        raise ValueError("finite points and valid integer tetrahedra required")
    t=raw.astype(np.int64)
    if np.any(np.diff(np.sort(t,axis=1),axis=1)==0) or len(np.unique(np.sort(t,axis=1),axis=0))!=len(t):
        raise ValueError("repeated or duplicate tetrahedron identities")
    def determinants(v,c):
        a=v[c].astype(np.float64);u=a[:,1:]-a[:,:1]
        return np.einsum("ij,ij->i",u[:,0],np.cross(u[:,1],u[:,2]))
    source_det=determinants(x,t)
    if not np.isfinite(source_det).all() or np.any(source_det<=0):
        raise ValueError("positive Float32 source tetrahedra required")
    face_owners=defaultdict(list);all_edges=set()
    for row,cell in enumerate(t):
        for local in _FACES:
            f=tuple(int(cell[j]) for j in local);face_owners[tuple(sorted(f))].append((row,f))
        all_edges.update(tuple(map(int,e)) for e in combinations(sorted(cell),2))
    boundary=set()
    for key,owners in face_owners.items():
        if len(owners)>2:raise ValueError("nonmanifold source face")
        if len(owners)==1:boundary.add(key)
        else:
            def parity(face):
                return sum(face[i]>face[j] for i in range(3) for j in range(i+1,3))%2
            if parity(owners[0][1])==parity(owners[1][1]):
                raise ValueError("inconsistent source face orientation")
    boundary_edges={tuple(e) for f in boundary for e in combinations(f,2)}
    def feature_set(value,width,name):
        a=np.asarray(value)
        if not a.size:return set()
        if (a.ndim!=2 or a.shape[1]!=width or a.dtype.kind not in "iu" or
            np.any(a<0) or np.any(a>=len(x)) or
            np.any(np.diff(np.sort(a,axis=1),axis=1)==0)):
            raise ValueError("invalid "+name+" identities")
        return {tuple(map(int,z)) for z in np.sort(a,axis=1)}
    edges=feature_set(selected_edges,2,"edge")
    faces=feature_set(selected_faces,3,"face")
    cells=np.asarray(selected_tetrahedra)
    if cells.size and (cells.ndim!=1 or cells.dtype.kind not in "iu" or np.any(cells<0) or np.any(cells>=len(t))):
        raise ValueError("invalid tetrahedron rows")
    selected=set(map(int,cells.ravel()))
    if not edges<=all_edges or not faces<=face_owners.keys():
        raise ValueError("selected support is absent from source mesh")
    if edges&boundary_edges or faces&boundary:
        raise ValueError("physical boundary support is immutable")
    # One shared triangulation per source face, including all adjacent cells.
    for f in face_owners:
        if any(tuple(e) in edges for e in combinations(f,2)):faces.add(f)
    touched=set(selected)
    for f in faces:touched.update(row for row,_ in face_owners[f])
    nodes=list(x.astype(float));indptr=list(range(len(x)+1))
    indices=list(range(len(x)));weights=[1.]*len(x)
    def center(ids, bary=None):
        row=len(nodes);ids=tuple(map(int,ids))
        bary=np.full(len(ids),1./len(ids)) if bary is None else np.asarray(bary,dtype=float)
        nodes.append((bary@x[list(ids)].astype(np.float64)).astype(np.float32).astype(float))
        indices.extend(ids);weights.extend(bary.tolist());indptr.append(len(indices))
        return row
    edge_nodes={e:center(e) for e in sorted(edges)}
    face_nodes={f:center(f) for f in sorted(faces)}
    face_triangles={}
    for key in faces:
        a,b,c=key;ring=[]
        for u,w in ((a,b),(b,c),(c,a)):
            ring.append(u);mid=edge_nodes.get(tuple(sorted((u,w))))
            if mid is not None:ring.append(mid)
        middle=face_nodes[key]
        face_triangles[key]=[(middle,ring[j],ring[(j+1)%len(ring)]) for j in range(len(ring))]
    def oriented_triangles(cell):
        triangles=[]
        for local in _FACES:
            face=tuple(int(cell[j]) for j in local);key=tuple(sorted(face))
            if key not in face_triangles:triangles.append(face);continue
            odd=sum(face[i]>face[j] for i in range(3) for j in range(i+1,3))%2
            for a,b,c in face_triangles[key]:
                triangles.append((a,c,b) if odd else (a,b,c))
        return np.asarray(triangles,dtype=np.int64)

    candidates=[np.full(4,.25)]
    for i in range(4):
        b=np.full(4,.2);b[i]=.4;candidates.append(b)
        b=np.full(4,.3);b[i]=.1;candidates.append(b)
    for i in range(4):
        for j in range(i+1,4):
            for sign in (-1,1):
                b=np.full(4,.25);b[i]+=sign*.125;b[j]-=sign*.125;candidates.append(b)
    candidates=np.asarray(candidates)
    cell_nodes={};cell_triangles={};noncentroid=0
    shared_points=np.asarray(nodes,dtype=np.float32)
    for row in sorted(touched):
        triangles=oriented_triangles(t[row]);cell_triangles[row]=triangles
        face=shared_points[triangles].astype(float)
        trial=(candidates@x[t[row]].astype(float)).astype(np.float32).astype(float)
        u=face[None,:,:,:]-trial[:,None,None,:]
        determinant=np.einsum("cfi,cfi->cf",u[:,:,0],np.cross(u[:,:,1],u[:,:,2]))
        longest=np.max(np.sum(u*u,axis=3),axis=2)
        for i,j in ((0,1),(0,2),(1,2)):
            longest=np.maximum(longest,np.sum((face[None,:,i]-face[None,:,j])**2,axis=2))
        q=determinant/np.power(longest,1.5)
        score=np.min(q,axis=1)
        valid=np.isfinite(q).all(axis=1)&(score>0)
        # Avoid unnecessary changes for already valid centroid constructions.
        choice=0
        if not valid[0] or score[0]<1e-5:
            if valid.any():choice=int(np.argmax(np.where(valid,score,-np.inf)))
        noncentroid+=int(choice!=0)
        cell_nodes[row]=center(t[row],candidates[choice])
    out=[];parents=[]
    for row,cell in enumerate(t):
        if row not in touched:
            out.append(tuple(map(int,cell)));parents.append(row);continue
        middle=cell_nodes[row]
        for triangle in cell_triangles[row]:
            out.append((middle,*triangle));parents.append(row)
    out=np.asarray(out,dtype=np.int64).reshape(-1,4);v=np.asarray(nodes,dtype=np.float32)
    det=determinants(v,out)
    if not np.isfinite(det).all() or np.any(det<=0):
        bad=np.flatnonzero(~np.isfinite(det)|(det<=0))
        raise InteriorSupportRefinementError(
            "interior support refinement degenerates after Float32 emission",
            {"failed_child_rows":bad.astype(int).tolist(),
             "failed_source_tetrahedron_rows":np.asarray(parents)[bad].astype(int).tolist(),
             "first_witnesses":[
                 {"child_row":int(row),"source_tetrahedron_row":int(parents[row]),
                  "child_node_ids":out[row].astype(int).tolist(),
                  "child_points_float32_m":v[out[row]].astype(float).tolist(),
                  "determinant_m3":float(det[row])} for row in bad[:16]],
             "physical_admission":False})
    output_boundary=defaultdict(int)
    for cell in out:
        for local in _FACES:output_boundary[tuple(sorted(int(cell[j]) for j in local))]+=1
    if any(n>2 for n in output_boundary.values()) or {f for f,n in output_boundary.items() if n==1}!=boundary:
        raise ValueError("interior refinement changed source boundary facets")
    u=v[out].astype(float);edge2=np.zeros(len(out))
    for i,j in combinations(range(4),2):edge2=np.maximum(edge2,np.sum((u[:,i]-u[:,j])**2,axis=1))
    quality=det/np.power(edge2,1.5)
    volume=np.bincount(np.asarray(parents),weights=det,minlength=len(t))
    rel=np.abs(volume-source_det)/source_det
    return {"points":v,"tetrahedra":out,"source_tetrahedron_rows":np.asarray(parents,dtype=np.int64),
        "node_source_indptr":np.asarray(indptr,dtype=np.int64),
        "node_source_indices":np.asarray(indices,dtype=np.int64),
        "node_source_weights":np.asarray(weights,dtype=np.float64),
        "boundary_faces":np.asarray(sorted(boundary),dtype=np.int64).reshape(-1,3),
        "report":{"source_nodes":len(x),"source_tetrahedra":len(t),
                  "added_edge_nodes":len(edge_nodes),"added_face_nodes":len(face_nodes),
                  "added_cell_nodes":len(cell_nodes),"noncentroid_cell_nodes":noncentroid,"output_nodes":len(v),"output_tetrahedra":len(out),
                  "minimum_float32_quality":float(quality.min()) if len(quality) else None,
                  "below_quality_floor_1e_5":int(np.count_nonzero(quality<1e-5)),
                  "maximum_source_cell_relative_volume_change":float(rel.max()) if len(rel) else 0.,
                  "physical_admission":False}}
