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

