"""Exact-level P1 extraction; identities are source nodes or undirected edges."""
from collections import Counter
import numpy as np


def parity(ids):
    a=list(map(int,ids)); inv=sum(a[i]>a[j] for i in range(len(a)) for j in range(i+1,len(a)))
    return -1 if inv%2 else 1


def area2(P, tri):
    a,b,c=(np.asarray(P[int(i)],np.float64) for i in tri)
    return float(np.linalg.norm(np.cross(b-a,c-a)))


def edge_hist(F):
    c=Counter()
    for a,b,d in np.asarray(F,np.int64).reshape(-1,3):
        for u,v in ((a,b),(b,d),(d,a)):
            c[tuple(sorted((int(u),int(v))))]+=1
    return dict(Counter(c.values()))


def extract_level_exact(V,T,phi,level,area2_floor=1e-16,allow_empty=False):
    V=np.asarray(V,np.float32); rawT=np.asarray(T)
    if not np.issubdtype(rawT.dtype,np.integer): raise ValueError("tetrahedron IDs must be integers")
    T=np.asarray(rawT,np.int64); phi=np.asarray(phi,np.float64); level=float(level)
    if not np.isfinite(area2_floor) or area2_floor<0.0: raise ValueError("invalid area floor")
    if V.ndim!=2 or V.shape[1]!=3 or T.ndim!=2 or T.shape[1]!=4 or phi.shape!=(len(V),):
        raise ValueError("expected V[N,3], T[M,4], phi[N]")
    if not np.isfinite(V).all() or not np.isfinite(phi).all() or not np.isfinite(level):
        raise ValueError("non-finite extraction input")
    if T.size and (T.min()<0 or T.max()>=len(V)): raise ValueError("tetrahedron node index outside V")
    edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    point_by_key={};points=[];point_kind=[];parent_node=[];parent_edge=[];edge_t=[];point_parent_tet=[]
    face_by_key={};face_records=[];cancelled=set();cancelled_exact=0

    def node_point(g,ti):
        g=int(g);key=("v",g);ix=point_by_key.get(key)
        if ix is None:
            ix=len(points);point_by_key[key]=ix;points.append(np.asarray(V[g],np.float32).copy())
            point_kind.append(1);parent_node.append(g);parent_edge.append((g,g));edge_t.append(0.0);point_parent_tet.append(int(ti))
        return ix

    def edge_point(ga,gb,ti):
        ga=int(ga);gb=int(gb);lo,hi=(ga,gb) if ga<gb else (gb,ga);key=("e",lo,hi);ix=point_by_key.get(key)
        if ix is None:
            den=float(phi[hi]-phi[lo])
            if den==0.0: raise ValueError(("zero scalar denominator on strict crossing edge",lo,hi))
            t=float((level-phi[lo])/den)
            if not (0.0<t<1.0): raise ValueError(("strict crossing parameter outside open edge",lo,hi,t))
            p=(np.asarray(V[lo],np.float64)+t*(np.asarray(V[hi],np.float64)-np.asarray(V[lo],np.float64))).astype(np.float32)
            ix=len(points);point_by_key[key]=ix;points.append(p);point_kind.append(0);parent_node.append(-1)
            parent_edge.append((lo,hi));edge_t.append(t);point_parent_tet.append(int(ti))
        return ix

    def add_or_cancel(tri,ti,is_exact_face):
        nonlocal cancelled_exact
        tri=tuple(map(int,tri));key=tuple(sorted(tri));par=parity(tri);prev=face_by_key.get(key)
        if prev is None:
            face_by_key[key]=len(face_records)
            face_records.append({"tri":tri,"tet":int(ti),"parity":par,"exact_face":bool(is_exact_face),"cancelled":False})
            return
        old=face_records[prev]
        if old["cancelled"]: raise ValueError(("third duplicate face after cancellation",key,old["tet"],int(ti)))
        if not (old["exact_face"] and is_exact_face): raise ValueError(("unexpected duplicate isosurface triangle",key,old["tet"],int(ti)))
        if par==old["parity"]: raise ValueError(("same-oriented duplicated exact-level face",key,old["tet"],int(ti)))
        old["cancelled"]=True;cancelled.add(prev);cancelled_exact+=1

    for ti,tet in enumerate(T):
        tet=np.asarray(tet,np.int64);vals=phi[tet]-level
        if np.all(vals==0.0): raise ValueError(("constant all-level tetrahedron is an ambiguous plateau",int(ti),tet.tolist()))
        if not np.any(vals<0.0): continue
        if np.all(vals<0.0): continue
        local=[]
        for j,g in enumerate(tet):
            if vals[j]==0.0: local.append(node_point(int(g),ti))
        for ia,ib in edges:
            va,vb=float(vals[ia]),float(vals[ib])
            if (va<0.0 and vb>0.0) or (vb<0.0 and va>0.0): local.append(edge_point(int(tet[ia]),int(tet[ib]),ti))
        local=list(dict.fromkeys(local))
        if len(local)<3: continue
        if len(local)>4: raise ValueError(("tet produced more than four exact-level polygon vertices",int(ti),local))
        X=np.asarray(V[tet],np.float64);A=np.stack((X[1]-X[0],X[2]-X[0],X[3]-X[0]),axis=0);ds=np.asarray(vals[1:]-vals[0],np.float64)
        try: grad=np.linalg.solve(A,ds)
        except np.linalg.LinAlgError as exc: raise ValueError(("singular field tetrahedron",int(ti))) from exc
        gn=float(np.linalg.norm(grad))
        if not np.isfinite(gn) or gn<=1e-16: raise ValueError(("zero or invalid scalar gradient",int(ti),gn))
        P=np.asarray([points[i] for i in local],np.float64);cent=P.mean(0);u=P[0]-cent;un=float(np.linalg.norm(u))
        if un<=1e-16: raise ValueError(("coincident polygon vertices",int(ti),local))
        u/=un;n=grad/gn;v=np.cross(n,u);angles=np.arctan2((P-cent)@v,(P-cent)@u);order=np.argsort(angles,kind="stable")
        poly=[local[int(j)] for j in order]
        if len(poly)==3: tris=[tuple(poly)]
        elif len(poly)==4:
            opts=[[(poly[0],poly[1],poly[2]),(poly[0],poly[2],poly[3])],[(poly[0],poly[1],poly[3]),(poly[1],poly[2],poly[3])]]
            scores=[]
            diagonals=((poly[0],poly[2]),(poly[1],poly[3]))
            for oi,op in enumerate(opts):
                areas=[area2(points,q) for q in op]
                da,db=diagonals[oi]
                diag=float(np.linalg.norm(np.asarray(points[da],np.float64)-np.asarray(points[db],np.float64)))
                scores.append((min(areas),-diag,-oi))
            tris=opts[max(range(2),key=lambda i:scores[i])]
        else: raise ValueError(("unexpected polygon cardinality",int(ti),len(poly)))
        exact_face=len(poly)==3 and all(point_kind[i]==1 for i in poly)
        for tri0 in tris:
            tri=tuple(tri0);a,b,c=(np.asarray(points[int(i)],np.float64) for i in tri);nd=float(np.dot(np.cross(b-a,c-a),grad))
            if abs(nd)<=area2_floor*gn: raise ValueError(("isosurface triangle orientation or area degenerate",int(ti),tri,nd))
            if nd<0.0: tri=(tri[0],tri[2],tri[1])
            add_or_cancel(tri,ti,exact_face)

    records=[r for i,r in enumerate(face_records) if i not in cancelled and not r["cancelled"]]
    if not records and not allow_empty: raise ValueError(f"empty level {level}")
    used=sorted({int(n) for r in records for n in r["tri"]});remap={old:new for new,old in enumerate(used)}
    F=np.asarray([[remap[int(n)] for n in r["tri"]] for r in records],np.int64).reshape(-1,3)
    P=np.asarray([points[i] for i in used],np.float32).reshape(-1,3)
    A2=np.asarray([area2(P,f) for f in F],np.float64)
    if len(A2) and np.any(A2<=area2_floor):
        wi=int(np.flatnonzero(A2<=area2_floor)[0]);raise ValueError(("F32 degenerate exact-level faces",int(np.sum(A2<=area2_floor)),"first",F[wi].tolist(),"area2",float(A2[wi])))
    vol=float(sum(np.dot(P[a].astype(np.float64),np.cross(P[b].astype(np.float64),P[c].astype(np.float64))) for a,b,c in F)/6.0)
    return {"points":P,"faces":F,"parent_edge":np.asarray([parent_edge[i] for i in used],np.int64).reshape(-1,2),
      "edge_t":np.asarray([edge_t[i] for i in used],np.float64),"parent_node_id":np.asarray([parent_node[i] for i in used],np.int64),
      "point_kind":np.asarray([point_kind[i] for i in used],np.int8),"parent_tet":np.asarray([point_parent_tet[i] for i in used],np.int64),
      "parent_face_tet":np.asarray([r["tet"] for r in records],np.int64),"cancelled_internal_exact_faces":int(cancelled_exact),
      "strict_sublevel_rule":"negative node required; exact-level vertices key by global node ID; strict crossings key by sorted global edge IDs; opposite duplicate exact faces cancel; all-level tet rejects",
      "edge_hist":edge_hist(F),"min_double_area2_m2":float(A2.min()) if len(A2) else None,"signed_volume_m3":vol}
