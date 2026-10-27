"""Fixed-coordinate vertex-star retriangulation proposals; all exact gates are downstream."""
from functools import lru_cache
import numpy as np

def ring_for_star(faces, rows, center):
    old=np.asarray(faces)[list(rows)]
    nxt={};incoming={}
    if len(old)<4:raise ValueError("star too small")
    for tri in old:
        at=np.flatnonzero(tri==center)
        if len(at)!=1:raise ValueError("not a center star")
        j=int(at[0]);a=int(tri[(j+1)%3]);b=int(tri[(j+2)%3])
        if a==b or a in nxt or b in incoming:raise ValueError("branched star")
        nxt[a]=b;incoming[b]=a
    if set(nxt)!=set(incoming) or center in nxt:raise ValueError("open star")
    ring=[min(nxt)]
    for _ in range(len(old)-1):
        v=nxt[ring[-1]]
        if v in ring:raise ValueError("multiple rings")
        ring.append(v)
    if nxt[ring[-1]]!=ring[0] or len(ring)!=len(old):raise ValueError("not a closed disk")
    return tuple(ring)

def orient(a,b,c):
    return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))

def segments_cross(a,b,c,d,eps):
    v=[orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)]
    return ((v[0]>eps and v[1]<-eps) or (v[0]<-eps and v[1]>eps)) and ((v[2]>eps and v[3]<-eps) or (v[2]<-eps and v[3]>eps))

def chart(points,ring,center):
    xyz=np.asarray(points,dtype=np.float64)[list(ring)]
    origin=xyz.mean(axis=0)
    _u,s,v=np.linalg.svd(xyz-origin,full_matrices=False)
    if len(s)<2 or s[1]<=1e-12:raise ValueError("degenerate chart")
    axes=v[:2].copy()
    p=(xyz-origin)@axes.T
    area=sum(orient(np.zeros(2),p[i],p[(i+1)%len(p)]) for i in range(len(p)))
    if area<0:axes[1]*=-1;p=(xyz-origin)@axes.T
    span=float(np.max(np.ptp(p,axis=0)))
    eps=max(span*span*1e-12,1e-16)
    if abs(area)<=eps:raise ValueError("zero chart area")
    n=len(p)
    for i in range(n):
        for j in range(i+1,n):
            if j==i or (i+1)%n==j or (j+1)%n==i:continue
            if segments_cross(p[i],p[(i+1)%n],p[j],p[(j+1)%n],eps):raise ValueError("non-simple chart")
    return p,(np.asarray(points,dtype=np.float64)[center]-origin)@axes.T,eps

def inside_polygon(pt,p,eps):
    winding=False
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        if abs(orient(a,b,pt))<=eps and np.all(pt>=np.minimum(a,b)-eps) and np.all(pt<=np.maximum(a,b)+eps):return True
        if (a[1]>pt[1])!=(b[1]>pt[1]):
            if pt[0] < (b[0]-a[0])*(pt[1]-a[1])/(b[1]-a[1])+a[0]:winding=not winding
    return winding

def polygon_triangulations(p,eps):
    n=len(p)
    def diagonal(i,j):
        if abs(i-j)==1 or abs(i-j)==n-1:return True
        if not inside_polygon((p[i]+p[j])*.5,p,eps):return False
        for k in range(n):
            l=(k+1)%n
            if i in (k,l) or j in (k,l):continue
            if segments_cross(p[i],p[j],p[k],p[l],eps):return False
        return True
    valid={(i,j):diagonal(i,j) for i in range(n) for j in range(i+1,n)}
    @lru_cache(None)
    def rec(i,j):
        if j<=i+1:return ((),)
        out=[]
        for k in range(i+1,j):
            if orient(p[i],p[k],p[j])<=eps:continue
            if not valid[min(i,k),max(i,k)] or not valid[min(k,j),max(k,j)]:continue
            if not inside_polygon((p[i]+p[k]+p[j])/3,p,eps):continue
            for left in rec(i,k):
                for right in rec(k,j):out.append(left+right+((i,k,j),))
        return tuple(out)
    return rec(0,n-1)

def star_candidates(points,reference,faces,rows,center,max_degree=9):
    ring=ring_for_star(faces,rows,center);k=len(ring)
    if k>max_degree:return []
    out={};oldset={tuple(sorted(map(int,t))) for t in np.asarray(faces)[list(rows)]}
    for geometry in (points,reference):
        try:p,c,eps=chart(geometry,ring,center)
        except ValueError:continue
        for tr in polygon_triangulations(p,eps):
            for j,t in enumerate(tr):
                a,b,d=(p[i] for i in t)
                if min(orient(a,b,c),orient(b,d,c),orient(d,a,c))<=eps:continue
                nf=[tuple(ring[i] for i in tri) for h,tri in enumerate(tr) if h!=j]
                aa,bb,dd=(ring[i] for i in t)
                nf.extend(((center,aa,bb),(center,bb,dd),(center,dd,aa)))
                key=tuple(sorted(tuple(sorted(z)) for z in nf))
                if set(key)==oldset:continue
                if len(nf)!=k or len(set(key))!=k:continue
                out[key]=np.asarray(nf,dtype=np.int64)
    return [out[k] for k in sorted(out)]

def external_edge_conflict(faces,rows,new_faces,edge_map):
    rset=set(map(int,rows))
    old_edges=set()
    for t in np.asarray(faces)[list(rows)]:
        for a,b in zip(t,np.roll(t,-1)):old_edges.add(tuple(sorted((int(a),int(b)))))
    for t in new_faces:
        for a,b in zip(t,np.roll(t,-1)):
            e=tuple(sorted((int(a),int(b))))
            if e not in old_edges and any(int(v[0]) not in rset for v in edge_map.get(e,())):return True
    return False

