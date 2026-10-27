import json, numpy as np
from pathlib import Path
from scipy.spatial import Delaunay
from collections import defaultdict
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
R=P/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided'
O=P/'partial-midpoint-continuation-root-001/bank74-cdt-root-001'
g=json.loads((R/'segment-graph-report.json').read_text())
w=json.loads((P/'per-bank-workers-20261027/b74-104/baseline-witnesses.json').read_text())
b=json.loads((P/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json').read_text())
z=np.load('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz',allow_pickle=False)
F=z['source_O_faces']; old=list(map(int,w['skin_child_faces'])); loop=[int(x['node']) for x in b['fixed_boundary_vertices']]
nb={}
for fr in old:
 for n,bc in zip(F[fr],z['refinement_face_parent_barycentric'][fr]):nb[int(n)]=np.asarray(bc,float)
outer=np.array([nb[n][1:] for n in loop]); inner=np.asarray(g['intersection_points_barycentric'],float)[:,1:]; p=np.vstack((outer,inner)); n=len(outer)
outer_edges=[tuple(sorted((i,(i+1)%n))) for i in range(n)]
inner_edges=[tuple(sorted((n+int(a),n+int(b)))) for a,b in g['segments_point_indices']]
constraints=outer_edges+inner_edges

def ori(a,b,c):
 x,y=p[b]-p[a],p[c]-p[a]; return float(x[0]*y[1]-x[1]*y[0])
def cross(e,f):
 a,b=e;c,d=f
 if len({a,b,c,d})<4:return False
 o1=ori(a,b,c);o2=ori(a,b,d);o3=ori(c,d,a);o4=ori(c,d,b)
 return o1*o2 < -1e-28 and o3*o4 < -1e-28

def make_tri(a,b,c):
 o=ori(a,b,c)
 if abs(o)<1e-15:raise RuntimeError('degenerate triangle '+str((a,b,c,o)))
 return (a,b,c) if o>0 else (a,c,b)
def adj(ts):
 m=defaultdict(list)
 for i,t in enumerate(ts):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):m[tuple(sorted((a,b)))].append(i)
 return m
T=[make_tri(*map(int,t)) for t in Delaunay(p).simplices]
original_n=len(T); flips=[]; protected=set(); fail=None
for target in constraints:
 for it in range(1000):
  edges=adj(T)
  if target in edges:protected.add(target);break
  crossing=[e for e in edges if cross(e,target)]
  switched=False
  for e in sorted(crossing):
   if e in protected or len(edges[e])!=2:continue
   i,j=edges[e]; c=next(x for x in T[i] if x not in e); d=next(x for x in T[j] if x not in e); new=tuple(sorted((c,d)))
   if new in edges or not cross(e,new):continue
   if cross(new,target) or any(cross(new,k) for k in protected):continue
   a,b=e
   try: t1=make_tri(c,d,a);t2=make_tri(d,c,b)
   except RuntimeError: continue
   oldarea=ori(*T[i])+ori(*T[j]);newarea=ori(*t1)+ori(*t2)
   if abs(oldarea-newarea)>1e-10:continue
   T[i]=t1;T[j]=t2;flips.append({'target':target,'old':e,'new':new});switched=True;break
  if not switched:
   fail={'target':target,'crossing_edges':crossing,'iteration':it,'reason':'no strictly progress-making convex flip'};break
 if fail: break
else: pass
edges=adj(T)
# Trace the degree-2 contour, preserving its exact ordered topology.
link=defaultdict(list)
for a,b in inner_edges:link[a].append(b);link[b].append(a)
cycle=[n];prev=-1;cur=n
while True:
 nxt=next(x for x in link[cur] if x!=prev)
 if nxt==cycle[0]:break
 if nxt in cycle: raise RuntimeError('unexpected repeated contour node')
 cycle.append(nxt);prev,cur=cur,nxt
if len(cycle)!=len(inner): raise RuntimeError('not one closed contour')
def area(poly):return abs(sum(p[poly[i],0]*p[poly[(i+1)%len(poly)],1]-p[poly[(i+1)%len(poly)],0]*p[poly[i],1] for i in range(len(poly)))/2)
def inside(q,poly):
 r=False
 for i in range(len(poly)):
  a=p[poly[i]];b=p[poly[(i+1)%len(poly)]]
  if ((a[1]>q[1]) != (b[1]>q[1])) and q[0] < (b[0]-a[0])*(q[1]-a[1])/(b[1]-a[1])+a[0]:r=not r
 return r
tri_area=np.array([ori(*t)/2 for t in T]); cap=np.array([inside(p[list(t)].mean(axis=0),cycle) for t in T]);outerarea=area(list(range(n)));innerarea=area(cycle)
all_edges=list(edges);crossings=[]
for i,e in enumerate(all_edges):
 for f in all_edges[i+1:]:
  if cross(e,f):crossings.append((e,f))
missing=[e for e in constraints if e not in edges]
unused=sorted(set(range(len(p)))-{v for t in T for v in t})
invalid_incidence=[(e,len(ts)) for e,ts in edges.items() if len(ts) not in (1,2)]
t_junctions=[]
for e in edges:
 a,b=e;vec=p[b]-p[a];den=float(vec@vec)
 for j,q in enumerate(p):
  if j in e:continue
  u=float((q-p[a])@vec/den); dist=float(np.linalg.norm(q-(p[a]+u*vec)))
  if 1e-12<u<1-1e-12 and dist<1e-12:t_junctions.append((e,j,u,dist))
invalid_bary=[i for i,q in enumerate(p) if q[0]<-1e-12 or q[1]<-1e-12 or q[0]+q[1]>1+1e-12]
wrong_boundary=[(e,len(edges.get(e,[]))) for e in outer_edges if len(edges.get(e,[]))!=1]
wrong_contour=[(e,len(edges.get(e,[]))) for e in inner_edges if len(edges.get(e,[]))!=2]
report={'method':'scipy Delaunay with deterministic convex edge flips and exact protected constraints','planar_only':True,'candidate_emitted':False,'failure':fail,'node_count':len(p),'triangle_count':len(T),'initial_triangle_count':original_n,'flips':flips,'missing_constraints':missing,'unused_nodes':unused,'invalid_edge_incidence':invalid_incidence,'t_junctions':t_junctions,'invalid_barycentric_nodes':invalid_bary,'wrong_boundary_edges':wrong_boundary,'wrong_contour_edges':wrong_contour,'nonadjacent_edge_crossings':crossings,'outer_area':outerarea,'contour_area':innerarea,'triangle_area':float(tri_area.sum()),'cap_area':float(tri_area[cap].sum()),'annulus_area':float(tri_area[~cap].sum()),'minimum_triangle_area':float(tri_area.min())}
report['pass']=bool(fail is None and not missing and not wrong_boundary and not wrong_contour and not crossings and not unused and not invalid_incidence and not t_junctions and not invalid_bary and abs(tri_area.sum()-outerarea)<1e-10 and abs(tri_area[cap].sum()-innerarea)<1e-10 and min(tri_area)>0)
(O/'planar-report.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(O/'planar-triangulation.npz',points_barycentric=np.column_stack((1-p.sum(axis=1),p)),triangles=np.array(T,dtype=np.int32),cap=cap,outer_nodes=np.array(loop),contour_edges=np.array(inner_edges,dtype=np.int32),outer_edges=np.array(outer_edges,dtype=np.int32))
print(json.dumps({k:v for k,v in report.items() if k!='flips'},indent=2));print('flip_count',len(flips))
