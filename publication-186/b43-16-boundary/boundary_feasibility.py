#!/usr/bin/env python3
"""Local boundary feasibility receipt for topology-only 43/16 patch support."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import hashlib,importlib.util,sys,json,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
PATCH=S/'per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy'
TOPO=S/'per-bank-workers-20261027/b43-16/attempt-001/complement-fill-check.json'
OUT=S/'per-bank-workers-20261027/b43-16/attempt-002/boundary-feasibility.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def ff(x):return Fraction.from_float(float(x))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Fraction(0))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def mul(s,a):return tuple(s*x for x in a)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def in_tri3(q,tri):
 a,b,c=tri;v0=sub(b,a);v1=sub(c,a);v2=sub(q,a);d00=dot(v0,v0);d01=dot(v0,v1);d11=dot(v1,v1);d20=dot(v2,v0);d21=dot(v2,v1);den=d00*d11-d01*d01
 if den==0:return False
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 return u>=0 and v>=0 and w>=0
def orient2(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def on_seg2(p,a,b):return orient2(a,b,p)==0 and min(a[0],b[0])<=p[0]<=max(a[0],b[0]) and min(a[1],b[1])<=p[1]<=max(a[1],b[1])
def seg_intersect2(a,b,c,d):
 o1=orient2(a,b,c);o2=orient2(a,b,d);o3=orient2(c,d,a);o4=orient2(c,d,b)
 if o1==0 and on_seg2(c,a,b):return True
 if o2==0 and on_seg2(d,a,b):return True
 if o3==0 and on_seg2(a,c,d):return True
 if o4==0 and on_seg2(b,c,d):return True
 return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)
def in_tri2(p,t):
 o=[orient2(t[0],t[1],p),orient2(t[1],t[2],p),orient2(t[2],t[0],p)]
 return all(x>=0 for x in o) or all(x<=0 for x in o)
def exact_segment_triangle(p0,p1,tri):
 a,b,c=tri;n=cross(sub(b,a),sub(c,a));d0=dot(n,sub(p0,a));d1=dot(n,sub(p1,a))
 if n==(0,0,0):return {'hit':False,'kind':'degenerate_triangle'}
 if d0==0 and d1==0:
  axis=max(range(3),key=lambda i:abs(n[i]));keep=[i for i in range(3) if i!=axis]
  pp0=(p0[keep[0]],p0[keep[1]]);pp1=(p1[keep[0]],p1[keep[1]]);tt=[(v[keep[0]],v[keep[1]]) for v in tri]
  hit=in_tri2(pp0,tt) or in_tri2(pp1,tt) or any(seg_intersect2(pp0,pp1,tt[i],tt[(i+1)%3]) for i in range(3))
  return {'hit':bool(hit),'kind':'coplanar_segment','point':None}
 if d0!=0 and d1!=0 and (d0>0)==(d1>0):return {'hit':False,'kind':'same_plane_side'}
 den=d0-d1
 if den==0:return {'hit':False,'kind':'parallel'}
 t=d0/den
 if t<0 or t>1:return {'hit':False,'kind':'outside_segment'}
 q=add(p0,mul(t,sub(p1,p0)))
 return {'hit':in_tri3(q,tri),'kind':'point','point':q if in_tri3(q,tri) else None}
def point_seg_dist(p,a,b):
 v=b-a;d=float(v@v);t=0.0 if d==0 else min(1.0,max(0.0,float((p-a)@v/d)));return float(np.linalg.norm(p-(a+t*v)))
def point_tri_dist(p,t):
 a,b,c=t;ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=float(n@n)
 if nn<1e-28:return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
 q=p-(n@(p-a))/nn*n;v0=ab;v1=ac;v2=q-a;d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 if min(u,v,w)>=-1e-12:return float(np.linalg.norm(p-q))
 return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
def seg_seg_dist(p1,q1,p2,q2):
 u=q1-p1;v=q2-p2;w=p1-p2;a=u@u;b=u@v;c=v@v;d=u@w;e=v@w;D=a*c-b*b;SMALL=1e-30
 sN=D;tN=D;sD=D;tD=D
 if D<SMALL:sN=0;sD=1;tN=e;tD=c if c>SMALL else 1
 else:
  sN=b*e-c*d;tN=a*e-b*d
  if sN<0:sN=0;tN=e;tD=c
  elif sN>sD:sN=sD;tN=e+b;tD=c
 if tN<0:
  tN=0
  if -d<0:sN=0
  elif -d>a:sN=sD
  else:sN=-d;sD=a
 elif tN>tD:
  tN=tD
  if -d+b<0:sN=0
  elif -d+b>a:sN=sD
  else:sN=-d+b;sD=a
 sc=0 if abs(sN)<SMALL else sN/sD;tc=0 if abs(tN)<SMALL else tN/tD
 return float(np.linalg.norm(w+sc*u-tc*v))
def seg_tri_dist(p0,p1,t):
 d=min(point_tri_dist(p0,t),point_tri_dist(p1,t))
 for v in t:d=min(d,point_seg_dist(v,p0,p1))
 for i in range(3):d=min(d,seg_seg_dist(p0,p1,t[i],t[(i+1)%3]))
 return d
def winding(p,tri):
 a=tri[:,0]-p;b=tri[:,1]-p;c=tri[:,2]-p;la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c));den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.sum(2*np.arctan2(num,den))/(4*np.pi))
started=time.monotonic()
assert sha(C)=='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
assert sha(A)=='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
assert sha(PATCH)=='ad2fb608a536ab8396b781baab4b01f62fc8ae1da81ad533fcb6a4c281131085'
patch=np.load(PATCH,allow_pickle=False).astype(np.int64);patch_set=set(map(int,patch))
with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float64);F=Z['source_O_faces'].astype(np.int64)
audit=mod('boundary43_audit',AUDIT);helper=mod('boundary43_helper',audit.BANKHELPER);bankdir=audit.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=helper.read_bank_inputs(bankdir,audit.E)
mask=(ids[:,0]==43)&(ids[:,1]==16);BT=np.asarray(T[mask],dtype=np.float64);BID=np.asarray(ids[mask,3],dtype=np.int64);bone_local=BT
# Candidate world F32 points in the pinned target body frame.
local=((P-pos[43])@M[43].T-off[43]);
# Recompute patch boundary from actual mesh edge incidence.
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
boundary=[];inside_boundary_faces=set();outside_boundary_faces=set()
for e,fs in edgefaces.items():
 ins=[f for f in fs if int(f) in patch_set];outs=[f for f in fs if int(f) not in patch_set]
 if ins and outs:
  boundary.append((e,ins,outs));inside_boundary_faces.update(ins);outside_boundary_faces.update(outs)
 elif ins and len(fs)==1:boundary.append((e,ins,[]));inside_boundary_faces.update(ins)
# One loop exactly, all nodes degree 2.
bg=defaultdict(list)
for e,_,_ in boundary:bg[e[0]].append(e[1]);bg[e[1]].append(e[0])
bnodes=sorted(bg);assert len(bnodes)==54 and all(len(bg[v])==2 for v in bg)
# Target bone exact triangle data promoted from the pinned Float32 table.
BTf=[tuple(tuple(ff(x) for x in vertex) for vertex in tri) for tri in BT]
# Classify every boundary point by closed-solid winding and nearest target bone triangle.
node_rows=[]
for n in bnodes:
 w=winding(local[n],bone_local);ds=[point_tri_dist(local[n],tr) for tr in bone_local];j=int(np.argmin(ds))
 node_rows.append({'skin_node_row':int(n),'winding':w,'inside_by_abs_winding_gt_half':bool(abs(w)>0.5),'nearest_target_bone_face_row':int(BID[j]),'nearest_distance_mm':float(ds[j]*1000.0)})
# Exact rational segment/triangle audit plus float64 closest distances for each fixed outer edge.
edge_rows=[];exact_edge_hits=[]
for e,ins,outs in sorted(boundary,key=lambda x:x[0]):
 u,v=map(int,e);p0=tuple(ff(x) for x in local[u]);p1=tuple(ff(x) for x in local[v]);p0f=local[u];p1f=local[v]
 best=np.inf;best_face=None;best_pair=None
 for j,tr in enumerate(bone_local):
  d=seg_tri_dist(p0f,p1f,tr)
  if d<best:best=d;best_face=int(BID[j]);best_pair=j
  hit=exact_segment_triangle(p0,p1,BTf[j])
  if hit['hit']:
   rec={'skin_edge_node_rows':[u,v],'target_bone_face_row':int(BID[j]),'intersection_kind':hit['kind'],'exact_point_local_m':[str(x) for x in hit['point']] if hit['point'] is not None else None}
   exact_edge_hits.append(rec)
 edge_rows.append({'skin_edge_node_rows':[u,v],'inside_patch_face_rows':sorted(map(int,ins)),'outside_patch_face_rows':sorted(map(int,outs)),'nearest_target_bone_face_row':best_face,'minimum_segment_to_triangle_distance_mm':float(best*1000.0)})
# Stored all-bank crossings touching any boundary node/edge/incident face; no new audit is run.
allbank_hits=[];boundary_node_set=set(bnodes);boundary_edge_set={tuple(sorted(map(int,e))) for e,_,_ in boundary};boundary_incident_faces=set(f for n in bnodes for f in nodefaces[n])
for br in json.loads(A.read_text())['banks']:
 body=int(br['body']);bone=int(br['stable_bone_id']);rr=br.get('result')
 if not rr:continue
 for sf,bf in rr['triangle_pairs']:
  sf=int(sf);bf=int(bf);fv=set(map(int,F[sf]));shared_nodes=sorted(fv&boundary_node_set)
  shared_edges=[]
  for a,b in ((F[sf,0],F[sf,1]),(F[sf,1],F[sf,2]),(F[sf,2],F[sf,0])):
   e=tuple(sorted((int(a),int(b))))
   if e in boundary_edge_set:shared_edges.append(list(e))
  if shared_nodes or shared_edges or sf in boundary_incident_faces:
   rel=[]
   if sf in inside_boundary_faces:rel.append('inside_patch_boundary_face')
   if sf in outside_boundary_faces:rel.append('outside_boundary_adjacent_face')
   if shared_nodes:rel.append('crossing_face_contains_boundary_node')
   if shared_edges:rel.append('crossing_face_shares_boundary_edge')
   allbank_hits.append({'typed_bank':[body,bone],'skin_face_row':sf,'bone_face_row':bf,'relation':rel,'boundary_nodes_shared':shared_nodes,'boundary_edges_shared':shared_edges})
# Compact status; no arbitrary clearance threshold is imposed.
inside_nodes=[r for r in node_rows if r['inside_by_abs_winding_gt_half']]
min_node=min(node_rows,key=lambda r:r['nearest_distance_mm'])
min_edge=min(edge_rows,key=lambda r:r['minimum_segment_to_triangle_distance_mm'])
result={'schema':'numi.human.bank43_16.fixed_boundary_feasibility.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'patch_face_rows_path':str(PATCH),'patch_face_rows_sha256':sha(PATCH),'topology_receipt_sha256':sha(TOPO),'typed_bank':[43,16],'patch_face_count':int(len(patch)),'crossing_exact_pair_count':249,'crossing_skin_face_count':183,'boundary_loop_count':1,'boundary_node_count':len(bnodes),'boundary_edge_count':len(boundary),'boundary_nodes_inside_target_bone':len(inside_nodes),'inside_boundary_node_rows':[r['skin_node_row'] for r in inside_nodes],'minimum_boundary_node_distance_mm':min_node['nearest_distance_mm'],'minimum_boundary_node_record':min_node,'exact_boundary_edge_target_bone_intersections':len(exact_edge_hits),'exact_boundary_edge_intersection_witnesses':exact_edge_hits,'minimum_boundary_edge_to_bone_distance_mm':min_edge['minimum_segment_to_triangle_distance_mm'],'minimum_boundary_edge_record':min_edge,'stored_neighbor_bank_crossing_pairs_touching_boundary':len(allbank_hits),'stored_neighbor_bank_boundary_crossing_witnesses':allbank_hits,'boundary_node_records':node_rows,'boundary_edge_records':edge_rows,'edge_predicate':'Exact rational segment-vs-triangle intersection in pinned target-body-local coordinates.','distance_predicate':'Float64 Euclidean closest distance to exact-emitted Float32 triangle values promoted to Float64; distance is diagnostic and no clearance margin threshold was imposed.','geometry_or_native_admission':False,'elapsed_seconds':time.monotonic()-started,'limitations':['Only the fixed boundary of the topology-only face set was checked against target bone 43/16 and stored audit crossings. No vertices moved, no complete candidate, self audit, all-bank rerun, or physical/native admission.']}
# Fail closed on any point containment, exact edge contact, or stored crossing touching boundary.
result['boundary_feasible_local']=bool(not inside_nodes and not exact_edge_hits and not allbank_hits)
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('boundary_node_records','boundary_edge_records')},indent=2,allow_nan=False))
