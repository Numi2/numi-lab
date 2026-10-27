#!/usr/bin/env python3
"""Exact local bone screen for the unchanged 653 topology-only patch boundary."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict, deque
import importlib.util, sys, json, hashlib, time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz')
A=C.parent/'audit-159/report.json'
PATCH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/complement-filled-patch-face-rows.npy')
TOPO=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/complement-fill-check.json')
OUT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/fixed-boundary-clearance.json')
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def ff(x):return Fraction.from_float(float(x))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Fraction(0))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def mul(s,a):return tuple(s*x for x in a)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def in_tri3(q,tri):
 a,b,c=tri;v0=sub(b,a);v1=sub(c,a);v2=sub(q,a)
 d00=dot(v0,v0);d01=dot(v0,v1);d11=dot(v1,v1);d20=dot(v2,v0);d21=dot(v2,v1);den=d00*d11-d01*d01
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
 if n==(0,0,0):return {'hit':False,'kind':'degenerate_triangle','point':None}
 if d0==0 and d1==0:
  axis=max(range(3),key=lambda i:abs(n[i]));keep=[i for i in range(3) if i!=axis]
  p0_2=(p0[keep[0]],p0[keep[1]]);p1_2=(p1[keep[0]],p1[keep[1]])
  t2=[(v[keep[0]],v[keep[1]]) for v in tri]
  hit=in_tri2(p0_2,t2) or in_tri2(p1_2,t2) or any(seg_intersect2(p0_2,p1_2,t2[i],t2[(i+1)%3]) for i in range(3))
  return {'hit':bool(hit),'kind':'coplanar_segment','point':None}
 if d0!=0 and d1!=0 and (d0>0)==(d1>0):return {'hit':False,'kind':'same_plane_side','point':None}
 den=d0-d1
 if den==0:return {'hit':False,'kind':'parallel','point':None}
 t=d0/den
 if t<0 or t>1:return {'hit':False,'kind':'outside_segment','point':None}
 q=add(p0,mul(t,sub(p1,p0)));hit=in_tri3(q,tri)
 return {'hit':hit,'kind':'point','point':q if hit else None}
def point_seg_dist(p,a,b):
 v=b-a;d=float(v@v);t=0.0 if d==0 else min(1.0,max(0.0,float((p-a)@v/d)));return float(np.linalg.norm(p-(a+t*v)))
def point_tri_dist(p,t):
 a,b,c=t;ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=float(n@n)
 if nn<1e-28:return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
 q=p-(n@(p-a))/nn*n;v0=ab;v1=ac;v2=q-a
 d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 if den==0:return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
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
 a=tri[:,0]-p;b=tri[:,1]-p;c=tri[:,2]-p
 la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c))
 den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.sum(2*np.arctan2(num,den))/(4*np.pi))
started=time.monotonic()
with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float64);F=Z['source_O_faces'].astype(np.int64)
audit_data=json.loads(A.read_text());assert audit_data['complete'] and audit_data['candidate_sha256']==sha(C)
bank=next(x for x in audit_data['banks'] if (int(x['body']),int(x['stable_bone_id']))==(109,114))
pairs=[tuple(map(int,x)) for x in bank['result']['triangle_pairs']]
crossing_faces=sorted({sf for sf,bf in pairs})
topo=json.loads(TOPO.read_text());assert topo['topology_only_fill']['all_crossings_still_inside']
patch=np.load(PATCH,allow_pickle=False).astype(np.int64);patch_set=set(map(int,patch))
if len(patch_set)!=len(patch):raise RuntimeError('patch face rows are not unique')
if not patch_set<=set(range(len(F))):raise RuntimeError('patch face row outside candidate faces')
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
boundary=[]
for e,fs in edgefaces.items():
 inside=[f for f in fs if f in patch_set];outside=[f for f in fs if f not in patch_set]
 if inside and outside:boundary.append((e,inside,outside))
 elif inside and len(fs)==1:boundary.append((e,inside,[]))
bg=defaultdict(list)
for e,ins,outs in boundary:bg[e[0]].append(e[1]);bg[e[1]].append(e[0])
unseen=set(bg);loops=[]
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];nodes=[]
 while q:
  v=q.pop();nodes.append(v)
  for w in bg[v]:
   if w in unseen:unseen.remove(w);q.append(w)
 ns=set(nodes);es=sorted(e for e,_,_ in boundary if e[0] in ns and e[1] in ns)
 loops.append({'nodes':sorted(ns),'edges':es,'simple_cycle':bool(len(es)==len(ns) and all(len(bg[v])==2 for v in ns))})
patch_adj=defaultdict(set);patch_faces=sorted(patch_set)
for e,fs in edgefaces.items():
 inside=[f for f in fs if f in patch_set]
 if len(inside)==2:patch_adj[inside[0]].add(inside[1]);patch_adj[inside[1]].add(inside[0])
unseen=set(patch_faces);patch_components=[]
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];c=[]
 while q:
  f=q.pop();c.append(f)
  for n in patch_adj[f]&unseen:unseen.remove(n);q.append(n)
 patch_components.append(sorted(c))
# Persist two independent face-row supports without editing the candidate coordinates or faces.
component_dir=OUT.parent/'patch-components';component_dir.mkdir(exist_ok=True)
component_files=[]
for ci,faces in enumerate(patch_components):
 cp=component_dir/f'component-{ci:02d}-face-rows.npy'
 np.save(cp,np.asarray(faces,np.int64))
 component_files.append({'component':ci,'face_rows_path':str(cp),'face_rows_sha256':sha(cp),'face_count':len(faces)})
patch_comp_of={f:ci for ci,fs in enumerate(patch_components) for f in fs}
boundary_inside={e:ins for e,ins,outs in boundary}
for li,L in enumerate(loops):
 cids=sorted({patch_comp_of[f] for e in L['edges'] for f in boundary_inside.get(tuple(e),[])})
 L['patch_component_ids']=cids
audit_mod=mod('bank109_clearance_audit',AUDIT);helper=mod('bank109_clearance_helper',audit_mod.BANKHELPER)
bankdir=audit_mod.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=helper.read_bank_inputs(bankdir,audit_mod.E)
mask=(ids[:,0]==109)&(ids[:,1]==114);BT=np.asarray(T[mask],dtype=np.float64);BID=np.asarray(ids[mask,3],dtype=np.int64)
local=(P-pos[109])@M[109].T-off[109]
BT_exact=[tuple(tuple(ff(x) for x in vertex) for vertex in tri) for tri in BT]
boundary_nodes=sorted(bg);node_records=[];inside_nodes=[]
for n in boundary_nodes:
 w=winding(local[n],BT);distances=[point_tri_dist(local[n],tri) for tri in BT];j=int(np.argmin(distances))
 rec={'skin_node_row':int(n),'winding':w,'inside_by_abs_winding_gt_half':bool(abs(w)>0.5),'nearest_target_bone_face_row':int(BID[j]),'nearest_distance_mm':float(distances[j]*1000)}
 node_records.append(rec)
 if rec['inside_by_abs_winding_gt_half']:inside_nodes.append(rec)
edge_records=[];exact_hits=[];edge_hit_by_loop=[]
for li,L in enumerate(loops):
 loop_hits=[]
 for e in L['edges']:
  u,v=map(int,e);p0=tuple(ff(x) for x in local[u]);p1=tuple(ff(x) for x in local[v])
  best=np.inf;best_f=None;hit_witnesses=[]
  for j,tri in enumerate(BT):
   d=seg_tri_dist(local[u],local[v],tri)
   if d<best:best=d;best_f=int(BID[j])
   hit=exact_segment_triangle(p0,p1,BT_exact[j])
   if hit['hit']:
    w={'target_bone_face_row':int(BID[j]),'kind':hit['kind'],'point_local_m':[str(x) for x in hit['point']] if hit['point'] is not None else None}
    hit_witnesses.append(w);exact_hits.append({'loop':li,'skin_edge_node_rows':[u,v],**w})
  rec={'loop':li,'skin_edge_node_rows':[u,v],'inside_patch_faces':sorted(map(int,next((ins for ee,ins,oo in boundary if ee==e),[]))),'outside_patch_faces':sorted(map(int,next((oo for ee,ins,oo in boundary if ee==e),[]))),'exact_target_bone_intersections':hit_witnesses,'nearest_target_bone_face_row':best_f,'minimum_segment_to_triangle_distance_mm':float(best*1000)}
  edge_records.append(rec)
  if hit_witnesses:loop_hits.extend(hit_witnesses)
 edge_hit_by_loop.append({'loop':li,'node_count':len(L['nodes']),'edge_count':len(L['edges']),'simple_cycle':L['simple_cycle'],'exact_intersection_count':sum(len(r['exact_target_bone_intersections']) for r in edge_records if r['loop']==li),'minimum_edge_clearance_record':min((r for r in edge_records if r['loop']==li),key=lambda x:x['minimum_segment_to_triangle_distance_mm']) if L['edges'] else None})
# Check stored audit crossing pairs from all banks that touch this boundary support.
boundary_node_set=set(boundary_nodes);boundary_edge_set={tuple(map(int,e)) for e,_,_ in boundary};boundary_incident_faces={f for v in boundary_nodes for f in nodefaces[v]}
inside_boundary_faces={f for _,ins,_ in boundary for f in ins};outside_boundary_faces={f for _,_,outs in boundary for f in outs}
neighbor_hits=[]
for br in audit_data['banks']:
 if not br.get('result'):continue
 bkey=[int(br['body']),int(br['stable_bone_id'])]
 for sf,bf in br['result']['triangle_pairs']:
  sf=int(sf);bf=int(bf);fv=set(map(int,F[sf]));shared_nodes=sorted(fv&boundary_node_set);shared_edges=[]
  tri=F[sf]
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   e=tuple(sorted((int(a),int(b))))
   if e in boundary_edge_set:shared_edges.append(list(e))
  if sf in boundary_incident_faces or shared_nodes or shared_edges:
   rel=[]
   if sf in inside_boundary_faces:rel.append('inside_patch_boundary_face')
   if sf in outside_boundary_faces:rel.append('outside_boundary_adjacent_face')
   if sf in boundary_incident_faces:rel.append('boundary_node_incident_face')
   if shared_edges:rel.append('shares_boundary_edge')
   touched_loops=[li for li,L in enumerate(loops) if (set(L['nodes']) & set(shared_nodes)) or any(tuple(e) in {tuple(x) for x in L['edges']} for e in shared_edges)]
   neighbor_hits.append({'typed_bank':bkey,'skin_face_row':sf,'bone_face_row':bf,'relation':rel,'boundary_nodes_shared':shared_nodes,'boundary_edges_shared':shared_edges,'boundary_loop_ids':touched_loops})
component_boundary=[];component_ancestry=[]
ancptr=Z['face_source_ancestry_indptr'];anc=Z['face_source_ancestry_face_rows']
detptr=Z['face_fine_detail_ancestry_indptr'];det=Z['face_fine_detail_ancestry_rows']
subptr=Z['face_fine_subface_ancestry_indptr'];sub=Z['face_fine_subface_ancestry_rows']
parent=Z['refinement_face_parent_face_row']
for ci,faces in enumerate(patch_components):
 lis=[li for li,L in enumerate(loops) if ci in L['patch_component_ids']]
 component_boundary.append({'component':ci,'face_count':len(faces),'boundary_loop_ids':lis,'boundary_loop_count':len(lis),'single_simple_closed_boundary':bool(len(lis)==1 and loops[lis[0]]['simple_cycle'])})
 component_ancestry.append({'component':ci,'source_ancestry_face_rows':sorted({int(v) for f in faces for v in anc[ancptr[f]:ancptr[f+1]]}),'fine_detail_ancestry_rows':sorted({int(v) for f in faces for v in det[detptr[f]:detptr[f+1]]}),'fine_subface_ancestry_rows':sorted({int(v) for f in faces for v in sub[subptr[f]:subptr[f+1]]}),'immediate_parent_face_rows':sorted(set(map(int,parent[faces])))})
single_closed_per_patch=bool(len(patch_components)>0 and all(x['single_simple_closed_boundary'] for x in component_boundary))
loop_clearance=[]
for li,L in enumerate(loops):
 loop_nodes=set(L['nodes'])
 exact_count=sum(x['loop']==li for x in exact_hits)
 inside_count=sum(x['skin_node_row'] in loop_nodes for x in inside_nodes)
 neighbor_count=sum(li in x['boundary_loop_ids'] for x in neighbor_hits)
 loop_nodes_records=[x for x in node_records if x['skin_node_row'] in loop_nodes]
 loop_edges_records=[x for x in edge_records if x['loop']==li]
 loop_clearance.append({'loop':li,'patch_component_ids':L['patch_component_ids'],'simple_cycle':L['simple_cycle'],'exact_target_bone_intersection_count':exact_count,'inside_boundary_node_count':inside_count,'stored_crossing_touch_count':neighbor_count,'minimum_boundary_node_distance_mm':min((x['nearest_distance_mm'] for x in loop_nodes_records),default=None),'minimum_boundary_node_record':min(loop_nodes_records,key=lambda x:x['nearest_distance_mm']) if loop_nodes_records else None,'minimum_boundary_edge_distance_mm':min((x['minimum_segment_to_triangle_distance_mm'] for x in loop_edges_records),default=None),'minimum_boundary_edge_record':min(loop_edges_records,key=lambda x:x['minimum_segment_to_triangle_distance_mm']) if loop_edges_records else None,'target_bone_clear':bool(exact_count==0 and inside_count==0),'stored_crossings_clear':neighbor_count==0,'fixed_boundary_clear':bool(L['simple_cycle'] and exact_count==0 and inside_count==0 and neighbor_count==0)})
result={'schema':'numi.human.bank109_114.fixed_boundary_clearance.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'topology_report_path':str(TOPO),'topology_report_sha256':sha(TOPO),'patch_face_rows_path':str(PATCH),'patch_face_rows_sha256':sha(PATCH),'typed_bank':[109,114],'exact_cross_pair_count':len(pairs),'crossing_skin_face_count':len(crossing_faces),'all_crossing_faces_in_patch':set(crossing_faces)<=patch_set,'patch_face_count':len(patch_set),'patch_edge_connected_component_count':len(patch_components),'patch_component_face_counts':sorted(len(c) for c in patch_components),'patch_component_face_row_files':component_files,'patch_component_boundaries':component_boundary,'patch_component_ancestry':component_ancestry,'boundary_loop_count':len(loops),'boundary_loop_node_counts':sorted(len(L['nodes']) for L in loops),'boundary_loops':[{'patch_component_ids':L['patch_component_ids'],'node_count':len(L['nodes']),'edge_count':len(L['edges']),'simple_cycle':L['simple_cycle'],'skin_node_rows':L['nodes']} for L in loops],'single_closed_fixed_boundary_per_patch':single_closed_per_patch,'per_loop_clearance':loop_clearance,'boundary_nodes_inside_target_bone':len(inside_nodes),'inside_boundary_node_rows':[x['skin_node_row'] for x in inside_nodes],'minimum_boundary_node_distance_mm':min((x['nearest_distance_mm'] for x in node_records),default=None),'minimum_boundary_node_record':min(node_records,key=lambda x:x['nearest_distance_mm']) if node_records else None,'exact_boundary_edge_target_bone_intersections':len(exact_hits),'exact_boundary_edge_intersection_witnesses':exact_hits,'minimum_boundary_edge_to_bone_distance_mm':min((x['minimum_segment_to_triangle_distance_mm'] for x in edge_records),default=None),'minimum_boundary_edge_record':min(edge_records,key=lambda x:x['minimum_segment_to_triangle_distance_mm']) if edge_records else None,'stored_neighbor_bank_crossing_pairs_touching_boundary':len(neighbor_hits),'stored_neighbor_bank_boundary_crossing_witnesses':neighbor_hits,'boundary_node_records':node_records,'boundary_edge_records':edge_records,'bone_triangle_count':len(BT),'bone_triangle_identity':'Direct exact-emitted Float32 triangle values promoted to Float64 in pinned owner-local coordinates; no node-table join or coordinate binning.','edge_predicate':'Exact rational segment-versus-triangle tests in owner-local coordinates derived from stored Float32 skin/bone inputs.','distance_predicate':'Float64 Euclidean closest distance diagnostic; no clearance margin threshold imposed.','coordinates_moved':False,'candidate_modified':False,'single_boundary_exact_clearance_per_patch':bool(single_closed_per_patch and all(x['fixed_boundary_clear'] for x in loop_clearance)),'per_loop_exact_clearance_conclusion':bool(all(x['fixed_boundary_clear'] for x in loop_clearance)),'elapsed_seconds':time.monotonic()-started,'limitations':['This only checks boundaries of the topology-only patch face set. The crossing support separates into two patches; each is reported with its own simple closed boundary and ancestry. No candidate edit, full/self/all-bank rerun, or physical/native admission.']}
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('boundary_node_records','boundary_edge_records','stored_neighbor_bank_boundary_crossing_witnesses')},indent=2,allow_nan=False))

