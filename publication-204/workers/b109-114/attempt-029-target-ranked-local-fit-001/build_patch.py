#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
from collections import defaultdict,deque
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';D=W/'b109-114/attempt-029-target-ranked-local-fit-001'
BASE=W/'combined-d071-b109-b61-001/candidate.npz'
AUD=W/'combined-d071-b109-b61-001/audit-159/report.json'
RANK=D/'target-ranking-corrected.json'
BASE_SHA='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
AMP_MM=0.05;RING=8;SEED=34547;TARGET_BONE_TRIANGLE_ROWS={111,112,116}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(BASE)!=BASE_SHA:raise RuntimeError('pinned baseline hash mismatch')
adoc=json.loads(AUD.read_text());b=next(x for x in adoc['banks'] if (int(x['body']),int(x['stable_bone_id']))==(109,114))
pairs=np.asarray(b['result']['triangle_pairs'],np.int64)
rank=json.loads(RANK.read_text());rr=next(x for x in rank['top_skin_faces'] if int(x['skin_face'])==SEED)
direction=np.asarray(rr['mean_outward_normal'],np.float64);direction/=np.linalg.norm(direction)
with np.load(BASE,allow_pickle=False) as z: p={k:z[k].copy() for k in z.files}
x=np.asarray(p['points_world_m'],np.float32);F=np.asarray(p['source_O_faces'],np.int64)
if len(x)!=21212 or len(F)!=42379:raise RuntimeError('unexpected baseline shape')
target_faces={int(fi) for fi,bi in pairs if int(bi) in TARGET_BONE_TRIANGLE_ROWS}
if SEED not in target_faces:raise RuntimeError('ranked seed no longer crosses selected bone set')
edge_faces=defaultdict(list);node_adj=[set() for _ in x]
for fi,face in enumerate(F):
 a,b,c=map(int,face)
 node_adj[a].update((b,c));node_adj[b].update((a,c));node_adj[c].update((a,b))
 for u,v in ((a,b),(b,c),(c,a)):edge_faces[tuple(sorted((u,v)))].append(fi)
face_adj=[set() for _ in F]
for fs in edge_faces.values():
 if len(fs)==2:u,v=fs;face_adj[u].add(v);face_adj[v].add(u)
cluster={SEED};q=deque([SEED])
while q:
 f=q.popleft()
 for g in face_adj[f]:
  if g in target_faces and g not in cluster:cluster.add(g);q.append(g)
seed_nodes=set(map(int,F[SEED]));target_nodes=set(map(int,F[list(sorted(cluster))].ravel()))
dist={i:0 for i in target_nodes};q=deque(target_nodes)
while q:
 u=q.popleft()
 if dist[u]>=RING:continue
 for v in node_adj[u]:
  if v not in dist:dist[v]=dist[u]+1;q.append(v)
anchors=np.asarray(p.get('typed_coarse_anchor_output_rows_018',np.empty(0)),np.int64)
anchor_set=set(map(int,anchors))
w=np.zeros(len(x),np.float64)
for node,d in dist.items():
 if node in anchor_set or d>=RING:continue
 z=1.0-float(d)/RING
 w[node]=z*z
y=x.astype(np.float64)
# Row-vector bank-local outward normal transformed to world with inverse of world-to-body.
# Bank 109 rotation is orthonormal; use the pinned matrix from exact emitted bodies.
from csv import DictReader
bodyfile=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.bodies.csv')
bodies={int(a['body_index']):a for a in DictReader(bodyfile.open(newline=''))}; rec=bodies[109]
M=np.asarray([[float(rec[f'world_to_body_m{i}{j}']) for j in range(3)] for i in range(3)],np.float64)
world_dir=M.T@direction;world_dir/=np.linalg.norm(world_dir)
y+=(w[:,None]*AMP_MM/1000.0)*world_dir[None,:]
y=np.asarray(y,dtype=np.float32)
changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1))
affected=np.flatnonzero(np.any(np.isin(F,changed),axis=1))
def metrics(T):
 a=T[:,1]-T[:,0];b=T[:,2]-T[:,0];c=T[:,2]-T[:,1];n=np.cross(a,b);twice=np.linalg.norm(n,axis=1)
 L=np.stack((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)),axis=1)
 A=[]
 for i in range(3):
  u=T[:,(i+1)%3]-T[:,i];v=T[:,(i+2)%3]-T[:,i]
  d=np.einsum('ij,ij->i',u,v)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-300)
  A.append(np.degrees(np.arccos(np.clip(d,-1,1))))
 return {'area':twice/2,'angle':np.min(np.stack(A,axis=1),axis=1),'aspect':np.max(L,axis=1)**2/(2*np.sqrt(3)*np.maximum(twice,1e-300)),'normal':n}
m0=metrics(x[F[affected]].astype(np.float64)*1000);m1=metrics(y[F[affected]].astype(np.float64)*1000)
area_ratio=m1['area']/np.maximum(m0['area'],1e-300);angle_ratio=m1['angle']/np.maximum(m0['angle'],1e-300);aspect_ratio=m1['aspect']/np.maximum(m0['aspect'],1e-300)
ndot=np.einsum('ij,ij->i',m0['normal'],m1['normal'])/np.maximum(np.linalg.norm(m0['normal'],axis=1)*np.linalg.norm(m1['normal'],axis=1),1e-300)
bad=(area_ratio<.95)|(angle_ratio<.95)|(aspect_ratio>1.05)|(ndot<=0)
disp=np.linalg.norm((y.astype(np.float64)-x.astype(np.float64))*1000,axis=1)
shape={'pass':bool(not bad.any() and disp.max(initial=0)<=3.0),'affected_faces':int(len(affected)),'changed_nodes':int(len(changed)),'max_displacement_mm':float(disp.max(initial=0)),'target_node_count':len(target_nodes),'target_cluster_face_count':len(cluster),'fixed_anchor_count':int(np.count_nonzero(w[np.asarray(list(anchor_set),np.int64)])) if anchor_set else 0,'support_nodes_with_nonzero_move':int(np.count_nonzero(w)),'max_ring':RING,'minimum_area_ratio':float(area_ratio.min(initial=np.inf)),'minimum_angle_ratio':float(angle_ratio.min(initial=np.inf)),'maximum_aspect_ratio':float(aspect_ratio.max(initial=-np.inf)),'minimum_normal_dot':float(ndot.min(initial=np.inf)),'maximum_normal_rotation_degrees':float(np.degrees(np.arccos(np.clip(ndot,-1,1))).max(initial=0)),'violating_face_rows':affected[bad].astype(int).tolist(),'thresholds':{'area_ratio_floor':.95,'angle_ratio_floor':.95,'aspect_ratio_ceiling':1.05,'positive_orientation':True,'max_step_mm':3.0}}
report={'schema':'numi.human.b109-114.target-ranked-normal-patch.v1','status':'SHAPE_PASS_CANDIDATE_EMITTED' if shape['pass'] else 'SHAPE_FAIL_NO_CANDIDATE','baseline_path':str(BASE),'baseline_sha256':BASE_SHA,'audit159_sha256':sha(AUD),'seed_skin_face':SEED,'seed_source_parent_face':int(rr['parent_face']),'target_bone_triangle_rows':sorted(TARGET_BONE_TRIANGLE_ROWS),'target_cluster_skin_faces':sorted(cluster),'target_cluster_nodes':sorted(target_nodes),'selected_direction_local':direction.tolist(),'selected_direction_world':world_dir.tolist(),'amplitude_mm':AMP_MM,'shape':shape,'candidate_path':None,'candidate_sha256':None,'geometry_admitted':False,'physical_steps':0}
if shape['pass']:
 p['points_world_m']=y
 n=len(x);m=len(F)
 p['remesh806_old_to_new_node_rows']=np.arange(n,dtype=np.int64)
 p['remesh806_new_to_old_node_rows']=np.arange(n,dtype=np.int64)
 p['remesh806_baseline_sha256']=np.asarray([BASE_SHA],dtype='<U64')
 p['remesh806_face_parent_indptr']=np.arange(m+1,dtype=np.int64)
 p['remesh806_face_parent_face_rows']=np.arange(m,dtype=np.int64)
 out=D/'candidate-private.npz'
 with out.open('xb') as f:np.savez_compressed(f,**p)
 report['candidate_path']=str(out);report['candidate_sha256']=sha(out)
 report['metadata_source_arrays_preserved']=True
report_path=D/'build-shape-report.json';report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('target_cluster_skin_faces','target_cluster_nodes')},indent=2,allow_nan=False))
