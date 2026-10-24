#!/usr/bin/env python3
"""One bilateral, rest-preserving weight-field counterfactual. Not admitted physics."""
from pathlib import Path
import importlib.util,sys,json,hashlib,copy,struct,time
from collections import defaultdict,Counter
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'bilateral-arm-rest-preserving-weight-repair-279'
if OUT.exists():raise RuntimeError('refuse overwrite')
OUT.mkdir()
start=time.monotonic()
helper=E/'batch-source-compose-252/tools/rebind_human_skin_reference.py'
spec=importlib.util.spec_from_file_location('rb279',helper);r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
source=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
snapshot=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json'
manifest=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
b=r.decode(source.read_bytes());base=json.loads(snapshot.read_text());F=np.asarray(b['inds'],np.int64);N=b['nv'];W=b['weights'].astype(float)
X0=r.registered_positions_float32(b,base,frame='rest').astype(float)
E0=np.stack([X0[F[:,1]]-X0[F[:,0]],X0[F[:,2]]-X0[F[:,0]]],axis=2)
_,R=np.linalg.qr(E0);Ri=np.linalg.inv(R)
names={int(x['core_body_index']):x['myosim_body'] for x in json.loads(manifest.read_text())['source']['anchors']}
owners=np.asarray([int(x['body']) for x in b['binds']])[np.argmax(W,axis=1)]
arm=np.array([any(t in names.get(int(o),'') for t in ('clavicle','scapula','humerus','radius','ulna')) for o in owners])
armfaces=np.flatnonzero(np.any(arm[F],axis=1))
posepaths={'current':snapshot,'shoulder_elevation':E/'joint-source-decomposition-053/shoulder-elevation-050/registered-skin-poses.json','elbow_flexion':E/'joint-source-decomposition-053/elbow-flexion-080/registered-skin-poses.json','shoulder_plane':Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/shoulder_plane_bilateral_q35q73_m0p3_at_elevation0p5.json')}
poses={}
def emitted(payload,p):return r.registered_positions_float32(payload,p,frame='current').astype(float)
def spans(X):
 ee=np.stack([X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]],axis=2)
 return np.linalg.svd(ee@Ri,compute_uv=False)
seed=set();baselines={}
for key,path in posepaths.items():
 p=json.loads(path.read_text())
 if 'source_poses' in p:
  pp=copy.deepcopy(base);by={int(x['core_body']):x for x in p['source_poses']}
  for row in pp.get('all_bodies',pp.get('bodies')):
   x=by[int(row['body_index'])];row['current']={'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']}
  p=pp
 poses[key]=p;X=emitted(b,p);sv=spans(X)
 bad=armfaces[(sv[armfaces,0]>3)|(sv[armfaces,1]<.5)]
 seed.update(map(int,bad));baselines[key]=(X,sv)
# Conforming source-connectivity neighborhood, never Euclidean cross-body links.
adj=[set() for _ in range(N)]
for t in F:
 for i,j in zip(t,np.roll(t,-1)):adj[int(i)].add(int(j));adj[int(j)].add(int(i))
seeds=np.unique(F[sorted(seed)]);dist={int(v):0 for v in seeds};front=set(map(int,seeds))
for ring in range(1,5):
 front={v for u in front for v in adj[u] if v not in dist};dist.update((v,ring) for v in front)
free=np.asarray(sorted(v for v,k in dist.items() if k<4),np.int64);fixed=np.asarray(sorted(v for v,k in dist.items() if k==4),np.int64)
lookup={int(v):i for i,v in enumerate(free)};fr=np.flatnonzero(np.any(np.isin(F,free),axis=1))
# Positive cotangent conductance, a scalar M-matrix. All 86 weights are solved
# together; nonnegativity/partition follow from the maximum principle, checked.
edgew=defaultdict(float)
for t in F[fr]:
 p=X0[t]
 for k in range(3):
  u=p[(k+1)%3]-p[k];v=p[(k+2)%3]-p[k]
  den=float(np.linalg.norm(np.cross(u,v)))
  if den<=1e-25:raise ValueError('degenerate source face')
  c=max(0.,float(u@v)/den)*.5
  i,j=sorted((int(t[(k+1)%3]),int(t[(k+2)%3])));edgew[(i,j)]+=c
rows=[];cols=[];vals=[];degree=np.zeros(len(free));rhs=np.zeros((len(free),b['nb']))
for (i,j),w in edgew.items():
 if w==0:continue
 for u,v in ((i,j),(j,i)):
  a=lookup.get(u)
  if a is None:continue
  degree[a]+=w
  z=lookup.get(v)
  if z is None:rhs[a]+=w*W[v]
  else:rows.append(a);cols.append(z);vals.append(-w)
if np.any(degree<=0):raise ValueError('isolated free row')
screen=.1*degree
A=coo_matrix((vals,(rows,cols)),shape=(len(free),len(free))).tocsc()+diags(degree+screen,format='csc')
rhs+=screen[:,None]*W[free]
factor=splu(A);Z=factor.solve(rhs)
res=np.linalg.norm(A@Z-rhs)/max(np.linalg.norm(rhs),1e-30)
if Z.min()<-1e-10 or np.max(np.abs(Z.sum(1)-1))>1e-6 or res>1e-10:raise ValueError(('weight solve',Z.min(),res))
Z=np.maximum(Z,0);Z/=Z.sum(1)[:,None]
weights=b['weights'].copy();weights[free]=Z.astype(np.float32)
rest={'source_poses':[{'core_body':int(row['body_index']),'position':row['rest']['position_world_m'],'quaternion_xyzw':row['rest']['orientation_world_xyzw']} for row in base.get('all_bodies',base.get('bodies'))]}
aa,bb=r.reference_transforms(b,rest,[0,0,0])
raw,info=r.rebind(b,weights,aa,bb,maximum_error_m=2e-6);cand=r.decode(raw)
changed=np.flatnonzero(np.any(cand['weights']!=b['weights'],axis=1))
normalrows=np.unique(F[np.any(np.isin(F,changed),axis=1)])
normals=r.registered_rest_vertex_normals_float32(cand,base,normalrows);blob=bytearray(raw)
for row,n in zip(normalrows,normals):struct.pack_into('<3f',blob,b['vertoff']+int(row)*56+12,*map(float,n))
raw=bytes(blob);cand=r.decode(raw)
outpath=OUT/'candidate.nhskin';outpath.write_bytes(raw)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'status':'counterfactual-unadmitted','method':'Positive-cotangent screened harmonic weights, screen=.1*row degree, four source-edge rings around every arm face with >3 maximum or <.5 minimum principal source-pose stretch in the four declared poses. Exact original binding IDs; existing inverse rebind preserves source-rest surface.','source_sha256':sha(source),'helper_sha256':sha(helper),'snapshot_sha256':sha(snapshot),'candidate_sha256':sha(outpath),'seed_source_faces':sorted(seed),'free_vertex_count':len(free),'fixed_boundary_vertices':len(fixed),'changed_weight_rows':len(changed),'normal_updated_rows':len(normalrows),'solve_relative_residual':float(res),'weight_change_l1_max':float(np.max(np.abs(cand['weights'].astype(float)-W).sum(1))),'rebind':info,'poses':{},'gates':{'no_physical_admission':True}}
Xr=r.registered_positions_float32(cand,base,frame='rest').astype(float);report['emitted_F32_rest_displacement_max_m']=float(np.linalg.norm(Xr-X0,axis=1).max())
for key,p in poses.items():
 X=emitted(cand,p);sv=spans(X);Xb,sb=baselines[key];delta=np.linalg.norm(X-Xb,axis=1)
 report['poses'][key]={'max_displacement_mm':float(delta.max()*1000),'baseline_max_stretch':float(sb[armfaces,0].max()),'candidate_max_stretch':float(sv[armfaces,0].max()),'baseline_faces_above_3':int(np.sum(sb[armfaces,0]>3)),'candidate_faces_above_3':int(np.sum(sv[armfaces,0]>3)),'baseline_faces_below_half':int(np.sum(sb[armfaces,1]<.5)),'candidate_faces_below_half':int(np.sum(sv[armfaces,1]<.5)),'candidate_min_stretch':float(sv[armfaces,1].min())}
np.savez_compressed(OUT/'patch-rows.npz',changed_weight_rows=changed,normal_updated_rows=normalrows,free_vertex_rows=free,fixed_boundary_vertex_rows=fixed,seed_source_face_rows=np.asarray(sorted(seed),np.int64))
report['elapsed_s']=time.monotonic()-start
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'build_weight279.py').write_text(Path(__file__).read_text())
print(json.dumps({k:v for k,v in report.items() if k not in ('rebind','seed_source_faces')}))

