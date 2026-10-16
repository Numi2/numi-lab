from pathlib import Path
import os,sys,json,hashlib,time,datetime
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-refined-regional-graph-240-003/matter/tools';sys.path[:0]=[str(s),str(q/'analysis/source198-strict-layer-replay-preparation-001')]
import strict_layers as layers
ref=layers.ref
d=q/'analysis/native596-final127-globalization-direction-study-001';d.mkdir(exist_ok=False)
c=q/'solver/strict-state-source240-control19-final-newton-diagnostic-596';capture=c/'diagnostics/capture-operator-fd-capture/capture'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
head,nodes,tets,integrity=layers.geometry.capture_records(capture)
nodes=sorted((r for r in nodes if r['environment']==0),key=lambda n:n['node_id']);tets=sorted((r for r in tets if r['environment']==0),key=lambda t:t['tet_id'])
a=np.load(q/'analysis/native596-final127-material-reference-003/arrays.npz');lin=np.load(q/'analysis/native596-final127-cached-tangent-conditioning-251-004/arrays.npz')
A=sparse.load_npz(q/'analysis/native596-final127-cached-tangent-conditioning-251-004/native_cached_operator_FP64.npz')
dt=float(np.float32(head['dt_seconds']));N=len(nodes);T=len(tets);ids=a['tet_nodes'];F=a['F_center'];old=a['accepted_paper_history']
material=a['material'];paper=np.flatnonzero(material!=2);glue=np.flatnonzero(material==2)
specs={m['index']:m for m in head['materials']};pt=layers.f32(head['parameters_by_environment'][0]);p=np.stack([pt[specs[int(material[i])]['parameter_offset']:specs[int(material[i])]['parameter_offset']+28] for i in paper])
gi=specs[2]['parameter_offset'];mu=np.full(len(glue),pt[gi+1]);lam=np.full(len(glue),pt[gi+2])
inv=layers.f32([t['inverse_rest_rows_per_m'] for t in tets]);vol=layers.f32([t['rest_volume_m3'] for t in tets]);rot=layers.f32([t['material_frame_rows_float32'] for t in tets])
mass=layers.f32([n['center']['mass_kg'] for n in nodes]);vc=layers.f32([n['center']['v_m_s'] for n in nodes]);vo=layers.f32([n['accepted']['v_m_s'] for n in nodes]);free_dofs=lin['free_dofs'];rhs=lin['native_rhs_Ns']
def evaluate(x,alpha):
 full=np.zeros(3*N);full[free_dofs]=x;dv=full.reshape(N,3)
 edges=np.stack([dv[ids[:,j]]-dv[ids[:,0]] for j in [1,2,3]],axis=2)
 H=edges@inv@rot
 ff=F+alpha*dt*H
 z,it,rn,cn,valid,reasons,passes=layers.project(p,ff[paper],old)
 if not valid.all():return {'alpha':alpha,'valid':False,'invalid_projections':int((~valid).sum())}
 P=np.empty_like(ff);P[paper]=ref.stress(p,ff[paper],z);P[glue]=ref.glue_stress(ff[glue],mu,lam)
 forces=layers.forces(P@np.swapaxes(rot,-1,-2),inv,vol)
 R=dt*layers.gather(forces,ids,N)-mass[:,None]*(vc+alpha*dv-vo)
 r=R.reshape(-1)[free_dofs]
 return {'alpha':alpha,'valid':bool(np.isfinite(r).all()),'residual_l2_Ns':float(np.linalg.norm(r)),'minimum_J':float(np.linalg.det(ff).min()),'maximum_scaled_projection_residual':float(rn.max()),'direction_l2_m_s':float(np.linalg.norm(x))}
base=evaluate(np.zeros(len(free_dofs)),0.)
directions=[('native_FGMRES',lin['native_correction'],None),('FP64_direct_cached_matrix',lin['FP64_solution'],None)]
for damping in [1e-3,1e-4,1e-5,1e-6,0.]:
 z=lsmr(A,rhs,damp=damping,atol=0.,btol=0.,conlim=1e30,maxiter=64)
 directions.append(('FP64_CPU_LSMR64_damping_'+str(damping),z[0],{'damping_kg':damping,'iterations':int(z[2]),'linear_defect_Ns':float(np.linalg.norm(A@z[0]-rhs))}))
rows=[];arrays={}
for name,x,meta in directions:
 grid=[evaluate(x,2.**(-i)) for i in range(0,21,2)]
 for v in grid:
  if v.get('valid'):v['relative_to_base']=v['residual_l2_Ns']/base['residual_l2_Ns']
 best=min((v for v in grid if v.get('valid')),key=lambda v:v['residual_l2_Ns'])
 rows.append({'direction':name,'linear_solve':meta,'grid':grid,'best':best});arrays[name]=x
 print(json.dumps({'direction':name,'best':best}),flush=True)
report={'schema':'numi.cardboard.offline-globalization-direction-study.v1','capture_sha256':sha(capture),'root':19,'newton':127,'base':base,'directions':rows,'classification':'Independent FP64 same-accepted-history candidate replay; CPU conditioning/globalization experiment only, not native geometry/history or completed control.','physical_operator_and_parameters_changed':False,'limits':['Ideal FP64 kinematic perturbations; native FP32 realization and local branch transitions require separate native trials.','LSMR is a CPU direction construction; it is not a native CGLS equivalence proof.','Cached matrix and actual accepted-history material reference remain fixed for all candidates.','No acceptance tolerance is changed and no complete coupon or shipping qualification is inferred.'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(d/'directions.npz',**arrays)
inputs=[capture,c/'receipt.json',q/'analysis/native596-final127-material-reference-003/arrays.npz',q/'analysis/native596-final127-cached-tangent-conditioning-251-004/arrays.npz',q/'analysis/native596-final127-cached-tangent-conditioning-251-004/native_cached_operator_FP64.npz',Path(__file__),Path(layers.__file__),Path(ref.__file__),Path(layers.strict.__file__),Path(layers.geometry.__file__)]
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):sha(p) for p in inputs},'artifacts':{p.name:sha(p) for p in d.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'base':base}),flush=True)
