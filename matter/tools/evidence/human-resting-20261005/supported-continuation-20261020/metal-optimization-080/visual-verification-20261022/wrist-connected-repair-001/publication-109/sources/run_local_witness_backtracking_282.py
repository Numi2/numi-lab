from pathlib import Path
import json,hashlib,subprocess,importlib.util,numpy as np,time,sys
from scipy import sparse
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(script,p,out):
 cmd=[PY,'/tmp/numi-human-geometry-slot.py',PY,str(script),str(p),str(out)]
 if script.name=='audit_skin_bone_surfaces_159.py':cmd+=['--max-seconds','600']
 st=time.monotonic()
 with Path(str(out)+'.log').open('x')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 write(str(out)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st))
 assert r.returncode==0
surface_damping=module('surface_damping',H/'surface-regression-witnesses-258.py')
assert sha(H/'surface-regression-witnesses-258.py')=='5099756b1244845be2fb61a476c15d3a2c6095e41b3e41f08a9517148ba7764d'
point_damping=module('point_damping',H/'point-regression-witnesses-254.py')
assert sha(H/'point-regression-witnesses-254.py')=='f78831353656e6b7e5bfee6d0edca89c876dac25ccd85eeaf522c350b58dd894'
configpath=Path(sys.argv[1]);cfg=read(configpath)
assert 1<=cfg.get('maximum_exact_audits',8)<=16 and 1<=cfg.get('maximum_construction_passes',40)<=64
r=cfg['state']
START=Path(r['candidate']);TRIAL=Path(cfg['trial'])
OUT=Path(cfg['out']);OUT.mkdir(exist_ok=False)
owner=module('bone_owner',D/'D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py')
# The tested helper path is supplied in the adjacent execution configuration.
config=cfg
helper=module('local_steps',Path(config['helper']))
assert sha(config['helper'])==config['helper_sha256']
z=np.load(START);t=np.load(TRIAL);X=z['points_world_m'].astype(float);Y=t['points_world_m'].astype(float);F=z['source_O_faces'].astype(np.int64);assert np.array_equal(F,t['source_O_faces'])
anchors=z['typed_coarse_anchor_output_rows_018'];payload={k:z[k]for k in z.files}
before=read(r['point_summary']);before_surface=read(r['surface_report'])
assert before['candidate_sha256']==before_surface['candidate_sha256']==sha(START)
base=np.load(D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
nold=len(base['points_before_joint_self_pair_repair_m']);prefixes=('', 'prior_input_006__', 'prior_run028__');P=sparse.eye(len(X),format='csr')
for level,prefix in enumerate(prefixes):
 ind=np.asarray(z[prefix+'source_prolongation_indptr'],np.int64);cols=np.asarray(z[prefix+'source_prolongation_node_rows'],np.int64);w=np.asarray(z[prefix+'source_prolongation_weights'],float)
 nc=len(z[prefixes[level+1]+'source_prolongation_indptr'])-1 if level+1<len(prefixes) else nold
 assert len(ind)-1==P.shape[1] and ind[0]==0 and ind[-1]==len(cols)==len(w) and (np.diff(ind)>0).all() and (cols>=0).all() and (cols<nc).all() and np.isfinite(w).all() and (w>=0).all()
 M=sparse.csr_matrix((w,cols,ind),shape=(len(ind)-1,nc));assert np.max(np.abs(np.asarray(M.sum(axis=1)).ravel()-1))<2e-10;P=P@M
R0=np.asarray(P@base['points_before_joint_self_pair_repair_m'],float)
refsource=Path(cfg['area_reference_npz']) if cfg.get('area_reference_npz') else D/'full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz'
if cfg.get('area_reference_npz'):
 assert sha(refsource)==cfg['area_reference_sha256'], 'matched refined area-reference hash mismatch'
refz=np.load(refsource)
reference_points_key=cfg.get('area_reference_points_key','points_world_m');reference_faces_key=cfg.get('area_reference_faces_key','source_O_faces')
assert np.array_equal(F,refz[reference_faces_key]), 'matched area-reference topology mismatch'
XR=refz[reference_points_key].astype(float)
assert XR.shape==X.shape and np.isfinite(XR).all(), 'matched area-reference coordinate domain mismatch'
fixed_prefix=cfg.get('fixed_prefix_nodes',0)
assert type(fixed_prefix)is int and 0<=fixed_prefix<=len(X)
assert np.array_equal(t['points_world_m'][:fixed_prefix].view(np.uint32),z['points_world_m'][:fixed_prefix].view(np.uint32)), 'trial changed fixed prefix'

reference_cross=np.cross(XR[F[:,1]]-XR[F[:,0]],XR[F[:,2]]-XR[F[:,0]]);reference_area=np.linalg.norm(reference_cross,axis=1)
ref=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]);den=np.einsum('ij,ij->i',ref,ref);assert (den>0).all()
assert (np.linalg.norm(ref,axis=1)/reference_area>=.1).all(), 'starting geometry fails fixed area magnitude gate'

assert np.array_equal(X[anchors].astype(np.float32).view(np.uint32),R0[anchors].astype(np.float32).view(np.uint32))
assert np.linalg.norm(X-R0,axis=1).max()<=.012002

scales=np.load(cfg['initial_scales']) if cfg.get('initial_scales') else np.full(len(X),cfg.get("initial_scale",.5))
assert scales.shape==(len(X),) and np.isfinite(scales).all() and ((scales>=0)&(scales<=1)).all()
scales[anchors]=0;scales[:fixed_prefix]=0;history=[];expensive_audits=0
reduce_scales=getattr(helper,cfg.get('helper_entry','reduce_witness_node_scales'))
write(OUT/'source-inputs.json',dict(start=str(START),start_sha256=sha(START),trial=str(TRIAL),trial_sha256=sha(TRIAL),helper_sha256=sha(config['helper']),source_point_summary_sha256=sha(r['point_summary']),source_surface_report_sha256=sha(r['surface_report']),native_admission=False,accepted_physical_steps=0))
for iteration in range(cfg.get('maximum_construction_passes',40)):
 out=OUT/f'pass-{iteration+1:03d}';out.mkdir()
 xp=(X+scales[:,None]*(Y-X)).astype(np.float32);xp[anchors]=X[anchors].astype(np.float32)
 cross=np.cross(xp[F[:,1]].astype(float)-xp[F[:,0]],xp[F[:,2]].astype(float)-xp[F[:,0]])
 ratios=np.einsum('ij,ij->i',cross,ref)/den
 magnitude_ratios=np.linalg.norm(cross,axis=1)/reference_area
 bad=np.flatnonzero((ratios<.1)|(magnitude_ratios<.1))
 stepbad=np.flatnonzero(np.linalg.norm(xp.astype(float)-X,axis=1)>0.001003)
 capbad=np.flatnonzero(np.linalg.norm(xp.astype(float)-R0,axis=1)>.012002)
 move=Y-X;proj=np.einsum('ij,ij->i',xp.astype(float)-X,move);regress=np.flatnonzero(proj < -.000002*np.linalg.norm(move,axis=1))
 badnodes=np.union1d(stepbad,np.union1d(capbad,regress))
 if len(badnodes):bad=np.union1d(bad,np.flatnonzero(np.isin(F,badnodes).any(axis=1)))
 assert np.isfinite(xp).all()
 p=out/'candidate.npz'
 if not len(bad):np.savez_compressed(p,**{**payload,'points_world_m':xp})
 np.save(out/'node-step-scales.npy',scales)
 rec=dict(iteration=iteration+1,candidate_sha256=sha(p) if p.exists() else None,emitted_points_sha256=hashlib.sha256(xp.tobytes()).hexdigest(),minimum_allface_oriented_area_ratio=float(ratios.min()),minimum_fixed_reference_area_magnitude_ratio=float(magnitude_ratios.min()),bad_area_faces=len(bad),anchors_bitwise_unchanged=bool(np.array_equal(xp[anchors].view(np.uint32),z['points_world_m'][anchors].view(np.uint32))),nonzero_step_nodes=int(np.count_nonzero(np.any(xp!=X.astype(np.float32),axis=1))),nonzero_construction_scale_nodes=int(np.count_nonzero(scales)),maximum_step_m=float(np.linalg.norm(xp.astype(float)-X,axis=1).max()),maximum_source_displacement_mm=float(np.linalg.norm(xp.astype(float)-R0,axis=1).max()*1000),source_cap_pass=not bool(len(capbad)),native_admission=False,accepted_physical_steps=0)
 assert rec['anchors_bitwise_unchanged']
 assert np.array_equal(xp[:fixed_prefix].view(np.uint32),z['points_world_m'][:fixed_prefix].view(np.uint32)), 'emitted fixed prefix changed'
 selfpairs=np.empty((0,2),np.int64)
 if not len(bad):
  if expensive_audits>=cfg.get('maximum_exact_audits',8):
   write(out/'result.json',dict(status='EXACT_AUDIT_BUDGET_REACHED',native_admission=False,accepted_physical_steps=0));break
  expensive_audits+=1
  run(D/'audit_connected_outer_geometry_130.py',p,out/'audit-130')
  after=read(out/'audit-130/summary.json');rec['point_summary_sha256']=sha(out/'audit-130/summary.json')
  assert after['candidate_sha256']==sha(p)
  if after.get('exact_PLC_surface_pairs',-1)>0:
   selfpairs=np.load(out/'audit-130/surface-self-pairs.npy')
   rec['exact_self_pairs']=len(selfpairs)
  elif after.get('complete_surface_audit')is True and after.get('exact_PLC_surface_pairs')==0:
   merit=owner.assess_point_clearance_construction_progress(before,after);rec['point_merit']=merit
   # Existing P093 joint merit owns construction selection; point count alone need not fall.
   if merit['maximum_depth_nonincreasing'] and merit['squared_depth_nonincreasing']:
    run(D/'audit_skin_bone_surfaces_159.py',p,out/'audit-159')
    sr=read(out/'audit-159/report.json')
    joint=owner.assess_joint_clearance_construction_progress(before,after,before_surface,sr)
    rec['joint_merit']=joint
    depth=owner.assess_construction_depth_progress(before,after,before_surface,sr);rec['depth_merit']=depth
    if joint['construction_progress'] or depth['construction_progress']:
     rec.update(status='CONSTRUCTION_PROGRESS_ONLY',candidate=str(p),point_summary=str(out/'audit-130/summary.json'),surface_report=str(out/'audit-159/report.json'),parent_fit=cfg['parent_fit'])
     write(out/'result.json',rec);write(OUT/'selected-result.json',rec);print(json.dumps({k:v for k,v in rec.items()if k not in ['scale_update','point_merit','joint_merit']}),flush=True);break
   if (not merit['maximum_depth_nonincreasing'] or not merit['squared_depth_nonincreasing']) and before.get('inside_node_bank_pairs')==0 and after.get('inside_node_bank_pairs',0)>0:
    regressed_rows,bad=point_damping.regressed_point_faces(before,after,F,len(X))
    rec['point_regression_witness_nodes']=regressed_rows.tolist()
    rec['point_regression_incident_faces']=bad.tolist()
    rec['construction_action']='halve only actual penetration witness incident-face node scales; rerun every geometry gate'
   elif merit['maximum_depth_nonincreasing'] and merit['squared_depth_nonincreasing'] and 'joint_merit' in rec:
    bad,introduced_count=surface_damping.introduced_surface_faces(before_surface,sr,len(F))
    if not len(bad):
     rec['status']='NO_COMBINED_PROGRESS';write(out/'result.json',rec);break
    rec['introduced_surface_pair_count_for_damping']=introduced_count
    rec['introduced_surface_witness_faces']=bad.tolist()
    rec['construction_action']='halve only exact introduced triangle witness face node scales; rerun every geometry gate'
   else:
    rec['status']='NO_COMBINED_PROGRESS';write(out/'result.json',rec);print(json.dumps({k:v for k,v in rec.items()if k not in ['scale_update','point_merit','joint_merit']}),flush=True);break
  else:raise RuntimeError('incomplete geometry audit')
 scales,update=reduce_scales(F,scales,bad,selfpairs,anchors=anchors,zero_below=cfg.get('zero_below',1/128))
 rec['scale_update']={k:v for k,v in update.items()if not isinstance(v,list)};history.append(rec);write(out/'result.json',rec);print(json.dumps({k:v for k,v in rec.items()if k not in ['scale_update','point_merit','joint_merit']}),flush=True)
else:write(OUT/'result.json',dict(status='BOUNDED_LOCAL_BACKTRACK_FAILED',history=history,native_admission=False,accepted_physical_steps=0))
