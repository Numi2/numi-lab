from pathlib import Path
import json,hashlib,subprocess,importlib.util,numpy as np,time
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
r=read(H/'refined-point-181-run001/round-001/result.json')
START=Path(r['candidate']);TRIAL=H/'refined-point-182-run001/round-001/fit/attempt-001/scp-04-alpha-1p0/candidate.npz'
OUT=H/'local-witness-backtracking-183-run001';OUT.mkdir(exist_ok=False)
owner=module('bone_owner',D/'publication-093/canonical/matter/tools/human_skin_fat_bone_audit.py')
# The tested helper path is supplied in the adjacent execution configuration.
config=read(H/'local-witness-backtracking-183-config.json')
helper=module('local_steps',Path(config['helper']))
assert sha(config['helper'])==config['helper_sha256']
z=np.load(START);t=np.load(TRIAL);X=z['points_world_m'].astype(float);Y=t['points_world_m'].astype(float);F=z['source_O_faces'].astype(np.int64);assert np.array_equal(F,t['source_O_faces'])
anchors=z['typed_coarse_anchor_output_rows_018'];payload={k:z[k]for k in z.files}
before=read(r['point_summary']);before_surface=read(r['surface_report'])
assert before['candidate_sha256']==before_surface['candidate_sha256']==sha(START)
ref=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]);den=np.einsum('ij,ij->i',ref,ref);assert (den>0).all()
scales=np.ones(len(X));scales[anchors]=0;history=[]
write(OUT/'source-inputs.json',dict(start=str(START),start_sha256=sha(START),trial=str(TRIAL),trial_sha256=sha(TRIAL),helper_sha256=sha(config['helper']),source_point_summary_sha256=sha(r['point_summary']),source_surface_report_sha256=sha(r['surface_report']),native_admission=False,accepted_physical_steps=0))
for iteration in range(20):
 out=OUT/f'pass-{iteration+1:03d}';out.mkdir()
 xp=(X+scales[:,None]*(Y-X)).astype(np.float32);xp[anchors]=X[anchors].astype(np.float32)
 cross=np.cross(xp[F[:,1]].astype(float)-xp[F[:,0]],xp[F[:,2]].astype(float)-xp[F[:,0]])
 ratios=np.einsum('ij,ij->i',cross,ref)/den;bad=np.flatnonzero(ratios<.1)
 stepbad=np.flatnonzero(np.linalg.norm(xp.astype(float)-X,axis=1)>0.001003)
 if len(stepbad):bad=np.union1d(bad,np.flatnonzero(np.isin(F,stepbad).any(axis=1)))
 assert np.isfinite(xp).all()
 p=out/'candidate.npz';np.savez_compressed(p,**{**payload,'points_world_m':xp})
 np.save(out/'node-step-scales.npy',scales)
 rec=dict(iteration=iteration+1,candidate_sha256=sha(p),minimum_allface_oriented_area_ratio=float(ratios.min()),bad_area_faces=len(bad),anchors_bitwise_unchanged=bool(np.array_equal(xp[anchors].view(np.uint32),z['points_world_m'][anchors].view(np.uint32))),nonzero_step_nodes=int(np.count_nonzero(scales)),maximum_step_m=float(np.linalg.norm(xp.astype(float)-X,axis=1).max()),native_admission=False,accepted_physical_steps=0)
 assert rec['anchors_bitwise_unchanged']
 selfpairs=np.empty((0,2),np.int64)
 if not len(bad):
  run(D/'audit_connected_outer_geometry_130.py',p,out/'audit-130')
  after=read(out/'audit-130/summary.json');rec['point_summary_sha256']=sha(out/'audit-130/summary.json')
  assert after['candidate_sha256']==sha(p)
  if after.get('exact_PLC_surface_pairs',-1)>0:
   selfpairs=np.load(out/'audit-130/surface-self-pairs.npy')
   rec['exact_self_pairs']=len(selfpairs)
  elif after.get('complete_surface_audit')is True and after.get('exact_PLC_surface_pairs')==0:
   merit=owner.assess_point_clearance_construction_progress(before,after);rec['point_merit']=merit
   if merit['construction_progress']:
    run(D/'audit_skin_bone_surfaces_159.py',p,out/'audit-159')
    sr=read(out/'audit-159/report.json')
    joint=owner.assess_joint_clearance_construction_progress(before,after,before_surface,sr)
    rec['joint_merit']=joint
    if joint['construction_progress']:
     rec.update(status='CONSTRUCTION_PROGRESS_ONLY',candidate=str(p),point_summary=str(out/'audit-130/summary.json'),surface_report=str(out/'audit-159/report.json'),parent_fit=str(H/'refined-point-182-run001/round-001/fit/attempt-001/fit-report.json'))
     write(out/'result.json',rec);write(OUT/'selected-result.json',rec);print(json.dumps(rec),flush=True);break
   rec['status']='NO_COMBINED_PROGRESS';write(out/'result.json',rec);print(json.dumps(rec),flush=True);break
  else:raise RuntimeError('incomplete geometry audit')
 scales,update=helper.reduce_witness_node_scales(F,scales,bad,selfpairs,anchors=anchors,zero_below=1/128)
 rec['scale_update']=update;history.append(rec);write(out/'result.json',rec);print(json.dumps(rec),flush=True)
else:write(OUT/'result.json',dict(status='BOUNDED_LOCAL_BACKTRACK_FAILED',history=history,native_admission=False,accepted_physical_steps=0))
