from pathlib import Path
import json,subprocess,hashlib,numpy as np,importlib.util,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=H/'whole-witness-replay-207-run001';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(args,log):
 cmd=[PY,'/tmp/numi-human-geometry-slot.py',*map(str,args)];st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
state=read(H/'joint-clusters-199-run001/selected-result.json')
raw=H/'whole-witness-connected-204-run001/proposal/fit/attempt-001/scp-01-alpha-1p0/candidate.npz'
pairs=np.load(H/'whole-witness-connected-204-run001/raw-audit130/surface-self-pairs.npy');allpairs={tuple(map(int,r))for r in pairs}
solver=D/'D/connected-local-witness-proposal-205/propose_all_witness_clusters_205.py'
assert sha(solver)=='019a4d54e8303303b0aa0ad13204a1dcda67af1e5d03d955b20bda1e0f2639c4'
ow=D/'D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py'
spec=importlib.util.spec_from_file_location('owner',ow);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
write(OUT/'input-pins.json',dict(state=state,solver_sha256=sha(solver),owner_sha256=sha(ow),raw_sha256=sha(raw)))
for i in range(4):
 rd=OUT/f'joint-{i+1:03d}';rd.mkdir();ps=rd/'self-pairs.npy';np.save(ps,np.asarray(sorted(allpairs),np.int64).reshape(-1,2))
 run([PY,solver,'--base-candidate',state['candidate'],'--raw-trial',raw,'--self-pairs',ps,'--source-base-npz',D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz','--area-reference-npz',D/'full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz','--output',rd/'proposal'],rd/'fit.log')
 candidate=rd/'proposal/candidate.npz'
 if not candidate.exists():print('NO_CANDIDATE',rd,flush=True);break
 run([PY,D/'audit_connected_outer_geometry_130.py',candidate,rd/'audit130'],rd/'audit130.log')
 after=read(rd/'audit130/summary.json')
 if after['exact_PLC_surface_pairs']:
  pp=np.load(rd/'audit130/surface-self-pairs.npy');old=len(allpairs);allpairs.update(tuple(map(int,p))for p in pp)
  write(rd/'result.json',dict(status='SELF_WITNESSES_LEARNED',self_pairs=len(pp),learned_pair_count=len(allpairs),native_admission=False))
  print(json.dumps(dict(iteration=i+1,self_pairs=len(pp),all_learned_pairs=len(allpairs))),flush=True)
  if len(allpairs)==old:break
  continue
 run([PY,D/'audit_skin_bone_surfaces_159.py',candidate,rd/'audit159','--max-seconds','600'],rd/'audit159.log')
 merit=owner.assess_joint_clearance_construction_progress(read(state['point_summary']),after,read(state['surface_report']),read(rd/'audit159/report.json'))
 depth=owner.assess_construction_depth_progress(read(state['point_summary']),after,read(state['surface_report']),read(rd/'audit159/report.json'))
 ok=merit['construction_progress']or depth['construction_progress']
 result=dict(status='CONSTRUCTION_PROGRESS_ONLY'if ok else'NO_COMBINED_PROGRESS',candidate=str(candidate),candidate_sha256=sha(candidate),point_summary=str(rd/'audit130/summary.json'),surface_report=str(rd/'audit159/report.json'),parent_fit=str(H/'whole-witness-connected-204-run001/proposal/fit/attempt-001/fit-report.json'),joint_merit=merit,depth_merit=depth,native_admission=False,accepted_physical_steps=0)
 write(rd/'result.json',result);print(json.dumps(dict(joint=merit,depth=depth)),flush=True)
 if ok:write(OUT/'selected-result.json',result)
 break
write(OUT/'completed.json',dict(status='BOUNDED_WHOLE_WITNESS_REPLAY_FINISHED',native_admission=False,accepted_physical_steps=0))
