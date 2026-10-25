from pathlib import Path
import json,subprocess,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python';OUT=H/'whole-witness-local-210-run001';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
state=read(H/'joint-clusters-199-run001/selected-result.json')
raw=H/'whole-witness-connected-204-run001/proposal/fit/attempt-001/scp-01-alpha-1p0/candidate.npz'
solver=D/'D/connected-local-witness-proposal-209/propose_all_witness_clusters_209.py'
cmd=[PY,'/tmp/numi-human-geometry-slot.py',PY,str(solver),'--base-candidate',state['candidate'],'--raw-trial',str(raw),'--self-pairs',str(H/'whole-witness-connected-204-run001/joint-001/self-pairs.npy'),'--source-base-npz',str(D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'),'--area-reference-npz',str(D/'full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz'),'--output',str(OUT/'joint-proposal')]
with(OUT/'joint-fit.log').open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0
sr=H/'run_local_witness_backtracking_191.py';s=sr.read_text().replace("D/'publication-093/canonical/matter/tools/human_skin_fat_bone_audit.py'","D/'D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py'")
s=s.replace("rec['joint_merit']=joint\n    if joint['construction_progress']:","rec['joint_merit']=joint\n    depth=owner.assess_construction_depth_progress(before,after,before_surface,sr);rec['depth_merit']=depth\n    if joint['construction_progress'] or depth['construction_progress']:")
s=s.replace("nonzero_step_nodes=int(np.count_nonzero(scales))","nonzero_step_nodes=int(np.count_nonzero(np.any(xp!=X.astype(np.float32),axis=1))),nonzero_construction_scale_nodes=int(np.count_nonzero(scales))")
f=H/'run_local_witness_backtracking_211.py';f.write_text(s);compile(s,str(f),'exec')
trial=OUT/'joint-proposal/raw-lp-candidate.npz';assert trial.exists()
helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
cfg=dict(state=state,trial=str(trial),out=str(OUT/'local-backtrack'),parent_fit=str(H/'whole-witness-connected-204-run001/proposal/fit/attempt-001/fit-report.json'),helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=1.,zero_below=1/16)
(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
with(OUT/'local.log').open('x')as f:r=subprocess.run([PY,str(H/'run_local_witness_backtracking_211.py'),str(OUT/'config.json')],stdout=f,stderr=subprocess.STDOUT)
print('RETURN',r.returncode)
if(OUT/'local-backtrack/selected-result.json').exists():print(json.dumps(read(OUT/'local-backtrack/selected-result.json')['joint_merit']))
