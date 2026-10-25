from pathlib import Path
import json,hashlib,numpy as np,subprocess
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';OUT=H/'joint-cluster-200-run001';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(H/'face-connected-continuation-195-run001/selected-result.json');j=read(D/'D/connected-local-witness-proposal-194/run-001/report.json')
a=np.load(r['candidate']);b=np.load(j['input_candidate']);c=np.load(j['candidate']);F=a['source_O_faces'];assert np.array_equal(F,b['source_O_faces']) and np.array_equal(F,c['source_O_faces'])
P=a['points_world_m'];delta=c['points_world_m'].astype(float)-b['points_world_m'].astype(float);support=np.flatnonzero(np.any(delta!=0,axis=1))
out={k:a[k]for k in a.files};out['points_world_m']=(P.astype(float)+delta).astype(np.float32)
assert np.array_equal(out['points_world_m'][a['typed_coarse_anchor_output_rows_018']],P[a['typed_coarse_anchor_output_rows_018']])
out['joint_cluster_200_delta_source_sha256']=np.array([sha(j['candidate'])]);out['joint_cluster_200_parent_sha256']=np.array([sha(r['candidate'])])
trial=OUT/'proposal.npz';np.savez_compressed(trial,**out)
helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
cfg=dict(state=r,trial=str(trial),out=str(OUT/'local-backtrack'),parent_fit=r['parent_fit'],helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=1.,zero_below=1/16)
(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(OUT/'composition.json').write_text(json.dumps(dict(parent_sha256=sha(r['candidate']),branch_start_sha256=sha(j['input_candidate']),branch_candidate_sha256=sha(j['candidate']),changed_nodes=len(support),maximum_delta_mm=float(np.linalg.norm(delta,axis=1).max()*1000),target_node_delta_mm=(delta[15971]*1000).tolist(),construction_only=True),indent=2)+'\n')
with(OUT/'local.log').open('x')as f:cp=subprocess.run(['/Users/n/numi-human-prep-venv-20261005/bin/python',str(H/'run_local_witness_backtracking_191.py'),str(OUT/'config.json')],stdout=f,stderr=subprocess.STDOUT)
print('RETURN',cp.returncode)
if(OUT/'local-backtrack/selected-result.json').exists():print(json.dumps(read(OUT/'local-backtrack/selected-result.json')['joint_merit']))
