from pathlib import Path
import json,subprocess,hashlib,importlib.util
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT=H/'connected-local-continuation-189-run001';OUT.mkdir(exist_ok=False)
state=read(H/'local-witness-backtracking-188-run001/selected-result.json')
helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py';expected='05b1beb2105a184deb885974db892fa0f8e519fc3e1fdfde5aa441c180c97b34'
assert sha(helper)==expected
for n in range(8):
 root=OUT/f'round-{n+1:03d}';root.mkdir()
 cmd=[PY,str(H/'prepare_refined_proposal_185.py'),'--candidate',state['candidate'],'--point',state['point_summary'],'--surface',state['surface_report'],'--parent',state['parent_fit'],'--out',str(root/'proposal'),'--rounds','1']
 with(root/'proposal.log').open('x')as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert cp.returncode==0,root/'proposal.log'
 fit=root/'proposal/round-001/fit/attempt-001'
 config=dict(state=state,trial=str(fit/'scp-01-alpha-1p0/candidate.npz'),out=str(root/'local-backtrack'),parent_fit=str(fit/'fit-report.json'),helper=str(helper),helper_sha256=expected,initial_scale=.5,zero_below=1/16,helper_entry="reduce_surface_witness_node_scales_183")
 assert Path(config['trial']).exists()
 write(root/'local-config.json',config)
 with(root/'local.log').open('x')as f:cp=subprocess.run([PY,str(H/'run_local_witness_backtracking_188.py'),str(root/'local-config.json')],stdout=f,stderr=subprocess.STDOUT)
 assert cp.returncode==0,root/'local.log'
 chosen=root/'local-backtrack/selected-result.json'
 if not chosen.exists():
  write(root/'result.json',dict(status='NO_COMBINED_PROGRESS',native_admission=False,accepted_physical_steps=0));break
 state=read(chosen);write(root/'result.json',state);write(OUT/'selected-result.json',state)
 joint=state['joint_merit'];print(json.dumps(dict(round=n+1,candidate_sha256=state['candidate_sha256'],before_crossings=joint['before_surface_pairs'],after_crossings=joint['after_surface_pairs'],before_inside=joint['before_point_pairs'],after_inside=joint['after_point_pairs'],maximum_depth_m=joint['after_maximum_point_depth_m'],native_admission=False)),flush=True)
 if joint['after_surface_pairs']==0 and joint['after_point_pairs']==0:break
