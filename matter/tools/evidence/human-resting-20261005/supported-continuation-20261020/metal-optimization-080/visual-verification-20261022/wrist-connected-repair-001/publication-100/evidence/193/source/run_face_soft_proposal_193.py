from pathlib import Path
import json,hashlib,importlib.util,subprocess,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
cfg=read(sys.argv[1]);r=cfg['state'];OUT=Path(cfg['out']);OUT.mkdir(exist_ok=False)
bn=Path(cfg['bary_npz']);bj=Path(cfg['bary_json'])
assert read(bj)['inputs']['candidate_sha256']==sha(r['candidate'])
sp=importlib.util.spec_from_file_location('prepare_owner',H/'prepare_refined_proposal_185.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.DRIVER=H/'refined-point-face-193-template/point_driver.py';old_emit=m.emit
def emit(template,dest,overrides,kind):
 if kind=='driver':
  overrides={**overrides,'BARY_TARGET_NPZ':bn,'BARY_TARGET_JSON':bj,'PIN':{**overrides['PIN'],'bary_npz':sha(bn),'bary_json':sha(bj)}}
 return old_emit(template,dest,overrides,kind)
m.emit=emit
m.prepare(Path(r['candidate']),Path(r['point_summary']),Path(r['surface_report']),Path(r['parent_fit']),OUT/'proposal',True)
fit=OUT/'proposal/fit/attempt-001';trial=fit/'scp-01-alpha-1p0/candidate.npz';assert trial.exists()
helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
lc=dict(state=r,trial=str(trial),out=str(OUT/'local-backtrack'),parent_fit=str(fit/'fit-report.json'),helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=.5,zero_below=1/16)
write(OUT/'local-config.json',lc)
with(OUT/'local.log').open('x')as f:p=subprocess.run([PY,str(H/'run_local_witness_backtracking_191.py'),str(OUT/'local-config.json')],stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
if(OUT/'local-backtrack/selected-result.json').exists():
 write(OUT/'selected-result.json',read(OUT/'local-backtrack/selected-result.json'))
 print(json.dumps(read(OUT/'selected-result.json')['joint_merit']),flush=True)
else:write(OUT/'result.json',dict(status='NO_COMBINED_PROGRESS',native_admission=False,accepted_physical_steps=0))
