from pathlib import Path
import json,subprocess,hashlib,importlib.util,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=H/'whole-connected-prep-216-run001';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(args,log):
 cmd=[PY,'/tmp/numi-human-geometry-slot.py',*map(str,args)];st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
state=read(H/'whole-witness-halfstep-215-run001/selected-result.json')
write(OUT/'input-state.json',state)
target=OUT/'targets'
run([PY,D/'H/nearest-face-union-190/build_nearest_face_union_targets_198.py','--candidate',state['candidate'],'--surface-report',state['surface_report'],'--output-dir',target],OUT/'targets.log')
spec=importlib.util.spec_from_file_location('prepare_185',H/'prepare_refined_proposal_185.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DRIVER=H/'refined-point-face-203-template/point_driver.py';emit0=m.emit
bn=target/'targets.npz';bj=target/'report.json'
def emit(t,d,ov,k):
 if k=='driver':ov={**ov,'BARY_TARGET_NPZ':bn,'BARY_TARGET_JSON':bj,'PIN':{**ov['PIN'],'bary_npz':sha(bn),'bary_json':sha(bj)}}
 return emit0(t,d,ov,k)
m.emit=emit
m.prepare(Path(state['candidate']),Path(state['point_summary']),Path(state['surface_report']),Path(state['parent_fit']),OUT/'proposal',True)
raw=OUT/'proposal/fit/attempt-001/scp-01-alpha-1p0/candidate.npz';assert raw.exists()
run([PY,D/'audit_connected_outer_geometry_130.py',raw,OUT/'raw-audit130'],OUT/'raw-audit130.log')
write(OUT/'prepared.json',dict(state=state,raw=str(raw),raw_sha256=sha(raw),pairs=str(OUT/'raw-audit130/surface-self-pairs.npy'),parent_fit=str(OUT/'proposal/fit/attempt-001/fit-report.json'),native_admission=False,accepted_physical_steps=0))
print((OUT/'prepared.json').read_text(),flush=True)
