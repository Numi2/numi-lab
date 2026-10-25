from pathlib import Path
import json,subprocess,hashlib,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
read=lambda p:json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
OUT=H/'face-connected-continuation-195-run001';OUT.mkdir(exist_ok=False)
state=read(H/'face-soft-193-run001/selected-result.json')
for i in range(3):
 rd=OUT/f'round-{i+1:03d}';rd.mkdir()
 target=rd/'targets'
 cmd=[PY,'/tmp/numi-human-geometry-slot.py',PY,str(H/'nearest-face-union-190-as-run002.py'),'--candidate',state['candidate'],'--surface-report',state['surface_report'],'--output-dir',str(target)]
 st=time.monotonic()
 with(rd/'targets.log').open('x')as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert cp.returncode==0
 cfg=dict(state=state,bary_npz=str(target/'targets.npz'),bary_json=str(target/'report.json'),out=str(rd/'fit'))
 write(rd/'config.json',cfg)
 with(rd/'fit.log').open('x')as f:cp=subprocess.run([PY,str(H/'run_face_soft_proposal_193.py'),str(rd/'config.json')],stdout=f,stderr=subprocess.STDOUT)
 assert cp.returncode==0
 if not(rd/'fit/selected-result.json').exists():
  write(rd/'result.json',dict(status='NO_COMBINED_PROGRESS',wall_seconds=time.monotonic()-st,native_admission=False));break
 state=read(rd/'fit/selected-result.json');write(rd/'result.json',state);write(OUT/'selected-result.json',state)
 print(json.dumps(dict(round=i+1,joint_merit=state['joint_merit'],wall_seconds=time.monotonic()-st)),flush=True)
write(OUT/'completed.json',dict(status='BOUNDED_CONTINUATION_FINISHED',final_candidate=state['candidate'],native_admission=False,accepted_physical_steps=0))
