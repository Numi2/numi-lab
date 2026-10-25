from pathlib import Path
import json,subprocess,hashlib,time,sys,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=Path(sys.argv[2]);OUT.mkdir(exist_ok=False);state=Path(sys.argv[1]);read=lambda p:json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(args,log):
 with Path(log).open('x')as f:r=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 assert r.returncode==0,(log,r.returncode)
for i in range(6):
 if shutil.disk_usage(OUT).free < 200*1024**2:
  write(OUT/'resource-stop.json',dict(status='DISK_HEADROOM_STOP',free_bytes=shutil.disk_usage(OUT).free,native_admission=False));break
 rd=OUT/f'round-{i+1:03d}';rd.mkdir();st=time.monotonic()
 run([PY,H/'run_whole_connected_prep_218.py',state,rd/'prep'],rd/'prep.log')
 prep=rd/'prep/prepared.json'
 run([PY,H/'run_whole_connected_joint_217.py',rd/'joint',prep,D/'D/connected-local-witness-proposal-214/propose_all_witness_clusters_214.py','bde46af1559e5884dd0c6f2521c82f3ead688a42c4c072aa0829b39424fdd53a'],rd/'joint.log')
 selected=rd/'joint/selected-result.json'
 if not selected.exists():write(rd/'result.json',dict(status='NO_COMBINED_PROGRESS',native_admission=False,accepted_physical_steps=0));break
 state=selected;r=read(state);write(OUT/'selected-result.json',r)
 print(json.dumps(dict(round=i+1,wall_seconds=time.monotonic()-st,joint_merit=r.get('joint_merit'),depth_merit=r.get('depth_merit'))),flush=True)
write(OUT/'completed.json',dict(status='BOUNDED_CONNECTED_CONTINUATION_FINISHED',final_state=str(state),native_admission=False,accepted_physical_steps=0))
