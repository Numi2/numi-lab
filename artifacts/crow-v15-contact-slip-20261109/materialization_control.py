from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';O=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-contact-slip-20261109/materialized';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cmd=json.loads((O/'feedback_live-bindings.json').read_text())['argv']
for k,v in [('--state-trace',str(B/'control.tsv')),('--crow-replay-pack',str(B/'control.json'))]:cmd[cmd.index(k)+1]=v
(A/'materialization-control-bindings.json').write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'materialization-control-plan.md'),'library_sha256':h(B/'libmetalrobo.dylib'),'baseline_binding':'../crow-v15-ground-contact-20261109/feedback_live-bindings.json'},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/'materialization-control.stdout').write_text(p.stdout)
with (A/'materialization-control.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-3000:])
raw=(B/'control.json').read_bytes()
with (A/'materialization-control.replay.json.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
frames=json.loads(raw)['payload']['frames'];prior=json.load(gzip.open(O/'feedback_live.replay.json.gz','rt'))['payload']['frames'];data=json.loads(p.stdout);keys=('q','v','body_states','actor_observation','reward');different={k:sum(x[k]!=y[k] for x,y in zip(frames,prior)) for k in keys};first={k:next((x['step'] for x,y in zip(frames,prior) if x[k]!=y[k]),None) for k in keys};r={'frames':len(frames),'different_frame_count':different,'first_different_step':first,'failed_environment_steps':data['failed_environment_steps'],'physical_failure_count':data['first_episode_navigation']['physical_failure_count'],'scope':'selected lane, one exposed start; other tasks not assessed'};(A/'materialization-control-summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
