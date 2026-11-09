from pathlib import Path
import gzip,hashlib,json,os,subprocess,tempfile
R=Path.cwd();A=R/'artifacts/crow-v15-physics-runtime-20261109';E=R/'artifacts/crow-v15-contact-slip-20261109';report={}
with tempfile.TemporaryDirectory(prefix='crow-native-restore-') as t:
 T=Path(t);out=T/'runtime';restore=subprocess.run(['python3',str(A/'restore_runtime.py'),str(out)],cwd=R,text=True,capture_output=True,check=True);(A/'restore-check.log').write_text(restore.stdout)
 cmd=json.loads((E/'materialized-late-route-reserved-final-i-bindings.json').read_text())['argv'];cmd[0]=str(out/'metalrobo_task_rollout')
 for k,v in [('--steps','350'),('--metallib',str(out/'boundary-handoff.metallib')),('--state-trace',str(T/'trace.tsv')),('--crow-replay-pack',str(T/'replay.json'))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(out),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 got=json.loads((T/'replay.json').read_bytes())['payload']['frames'];want=json.load(gzip.open(E/'materialized-late-route-reserved-final-i.replay.json.gz','rt'))['payload']['frames'][:350];keys=('q','v','body_states');same=len(got)==len(want) and all(all(a[k]==b[k] for k in keys) for a,b in zip(got,want));report={'restored_runtime_physical_replay_equal':same,'frames':len(got),'scope':'same macmini, actor packs, exposed layout and seed; physical states only'}
 if not same:raise RuntimeError('restored runtime differs in physical replay')
(A/'restore-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
