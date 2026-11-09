from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for old in ['launch-perturb-control','launch-perturb-push']:
 name=old+'-complete';bind=json.load(open(A/(old+'-bindings.json')));cmd=bind['argv'];
 for k,v in [('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'source_binding':old+'-bindings.json','observer_plan_sha256':h(A/'observer-plan.md'),'observer_sha256':h(C/'libmetalrobo.dylib')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-2000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 new=json.loads(raw)['payload']['frames'];prior=json.load(gzip.open(A/(old+'.replay.json.gz'),'rt'))['payload']['frames'];same=len(new)==len(prior) and all(all(a[k]==b[k] for k in ('q','v','body_states')) for a,b in zip(new,prior))
 report={'source':old,'frames':len(new),'physical_state_replay_equal':same,'scope':'selected lane q/v and body states; does not establish parity of every lane'};(A/(name+'-parity.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
 if not same:raise RuntimeError('observer changed selected-lane physical states')
