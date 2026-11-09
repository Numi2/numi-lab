from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys
R=Path.cwd();A=R/'artifacts/crow-v15-branch-perch-20261109';B=R/'build/crow-v15-branch-perch-20261109';O=R/'artifacts/crow-v15-grip-search-20261109'
mode=sys.argv[1] if len(sys.argv)>1 else 'smoke'
cases=[('training',20261126,350)] if mode=='smoke' else [('reserved-final-i',4038889268,3000),('reserved-final-j',3193706597,3000)]
for course,seed,steps in cases:
 for arm in ('active','sensor-zero'):
  name=mode+'-'+course+'-'+arm;x=json.loads((O/'toe-envelope-neutral-bindings.json').read_text());cmd=x['argv']
  for k,v in [('--metallib',str(B/(arm+'.metallib'))),('--policy-pack',str(B/'landing.policypack')),('--base-policy-pack',str(B/'route.policypack')),('--birdflow-navigation-course',course),('--seed',str(seed)),('--envs','1'),('--steps',str(steps)),('--state-trace-environment','0'),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
  h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();x.update({'argv':cmd,'shader_sha256':h(B/(arm+'.metallib')),'library_sha256':h(B/'libmetalrobo.dylib'),'plan_sha256':h(A/'plan.md'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')});(A/(name+'-bindings.json')).write_text(json.dumps(x,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
  p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600);(A/(name+'.stdout')).write_text(p.stdout)
  for suffix,data in [('stderr',p.stderr.encode()),('replay.json',(B/(name+'.json')).read_bytes() if (B/(name+'.json')).exists() else b''),('trace.tsv',(B/(name+'.tsv')).read_bytes() if (B/(name+'.tsv')).exists() else b'')]:
   with (A/(name+'.'+suffix+'.gz')).open('wb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
  print(name,'exit',p.returncode,flush=True)
  if p.returncode:print(p.stderr[-3000:],flush=True);raise SystemExit(1)
if mode=='smoke':
 a=json.loads((B/'smoke-training-active.json').read_text())['payload']['frames'];b=json.loads((B/'smoke-training-sensor-zero.json').read_text())['payload']['frames'];equal=len(a)==len(b)==350 and all(x['q']==y['q'] and x['v']==y['v'] for x,y in zip(a,b));(A/'smoke-comparison.json').write_text(json.dumps({'qv_exact':equal,'steps':len(a)},indent=2)+'\n');print('identical q/v',equal,flush=True)
 if not equal:raise SystemExit('Smoke mismatch')
