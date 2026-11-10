from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-heavy-wing-gait-20261110';B=R/'build/crow-v15-heavy-wing-gait-20261110';base=json.loads((R/'artifacts/crow-v15-damping-launch-conditional-20261110/20261126-push-trigger-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,seed,arm,observe=True):
 cmd=list(base)
 for k,v in [('--policy-pack',str(B/'zero-primary.policypack')),('--base-policy-pack',str(B/'zero-primary.policypack')),('--metallib',str(B/(arm+'.metallib'))),('--steps','900'),('--seed',str(seed)),('--state-trace',str(B/(name+'.tsv')))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_TASK_STATE_DIAGNOSTIC':'1','PYTHONDONTWRITEBYTECODE':'1'}
 if observe:env['CROW_GAIT_TRACE']='1'
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'gait_observer':observe,'library_sha256':h(B/'libmetalrobo.dylib'),'shader_sha256':h(B/(arm+'.metallib')),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal'),'plan_sha256':h(A/'plan.md')},indent=2)+'\n')
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 trace=B/(name+'.tsv')
 if trace.exists():
  with (A/(name+'.trace.tsv.gz')).open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(trace.read_bytes())
 print(name,'exit',p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
 return p.stderr
def samples(text):return {(x['lane'],x['step']):x for l in text.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(l.split(' ',1)[1])]}
x=run('observer-off',38942573,'push-trigger',False);y=run('38942573-push-trigger',38942573,'push-trigger');a=samples(x);b=samples(y);common=set(a)&set(b);diff=max((abs(x-y) for k in common for n in ('q','v') for x,y in zip(a[k][n],b[k][n])),default=None);(A/'observer-parity.json').write_text(json.dumps({'matched_samples':len(common),'max_abs_state_difference':diff},indent=2)+'\n');print('observer parity',len(common),diff,flush=True)
if len(common)!=72 or diff!=0.0:raise SystemExit('observer failed state parity')
for seed in (38942573,38942609):
 for arm in ('push-trigger','control-trigger'):
  if seed==38942573 and arm=='push-trigger':continue
  run(str(seed)+'-'+arm,seed,arm)
