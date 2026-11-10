from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-damping-launch-conditional-20261110';B=R/'build/crow-v15-damping-launch-conditional-20261110';L=R/'build/crow-v15-causal-launch-20261110';H=R/'build/crow-v15-damping-launch-20261110';base=json.loads((A/'zero-probe.json').read_text())['argv'];idx=base.index('--crow-replay-pack');del base[idx:idx+2];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,seed,shader,observe=True):
 cmd=list(base)
 for k,v in [('--metallib',str(shader)),('--envs','8'),('--steps','400'),('--seed',str(seed)),('--state-trace',str(B/(name+'.tsv'))),('--state-trace-environment','1')]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(H),'CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 if observe:env['CROW_TASK_STATE_DIAGNOSTIC']='1'
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'task_state_observer':observe,'library_sha256':h(H/'libmetalrobo.dylib'),'shader_sha256':h(shader),'primary_sha256':h(B/'zero-primary.policypack'),'base_sha256':h(B/'zero-base.policypack'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal'),'plan_sha256':h(A/'plan.md')},indent=2)+'\n')
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600);(A/(name+'.stdout')).write_text(p.stdout)
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
x=run('observer-off',20261126,L/'push-trigger.metallib',False);y=run('20261126-push-trigger',20261126,L/'push-trigger.metallib');a=samples(x);b=samples(y);common=set(a)&set(b);diff=max((abs(x-y) for k in common for n in ('q','v') for x,y in zip(a[k][n],b[k][n])),default=None);(A/'observer-parity.json').write_text(json.dumps({'matched_samples':len(common),'max_abs_state_difference':diff},indent=2)+'\n');print('observer parity',len(common),diff,flush=True)
if len(common)!=32 or diff!=0.0:raise SystemExit('observer failed state parity')
for seed in (20261126,18647371,18647393):
 for arm in ('push-trigger','push-timer','control-trigger'):
  if seed==20261126 and arm=='push-trigger':continue
  run(str(seed)+'-'+arm,seed,L/(arm+'.metallib'))
