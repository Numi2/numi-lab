from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-damping-launch-20261110';B=R/'build/crow-v15-damping-launch-20261110';L=R/'build/crow-v15-causal-launch-20261110';base=json.loads((R/'artifacts/crow-v15-toe-free-flight-20261110/smoke-command.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_TASK_STATE_DIAGNOSTIC':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
def run(name,seed,shader,steps):
 cmd=list(base)
 for k,v in [('--metallib',str(shader)),('--envs','8'),('--steps',str(steps)),('--seed',str(seed)),('--policy-pack',str(R/'build/crow-v15-toe-damping-isolation-20261110/zero-4/zero.policypack')),('--state-trace',str(B/(name+'.tsv'))),('--state-trace-environment','1')]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'library_sha256':h(B/'libmetalrobo.dylib'),'shader_sha256':h(shader),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal'),'plan_sha256':h(A/'plan.md')},indent=2)+'\n')
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
s=run('observer-control',20261120,R/'build/crow-v15-toe-free-flight-20261110/tracking.metallib',400)
def samples(text):return {(x['lane'],x['step']):x for l in text.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(l.split(' ',1)[1])]}
a=samples(gzip.open(R/'artifacts/crow-v15-toe-damping-isolation-20261110/zero-4.stderr.gz','rt').read());b=samples(s);common=set(a)&set(b);diff=max((abs(x-y) for k in common for n in ('q','v') for x,y in zip(a[k][n],b[k][n])),default=None);(A/'observer-parity.json').write_text(json.dumps({'matched_samples':len(common),'max_abs_state_difference':diff},indent=2)+'\n');print('observer parity',len(common),diff,flush=True)
if len(common)!=32 or diff!=0.0:raise SystemExit('observer failed state parity')
for seed in (18647371,18647393):
 for arm in ('push-trigger','push-timer','control-trigger'):
  run(str(seed)+'-'+arm,seed,L/(arm+'.metallib'),400)
