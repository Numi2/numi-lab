from pathlib import Path
import gzip,hashlib,json,os,subprocess,shutil
R=Path.cwd();A=R/'artifacts/crow-v15-stride-balance-20261110';B=R/'build/crow-v15-hybrid-gait-causality-20261110';G=R/'build/crow-v15-stride-balance-20261110';H=R/'build/crow-v15-hybrid-route-probe-20261110';base=json.loads((R/'artifacts/crow-v15-gait-coordinated-confirmation-20261110/38942573-push-trigger-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ('reach34','reach36'):
 name='38942573-'+arm;cmd=list(base);shader=G/(arm+'.metallib')
 for k,v in [('--metallib',str(shader)),('--steps','900'),('--seed','38942573'),('--state-trace',str(G/(name+'.tsv'))),('--policy-pack',str(B/'zero-primary.policypack')),('--base-policy-pack',str(B/'zero-base.policypack'))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(H),'CROW_GAIT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_TASK_STATE_DIAGNOSTIC':'1','CROW_SUBSTEP_FOOT_SENSOR':'1','PYTHONDONTWRITEBYTECODE':'1'};(A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'library_sha256':h(H/'libmetalrobo.dylib'),'shader_sha256':h(shader),'zero_primary_sha256':h(B/'zero-primary.policypack'),'zero_base_sha256':h(B/'zero-base.policypack'),'teacher_enabled':False},indent=2)+'\n')
 with (A/(name+'.stdout')).open('wb') as out,(A/(name+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.PIPE);shutil.copyfileobj(proc.stderr,z);rc=proc.wait()
 p=G/(name+'.tsv')
 if p.exists():
  with p.open('rb') as x,(A/(name+'.trace.tsv.gz')).open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:shutil.copyfileobj(x,z)
 print(name,'exit',rc,flush=True)
