from pathlib import Path
import gzip,json,os,subprocess,shutil,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-gentle-wing-transfer-20261110';B=R/'build/crow-v15-gentle-wing-transfer-20261110';H=R/'build/crow-v15-heavy-wing-route-20261110';base=json.loads((R/'artifacts/crow-v15-integrated-route-baseline-20261110/contract-bindings.json').read_text())['argv'];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ('gentle15','gentle30'):
 name='development-reserved-final-j-'+arm;cmd=base[:]
 for k,v in [('--metallib',str(B/(arm+'.metallib'))),('--birdflow-navigation-course','reserved-final-j'),('--envs','8'),('--seed','3193706597'),('--steps','4500'),('--policy-pack',str(H/'landing.policypack')),('--base-policy-pack',str(H/'route.policypack')),('--state-trace',str(B/(name+'.tsv'))),('--state-trace-environment','0'),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 cmd.append('--birdflow-journey-teacher');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_SUBSTEP_FOOT_SENSOR':'1','CROW_SUBSTEP_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','CROW_TASK_STATE_DIAGNOSTIC':'1','PYTHONDONTWRITEBYTECODE':'1'};(A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'host_sha256':h(B/'libmetalrobo.dylib'),'shader_sha256':h(B/(arm+'.metallib')),'teacher_enabled':True,'note':'Privileged guide exploratory diagnostic, no autonomous claim'},indent=2)+'\n')
 with (A/(name+'.stdout')).open('wb') as out,(A/(name+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.PIPE);shutil.copyfileobj(p.stderr,gz);rc=p.wait()
 if rc:raise RuntimeError(name+' exit '+str(rc))
 for ext in ('tsv','json'):
  src=B/(name+'.'+ext)
  with src.open('rb') as inp,(A/(name+'.'+ext+'.gz')).open('wb') as out:
   with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as gz:shutil.copyfileobj(inp,gz)
 print(name,'exit',rc,flush=True)
