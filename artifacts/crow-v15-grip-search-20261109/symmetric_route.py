from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for course,seed in [('reserved-final-i',4038889268),('reserved-final-j',3193706597)]:
 for arm in ['control','push']:
  name='symmetric-route-'+course+'-'+arm
  binding=json.loads((A/('symmetric-launch-wings-on-'+arm+'-bindings.json')).read_text());cmd=binding['argv']
  for k,v in [('--birdflow-navigation-course',course),('--seed',str(seed)),('--envs','2'),('--steps','3000'),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
  binding['argv']=cmd;binding['plan_sha256']=h(A/'symmetric-route-plan.md');(A/(name+'-bindings.json')).write_text(json.dumps(binding,indent=2)+'\n')
  env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
  p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600);(A/(name+'.stdout')).write_text(p.stdout)
  for suffix,data in [('stderr',p.stderr.encode()),('replay.json',(B/(name+'.json')).read_bytes() if (B/(name+'.json')).exists() else b''),('trace.tsv',(B/(name+'.tsv')).read_bytes() if (B/(name+'.tsv')).exists() else b'')]:
   with (A/(name+'.'+suffix+'.gz')).open('wb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
  print(name,'exit',p.returncode,flush=True)
  if p.returncode: print(p.stderr[-2000:],flush=True)
