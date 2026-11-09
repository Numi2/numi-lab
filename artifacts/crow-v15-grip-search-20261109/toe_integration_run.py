from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'toe-integration';name='toe-integration-neutral'
x=json.loads((A/'symmetric-launch-wings-on-push-bindings.json').read_text());cmd=x['argv']
for k,v in [('--envs','1'),('--state-trace-environment','0'),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
x['argv']=cmd;x['observer_sha256']=hashlib.sha256((C/'libmetalrobo.dylib').read_bytes()).hexdigest();x['plan_sha256']=hashlib.sha256((A/'toe-integration-plan.md').read_bytes()).hexdigest();(A/(name+'-bindings.json')).write_text(json.dumps(x,indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
for suffix,data in [('stderr',p.stderr.encode()),('replay.json',(B/(name+'.json')).read_bytes() if (B/(name+'.json')).exists() else b''),('trace.tsv',(B/(name+'.tsv')).read_bytes() if (B/(name+'.tsv')).exists() else b'')]:
 with (A/(name+'.'+suffix+'.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
(A/(name+'-return.json')).write_text(json.dumps({'returncode':p.returncode,'stderr_tail':p.stderr[-4000:]},indent=2)+'\n');print('return',p.returncode,'stderr tail',p.stderr[-1000:])
