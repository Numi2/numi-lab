from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-head-contact-fix-20261110';B=R/'build/crow-v15-head-contact-fix-20261110';old=R/'artifacts/crow-v15-latched-balance-20261110';data=json.loads((old/'layout-j-positive-bindings.json').read_text());cmd=data['argv']
for k,v in [('--policy-pack',str(B/'landing.policypack')),('--base-policy-pack',str(B/'route.policypack')),('--state-trace',str(B/'corrected.tsv')),('--crow-replay-pack',str(B/'corrected.json'))]:cmd[cmd.index(k)+1]=v
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();data.update({'argv':cmd,'library_sha256':h(B/'libmetalrobo.dylib'),'plan_sha256':h(A/'plan.md'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')});(A/'bindings.json').write_text(json.dumps(data,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600);(A/'stdout').write_text(p.stdout)
for suffix,content in [('stderr',p.stderr.encode()),('replay.json',(B/'corrected.json').read_bytes()),('trace.tsv',(B/'corrected.tsv').read_bytes())]:
 with (A/(suffix+'.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(content)
print('exit',p.returncode);raise SystemExit(p.returncode)
