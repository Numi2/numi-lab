from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd(); A=R/'artifacts/crow-support-observer-20261109'; B=R/'build/crow-support-observer-20261109'
cmd=json.loads((R/'artifacts/crow-route-teacher-v1-20261109/route-results.json').read_text())['argv']
for key,val in [('--metallib',str(R/'build/crow-route-teacher-v1-20261109/tracking.metallib')),('--policy-pack',str(R/'build/crow-extended-flight-20261109/collision-on/extended.policypack')),('--state-trace',str(B/'witness.tsv')),('--envs','16'),('--steps','1600')]:
 cmd[cmd.index(key)+1]=val
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480)
with (A/'witness.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
(A/'witness.stdout').write_text(p.stdout)
if p.returncode:raise RuntimeError(p.stderr[-5000:])
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
terms=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
report={'argv':cmd,'support_sample_count':len(samples),'lanes_with_perch_samples':sorted({x['lane'] for x in samples}),'first_sample':samples[0] if samples else None,'contact_terminations':len(terms),'qualification':'observer witness only; contacts here include crashes and do not establish foot support'}
(A/'witness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='argv'},indent=2))
