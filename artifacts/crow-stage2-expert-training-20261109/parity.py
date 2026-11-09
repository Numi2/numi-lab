from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-stage2-expert-training-20261109';B=R/'build/crow-stage2-expert-training-20261109'
old=R/'artifacts/crow-stage2-fresh-eval-20261109/stage2/held-out-a';r=json.loads((old/'results.json').read_text());cmd=r['argv'].copy()
for key,val in [('--metallib',str(R/'build/crow-stage2-head-experiment-20261109/stage2.metallib')),('--base-policy-pack',str(B/'combined-identity.policypack')),('--state-trace',str(B/'parity.tsv'))]:cmd[cmd.index(key)+1]=val
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','DYLD_LIBRARY_PATH':str(R/'build/crow-production-observer-20261109'),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
(A/'parity.stdout').write_text(p.stdout)
with (A/'parity.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-5000:])
def sample(lines):
 return {(z['lane'],z['step']):z for line in lines if line.startswith('CROW_FLIGHT_SAMPLE ') for z in [json.loads(line.split(' ',1)[1])]}
with gzip.open(old/'rollout.stderr.gz','rt') as f:a=sample(f)
b=sample(p.stderr.splitlines());first=json.loads(p.stdout)['first_episode_navigation'];ends=r['first_episode']['steps_by_environment']
keys=[k for k in a if k[1]<=ends[k[0]]]
bad=[k for k in keys if k not in b or a[k]['q']!=b[k]['q'] or a[k]['v']!=b[k]['v']]
report={'first_episode_equal':first==r['first_episode'],'compared_samples':len(keys),'qv_mismatches':len(bad),'first_mismatches':bad[:10],'argv':cmd,'shader_sha256':hashlib.sha256((R/'build/crow-stage2-head-experiment-20261109/stage2.metallib').read_bytes()).hexdigest(),'scope':'block-diagonal identity expert vs production stage2 actor; teacher off, first episodes'}
(A/'parity-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['first_episode_equal','compared_samples','qv_mismatches','first_mismatches']}),flush=True)
assert keys and not bad and report['first_episode_equal']
