from pathlib import Path
import gzip,hashlib,json,subprocess
A=Path('artifacts/crow-v15-platform-release-continuation-20261112')
old=Path('artifacts/crow-v15-localized-route-autonomous-eval-20261111/autonomous-20261126')
new=A/'autonomous-20261126-extended'
ends=json.loads((old/'run.stdout').read_text())['first_episode_navigation']['steps_by_environment']
def read(f):
 hs={(e,k):hashlib.sha256() for e in range(8) for k in ('state','contact')};n={(e,k):0 for e in range(8) for k in ('state','contact')}
 with gzip.open(f,'rt') as z:
  for line in z:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];k='state'
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=line.split(None,4);t,sub,e=map(int,a[1:4]);k='contact'
   else: continue
   if e<8 and t<=ends[e]:hs[e,k].update(line.encode());n[e,k]+=1
 return {f'{e}-{k}':{'count':n[e,k],'sha256':hs[e,k].hexdigest()} for e in range(8) for k in ('state','contact')}
parts=sorted(old.glob('run.stderr.gz.part[0-9][0-9][0-9]'));p=subprocess.Popen(['cat',*[str(x) for x in parts]],stdout=subprocess.PIPE)
a=read(p.stdout);assert p.wait()==0
p=subprocess.Popen(['cat',*[str(x) for x in sorted(new.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE);b=read(p.stdout);assert p.wait()==0
r={'source':a,'extended':b,'identical':{k:a[k]==b[k] for k in a},'note':'Only first-episode state and full contact-substep lines through original termination/horizon are compared; per-lane episode ends differ after physical failure.'}
(A/'prefix-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['identical'],indent=2))
