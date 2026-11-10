from pathlib import Path
import gzip,json,subprocess
A=Path('artifacts/crow-v15-approach-sensor-observer-20261110');p=A/'diagnostic-reserved-final-j-newstart.stderr.gz';lines=[]
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   d=json.loads(l.split(' ',1)[1]);s=d['step']
   if s in (2300,2400,2500,3000):lines.append('new_'+str(s)+' '+' '.join(map(str,d['q']))+'\n')
r=subprocess.run(['build/crow-v15-landing-kinematics-20261110/probe'],input=''.join(lines),capture_output=True,text=True,check=True)
out='\n'.join(l.split(' crow_left_front')[0] for l in r.stdout.splitlines() if ' -1 ' in l)+'\n'
(A/'joint-reference.log').write_text(out)
print(out,end='')
