from pathlib import Path
import gzip,json,subprocess,itertools
A=Path('artifacts/crow-v15-leg-travel-kinematics-20261110');p=Path('artifacts/crow-v15-toe-height-bracket-20261110/development-reserved-final-j.stderr.gz');sample=None
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1])
   if x['lane']==0 and x['step']==3400:sample=x;break
assert sample is not None
q=sample['q'];lines=[]
for idx,(h,k,a) in enumerate(itertools.product([-.2,0,.2],repeat=3)):
 v=q[:]
 for j,d in enumerate([h,k,a,h,k,a]):v[14+j]+=d
 lines.append(f'pose{idx}_h{h:+.1f}_k{k:+.1f}_a{a:+.1f} '+' '.join(map(str,v))+'\n')
(A/'input.txt').write_text(''.join(lines));r=subprocess.run(['build/crow-v15-landing-kinematics-20261110/probe'],input=''.join(lines),text=True,capture_output=True,check=True);(A/'full-output.txt').write_text(r.stdout)
rows=[]
for line in r.stdout.splitlines():
 parts=line.split()
 if parts[1]!='-1':continue
 p=next(z for z in parts if z.startswith('crow_left_foot:')).split(':')[1].split(',');rows.append({'pose':parts[0],'x':float(p[0]),'z':float(p[2])})
base=next(z for z in rows if '_h+0.0_k+0.0_a+0.0' in z['pose'])
for z in rows:z['dx_m']=z['x']-base['x'];z['dz_m']=z['z']-base['z']
rows.sort(key=lambda z:z['dz_m']);out={'source_lane':0,'source_step':3400,'base':base,'candidates':rows,'scope':'static FP64 geometry only, not dynamics, contact, drive limits, or biology'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('base',base);print('lowest',rows[:5])
