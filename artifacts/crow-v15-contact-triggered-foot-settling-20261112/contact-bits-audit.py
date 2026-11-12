from pathlib import Path
import collections,gzip,json,sys
A=Path('artifacts/crow-v15-contact-triggered-foot-settling-20261112');seed=int(sys.argv[1]);D=A/f'candidate-{seed}-rebound-phase';bits={};toes={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_SENSOR_BITS':
   t,e=map(int,z[1:3]);assert (e,t) not in bits;bits[e,t]=int(z[3])
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4])
   if 201<=t<=300 and k==15:
    assert (e,t) not in toes;toes[e,t]=list(map(float,z[11:27]))
assert len(bits)==len(toes)==800 and bits.keys()==toes.keys()
rows=[]
for e in range(8):
 for side in (0,1):
  valid=rearonly=disagree=0;examples=[]
  for t in range(201,301):
   b=bits[e,t];isvalid=bool(b&(1<<20));rear=bool(b&(1<<(21+2*side)));front=bool(b&(1<<(22+2*side)));z=toes[e,t];base=8*side;rear_ref=any(z[base+i]>1e-7 for i in (6,7));front_ref=any(z[base+i]>1e-7 for i in (2,3));valid+=isvalid;rearonly+=isvalid and rear and not front
   if isvalid and (rear!=rear_ref or front!=front_ref):
    disagree+=1
    if len(examples)<3:examples.append({'step':t,'sensor_rear_front':[rear,front],'trace_rear_front':[rear_ref,front_ref]})
  rows.append({'lane':e,'side':['left','right'][side],'valid_samples':valid,'rear_only_samples':rearonly,'active_contact_vs_recorded_disagreements':disagree,'examples':examples})
(A/f'contact-bits-{seed}.json').write_text(json.dumps({'seed':seed,'rows':rows,'scope':'Sensor uses active contacts after publishing; trace uses final-substep required constraints. Different buffers and timing can disagree. This is not force confirmation at action application.'},indent=2)+'\n')
for r in rows:
 if r['lane']>=2:print(seed,r['lane'],r['side'],'rear-only',r['rear_only_samples'],'mismatches',r['active_contact_vs_recorded_disagreements'])
