from pathlib import Path
import gzip,json,sys
R=Path('artifacts');A=R/'crow-v15-slewed-foot-settling-20261112';P=R/'crow-v15-negative-ankle-matched-control-20261112';seed=int(sys.argv[1]);paths={'previous':P/f'candidate-{seed}-rebound-phase/run.stderr.gz','trial':A/f'candidate-{seed}-rebound-phase/run.stderr.gz'}
def read(path):
 d={};bits={}
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);assert (e,t) not in d;d[e,t]=list(map(float,z[3:]));assert len(d[e,t])==40
   if z[0]=='CROW_SENSOR_BITS':
    t,e=map(int,z[1:3]);bits[e,t]=int(z[3])
 return d,bits
old,_=read(paths['previous']);new,bits=read(paths['trial']);assert len(old)==len(new)==800 and len(bits)==800
rows=[]
for e in range(4,8):
 for side,(ki,ai) in enumerate(((14,15),(17,18))):
  found=[]
  for t in range(202,301):
   dk=new[e,t][ki]-old[e,t][ki];da=new[e,t][ai]-old[e,t][ai]
   if max(abs(dk),abs(da))>1e-6:
    b=bits[e,t-1];found.append({'step':t,'knee_target_delta_rad':dk,'ankle_target_delta_rad':da,'previous_recorded_valid':bool(b&(1<<20)),'previous_recorded_rear':bool(b&(1<<(21+2*side))),'previous_recorded_front':bool(b&(1<<(22+2*side)))})
  rows.append({'lane':e,'side':['left','right'][side],'first_target_difference':found[0] if found else None,'differing_steps':len(found)})
(A/f'target-activation-{seed}.json').write_text(json.dumps({'seed':seed,'rows':rows,'scope':'Targets compared with same-lane prior trajectories. Once trajectories diverge, differences do not isolate the added command. Prior recorded sensor sample is a timing proxy, not proof of the exact consumption boundary.'},indent=2)+'\n')
for r in rows:print(seed,r['lane'],r['side'],r['first_target_difference'])
