from pathlib import Path
import gzip,json,sys,math
A=Path('artifacts/crow-v15-toe-regrip-fk-balance-20261113');old=Path('artifacts/crow-v15-toe-release-regrip-20261111/trace.log.gz'); now=A/'trace.log.gz';data={};ref={};hips={};iks={};events={}
def scan(path,store):
 with gzip.open(path,'rt') as f:
  for s in f:
   z=s.split()
   if not z:continue
   k=z[0]
   if k in ('step','control','event'):
    e,t=map(int,z[1:3]);store[e,t,k]=z[3:]
    if path==now and k=='event':events[e,t]=z[3:]
   elif path==now and k=='hiptarget':
    e,t,qi=map(int,z[1:4]);assert (e,t,qi) not in hips; hips[e,t,qi]=list(map(float,z[4:]))
   elif path==now and k=='ik':
    e,t,side=map(int,z[1:4]);assert (e,t,side) not in iks;iks[e,t,side]=list(map(float,z[4:]))
scan(now,data);scan(old,ref);analysis=json.loads((A/'analysis.json').read_text()); rows=[]
for e in range(8):
 finish=next((t for t in range(350) if (e,t) in events and int(events[e,t][8])==1),None);end=finish if finish is not None else 349
 missing=[];different=[]
 for t in range(end+1):
  for k in ('step','control'):
   current=data.get((e,t,k));same_run=data.get((e%2,t,k));prior=ref.get((e%2,t,k))
   if current is None or same_run is None or prior is None:missing.append((t,k));continue
   if current!=same_run or current!=prior:different.append((t,k))
 hip_missing=[];hip_bad=[];ik_missing=[];ik_bad=[];gain=[0,0,.25,.25,.5,.5,1,1][e]
 for t in range(350):
  for side,qi in enumerate((14,17)):
   v=hips.get((e,t,qi))
   if v is None:hip_missing.append((t,qi))
   elif len(v)!=3 or not all(map(math.isfinite,v)) or not(v[1]-1e-5<=v[0]<=v[2]+1e-5):hip_bad.append((t,qi,v))
   v=iks.get((e,t,side))
   if v is None:ik_missing.append((t,side));continue
   foot,goal,jac,corr=v; want=max(-.1,min(.1,gain*(goal-foot)/jac)) if finish is not None and t>=finish and abs(jac)>.005 else 0
   if not all(map(math.isfinite,v)) or abs(corr-want)>1e-5:ik_bad.append((t,side,v,want))
 row=analysis['lanes'][e];passed=row['pass'] and not missing and not different and not hip_missing and not hip_bad and not ik_missing and not ik_bad and analysis['missing_substeps']==0 and analysis['bad_substeps']==0 and not analysis['errors'];rows.append({'lane':e,'return_step_zero_based':finish,'pre_return_missing':len(missing),'pre_return_differences':len(different),'hip_target_missing':len(hip_missing),'hip_target_out_of_range':len(hip_bad),'hip_target_first_bad':hip_bad[:1],'ik_missing':len(ik_missing),'ik_invalid':len(ik_bad),'ik_first_bad':ik_bad[:1],'previous_grip_screen_pass':row['pass'],'passed':passed})
out={'rows':rows,'any_pass':any(r['passed'] for r in rows),'scope':'Constructed development perch, not airborne landing. Effort and full physical integration remain unqualified.'};(A/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
