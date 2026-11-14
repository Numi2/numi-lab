from pathlib import Path
import gzip,json,sys
p=Path('artifacts/crow-v15-actor-command-flight-20261114')/sys.argv[1];records={};prev={};changes=[]
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0]=='CROW_PRONATION_EFFORT':
   t,sub,e=map(int,x[1:4]);v=list(map(float,x[4:]));
   if len(v)!=14:continue
   for side,j in enumerate((0,7)):
    r={'lane':e,'interval':t,'substep':sub,'side':side,'effort_Nm':v[j],'target_rad':v[j+1],'position_rad':v[j+2],'speed_rad_s':v[j+4],'absolute_error_rad':abs(v[j+1]-v[j+2])}
    key=(e,side)
    if key not in records or r['absolute_error_rad']>records[key]['absolute_error_rad']:records[key]=r
  elif x[0]=='CROW_DRIVE_TARGET':
   t,e=int(x[1]),int(x[2]);v=list(map(float,x[3:]));
   if len(v)<12:continue
   for side,j in enumerate((10,11)):
    key=(e,side)
    if key in prev:
     t0,v0=prev[key]
     if t==t0+1 and t>=500:changes.append({'lane':e,'interval':t,'side':side,'previous_target':v0,'target':v[j],'absolute_interval_change_rad':abs(v[j]-v0)})
    prev[key]=(t,v[j])
rows=sorted(records.values(),key=lambda r:(r['lane'],r['side']));top=sorted(changes,key=lambda x:-x['absolute_interval_change_rad'])[:24];out={'worst_substep_by_lane_and_side':rows,'largest_interval_target_changes':top,'scope':'recorded pronation observer and drive target; maximum errors are over observer coverage'}
(p/'tracking-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
