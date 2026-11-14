from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-earlier-foot-descent-20261114');times={270,278,284,287,290,293,294,297,300};data={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x or x[0] not in ('step','event','balance','pose','placement','jointtarget','rightHipCommand'):continue
  e,t=int(x[1]),int(x[2])
  if e not in (0,1) or t not in times:continue
  d=data.setdefault((e,t),{})
  if x[0]=='step':
   qx=x.index('q');vx=x.index('v');q=list(map(float,x[qx+1:vx]));v=list(map(float,x[vx+1:]));d.update(root_position=q[:3],right_toe_q=q[31:39],right_toe_v=v[30:38])
  elif x[0]=='event':d['up_and_opposing_impulses_Ns']=list(map(float,x[5:11]))
  elif x[0]=='balance':d['balance_relative_state']=list(map(float,x[5:]))
  elif x[0]=='pose' and x[3]=='crow_right_foot':d['stance_foot_position']=list(map(float,x[4:7]))
  elif x[0]=='placement':d['placement_state']=list(map(float,x[3:]))
  elif x[0]=='jointtarget':d.setdefault('return_joint_targets',{})[x[3]]=float(x[4])
  elif x[0]=='rightHipCommand':d['stance_hip_target']=float(x[3])
out={'samples':[{'lane':e,'zero_based_interval':t,**d} for (e,t),d in sorted(data.items())],'scope':'accepted interval samples; contact/state epochs not independently synchronized'}
(A/'transition-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('samples',len(out['samples']))
