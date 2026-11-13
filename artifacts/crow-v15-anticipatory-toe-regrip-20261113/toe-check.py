from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-anticipatory-toe-regrip-20261113');state={0:[],1:[]};cmd={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='step':
   e=int(x[1]);qi=x.index('q');vi=x.index('v');q=list(map(float,x[qi+1:vi]));v=list(map(float,x[vi+1:]));state[e].append((int(x[2]),q[23:31],v[22:30],q[2]))
  elif x[0]=='control':cmd[int(x[1])].append((int(x[2]),float(x[4])))
r={}
for e in (0,1):
 c=cmd[e];s=state[e];fall=next((t for t,_,_,z in s if z<.1),None);incr=[max(0.,b[1]-a[1]) for a,b in zip(c,c[1:])];r[str(e)]={'control_samples':len(c),'states':len(s),'max_front_target_rad':max(x for _,x in c),'max_increase_per_interval_rad':max(incr),'target_cap_ok':len(c)==450 and max(x for _,x in c)<=1.050001,'rate_cap_ok':len(c)==450 and max(incr)<=(.045001 if e==0 else .060001),'max_accepted_left_toe_speed_rad_s':max(abs(v) for _,_,vs,_ in s for v in vs),'max_accepted_left_toe_speed_before_fall_rad_s':max(abs(v) for t,_,vs,z in s if fall is None or t<fall for v in vs),'accepted_speed_cap_samples_before_fall':sum(max(map(abs,vs))>=11.99 for t,_,vs,z in s if fall is None or t<fall),'accepted_speed_cap_samples_after_fall':sum(max(map(abs,vs))>=11.99 for t,_,vs,z in s if fall is not None and t>=fall),'scope':'accepted states only; no substep toe-speed or torque bound'}
(A/'toe-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
