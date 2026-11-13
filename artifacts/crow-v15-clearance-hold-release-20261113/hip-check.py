from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-clearance-hold-release-20261113');out={}
for e in (0,1):
 cmds=[];pos=[];speed=[]
 with gzip.open(A/'trace.log.gz','rt') as f:
  for l in f:
   x=l.split()
   if not x or x[0] not in ('hipcommand','step') or int(x[1])!=e:continue
   if x[0]=='hipcommand':cmds.append(float(x[3]))
   if x[0]=='step':
    qi=x.index('q');vi=x.index('v');q=list(map(float,x[qi+1:vi]));v=list(map(float,x[vi+1:]));pos.append(q[14]);speed.append(abs(v[13]))
 out[str(e)]={'command_count':len(cmds),'accepted_state_count':len(pos),'command_range_rad':[min(cmds),max(cmds)],'position_range_rad':[min(pos),max(pos)],'max_control_state_speed_rad_s':max(speed),'within_model_position_limits':all(-.850001<=x<=.850001 for x in cmds+pos),'scope':'control-state positions and targets; not a substep actuator-effort bound'}
(A/'hip-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
