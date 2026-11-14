from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stance-toe-limit-audit-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114');out={}
for e in (0,1):
 states={};controls={};events={}
 for line in gzip.open(P/'trace.log.gz','rt'):
  z=line.split()
  if not z or len(z)<3 or z[0] not in ('step','control','event') or int(z[1])!=e:continue
  t=int(z[2])
  if z[0]=='step':
   i=z.index('q');j=z.index('v');states[t]={'q':list(map(float,z[i+1:j])),'v':list(map(float,z[j+1:]))}
  elif z[0]=='control':controls[t]=list(map(float,z[3:]))
  elif z[0]=='event':events[t]=list(map(float,z[3:]))
 samples=[]
 for t in (270,280,290,295,300,302,304,306,310,320):
  q=states[t]['q'];ev=events[t];samples.append({'zero_based_interval':t,'root_position_m':q[:3],'right_foretoe_joint_positions_rad':q[31:37],'right_rear_toe_joint_positions_rad':q[37:39],'right_rear_target_rad':controls[t][3],'right_fore_target_rad':controls[t][4],'right_front_signed_normal_Ns':ev[6],'right_rear_signed_normal_Ns':ev[7]})
 first=next((t for t in range(273,370) if sum(q>=1.199 for q in states[t]['q'][31:37])>=3),None)
 out[str(e)]={'samples':samples,'first_three_right_foretoe_joints_at_1_199_rad_zero_based':first}
(A/'analysis.json').write_text(json.dumps({'lanes':out,'scope':'Recorded endpoint q and signed normals, not proof of mechanical causation; 1.2 rad is the observed upper toe angle in this run.'},indent=2)+'\n');print(json.dumps(out,indent=2))
