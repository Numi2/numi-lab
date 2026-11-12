from pathlib import Path
import json,math,statistics,hashlib
A=Path('artifacts/crow-v15-aerodynamic-load-replay-20261112');data={}
for l in (A/'output.txt').read_text().splitlines():
 z=l.split();assert len(z)==12
 a,e,t=map(int,z[:3]);v=list(map(float,z[3:]));assert all(map(math.isfinite,v));key=a,e,t;assert key not in data;data[key]=v
assert len(data)==144
rows=[]
for e in range(6):
 times=sorted({t for a,i,t in data if a==0 and i==e}&{t for a,i,t in data if a==1 and i==e});assert times==list(range(500,1601,100))
 paired=[]
 for t in times:
  b=data[0,e,t];c=data[1,e,t];paired.append({'step':t,'baseline_left_force_N':b[:3],'baseline_left_yaw_Nm':b[3],'baseline_right_force_N':b[4:7],'baseline_right_yaw_Nm':b[7],'baseline_root_yaw_Nm':b[8], 'enhanced_left_force_N':c[:3],'enhanced_left_yaw_Nm':c[3],'enhanced_right_force_N':c[4:7],'enhanced_right_yaw_Nm':c[7],'enhanced_root_yaw_Nm':c[8], 'wing_yaw_delta_Nm':c[3]+c[7]-b[3]-b[7]})
 rows.append({'lane':e,'samples':paired,'mean_sampled_wing_yaw_delta_Nm':statistics.mean(x['wing_yaw_delta_Nm'] for x in paired),'max_abs_sampled_wing_yaw_delta_Nm':max(abs(x['wing_yaw_delta_Nm']) for x in paired)})
out={'rows':rows,'scope':'GPU replay of a reconstructed v15 program at sparse recorded control endpoints, not microstep integrals or measured bird aerodynamics; source parity to runtime must be independently checked. Root torque includes fuselage and conditional root yaw closure, not tail-induced torque.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows: print(x['lane'],'mean yaw delta Nm',round(x['mean_sampled_wing_yaw_delta_Nm'],7),'max abs',round(x['max_abs_sampled_wing_yaw_delta_Nm'],7))
