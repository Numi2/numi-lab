from pathlib import Path
import gzip,json,collections
p=Path('artifacts/crow-v15-rear-toe-impact-audit-20261112');src=Path('artifacts/crow-v15-crouched-walk-screen-20261112/candidate-20261127-rebound/run.stderr.gz');data=collections.defaultdict(dict);drive={};hits=collections.Counter()
with gzip.open(src,'rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_DRIVE_TARGET':
   t,e=map(int,z[1:3]);drive[e,t]=list(map(float,z[3:]));continue
  if z[0] not in ('CROW_TOE_V_SUBSTEP','CROW_TWO_TOE_SUBSTEP','CROW_Q_SUBSTEP'):continue
  t,k,e=map(int,z[1:4])
  if 201<=t<=300 and z[0]=='CROW_TOE_V_SUBSTEP':
   v=list(map(float,z[4:]));
   for i,x in enumerate(v):
    if abs(x)>=11.99:hits[e,i]+=1
  if e==4 and t in (276,277,278):data[t,k][z[0]]=z
sample=[]
for t,k in sorted(data):
 z=data[t,k];assert len(z)==3
 v=list(map(float,z['CROW_TOE_V_SUBSTEP'][4:]));q=list(map(float,z['CROW_Q_SUBSTEP'][4:]));c=list(map(float,z['CROW_TWO_TOE_SUBSTEP'][6:]));sample.append({'step':t,'substep':k,'rear_toe_proximal_q_rad':q[37],'rear_toe_proximal_velocity_rad_s':v[14],'rear_toe_distal_velocity_rad_s':v[15],'rear_toe_normal_impulse_Ns':c[19]+c[20],'right_foot_normal_impulse_Ns':c[3],'left_foot_normal_impulse_Ns':c[2],'rear_toe_position_target_rad':drive[4,t][36]})
out={'scope':'Illustrative retained used-seed lane 4, 1.25 ms substeps, not causality or biological validation. Target is per-control-step position drive, not an applied force. Toe normal impulse is sum of two segments. q and contacts are recorded at solver substeps; timestamps require established observer ordering. Coordinate speed cap is 12 rad/s and negative configured position limit is -0.3 rad. Ground contact forcing the toe toward the bound is an inference from these coincident samples, not a solver diagnosis.','speed_cap_hits_by_lane_and_toe':[{ 'lane':e,'toe_coordinate':i,'hits':n} for (e,i),n in sorted(hits.items())],'samples':sample}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'hits':out['speed_cap_hits_by_lane_and_toe'],'selected':[s for s in sample if s['step']==277 and s['substep'] in (0,9,10,11,15)]},indent=2))
