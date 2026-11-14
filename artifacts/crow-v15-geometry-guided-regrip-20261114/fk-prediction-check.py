from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-geometry-guided-regrip-20261114');s={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='step' and int(x[2]) in (300,305,310,315,320,325):
   e,t=map(int,x[1:3]);a=x.index('q');b=x.index('v');s[e,t]=list(map(float,x[a+1:b]))
rows=[]
for t in (300,305,310,315,320,325):
 a,b=s[0,t],s[1,t];rows.append({'tick':t,'candidate_minus_control_root_position_m':[b[i]-a[i] for i in range(3)],'candidate_minus_control_root_quaternion':[b[i]-a[i] for i in range(3,7)],'candidate_minus_control_hip_knee_ankle_rad':[b[i]-a[i] for i in (14,15,16)],'candidate_minus_control_left_toe_rad':[b[i]-a[i] for i in range(23,31)]})
grid=json.loads(Path('artifacts/crow-v15-coordinated-grip-fk-grid-20261114/analysis.json').read_text());prediction=next(x for x in grid['rows'] if x['tick']==310)['smallest_displacement_examples'][0];geo=json.loads((A/'geometry-analysis.json').read_text())['placement'];v=next(x for x in geo['samples'] if x['zero_based_tick']==310);observed=[v['groups'][k]['min_box_gap_m'] for k in ('foot','front','rear')]
out={'rows':rows,'tick310_fixed_state_prediction_m':prediction['gaps_m'],'tick310_dynamic_observed_m':observed,'dynamic_minus_fixed_prediction_m':[x-y for x,y in zip(observed,prediction['gaps_m'])],'scope':'matched lanes, accepted state and pose; differences in body and other joints confound local FK prediction'};(A/'fk-prediction-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
