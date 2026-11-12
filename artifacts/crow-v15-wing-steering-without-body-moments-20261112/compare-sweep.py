from pathlib import Path
import gzip,json,statistics
p=Path('artifacts/crow-v15-wing-steering-without-body-moments-20261112');s={}
for arm in ('baseline','enhanced'):
 with gzip.open(p/(arm+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);t=x['step'];e=x['lane']
    if t>=500:s[arm,e,t]=x['q'][7:9]
rows=[]
for e in range(6):
 ts=sorted({t for a,i,t in s if a=='baseline' and i==e}&{t for a,i,t in s if a=='enhanced' and i==e})
 diff=[[s['enhanced',e,t][i]-s['baseline',e,t][i] for i in range(2)] for t in ts]
 rows.append({'lane':e,'matched_steps':ts,'mean_left_sweep_delta_rad':statistics.mean(d[0] for d in diff),'mean_right_sweep_delta_rad':statistics.mean(d[1] for d in diff),'final_delta_rad':diff[-1]})
out={'rows':rows,'limits':'sampled joint angles only, not commanded torque or aerodynamic force; divergence can include closed-loop trajectory effects'}
(p/'sweep-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['lane'],round(x['mean_left_sweep_delta_rad'],5),round(x['mean_right_sweep_delta_rad'],5),x['final_delta_rad'])
