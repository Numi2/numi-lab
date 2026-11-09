from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-flight-envelope-20261109')
out={}
for name in ['yaw-negative','yaw-zero','yaw-positive']:
 raw=json.loads((A/(name+'-analysis.json')).read_text())
 ss=[json.loads(line.split(' ',1)[1]) for line in gzip.open(A/(name+'.stderr.gz'),'rt') if line.startswith('CROW_FLIGHT_SAMPLE ')]
 rows=[]
 for lane in range(32):
  end=raw['first_episode']['steps_by_environment'][lane]
  xs=sorted((s for s in ss if s['lane']==lane and 1000<=s['step']<=1500 and s['step']<=end),key=lambda v:v['step'])
  headings=[]
  for x in xs:
   q=x['q'];a,b,c,d=q[3:7]
   yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
   headings.append(yaw)
  rates=[math.atan2(math.sin(b-a),math.cos(b-a))/2 for a,b in zip(headings,headings[1:])]
  rows.append({'lane':lane,'steps':end,'physical_failure':raw['first_episode']['physical_failure_by_environment'][lane],'sample_count':len(xs),'mean_yaw_rate_rad_s':statistics.mean(rates) if len(rates)==5 else None})
 complete=all(r['sample_count']==6 and r['steps']==1600 for r in rows)
 out[name]={'failed_environment_steps':raw['failed_environment_steps'],'physical_failures':sum(r['physical_failure'] for r in rows),'complete':complete,'median_mean_yaw_rate_rad_s':statistics.median(r['mean_yaw_rate_rad_s'] for r in rows) if complete else None,'lanes':rows}
(A/'yaw-authority-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for name,x in out.items():
 print(name,'fail',x['physical_failures'],'complete',x['complete'],'median rate',x['median_mean_yaw_rate_rad_s'])
