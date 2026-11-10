from pathlib import Path
import gzip,json,statistics,math
A=Path('artifacts/crow-v15-vertical-rate-bench-20261110');data={}
for variant in ('baseline','damped'):
 name='18722533-'+variant;report=json.loads((A/(name+'-analysis.json')).read_text());first=report['first_episode'];vals={i:[] for i in range(16)};last={};closed=set()
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for line in f:
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   x=json.loads(line.split(' ',1)[1]);i=x['lane'];step=x['step']
   if i in last and step<last[i]:closed.add(i)
   last[i]=step
   if i not in closed and 1000<=step<=1500 and step<=first['steps_by_environment'][i]:vals[i].append(x['v'][2])
 lanes=[]
 for i,v in vals.items():lanes.append({'lane':i,'samples':len(v),'rms_vertical_m_s':math.sqrt(sum(x*x for x in v)/len(v)) if v else None,'max_abs_vertical_m_s':max(map(abs,v),default=None),'screen_pass':report['lanes'][i]['screen_pass']})
 data[variant]={'lanes':lanes,'physical_failures':sum(first['physical_failure_by_environment']),'failed_environment_steps':report['failed_environment_steps']}
r=[]
for i in range(16):
 a=data['baseline']['lanes'][i]['rms_vertical_m_s'];b=data['damped']['lanes'][i]['rms_vertical_m_s'];r.append((a-b)/a if a is not None and b is not None and a>0 else None)
valid=all(x is not None for x in r) and all(z['samples']==501 for k in data.values() for z in k['lanes']);med=statistics.median(r) if valid else None
out={'valid_full_observation_window':valid,'median_paired_fractional_rms_reduction':med,'paired_reductions':r,'arms':data,'prediction_pass':bool(valid and med>=.2 and all(a['physical_failures']==0 and a['failed_environment_steps']==0 and all(z['screen_pass'] for z in a['lanes']) for a in data.values()))}
(A/'paired-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('valid',valid,'median fractional improvement',med,'prediction pass',out['prediction_pass'])
