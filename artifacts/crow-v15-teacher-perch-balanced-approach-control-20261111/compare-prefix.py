from pathlib import Path
import gzip,json
ref=Path('artifacts/crow-v15-teacher-landing-flare-control-20261111/teacher-20261126')
cur=Path('artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126')
d=json.loads((ref/'run.stdout').read_text());ends=d['first_episode_navigation']['first_completion_step_by_environment']
def read(p):
 out={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1])
    if x['step']<=ends[x['lane']]:out[x['lane'],x['step']]=x
 return out
a=read(ref);b=read(cur);rows=[]
for e,end in enumerate(ends):
 miss=[t for t in range(1,end+1) if (e,t) not in a or (e,t) not in b]
 diff=[t for t in range(1,end+1) if (e,t) in a and (e,t) in b and a[e,t]!=b[e,t]]
 rows.append({'lane':e,'through_reference_waypoint_step':end,'missing_count':len(miss),'difference_count':len(diff),'first_difference':diff[0] if diff else None})
Path('artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/prefix-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
