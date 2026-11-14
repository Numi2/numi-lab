from pathlib import Path
import gzip,json,sys,math,collections
p=Path(sys.argv[1]);out={'directory':str(p),'run_result':json.loads((p/'run-result.json').read_text()),'meta':None,'state_counts':{},'first_nonfinite':[],'first_ground':{},'observer_errors':[]}
try:out['meta']=json.loads((p/'run.stdout').read_text()).get('first_episode_navigation')
except Exception as e:out['stdout_parse_error']=str(e)
counts=collections.Counter()
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e=x['lane'];counts[e]+=1
   if not all(math.isfinite(v) for v in x['q']+x['v']):out['first_nonfinite'].append([e,x['step']])
  elif line.startswith('CROW_GROUND_CONTACT '):
   x=line.split();out['first_ground'].setdefault(x[3],int(x[1]))
  elif 'OBSERVER_ERROR' in line or line.startswith('CROW_SUBSTEP_ERROR'):out['observer_errors'].append(line.strip())
out['state_counts']=dict(counts);(p/'initial-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
