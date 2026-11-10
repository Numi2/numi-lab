from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-student-landing-eval-20261110/j-student-landing')
with gzip.open(str(p)+'.json.gz','rt') as f:frames=json.load(f)['payload']['frames']
with gzip.open(str(p)+'.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   z=json.loads(line.split(' ',1)[1]);t=z['step']
   if t==4433:break
errors={str(k):max(abs(a-b) for a,b in zip(z['q'],frames[k-1]['q'])) for k in (t-1,t,t+1)}
r={'support_step':t,'max_q_error_by_replay_step':errors,'matched_replay_step':min(errors,key=lambda k:errors[k]),'explanation':'Text support floats round to 9 significant digits. Support sample t corresponds to accepted replay frame t-1 in this trace.'};Path('artifacts/crow-v15-alternating-foot-release-20261110/alignment.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
