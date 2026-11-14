from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-contact-load-transfer-20261114')
lanes={0:[],1:[]}
with gzip.open(p/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='returntransfer':
   lane,t,flag=int(x[1]),int(x[2]),int(x[3]);lanes[lane].append({'interval':t,'latch':flag,'previous_interval_upward_impulse_Ns':float(x[4])})
m=json.loads((p/'matched-audit.json').read_text());first=min(x['first_difference_zero_based'] for x in m['rows']);hit=next(x for x in lanes[1] if x['latch']);previous=lanes[1][hit['interval']-1]
r={'control_never_latched':all(not x['latch'] for x in lanes[0]),'first_candidate_latch':hit,'previous_candidate_sample':previous,'first_logged_state_or_target_difference':first,'difference_not_before_latch':first>=hit['interval'],'interpretation':'Control goal is calculated before latch in this loop; logged contact and state epochs are not independently synchronized.'}
(p/'latch-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
