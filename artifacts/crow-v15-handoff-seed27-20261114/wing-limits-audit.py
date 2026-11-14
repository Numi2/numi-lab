from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-handoff-seed27-20261114');rows=[]
for arm in ('push','hold'):
 d=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'))
 for e in range(8):
  state=[x for x in d['state'] if x['lane']==e and 371<=x['step']<=500];targets=[x for x in d['target'] if x[0]==e and 371<=x[1]<=500];assert len(state)==len(targets)==130
  q=[x['q'][i] for x in state for i in (9,10)];v=[x['v'][i] for x in state for i in (8,9)];cmd=[x[2+i] for x in targets for i in (8,9)]
  rows.append({'arm':arm,'lane':e,'min_target':min(cmd),'max_target':max(cmd),'min_accepted_q':min(q),'max_accepted_q':max(q),'max_abs_accepted_v':max(map(abs,v)),'accepted_state_limit_hit':any(x<=-1.349 or x>=1.349 for x in q),'accepted_velocity_limit_hit':any(abs(x)>=89.99 for x in v)})
out={'rows':rows,'scope':'Endpoint state values only; physical substep wing-angle and speed limits are not independently observed here. Target binding clamp and v15 actuator limits remain active.'};(A/'wing-limits-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
