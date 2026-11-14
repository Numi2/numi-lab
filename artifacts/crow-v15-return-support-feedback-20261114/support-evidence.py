from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-rear-contact-hip-advance-20261114/trace.log.gz'); rows=[]
with gzip.open(p,'rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='event' and int(x[1])==0 and 283<=int(x[2])<359:
   t=int(x[2]);up,front,rear=map(float,(x[6],x[9],x[10]))
   if up>.001 and not (front>.000001 and rear<-.000001):rows.append({'tick':t,'up_Ns':up,'front_x_Ns':front,'rear_x_Ns':rear})
out={'total_intervals':76,'loaded_but_opposing_gate_false':len(rows),'rows':rows,'scope':'last recorded control-step contact summaries; not proof of grip or simultaneous substep/state timing'};Path('artifacts/crow-v15-return-support-feedback-20261114/support-evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
