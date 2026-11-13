from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-active-clearance-regrip-20261113');r={0:[],1:[]}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='liftfeedback':r[int(x[1])].append((int(x[2]),bool(int(x[3])),float(x[4]),float(x[5])))
o={}
for e,z in r.items():
 a=[x for x in z if x[1]];inc=[(p,q) for p,q in zip(z,z[1:]) if q[3]>p[3]+1e-8 and q[1]];o[str(e)]={'samples_in_release':len(z),'feedback_active_intervals':len(a),'first_active_zero_based':a[0][0] if a else None,'peak_lift_rad':max((x[3] for x in z),default=None),'max_feedback_increment_rad':max((q[3]-p[3] for p,q in inc),default=None),'feedback_active_above_clearance_threshold':sum(x[2]>=.0025 for x in a),'scope':'control-state target and sensor; not substep contact effort'}
(A/'lift-feedback-check.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
