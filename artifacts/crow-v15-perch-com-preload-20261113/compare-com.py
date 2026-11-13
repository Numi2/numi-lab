from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-perch-com-preload-20261113'); measurements={}; computed={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='com':
   e,t=map(int,z[1:3]);measurements[e,t]=tuple(map(float,z[3:5]))
with gzip.open(A/'fk-com.tsv.gz','rt') as f:
 for line in f:
  z=line.split();e,t=map(int,z[:2]);computed[e,t]=(float(z[3]),float(z[5]))
missing=set(measurements)^set(computed);errs=[(abs(measurements[k][0]-computed[k][0]),abs(measurements[k][1]-computed[k][1])) for k in set(measurements)&set(computed)];out={'logged_count':len(measurements),'fk_count':len(computed),'missing_count':len(missing),'max_com_x_error_m':max((x for x,y in errs),default=None),'max_com_vx_error_m_s':max((y for x,y in errs),default=None),'passed':not missing and bool(errs) and max(x for x,y in errs)<2e-6 and max(y for x,y in errs)<2e-6,'scope':'Independent executable compares accepted-state FK values; same underlying kinematics implementation and model; no biological claim.'};(A/'fk-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
