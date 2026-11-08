import json,math,sys
from collections import defaultdict
from pathlib import Path
p=Path(__file__).parent; name=sys.argv[1]; data=json.loads((p/name).read_text())['cases'];assert len(data)==36
out=defaultdict(lambda:{'cases':0,'mismatches':0,'max_error':0.,'max_observed':0.,'positive_sub_epsilon_rows':0,'false_zero_cases':0,'failures':0})
rows=[]
for c in data:
 v=c['v'];cx=0;cy=c['initial_y']; errs=[]
 for r in c['rows']:
  assert r['bodyA']==0 and r['bodyB']==1
  x,y,z=r['point']; rx,ry,rz=x-cx,y-cy,z
  point=[v[0]+v[4]*rz-v[5]*ry,v[1]+v[5]*rx-v[3]*rz,v[2]+v[3]*ry-v[4]*rx]
  equation=sum(a*b for a,b in zip(point,r['direction']))-r['target']+r['reg']*r['lambda']
  errs.append(abs(equation) if r['lambda']>0 else max(-equation,0.))
 independent=max(errs,default=0); error=abs(independent-c['solver_residual']);bucket=out[str(c['mass'])];bucket['cases']+=1
 bucket['max_error']=max(bucket['max_error'],error);bucket['max_observed']=max(bucket['max_observed'],independent)
 bucket['mismatches']+=error>2e-5;bucket['false_zero_cases']+=c['solver_residual']<1e-6 and independent>0.002
 bucket['positive_sub_epsilon_rows']+=sum(0<r['lambda']<=1e-7 for r in c['rows']);bucket['failures']+=not c['success'] or not c['repeat'] or c['contacts']!=2
 rows.append({'mass':c['mass'],'angle':c['angle'],'iterations':c['iterations'],'case':c['case'],'independent':independent,'reported':c['solver_residual'],'error':error,'impulses':[r['lambda'] for r in c['rows']]})
summary={'summary':dict(out),'rows':rows};(p/(name.removesuffix('.json')+'-analysis.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['summary'],indent=2))
