#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
base=Path('docs/evidence/rod-scaled-primal-20261108')
ns={};exec((base/'vector_full_newton.py').read_text().rsplit('\nspec=[',1)[0],ns)
make=ns['make_case'];evaluate=ns['evaluate']
refs=json.loads((base/'metal-reference.json').read_text())['cases']
raw=(base/'metal-output.bin').read_bytes()
if len(raw)!=len(refs)*560:raise ValueError('output length mismatch')
outputs=np.frombuffer(raw,dtype='<f4').reshape((-1,35,4)); rows=[]
for i,ref in enumerate(refs):
 n=ref['nodes'];kind=ref['case'];o=outputs[i];q=o[:n,:3].astype(np.float64);reactions=o[17:17+n,:3].astype(np.float64)
 info=o[34];status=int(info[2]);iterations=int(info[0]);working=float(info[1])
 expected=np.array(ref['positions']);poserr=float(np.max(np.linalg.norm(q-expected,axis=1)))
 p,hard,soft,rest,k,m=make(n,kind,np.float64)
 _,g,_,_=evaluate(q,p,soft,rest,k,m,False)
 free=[j for j in range(n) if j not in hard]
 residual=float(np.max(np.linalg.norm(g[free],axis=1)))
 errors={};limits={}
 for key,value in ref['reactions'].items():
  node=int(key.split(':')[1]);err=float(np.linalg.norm(reactions[node]-value))
  limit=max(1e-6,.01*float(np.linalg.norm(value)));errors[key]=err;limits[key]=limit
 hard_error=max(float(np.linalg.norm(q[j]-v)) for j,v in hard.items())
 passed=bool(status==0 and np.all(np.isfinite(o)) and poserr<=1e-6 and residual<=1e-5 and all(errors[key]<=limits[key] for key in errors) and hard_error<=1e-6)
 rows.append(dict(nodes=n,case=kind,status=status,iterations=iterations,working_residual_N=working,
  published_residual_N=residual,position_error_m=poserr,hard_target_error_m=hard_error,
  reaction_error_N=errors,reaction_limit_N=limits,passed=passed))
result=dict(cases=rows,all_pass=all(r['passed'] for r in rows))
print(json.dumps(result,indent=2,sort_keys=True))
raise SystemExit(0 if result['all_pass'] else 1)
