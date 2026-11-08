#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

s=Path('docs/evidence/rod-scaled-primal-20261108/vector_stable_difference.py').read_text().rsplit('\nspec=[',1)[0]
fns={};exec(s,fns)
make_case=fns['make_case']; observed=fns['energy_change']; H=fns['H']
rng=np.random.default_rng(20261108)

def reference(d,trial,p,soft,rest,k,m):
    # Treat FP32-rounded input edge and soft-target offsets as the working
    # state; evaluate old/new term values independently in FP64.
    old=d.astype(np.float64);new=trial.astype(np.float64)
    inert=m.astype(np.float64)/float(np.float32(H)**np.float32(2))
    old_e=.5*np.sum(inert[:,None]*old*old);new_e=.5*np.sum(inert[:,None]*new*new)
    for i in range(len(d)-1):
        base=(p[i+1]-p[i]).astype(np.float64)
        a=base+old[i+1]-old[i];b=base+new[i+1]-new[i]
        old_e+=.5*float(k)*(np.linalg.norm(a)-float(rest))**2
        new_e+=.5*float(k)*(np.linalg.norm(b)-float(rest))**2
    for i,target,stiffness in soft:
        base=(p[i]-target).astype(np.float64)
        old_e+=.5*float(stiffness)*np.dot(base+old[i],base+old[i])
        new_e+=.5*float(stiffness)*np.dot(base+new[i],base+new[i])
    return float(new_e-old_e)

rows=[]
for n,kind in ((9,'curved'),(17,'curved'),(9,'interior_soft'),(9,'soft_endpoint')):
    p,hard,soft,rest,k,m=make_case(n,kind,np.float32)
    d=(rng.standard_normal((n,3))*3e-5).astype(np.float32)
    for i,target in hard.items():d[i]=target-p[i]
    for scale in (1e-6,1e-7,1e-8,1e-9,1e-10):
        direction=rng.standard_normal((n,3)).astype(np.float32)
        for i in hard:direction[i]=0
        for sign in (-1,1):
            trial=np.asarray(d+np.float32(sign*scale)*direction,dtype=np.float32)
            got=observed(d,trial,p,soft,rest,k,m)
            expected=reference(d,trial,p,soft,rest,k,m)
            error=abs(got-expected); limit=1e-15+.005*abs(expected)
            informative=abs(expected)>1e-13
            sign_ok=not informative or np.sign(got)==np.sign(expected)
            rows.append(dict(nodes=n,case=kind,scale_m=scale,sign=sign,
                actual_J=got,expected_J=expected,absolute_error_J=error,limit_J=limit,
                informative=bool(informative),passed=bool(error<=limit and sign_ok)))
    zero=observed(d,d,p,soft,rest,k,m)
    rows.append(dict(nodes=n,case=kind,scale_m=0,sign=0,actual_J=zero,expected_J=0,
        absolute_error_J=abs(zero),limit_J=0,informative=False,passed=bool(zero==0)))
negative_rejected=sum(1 for r in rows if r['informative'] and np.sign(-r['actual_J']) != np.sign(r['expected_J']))
result=dict(cases=rows,total=len(rows),passed_count=sum(r['passed'] for r in rows),
    informative_signs=sum(r['informative'] for r in rows),negative_control_rejected=negative_rejected,
    all_pass=all(r['passed'] for r in rows) and negative_rejected>0)
print(json.dumps(result,indent=2,sort_keys=True))
raise SystemExit(0 if result['all_pass'] else 1)
