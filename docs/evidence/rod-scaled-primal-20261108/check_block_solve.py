#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
s=Path('docs/evidence/rod-scaled-primal-20261108/vector_coupling.py').read_text().split('spec=[')[0]
d={};exec(s,d)
p,hard,soft,rest,k,m=d['make_case'](9,'curved',np.float64)
x=p.copy()
for i,v in hard.items(): x[i]=v
_,g,diag,off=d['evaluate'](x,p,soft,rest,k,m,False)
diag=diag/k;off=off/k;rhs=-g/k
for i in hard:
    diag[i]=np.eye(3);rhs[i]=0
    if i>0:off[i-1]=0
    if i<8:off[i]=0
full=d['make_full'](diag,off); b=rhs.copy()
expected=np.linalg.solve(full,b.reshape(-1)).reshape((-1,3))
for i in range(1,9):
    factor=np.linalg.solve(diag[i-1].T,off[i-1]).T
    diag[i]-=factor@off[i-1];rhs[i]-=factor@rhs[i-1]
step=np.zeros_like(x);step[-1]=np.linalg.solve(diag[-1],rhs[-1])
for i in range(7,-1,-1):step[i]=np.linalg.solve(diag[i],rhs[i]-off[i]@step[i+1])
difference=float(np.max(np.abs(step-expected)))
residual=float(np.max(np.abs(full@step.flatten()-b.flatten())))
passed=difference<=1e-12 and residual<=1e-12
print(json.dumps(dict(passed=passed,maximum_step=float(np.max(np.abs(step))),
    block_dense_difference=difference,linear_residual=residual),indent=2,sort_keys=True))
raise SystemExit(0 if passed else 1)
