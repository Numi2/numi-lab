#!/usr/bin/env python3
import json,struct
from pathlib import Path
import numpy as np
ns={}
s=Path('docs/evidence/rod-scaled-primal-20261108/vector_full_newton.py').read_text().rsplit('\nspec=[',1)[0]
exec(s,ns)
make=ns['make_case'];solve=ns['solve'];H=ns['H']
spec=[(3,'straight'),(9,'straight'),(17,'straight'),(9,'rotated'),(9,'curved'),(17,'curved'),(9,'moving'),(9,'moving_endpoint'),(9,'soft_endpoint'),(9,'interior_soft')]
blob=bytearray();cases=[]
for n,kind in spec:
    p,hard,soft,rest,k,m=make(n,kind,np.float32)
    target=np.zeros((17,4),dtype=np.float32);nodes=np.zeros((17,4),dtype=np.float32)
    nodes[:n,:3]=p;nodes[:n,3]=m
    for i,t in hard.items():target[i,:3]=t;target[i,3]=-1
    for i,t,s in soft:
        if target[i,3]!=0:raise RuntimeError('duplicate attachment')
        target[i,:3]=t;target[i,3]=s
    reference=solve(kind,n,np.float64,True)
    if reference['status']!='converged':raise RuntimeError('reference failed '+kind)
    for arr in (nodes,target):blob.extend(arr.astype('<f4').tobytes())
    blob.extend(struct.pack('<4f4I',float(rest),float(k),H,1e-6,n,0,0,0))
    cases.append(dict(nodes=n,case=kind,positions=reference['x'].tolist(),
        reactions=reference['reactions'],reference_residual_N=reference['grad']))
out=Path('docs/evidence/rod-scaled-primal-20261108')
(out/'metal-input.bin').write_bytes(blob)
(out/'metal-reference.json').write_text(json.dumps(dict(cases=cases),indent=2,sort_keys=True)+'\n')
print('cases',len(cases),'input_bytes',len(blob),'input_stride',len(blob)//len(cases))
