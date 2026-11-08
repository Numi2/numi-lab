#!/usr/bin/env python3
import json,math,struct,sys
from pathlib import Path
import numpy as np
b=Path('docs/evidence/rod-scaled-primal-20261108');ns={}
exec((b/'vector_full_newton.py').read_text().rsplit('\nspec=[',1)[0],ns)
evaluate=ns['evaluate'];make_full=ns['make_full'];H=ns['H']

def refsolve(p,hard,soft,rest,k,m):
    n=len(p);x=p.copy()
    for i,v in hard.items():x[i]=v
    free=[i for i in range(n) if i not in hard]
    indices=np.array([3*i+j for i in free for j in range(3)])
    status='limit'
    for it in range(100):
        energy,g,diag,off=evaluate(x,p,soft,rest,k,m,True)
        residual=float(np.max(np.linalg.norm(g[free],axis=1)))
        if residual<=1e-8:status='converged';break
        full=make_full(diag,off);a=full[np.ix_(indices,indices)]
        ev=float(np.min(np.linalg.eigvalsh(a)))
        if ev<=0:a=a+np.eye(len(a))*(-ev+1e-8)
        step=np.zeros_like(x);step[free]=np.linalg.solve(a,-g[free].reshape(-1)).reshape((-1,3))
        accepted=False
        for j in range(32):
            trial=x+(2.**(-j))*step
            for i,v in hard.items():trial[i]=v
            next_energy,*_=evaluate(trial,p,soft,rest,k,m,False)
            if next_energy<=energy:
                if np.array_equal(x,trial):break
                x=trial;accepted=True;break
        if not accepted:status='stalled';break
    _,g,_,_=evaluate(x,p,soft,rest,k,m,False)
    reactions={f'hard:{i}':list(map(float,-g[i])) for i in hard}
    reactions.update({f'soft:{i}':list(map(float,stiffness*(x[i]-target))) for i,target,stiffness in soft})
    return dict(status=status,iterations=it+1,residual_N=residual,positions=x.tolist(),reactions=reactions)

if sys.argv[1]=='generate':
    rng=np.random.default_rng(20261109);cases=[];blob=bytearray()
    for index in range(12):
        n=(5,9,13,17)[index%4];t=np.linspace(0,1,n)
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        side=rng.normal(size=3);side-=axis*np.dot(axis,side);side/=np.linalg.norm(side)
        cross=np.cross(axis,side)
        translation=rng.uniform(-.01,.01,size=3)
        amp=rng.uniform(.0005,.004);amp2=rng.uniform(-.002,.002)
        p=translation[None,:]+.122*t[:,None]*axis+amp*np.sin(np.pi*t)[:,None]*side+amp2*np.sin(2*np.pi*t)[:,None]*cross
        p[n//2]+=rng.normal(size=3)*.00012
        hard={0:p[0].copy(),n-1:p[-1]+rng.normal(size=3)*.00002}
        soft=[]
        if index%2:soft.append((n//2,p[n//2]+rng.normal(size=3)*.00012,50.))
        rest=.11758755785232791/(n-1);k=2.3818718058345718/rest
        edge_mass=1100*math.pi*(.000020**2)*rest
        m=np.full(n,edge_mass);m[[0,-1]]*=.5
        reference=refsolve(p,hard,soft,rest,k,m)
        nodes=np.zeros((17,4),dtype=np.float32);targets=np.zeros((17,4),dtype=np.float32)
        nodes[:n,:3]=p;nodes[:n,3]=m
        for i,v in hard.items():targets[i,:3]=v;targets[i,3]=-1
        for i,v,stiffness in soft:targets[i,:3]=v;targets[i,3]=stiffness
        blob.extend(nodes.astype('<f4').tobytes());blob.extend(targets.astype('<f4').tobytes())
        blob.extend(struct.pack('<4f4I',rest,k,H,1e-6,n,0,0,0))
        cases.append(dict(index=index,nodes=n,p=p.tolist(),hard={str(i):v.tolist() for i,v in hard.items()},
            soft=[dict(node=i,target=v.tolist(),stiffness=stiffness) for i,v,stiffness in soft],
            rest=rest,stiffness=k,mass=m.tolist(),reference=reference))
    (b/'metal-heldout-input.bin').write_bytes(blob)
    (b/'metal-heldout-reference.json').write_text(json.dumps(dict(seed=20261109,cases=cases),indent=2,sort_keys=True)+'\n')
    print('cases',len(cases),'reference_qualified',sum(c['reference']['status']=='converged' for c in cases))
elif sys.argv[1]=='analyze':
    cases=json.loads((b/'metal-heldout-reference.json').read_text())['cases']
    raw=(b/'metal-heldout-output.bin').read_bytes()
    if len(raw)!=len(cases)*560:raise ValueError('output length')
    outputs=np.frombuffer(raw,dtype='<f4').reshape((-1,35,4));rows=[]
    for c,o in zip(cases,outputs):
        n=c['nodes'];q=o[:n,:3].astype(np.float64);f=o[17:17+n,:3].astype(np.float64)
        info=o[34];status=int(info[2]);ref=c['reference'];pos=float(np.max(np.linalg.norm(q-np.array(ref['positions']),axis=1)))
        hard={int(i):np.array(v) for i,v in c['hard'].items()}
        soft=[(v['node'],np.array(v['target']),v['stiffness']) for v in c['soft']]
        _,g,_,_=evaluate(q,np.array(c['p']),soft,c['rest'],c['stiffness'],np.array(c['mass']),False)
        free=[i for i in range(n) if i not in hard];local=float(np.max(np.linalg.norm(g[free],axis=1)))
        errors={};limits={}
        for name,force in ref['reactions'].items():
            j=int(name.split(':')[1]);errors[name]=float(np.linalg.norm(f[j]-force));limits[name]=max(1e-6,.01*float(np.linalg.norm(force)))
        passed=bool(ref['status']=='converged' and status==0 and np.all(np.isfinite(o)) and pos<=1e-6 and local<=1e-5 and all(errors[k]<=limits[k] for k in errors))
        rows.append(dict(index=c['index'],nodes=n,reference_status=ref['status'],gpu_status=status,
            iterations=int(info[0]),position_error_m=pos,published_residual_N=local,
            reaction_error_N=errors,reaction_limit_N=limits,passed=passed))
    result=dict(cases=rows,all_pass=all(r['passed'] for r in rows))
    print(json.dumps(result,indent=2,sort_keys=True));sys.exit(0 if result['all_pass'] else 1)
else:raise ValueError('generate or analyze')
