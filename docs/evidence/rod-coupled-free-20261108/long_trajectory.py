#!/usr/bin/env python3
import ast,copy,json,math,subprocess,sys
from pathlib import Path
import numpy as np
probe,shader,native,prefix=sys.argv[1:]
p=Path(__file__).resolve().parent
source=ast.parse((p/'free_3d_globalized.py').read_text())
ns=dict(np=np,subprocess=subprocess,json=json,native=native,L=.11758755785232791,R=2e-5,RHO=1100.,g=np.array([0.,float(np.float32(-9.81)),0.]))
for name in ('invoke','line','gradient','native_rows','mass','residual','reference','jsonable'):
    f=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'independent-reference','exec'),ns)
invoke=ns['invoke']; native_rows=ns['native_rows'];mass=ns['mass'];residual=ns['residual'];reference=ns['reference'];jsonable=ns['jsonable']
seeds=(92101,92105,92109)
inputs=json.loads((p/'free-3d-global-1-inputs.json').read_text())
results=[]
for seed in seeds:
    initial=next(x for x in inputs if x['seed']==seed)
    for k in ('q','v','theta','omega'):initial[k]=np.array(initial[k],dtype=float)
    metal=copy.deepcopy(initial);ref=copy.deepcopy(initial);m,_=mass(initial);h=initial['h']
    initial_momentum=np.sum(m[:,None]*initial['v'],axis=0)
    for step in range(1,101):
        try:
            vals=[metal['n'],30,h,*metal['q'].ravel(),*metal['v'].ravel(),*metal['theta'],*metal['omega']]
            a=json.loads(invoke([probe,shader],' '.join(format(float(x),'.17g') for x in vals)+'\n'))
            b=reference(ref)
            record=dict(seed=seed,step=step,actual=a,reference=b)
            good=a['published'] and a['status']==0 and a['gpu']==0 and a['reactions']==0 and b['status']=='converged'
            if good:
                q=np.array(a['q']);v=np.array(a['v']);t=np.array(a['theta']);w=np.array(a['omega'])
                row=native_rows(metal,[(q,t)])[0]
                rv,rw=residual(metal,q,t,v,w,row,1);rv0,rw0=residual(metal,q,t,v,w,row,0)
                stepmom=np.sum(m[:,None]*(v-metal['v']-h*ns['g']),axis=0)
                cumulative=np.sum(m[:,None]*v,axis=0)-initial_momentum-step*h*np.sum(m)*ns['g']
                neg=rv.copy();neg.flat[0]+=math.copysign(.002,neg.flat[0] if neg.flat[0] else 1.)
                metrics=dict(position=float(np.max(np.abs(q-b['q']))),velocity=float(np.max(np.abs(v-b['v']))),
                    twist=float(np.max(np.abs(t-b['theta']))),twist_rate=float(np.max(np.abs(w-b['omega']))),
                    linear=float(np.max(np.abs(rv))),angular=float(np.max(np.abs(rw))),
                    linear_agreement=float(np.max(np.abs(rv-rv0))),angular_agreement=float(np.max(np.abs(rw-rw0))),
                    momentum=float(np.max(np.abs(stepmom))),cumulative_momentum=float(np.max(np.abs(cumulative))),
                    negative_detected=bool(np.max(np.abs(neg))>.001))
                good=(metrics['position']<=1e-6 and metrics['velocity']<=.001 and metrics['twist']<=1e-4 and metrics['twist_rate']<=.001
                    and metrics['linear']<=.001 and metrics['angular']<=.001 and metrics['linear_agreement']<=.0001
                    and metrics['angular_agreement']<=.0001 and metrics['momentum']<=1e-12
                    and metrics['cumulative_momentum']<=1e-11 and metrics['negative_detected'])
                record['metrics']=metrics
                if good:
                    for key,val in (('q',q),('v',v),('theta',t),('omega',w)):metal[key]=val
                    for key in ('q','v','theta','omega'):ref[key]=np.array(b[key])
            record['passed']=bool(good)
        except Exception as e:
            record=dict(seed=seed,step=step,passed=False,error=type(e).__name__+': '+str(e))
        results.append(record)
        print(json.dumps({'seed':seed,'step':step,'passed':record['passed'],'error':record.get('error')},allow_nan=False),flush=True)
        if not record['passed']:break
Path(prefix+'-results.json').write_text(json.dumps(results,default=jsonable,indent=2,allow_nan=False)+'\n')
sys.exit(0 if len(results)==300 and all(x['passed'] for x in results) else 1)
