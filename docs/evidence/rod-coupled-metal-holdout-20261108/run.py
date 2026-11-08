#!/usr/bin/env python3
import json, math, subprocess, sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3]
p=Path(__file__).resolve().parent
cases=json.loads((p/'cases.json').read_text())
refs=json.loads((p/'reference.json').read_text())['cases']
assert len(cases)==len(refs)==12 and all(r['status']=='converged' and r['passed'] for r in refs)
probe,metal,native=sys.argv[1:]
def call(cmd, data):
    result=subprocess.run(cmd,input=data,text=True,capture_output=True,timeout=30)
    if result.returncode or result.stderr:
        raise RuntimeError(f'owner command failed {cmd[0]}: exit={result.returncode} stderr={result.stderr!r} stdout={result.stdout!r}')
    return json.loads(result.stdout, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def flat(x): return np.asarray(x,dtype=np.float64).reshape(-1).tolist()
def packed(c,cap):
    xs=[c['n'],cap]+sum((flat(c[k]) for k in ('q','v','theta','omega','target')),[])
    return ' '.join(format(x,'.17g') for x in xs)+'\n'
def native_metrics(c,a):
    n=c['n']; vals=[n]+flat(a['q'])+flat(a['theta'])
    nr=call([native], ' '.join(format(x,'.17g') for x in vals)+'\n')
    h=.001;ell=.11758755785232791/(n-1); radius=2e-5;rho=1100
    mass=np.full(n,rho*np.pi*radius**2*ell);mass[[0,-1]]*=.5
    inert=np.full(n-1,rho*np.pi*radius**4/2*ell)
    rv=[];rw=[]
    for k in (0,1):
        bend=np.array(nr[f'bendGradient{k}']); force=(np.array(nr[f'stretchGradient{k}'])+bend[:3*n]).reshape(n,3)
        torque=bend[3*n:]+np.array(nr[f'twistGradient{k}'])
        rv.append(np.array(a['v'])-np.array(c['v'])+h*force/mass[:,None])
        rw.append(np.array(a['omega'])-np.array(c['omega'])+h*torque/inert)
    linear=float(np.max(np.abs(rv[1][1:-1])));angular=float(np.max(np.abs(rw[1])))
    lagree=float(np.max(np.abs((rv[1]-rv[0])[1:-1])));aagree=float(np.max(np.abs(rw[1]-rw[0])))
    negative=rv[1][1:-1].copy();v=negative.flat[0];negative.flat[0]=v+math.copysign(.002,v if v else 1.)
    negative_angle=rw[1].copy();z=negative_angle[0];negative_angle[0]=z+math.copysign(.002,z if z else 1.)
    return dict(linear=linear,angular=angular,linear_agreement=lagree,angular_agreement=aagree,negative_detected=bool(np.max(np.abs(negative))>.001),angular_negative_detected=bool(np.max(np.abs(negative_angle))>.001))
for cap in (30,):
    for idx,(c,ref) in enumerate(zip(cases,refs)):
        a=call([probe,metal],packed(c,cap)); item={'cap':cap,'index':idx,'seed':c['seed'],'actual':a}
        pub=a['status']==0 and a['gpu']==0 and a['published'] and a['states']==1 and a['reactions']==2
        if a['published'] != pub: raise RuntimeError(f'inconsistent publication {cap}/{idx}: {a}')
        item['accepted']=pub
        if pub:
            for key in ('q','v','theta','omega','reaction'):
                if not np.isfinite(np.array(a[key])).all():raise RuntimeError(f'nonfinite {cap}/{idx}/{key}')
            err={k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(ref[k])))) for k in ('q','v','theta','omega')}
            ferror=np.linalg.norm(np.asarray(a['reaction'])-np.asarray(ref['reaction']),axis=1)
            flimit=np.maximum(1e-6,.01*np.linalg.norm(np.asarray(ref['reaction']),axis=1))
            m=native_metrics(c,a); item.update(errors=err,force_errors=ferror.tolist(),force_limits=flimit.tolist(),momentum=m)
            item['physical_pass']=bool(err['q']<=1e-6 and err['v']<=.001 and err['theta']<=.0001 and err['omega']<=.001 and np.all(ferror<=flimit) and m['linear']<=.001 and m['angular']<=.001 and m['linear_agreement']<=.0001 and m['angular_agreement']<=.0001 and m['negative_detected'] and m['angular_negative_detected'])
        else:item['physical_pass']=False
        print(json.dumps(item,sort_keys=True,allow_nan=False),flush=True)
