#!/usr/bin/env python3
"""Nine preregistered free-end cases compared with independent native-FP64 equations."""
import gzip, json, math, subprocess, sys
from pathlib import Path
import numpy as np

probe, shader, native, output = sys.argv[1:]
out=Path(output)
L=.11758755785232791
R=2e-5
RHO=1100.
g=np.array([0.,float(np.float32(-9.81)),0.])

def invoke(args, text):
    r=subprocess.run(args,input=text,text=True,capture_output=True,timeout=30)
    if r.returncode or r.stderr:
        raise RuntimeError(f"exit={r.returncode} stderr={r.stderr!r} stdout={r.stdout!r}")
    return r.stdout

def make(n,h,seed):
    rng=np.random.default_rng(seed)
    q=np.zeros((n,3));q[:,0]=np.linspace(0,L,n)
    q+=rng.uniform(-.002,.002,(n,3))*(L/(n-1))
    v=rng.uniform(-.004,.004,(n,3))
    t=rng.uniform(-.02,.02,n-1)
    w=rng.uniform(-.01,.01,n-1)
    return dict(n=n,h=float(np.float32(h)),seed=seed,q=q,v=v,theta=t,omega=w)

def line(c,q,t):
    return ' '.join(format(float(x),'.17g') for x in [c['n'],*q.ravel(),*t])

def gradient(row,n,k=1):
    a=np.array(row[f'bendGradient{k}'],dtype=np.float64)
    a[:3*n]+=np.array(row[f'stretchGradient{k}'])
    a[3*n:]+=np.array(row[f'twistGradient{k}'])
    return a

def native_rows(c,states):
    data='\n'.join(line(c,q,t) for q,t in states)+'\n'
    text=invoke([native],data)
    rows=[json.loads(x) for x in text.splitlines()]
    if len(rows)!=len(states):raise ValueError('native row count mismatch')
    return rows

def mass(c):
    n=c['n']; ell=L/(n-1)
    m=np.full(n,RHO*np.pi*R**2*ell);m[[0,-1]]*=.5
    inertia=np.full(n-1,RHO*np.pi*R**4/2*ell)
    return m,inertia

def residual(c,q,t,v,w,row,k=1):
    n=c['n'];h=c['h'];m,I=mass(c);grad=gradient(row,n,k)
    rv=v-c['v']-h*g+h*grad[:3*n].reshape(n,3)/m[:,None]
    rw=w-c['omega']+h*grad[3*n:]/I
    return rv,rw

def reference(c):
    n=c['n']; N=4*n-1; M=3*n;h=c['h'];m,I=mass(c)
    diag=np.r_[np.repeat(m/h**2,3),I/h**2]; scale=1/np.sqrt(diag)
    d=h*c['v']+h*h*g; dt=h*c['omega']
    accepted_shifts=[]
    def state_and_gradient(dd,tt):
        q=c['q']+dd; t=c['theta']+tt
        row=native_rows(c,[(q,t)])[0]
        grad=gradient(row,n)+diag*np.r_[(dd-h*c['v']-h*h*g).ravel(),tt-h*c['omega']]
        return q,t,row,grad
    for iteration in range(25):
        q=c['q']+d;t=c['theta']+dt
        states=[(q,t)]
        for j in range(N):
            eps=1e-6 if j<M else 1e-4
            for sign in (1.,-1.):
                qq=q.copy();tt=t.copy()
                if j<M:qq[j//3,j%3]+=sign*eps
                else:tt[j-M]+=sign*eps
                states.append((qq,tt))
        rows=native_rows(c,states);v=d/h;w=dt/h
        rv,rw=residual(c,q,t,v,w,rows[0])
        err=max(float(np.max(np.abs(rv))),float(np.max(np.abs(rw))))
        J=np.empty((N,N))
        for j in range(N):
            eps=1e-6 if j<M else 1e-4
            J[:,j]=(gradient(rows[1+2*j],n)-gradient(rows[2+2*j],n))/(2*eps)
        J+=np.diag(diag);J=(J+J.T)*.5
        A=scale[:,None]*J*scale[None,:]
        eig=float(np.min(np.linalg.eigvalsh(A)))
        if err<=1e-4:
            if eig<=0:return dict(status='nonpositive_final_curvature',eigenvalue=eig,iterations=iteration+1)
            return dict(status='converged',q=q,v=v,theta=t,omega=w,iterations=iteration+1,residual=err,eigenvalue=eig,shifts=accepted_shifts)
        full=gradient(rows[0],n)+diag*np.r_[(d-h*c['v']-h*h*g).ravel(),dt-h*c['omega']]
        merit=float(np.linalg.norm(scale*full))
        updated=False
        for shift in (0.,.01,.1,1.,10.,100.,1000.):
            matrix=A+shift*np.eye(N)
            try:
                np.linalg.cholesky(matrix)
                direction=scale*np.linalg.solve(matrix,-scale*full)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(direction).all():continue
            for k in range(20):
                alpha=2.**(-k)
                nd=d+alpha*direction[:M].reshape(n,3);nt=dt+alpha*direction[M:]
                _,_,_,nextgrad=state_and_gradient(nd,nt)
                if float(np.linalg.norm(scale*nextgrad))<merit:
                    d=nd;dt=nt;accepted_shifts.append(shift);updated=True;break
            if updated:break
        if not updated:return dict(status='line_search_failure',iterations=iteration+1,eigenvalue=eig,shifts=accepted_shifts)
    return dict(status='iteration_limit',iterations=25,shifts=accepted_shifts)

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(str(type(x)))

cases=[]
for i,n in enumerate((3,5,8)):
    for j,h in enumerate((.0005,.001,.002)):
        cases.append(make(n,h,92101+i*3+j))
(out.parent/(out.name+'-inputs.json')).write_text(json.dumps(cases,default=jsonable,indent=2,allow_nan=False)+'\n')
results=[]
for c in cases:
    try:
        vals=[c['n'],30,c['h'],*c['q'].ravel(),*c['v'].ravel(),*c['theta'],*c['omega']]
        a=json.loads(invoke([probe,shader],' '.join(format(float(x),'.17g') for x in vals)+'\n'))
        b=reference(c)
        item=dict(n=c['n'],h=c['h'],seed=c['seed'],actual=a,reference=b)
        good=a['published'] and a['status']==0 and a['gpu']==0 and a['reactions']==0 and b['status']=='converged'
        if good:
            n=c['n'];q=np.array(a['q']);v=np.array(a['v']);t=np.array(a['theta']);w=np.array(a['omega'])
            row=native_rows(c,[(q,t)])[0]
            rv,rw=residual(c,q,t,v,w,row,1);rv0,rw0=residual(c,q,t,v,w,row,0)
            m,_=mass(c)
            momentum=np.sum(m[:,None]*(v-c['v']-c['h']*g),axis=0)
            neg=rv.copy();neg.flat[0]+=math.copysign(.002,neg.flat[0] if neg.flat[0] else 1)
            metrics=dict(position=float(np.max(np.abs(q-b['q']))),velocity=float(np.max(np.abs(v-b['v']))),
                twist=float(np.max(np.abs(t-b['theta']))),twist_rate=float(np.max(np.abs(w-b['omega']))),
                linear=float(np.max(np.abs(rv))),angular=float(np.max(np.abs(rw))),
                linear_agreement=float(np.max(np.abs(rv-rv0))),angular_agreement=float(np.max(np.abs(rw-rw0))),
                momentum=float(np.max(np.abs(momentum))),negative_detected=bool(np.max(np.abs(neg))>.001))
            good=(metrics['position']<=1e-6 and metrics['velocity']<=.001 and metrics['twist']<=1e-4 and metrics['twist_rate']<=.001
                and metrics['linear']<=.001 and metrics['angular']<=.001 and metrics['linear_agreement']<=.0001
                and metrics['angular_agreement']<=.0001 and metrics['momentum']<=1e-12 and metrics['negative_detected'])
            item['metrics']=metrics
        item['passed']=bool(good)
    except Exception as e:
        item=dict(n=c['n'],h=c['h'],seed=c['seed'],passed=False,error=type(e).__name__+': '+str(e))
    results.append(item)
    print(json.dumps({'seed':c['seed'],'passed':item['passed'],'error':item.get('error'),'reference':item.get('reference',{}).get('status')},allow_nan=False),flush=True)
(out.parent/(out.name+'-results.json')).write_text(json.dumps(results,default=jsonable,indent=2,allow_nan=False)+'\n')
sys.exit(0 if len(results)==9 and all(x['passed'] for x in results) else 1)
