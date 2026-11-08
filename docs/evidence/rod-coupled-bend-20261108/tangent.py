#!/usr/bin/env python3
"""Coupled straight-rest bend/twist Hessian-vector comparison."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

p=Path(__file__).resolve().parent
L=0.11758755785232791
cases=[]
for idx in range(12):
    n=(3,5,9)[idx%3]; rng=np.random.default_rng(901+idx)
    axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
    x=[np.full(3,0.5 if idx%2 else 0.)]
    for _ in range(n-1):
        axis=axis+rng.normal(size=3)*.22; axis/=np.linalg.norm(axis)
        x.append(x[-1]+axis*L/(n-1)*rng.uniform(.97,1.03))
    x=np.array(x); t=rng.uniform(-1.4,1.4,n-1)
    v=np.r_[rng.normal(size=3*n)*1e-3,rng.normal(size=n-1)*.5]
    w=np.r_[rng.normal(size=3*n)*1e-3,rng.normal(size=n-1)*.5]
    cases.append((x,t,v,w))

def encode(x,t): return ' '.join(format(float(a),'.17g') for a in [len(x),*x.ravel(),*t])
lines=[]
for x,t,v,w in cases:
    n=len(x); m=3*n
    for h in (1e-3,5e-4):
        for sign in (1.,-1.): lines.append(encode(x+sign*h*v[:m].reshape(n,3),t+sign*h*v[m:]))
data='\n'.join(lines)+'\n'; (p/'tangent-inputs.txt').write_text(data)
run=subprocess.run([sys.argv[1]],input=data,capture_output=True,text=True)
(p/'tangent-native.stdout').write_text(run.stdout); (p/'tangent-native.stderr').write_text(run.stderr)
if run.returncode: raise RuntimeError(f'native exit {run.returncode}')
native=[json.loads(line) for line in run.stdout.splitlines()]
assert len(native)==48

def grad_and_hv(x,t,v,ei,gj,l):
    n=len(x); vp=v[:3*n].reshape(n,3); vt=v[3*n:]
    g=np.zeros_like(x); dg=np.zeros_like(x); gt=np.zeros(n-1); dgt=np.zeros(n-1)
    mn=1.
    for i in range(n-2):
        e=x[i+1]-x[i]; f=x[i+2]-x[i+1]
        de=vp[i+1]-vp[i]; df=vp[i+2]-vp[i+1]
        a=np.linalg.norm(e); b=np.linalg.norm(f); u=e/a; z=f/b
        da=u@de; db=z@df; du=(de-u*da)/a; dz=(df-z*db)/b
        c=u@z; dc=du@z+u@dz; mn=min(mn,float(c))
        d=t[i+1]-t[i]; dd=vt[i+1]-vt[i]; k=ei/l
        r=(1-c)/(1+c); dr=-2*dc/(1+c)**2
        fc=-2*k*(1+np.cos(d))/(1+c)**2
        dfc=2*k*np.sin(d)*dd/(1+c)**2+4*k*(1+np.cos(d))*dc/(1+c)**3
        uc=(z-c*u)/a; zc=(u-c*z)/b
        duc=(dz-dc*u-c*du)/a-uc*da/a
        dzc=(du-dc*z-c*dz)/b-zc*db/b
        ge=fc*uc; gf=fc*zc; dge=dfc*uc+fc*duc; dgf=dfc*zc+fc*dzc
        g[i]-=ge; g[i+1]+=ge-gf; g[i+2]+=gf
        dg[i]-=dge; dg[i+1]+=dge-dgf; dg[i+2]+=dgf
        gd=-k*r*np.sin(d)+gj/l*d
        dgd=-k*(dr*np.sin(d)+r*np.cos(d)*dd)+gj/l*dd
        gt[i]-=gd; gt[i+1]+=gd; dgt[i]-=dgd; dgt[i+1]+=dgd
    return np.r_[g.ravel(),gt],np.r_[dg.ravel(),dgt],mn

def native_gradient(record,n):
    b=np.array(record['bendGradient1']); b[3*n:]+=record['twistGradient1']; return b

def check(a,b,n):
    a=np.asarray(a);b=np.asarray(b);ab=np.abs(a-b)
    tolerance=np.r_[np.full(3*n,1e-10),np.full(n-1,1e-14)]
    gate=np.maximum(tolerance,.002*np.maximum(np.abs(a),np.abs(b)))
    return bool(np.all(ab<=gate)),float(np.max(ab/gate)),float(np.max(ab[:3*n])),float(np.max(ab[3*n:]))

rows=[]; negatives=0
for idx,(x,t,v,w) in enumerate(cases):
    n=len(x); m=3*n; rr=native[4*idx:4*idx+4]; ei,gj,l=rr[0]['EI'],rr[0]['GJ'],rr[0]['restLength']
    _,hv,mn=grad_and_hv(x,t,v,ei,gj,l); assert mn>=-.5
    n0=(native_gradient(rr[0],n)-native_gradient(rr[1],n))/(2e-3)
    n1=(native_gradient(rr[2],n)-native_gradient(rr[3],n))/(1e-3)
    checks=[check(n0,n1,n),check(hv,n1,n)]
    for h in (1e-3,5e-4):
        gp,_,_=grad_and_hv(x+h*v[:m].reshape(n,3),t+h*v[m:],v,ei,gj,l)
        gm,_,_=grad_and_hv(x-h*v[:m].reshape(n,3),t-h*v[m:],v,ei,gj,l)
        checks.append(check(hv,(gp-gm)/(2*h),n))
    _,hw,_=grad_and_hv(x,t,w,ei,gj,l)
    a=float(v@hw);b=float(w@hv); symmetry=abs(a-b)<=max(1e-16,1e-8*max(abs(a),abs(b)))
    vp=np.r_[v[:m],np.zeros(n-1)]; vt=np.r_[np.zeros(m),v[m:]]
    _,hvp,_=grad_and_hv(x,t,vp,ei,gj,l); _,hvt,_=grad_and_hv(x,t,vt,ei,gj,l)
    bad=np.r_[hvp[:m],hvt[m:]]
    negative=not check(bad,n1,n)[0]
    negatives+=negative
    rows.append(dict(index=idx,n=n,minimum_cosine=mn,checks=[dict(passed=a,max_gate_ratio=b,max_force_error_N=c,max_torque_error_Nm=d) for a,b,c,d in checks],symmetry_pass=bool(symmetry),symmetry_error_J=abs(a-b),negative_detected=bool(negative),passed=bool(all(q[0] for q in checks) and symmetry)))
result=dict(cases=rows,negative_detected=negatives,all_pass=bool(all(r['passed'] for r in rows) and negatives>0))
(p/'tangent-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_pass':result['all_pass'],'negative_detected':negatives,'failed_indices':[r['index'] for r in rows if not r['passed']]},indent=2))
sys.exit(0 if result['all_pass'] else 1)
