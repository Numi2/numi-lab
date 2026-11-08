#!/usr/bin/env python3
"""FP64 research-only implicit straight-rest DER step, with native audit."""
import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

p=Path(__file__).resolve().parent
# Extract only the previously qualified analytical derivative, with no script side effects.
module=ast.parse((p/'tangent.py').read_text())
fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='grad_and_hv')
ns={'np':np}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'tangent.py','exec'),ns)
bend=ns['grad_and_hv']
L=.11758755785232791; EA=2.3818718058345718; R=2e-5; RHO=1100.; POISSON=.35; H=.001
EI=EA*R*R/4; GJ=EI/(1+POISSON); AREA=np.pi*R*R

def make_case(n,seed,bend_amp=None):
    if seed<0:
        axis=np.array([1.,0.,0.]); a=np.array([0.,1.,0.]); b=np.array([0.,0.,1.]); offset=np.zeros(3)
        amp=bend_amp; rng=np.random.default_rng(1)
    else:
        rng=np.random.default_rng(seed); axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        z=rng.normal(size=3);a=z-axis*(axis@z);a/=np.linalg.norm(a); b=np.cross(axis,a)
        offset=np.full(3,.5 if seed%2 else 0.);amp=.0004+.0001*(seed%5)
    t=np.linspace(0,1,n); q=offset+np.outer(.122*t,axis)+np.outer(amp*np.sin(np.pi*t),a)
    if seed>=0: q+=np.outer(.0002*np.sin(2*np.pi*t),b)
    v=np.zeros_like(q) if seed<0 else rng.normal(size=(n,3))*.002
    theta=np.zeros(n-1) if seed<0 else rng.uniform(-.35,.35,n-1)
    omega=np.zeros(n-1) if seed<0 else rng.uniform(-.08,.08,n-1)
    target=q[[0,-1]].copy()
    if seed>=0 and seed%2==0: target[-1]+=axis*2e-5+a*1e-5
    return dict(n=n,seed=seed,q=q,v=v,theta=theta,omega=omega,target=target)

cases=[make_case(3,-1,0.),make_case(3,-2,1e-7)] + [make_case((5,8,9)[i%3],1101+i) for i in range(10)]

def setup(c):
    n=c['n'];ell=L/(n-1); mass=np.full(n,RHO*AREA*ell);mass[[0,-1]]*=.5
    inertia=np.full(n-1,RHO*(np.pi*R**4/2)*ell)
    return ell,mass,inertia

def potential(q,t,ell):
    edge=np.diff(q,axis=0); lengths=np.linalg.norm(edge,axis=1); u=edge/lengths[:,None]
    stretch=.5*EA/ell*np.sum((lengths-ell)**2)
    c=np.sum(u[:-1]*u[1:],axis=1);dt=np.diff(t)
    b=EI/ell*np.sum((1-c)/(1+c)*(1+np.cos(dt)))
    tw=.5*GJ/ell*np.sum(dt*dt)
    return float(stretch+b+tw),float(np.min(c))

def grad_hv(q,t,direction,ell,mass,inertia,dpred,tpred):
    n=len(q);zero=np.zeros(3*n+n-1)
    gb,hb,minc=bend(q,t,direction,EI,GJ,ell)
    dx=direction[:3*n].reshape(n,3);td=direction[3*n:]
    g=gb.copy(); hv=hb.copy(); gq=g[:3*n].reshape(n,3);hq=hv[:3*n].reshape(n,3)
    for i in range(n-1):
        edge=q[i+1]-q[i]; length=np.linalg.norm(edge); u=edge/length; ext=length-ell
        f=EA/ell*ext*u; gq[i]-=f;gq[i+1]+=f
        d=dx[i+1]-dx[i]; df=EA/ell*(u*(u@d)+(ext/length)*(d-u*(u@d)))
        hq[i]-=df;hq[i+1]+=df
    gq[:]+=mass[:,None]/(H*H)*dpred; hq[:]+=mass[:,None]/(H*H)*dx
    g[3*n:]+=inertia/(H*H)*tpred;hv[3*n:]+=inertia/(H*H)*td
    return g,hv,minc

def energy_delta(c,d,t,nd,nt,ell,mass,inertia):
    delta=nd-d;td=nt-t
    ans=.5*np.sum(mass[:,None]/H**2*(2*(d-H*c['v'])*delta+delta*delta))
    ans+=.5*np.sum(inertia/H**2*(2*(t-H*c['omega'])*td+td*td))
    base=np.diff(c['q'],axis=0);old=np.diff(d,axis=0);dd=np.diff(delta,axis=0);e=base+old
    a=np.linalg.norm(e,axis=1);newlen=np.linalg.norm(e+dd,axis=1)
    dl=(2*np.sum(e*dd,axis=1)+np.sum(dd*dd,axis=1))/(a+newlen)
    ans+=.5*EA/ell*np.sum(dl*(2*(a-ell)+dl))
    # Bend/twist changes are evaluated per adjacent pair, avoiding cancellation
    # against the much larger absolute stretch and inertia energies.
    def bt(x,theta):
        edges=np.diff(x,axis=0);u=edges/np.linalg.norm(edges,axis=1)[:,None]
        co=np.sum(u[:-1]*u[1:],axis=1);dt=np.diff(theta)
        return EI/ell*(1-co)/(1+co)*(1+np.cos(dt))+.5*GJ/ell*dt*dt
    ans+=np.sum(bt(c['q']+nd,c['theta']+nt)-bt(c['q']+d,c['theta']+t))
    return float(ans)

def solve(c):
    n=c['n'];ell,mass,inertia=setup(c)
    if n<3 or not all(np.isfinite(c[k]).all() for k in ('q','v','theta','omega','target')):
        return {'status':'invalid'}
    if np.any(np.linalg.norm(np.diff(c['q'],axis=0),axis=1)<=1e-14):return {'status':'invalid'}
    d=H*c['v']; t=H*c['omega'];d[[0,-1]]=c['target']-c['q'][[0,-1]]
    free=np.r_[np.arange(3,3*(n-1)),np.arange(3*n,4*n-1)]
    lines=[]; status='iterations'; mineig=None
    for iteration in range(60):
        q=c['q']+d;theta=c['theta']+t;N=4*n-1
        g,_,minc=grad_hv(q,theta,np.zeros(N),ell,mass,inertia,d-H*c['v'],t-H*c['omega'])
        if minc<.5 or not np.isfinite(g).all():status='invalid';break
        J=np.empty((N,N))
        for j in range(N):
            e=np.zeros(N);e[j]=1
            _,J[:,j],_=grad_hv(q,theta,e,ell,mass,inertia,d-H*c['v'],t-H*c['omega'])
        Jf=J[np.ix_(free,free)]; diag=np.diag(Jf)
        if np.any(diag<=0) or not np.isfinite(Jf).all():status='curvature';break
        D=1/np.sqrt(diag);A=(D[:,None]*Jf)*D[None,:];mineig=float(np.min(np.linalg.eigvalsh(A)))
        if mineig<=0:status='curvature';break
        force=float(np.max(np.abs(g[3:3*(n-1)]))); torque=float(np.max(np.abs(g[3*n:])))
        if force<=1e-10 and torque<=1e-19:status='converged';break
        try: step=np.zeros(N);step[free]=D*np.linalg.solve(A,-D*g[free])
        except np.linalg.LinAlgError:status='linear_failure';break
        accepted=False
        for j in range(30):
            f=2.**(-j);nd=d+f*step[:3*n].reshape(n,3);nt=t+f*step[3*n:]
            nd[[0,-1]]=c['target']-c['q'][[0,-1]]
            change=energy_delta(c,d,t,nd,nt,ell,mass,inertia)
            if np.isfinite(change) and change<0 and (np.any(nd!=d) or np.any(nt!=t)):
                accepted=True;break
        lines.append(j+1)
        if not accepted:status='line_search';break
        d,t=nd,nt
    out=dict(status=status,iterations=iteration+1,line_search=lines,min_scaled_eigenvalue=mineig)
    if status!='converged':return out
    q=c['q']+d;theta=c['theta']+t;vel=d/H;om=t/H
    g,_,minc=grad_hv(q,theta,np.zeros(4*n-1),ell,mass,inertia,d-H*c['v'],t-H*c['omega'])
    reaction=-g[:3*n].reshape(n,3)[[0,-1]]
    out.update(q=q.tolist(),theta=theta.tolist(),v=vel.tolist(),omega=om.tolist(),reaction=reaction.tolist(),impulse=(reaction*H).tolist(),work=float(np.sum(reaction*(c['target']-c['q'][[0,-1]]))),min_cosine=float(minc),working_force=float(np.max(np.abs(g[3:3*(n-1)]))),working_torque=float(np.max(np.abs(g[3*n:]))))
    return out

records=[solve(c) for c in cases]
def input_line(c,r):
    return ' '.join(format(float(a),'.17g') for a in [c['n'],*np.array(r['q']).ravel(),*r['theta']])
accepted=[i for i,r in enumerate(records) if r['status']=='converged']
inputs='\n'.join(input_line(cases[i],records[i]) for i in accepted)+'\n'
(p/'solve-native-inputs.txt').write_text(inputs)
run=subprocess.run([sys.argv[1]],input=inputs,capture_output=True,text=True)
(p/'solve-native.stdout').write_text(run.stdout);(p/'solve-native.stderr').write_text(run.stderr)
if run.returncode:raise RuntimeError(f'native exit {run.returncode}')
native=[json.loads(s) for s in run.stdout.splitlines()];assert len(native)==len(accepted)
for idx,ref in zip(accepted,native):
    c=cases[idx];r=records[idx];n=c['n'];ell,mass,inertia=setup(c); q=np.array(r['q']);v=np.array(r['v']);om=np.array(r['omega']); reaction=np.array(r['reaction'])
    residuals=[];anchors=[]
    for qual in (0,1):
        bg=np.array(ref[f'bendGradient{qual}']); sg=np.array(ref[f'stretchGradient{qual}']);tg=np.array(ref[f'twistGradient{qual}'])
        f=(sg+bg[:3*n]).reshape(n,3);torque=bg[3*n:]+tg
        rv=v-c['v']+H*f/mass[:,None];rw=om-c['omega']+H*torque/inertia
        aref=-(mass[:,None]/H**2*((q-c['q'])-H*c['v'])+f)[[0,-1]]
        residuals.append((rv,rw));anchors.append(aref)
    r0,r1=residuals
    linear=float(np.max(np.abs(r1[0][1:-1]))); angular=float(np.max(np.abs(r1[1])))
    agreement_v=float(np.max(np.abs((r1[0]-r0[0])[1:-1])));agreement_w=float(np.max(np.abs(r1[1]-r0[1])))
    err=np.linalg.norm(reaction-anchors[1],axis=1);lim=np.maximum(1e-6,.01*np.linalg.norm(anchors[1],axis=1))
    negv=np.array(r1[0]);negv[1,0]+=.002;negw=np.array(r1[1]);negw[0]+=.002
    passed=bool(linear<=.001 and angular<=.001 and agreement_v<=.0001 and agreement_w<=.0001 and np.all(err<=lim) and np.array_equal(q[[0,-1]],c['target']) and np.max(np.abs(negv[1:-1]))>.001 and np.max(np.abs(negw))>.001)
    r.update(linear_momentum_residual_m_s=linear,angular_momentum_residual_rad_s=angular,linear_reference_agreement_m_s=agreement_v,angular_reference_agreement_rad_s=agreement_w,anchor_error_N=err.tolist(),anchor_limit_N=lim.tolist(),negative_control_pass=bool(np.max(np.abs(negv[1:-1]))>.001 and np.max(np.abs(negw))>.001),passed=passed)
invalid=[]
for mode in ('nan','edge'):
    c=make_case(5,1101)
    if mode=='nan':c['q'][1,0]=np.nan
    else:c['q'][1]=c['q'][0]
    invalid.append(dict(mode=mode,result=solve(c)))
inputs_json=[]
for c in cases:
    inputs_json.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()})
(p/'solve-cases.json').write_text(json.dumps(inputs_json,indent=2,allow_nan=False)+'\n')
result={'cases':records,'invalid_controls':invalid,'all_pass':bool(len(accepted)==12 and all(r.get('passed',False) for r in records) and all(x['result']=={'status':'invalid'} for x in invalid))}
(p/'solve-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_pass':result['all_pass'],'statuses':[x['status'] for x in records],'failed':[i for i,r in enumerate(records) if not r.get('passed',False)]},indent=2))
sys.exit(0 if result['all_pass'] else 1)
