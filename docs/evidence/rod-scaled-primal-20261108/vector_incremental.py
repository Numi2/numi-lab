#!/usr/bin/env python3
"""Temporary local-coordinate vector solve versus dense FP64 reference."""
import json
import math
from pathlib import Path
import numpy as np
# Reuse only fixture, independent dense reference, and common post-checks.
namespace={}
source=Path('docs/evidence/rod-scaled-primal-20261108/vector_full_newton.py').read_text()
exec(source.split('spec=[')[0],namespace)
make_case=namespace['make_case']; evaluate=namespace['evaluate']; reference=namespace['solve']; H=namespace['H']

def evaluate_delta(d,p,soft,rest,k,m):
    T=np.float32; n=len(d); eye=np.eye(3,dtype=T)
    inert=np.asarray(m/(T(H)*T(H)),dtype=T)
    energy=T(T(.5)*np.sum(inert[:,None]*d*d,dtype=T))
    g=np.asarray(inert[:,None]*d,dtype=T)
    diag=np.asarray([eye*w for w in inert],dtype=T)
    off=np.zeros((n-1,3,3),dtype=T)
    for i in range(n-1):
        edge=np.asarray((p[i+1]-p[i])+(d[i+1]-d[i]),dtype=T)
        length=T(np.sqrt(np.dot(edge,edge))); u=edge/length
        stretch=T(length-rest)
        energy=T(energy+T(.5)*k*stretch*stretch)
        force=k*stretch*u
        g[i]-=force;g[i+1]+=force
        tangent=k*(np.outer(u,u)+(stretch/length)*(eye-np.outer(u,u)))
        diag[i]+=tangent;diag[i+1]+=tangent;off[i]-=tangent
    for i,target,stiffness in soft:
        rel=np.asarray((p[i]-target)+d[i],dtype=T)
        energy=T(energy+T(.5)*stiffness*np.dot(rel,rel))
        g[i]+=stiffness*rel;diag[i]+=stiffness*eye
    return float(energy),g,diag,off

def solve_incremental(kind,n):
    T=np.float32
    p,hard,soft,rest,k,m=make_case(n,kind,T)
    d=np.zeros_like(p)
    for i,target in hard.items():d[i]=target-p[i]
    free=[i for i in range(n) if i not in hard]
    objective=[];status='limit'
    for iteration in range(100):
        e,g,diag,off=evaluate_delta(d,p,soft,rest,k,m);objective.append(e)
        grad=float(np.max(np.linalg.norm(g[free].astype(np.float64),axis=1)))
        if grad <= 1e-6:status='converged';break
        diag=diag/k;off=off/k;rhs=-g/k
        for i in hard:
            diag[i]=np.eye(3,dtype=T);rhs[i]=0
            if i>0:off[i-1]=0
            if i<n-1:off[i]=0
        try:
            for i in range(1,n):
                pivot=np.linalg.eigvalsh(diag[i-1])
                if not np.all(np.isfinite(pivot)) or np.min(pivot)<=0: raise np.linalg.LinAlgError('nonpositive pivot')
                factor=np.linalg.solve(diag[i-1].T,off[i-1]).T
                diag[i]-=factor@off[i-1];rhs[i]-=factor@rhs[i-1]
            pivot=np.linalg.eigvalsh(diag[-1])
            if not np.all(np.isfinite(pivot)) or np.min(pivot)<=0: raise np.linalg.LinAlgError('nonpositive pivot')
            step=np.zeros_like(d);step[-1]=np.linalg.solve(diag[-1],rhs[-1])
            for i in range(n-2,-1,-1):step[i]=np.linalg.solve(diag[i],rhs[i]-off[i]@step[i+1])
        except np.linalg.LinAlgError:status='singular';break
        if not np.all(np.isfinite(step)):status='nonfinite';break
        accepted=False
        for j in range(24):
            trial=np.asarray(d+T(2.**(-j))*step,dtype=T)
            for i,target in hard.items():trial[i]=target-p[i]
            e2,*_=evaluate_delta(trial,p,soft,rest,k,m)
            if e2<=e:
                if np.array_equal(trial,d):status='stagnated';break
                d=trial;accepted=True;break
        if not accepted:
            if status!='stagnated':status='line_search_failed'
            break
    e,g,_,_=evaluate_delta(d,p,soft,rest,k,m)
    grad=float(np.max(np.linalg.norm(g[free].astype(np.float64),axis=1)))
    reactions={f'hard:{i}':list(map(float,-g[i])) for i in hard}
    reactions.update({f'soft:{i}':list(map(float,stiffness*((p[i]-target)+d[i]))) for i,target,stiffness in soft})
    x=np.asarray(p+d,dtype=T)
    for i,target in hard.items():x[i]=target
    return dict(x=x,status=status,iterations=iteration+1,grad=grad,energy=e,objective=objective,reactions=reactions)

spec=[(3,'straight'),(9,'straight'),(17,'straight'),(9,'rotated'),(9,'curved'),(17,'curved'),(9,'moving'),(9,'moving_endpoint'),(9,'soft_endpoint'),(9,'interior_soft')]
rows=[]
for n,kind in spec:
    a=solve_incremental(kind,n);b=reference(kind,n,np.float64,True)
    pos=float(np.max(np.linalg.norm(a['x'].astype(np.float64)-b['x'],axis=1)))
    err={};limits={}
    for key in b['reactions']:
        err[key]=float(np.linalg.norm(np.array(a['reactions'][key])-b['reactions'][key]))
        limits[key]=max(1e-6,.01*float(np.linalg.norm(b['reactions'][key])))
    p,hard,soft,rest,k,m=make_case(n,kind,np.float64)
    _,g,_,_=evaluate(a['x'].astype(np.float64),p,soft,rest,k,m,False)
    free=[i for i in range(n) if i not in hard]
    independent=float(np.max(np.linalg.norm(g[free],axis=1)))
    passed=bool(a['status']=='converged' and b['status']=='converged' and pos<=1e-6 and all(err[k]<=limits[k] for k in err) and independent<=1e-5)
    rows.append(dict(nodes=n,case=kind,passed=passed,position_error_m=pos,reaction_error_N=err,
        reaction_limit_N=limits,candidate_status=a['status'],reference_status=b['status'],
        candidate_iterations=a['iterations'],reference_iterations=b['iterations'],
        candidate_local_residual_N=a['grad'],independent_local_residual_N=independent,
        reference_local_residual_N=b['grad'],candidate_energy=a['energy'],reference_energy=b['energy'],
        candidate_objective=a['objective'],reference_objective=b['objective']))
print(json.dumps(dict(cases=rows,all_pass=all(r['passed'] for r in rows)),indent=2,sort_keys=True))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
