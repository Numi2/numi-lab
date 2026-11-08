#!/usr/bin/env python3
"""Research-only vector stretch solve; no bend, twist, or contact."""
import json
import math
import numpy as np

TOTAL_REST=.11758755785232791
EA=2.3818718058345718
AREA_DENSITY=1100*math.pi*(.000020**2)
H=.001

def make_case(n,kind,dtype):
    t=np.linspace(0.,1.,n,dtype=np.float64)
    p=np.column_stack((.122*t,np.zeros(n),np.zeros(n)))
    if kind in ('curved','interior_soft','soft_endpoint'):
        p[:,1]=.004*np.sin(np.pi*t); p[:,2]=.002*np.sin(2*np.pi*t)
    if kind in ('moving','interior_soft'):
        p[n//2]+=np.array([-.0005,.0003,-.0002])+H*np.array([.01,-.02,.015])
    if kind=='rotated':
        axis=np.array([1.,2.,3.]); axis/=np.linalg.norm(axis)
        v=np.cross(np.array([0.,0.,1.]),axis); s=np.linalg.norm(v); c=axis[2]
        cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        r=np.eye(3)+cross+cross@cross*((1-c)/(s*s))
        p=p@r.T
    hard={0:p[0].copy()}
    soft=[]
    if kind=='soft_endpoint':
        soft.append((n-1,p[-1]+np.array([.00002,-.00001,.00001]),100.))
    else:
        target=p[-1].copy()
        if kind=='moving_endpoint': target+=np.array([.00005,-.00001,.00002])
        hard[n-1]=target
    if kind=='interior_soft': soft.append((n//2,p[n//2]+np.array([.0002,-.0001,.00005]),50.))
    rest=dtype(TOTAL_REST/(n-1)); k=dtype(EA/float(rest))
    m=np.full(n,dtype(AREA_DENSITY*(TOTAL_REST/(n-1))),dtype=dtype); m[[0,-1]]*=dtype(.5)
    return p.astype(dtype),{i:v.astype(dtype) for i,v in hard.items()},[(i,v.astype(dtype),dtype(kv)) for i,v,kv in soft],rest,k,m

def evaluate(x,p,soft,rest,k,m,exact):
    dtype=x.dtype.type; n=len(x); eye=np.eye(3,dtype=dtype)
    inert=np.asarray(m/(dtype(H)*dtype(H)),dtype=dtype)
    d=x-p
    energy=dtype(dtype(.5)*np.sum(inert[:,None]*d*d,dtype=dtype))
    g=np.asarray(inert[:,None]*d,dtype=dtype)
    blocks=np.asarray([eye*w for w in inert],dtype=dtype)
    off=np.zeros((n-1,3,3),dtype=dtype)
    for i in range(n-1):
        edge=x[i+1]-x[i]; length=dtype(np.sqrt(np.dot(edge,edge))); u=edge/length
        stretch=dtype(length-rest)
        energy=dtype(energy+dtype(.5)*k*stretch*stretch)
        force=k*stretch*u
        g[i]-=force; g[i+1]+=force
        tangent=k*np.outer(u,u)
        if exact:
            tangent=tangent+k*(stretch/length)*(eye-np.outer(u,u))
        blocks[i]+=tangent; blocks[i+1]+=tangent; off[i]-=tangent
    for i,target,stiffness in soft:
        delta=x[i]-target
        energy=dtype(energy+dtype(.5)*stiffness*np.dot(delta,delta))
        g[i]+=stiffness*delta; blocks[i]+=stiffness*eye
    return float(energy),g,blocks,off

def make_full(diag,off):
    n=len(diag); a=np.zeros((3*n,3*n),dtype=diag.dtype)
    for i in range(n):
        a[3*i:3*i+3,3*i:3*i+3]=diag[i]
        if i<n-1:
            a[3*i:3*i+3,3*i+3:3*i+6]=off[i]
            a[3*i+3:3*i+6,3*i:3*i+3]=off[i].T
    return a

def solve(kind,n,dtype,exact):
    p,hard,soft,rest,k,m=make_case(n,kind,dtype)
    x=p.copy()
    for i,target in hard.items(): x[i]=target
    free=[i for i in range(n) if i not in hard]
    objective=[]; status='limit'; max_iterations=100
    for iteration in range(max_iterations):
        e,g,diag,off=evaluate(x,p,soft,rest,k,m,exact)
        objective.append(e)
        grad=float(np.max(np.linalg.norm(g[free].astype(np.float64),axis=1))) if free else 0.
        if grad <= (1e-5 if dtype==np.float32 else 1e-10):
            status='converged'; break
        if exact:
            full=make_full(diag,off)
            indices=np.array([3*i+j for i in free for j in range(3)])
            a=full[np.ix_(indices,indices)]
            eigen=float(np.min(np.linalg.eigvalsh(a)))
            if eigen<=0: a=a+np.eye(len(a))*(-eigen+1e-8)
            step=np.zeros_like(x); step[free]=np.linalg.solve(a,-g[free].reshape(-1)).reshape((-1,3))
        else:
            # Dimensionless local blocks and RHS; pinned rows are identity.
            scale=dtype(k)
            diag=diag/scale; off=off/scale; rhs=-g/scale
            for i in hard:
                diag[i]=np.eye(3,dtype=dtype); rhs[i]=0
                if i>0: off[i-1]=0
                if i<n-1: off[i]=0
            try:
                for i in range(1,n):
                    factor=np.linalg.solve(diag[i-1].T,off[i-1]).T
                    diag[i]-=factor@off[i-1]
                    rhs[i]-=factor@rhs[i-1]
                step=np.zeros_like(x); step[-1]=np.linalg.solve(diag[-1],rhs[-1])
                for i in range(n-2,-1,-1): step[i]=np.linalg.solve(diag[i],rhs[i]-off[i]@step[i+1])
            except np.linalg.LinAlgError:
                status='singular'; break
        if not np.all(np.isfinite(step)): status='nonfinite'; break
        accepted=False
        for j in range(24):
            trial=np.asarray(x+dtype(2.**(-j))*step,dtype=dtype)
            for i,target in hard.items(): trial[i]=target
            e2,*_=evaluate(trial,p,soft,rest,k,m,False)
            if e2 <= e:
                if np.array_equal(trial,x): status='stagnated'; break
                x=trial; accepted=True; break
        if not accepted:
            if status != 'stagnated': status='line_search_failed'
            break
    e,g,_,_=evaluate(x,p,soft,rest,k,m,False)
    grad=float(np.max(np.linalg.norm(g[free].astype(np.float64),axis=1))) if free else 0.
    reactions={f'hard:{i}':list(map(float,-g[i])) for i in hard}
    reactions.update({f'soft:{i}':list(map(float,stiffness*(x[i]-target))) for i,target,stiffness in soft})
    return dict(x=x, status=status,iterations=iteration+1,grad=grad,energy=e,objective=objective,reactions=reactions)

spec=[(3,'straight'),(9,'straight'),(17,'straight'),(9,'rotated'),(9,'curved'),(17,'curved'),(9,'moving'),(9,'moving_endpoint'),(9,'soft_endpoint'),(9,'interior_soft')]
rows=[]
for n,kind in spec:
    a=solve(kind,n,np.float32,False); b=solve(kind,n,np.float64,True)
    pos=float(np.max(np.linalg.norm(a['x'].astype(np.float64)-b['x'],axis=1)))
    err={};limits={}
    for key in b['reactions']:
        err[key]=float(np.linalg.norm(np.array(a['reactions'][key])-b['reactions'][key]))
        limits[key]=max(1e-6,.01*float(np.linalg.norm(b['reactions'][key])))
    p,hard,soft,rest,k,m=make_case(n,kind,np.float64)
    _,actual_g,_,_=evaluate(a['x'].astype(np.float64),p,soft,rest,k,m,False)
    free=[i for i in range(n) if i not in hard]
    independent_grad=float(np.max(np.linalg.norm(actual_g[free],axis=1)))
    passed=bool(a['status']=='converged' and b['status']=='converged' and pos<=1e-6 and all(err[k]<=limits[k] for k in err) and independent_grad<=1e-5)
    rows.append(dict(nodes=n,case=kind,passed=passed,position_error_m=pos,reaction_error_N=err,
        reaction_limit_N=limits,candidate_status=a['status'],reference_status=b['status'],
        candidate_iterations=a['iterations'],reference_iterations=b['iterations'],
        candidate_local_residual_N=a['grad'],independent_local_residual_N=independent_grad,
        reference_local_residual_N=b['grad'],candidate_energy=a['energy'],reference_energy=b['energy'],
        candidate_objective=a['objective'],reference_objective=b['objective']))
print(json.dumps(dict(cases=rows,all_pass=all(r['passed'] for r in rows)),indent=2,sort_keys=True))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
