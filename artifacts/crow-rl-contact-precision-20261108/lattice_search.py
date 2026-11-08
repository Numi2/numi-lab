from pathlib import Path
import json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
source=(A/'reference-source.py').read_text().split('initial=evaluate(failed)')[0].replace("B=R/'build/crow-quality-representability-build-01'","B=R/'build/crow-contact-precision-build-01'")
ns={};exec(compile(source,'retained-independent-reference','exec'),ns);evaluate=ns['evaluate'];nv=ns['nv'];L=np.longdouble
def gs(b):
    n=b.shape[1];orth=np.zeros_like(b);mu=np.zeros((n,n),dtype=L);norm=np.zeros(n,dtype=L)
    for i in range(n):
        v=b[:,i].copy()
        for repeat in range(2):
            for j in range(i):
                coefficient=(v@orth[:,j])/norm[j];mu[i,j]+=coefficient;v-=coefficient*orth[:,j]
        orth[:,i]=v;norm[i]=v@v
        assert norm[i]>0 and np.isfinite(norm[i])
    return orth,mu,norm
def reduce_basis(original,begin):
    n=original.shape[1];u=np.eye(n,dtype=object);b=original.astype(L);orth,mu,norm=gs(b);k=1;operations=0
    while k<n:
        if operations>=20000 or time.monotonic()-begin>100:raise TimeoutError('LLL budget')
        for j in range(k-1,-1,-1):
            coefficient=int(np.rint(mu[k,j]))
            if coefficient:
                u[:,k]-=coefficient*u[:,j];b=original.astype(L)@u.astype(L);orth,mu,norm=gs(b);operations+=1
        if norm[k]>=(L(.75)-mu[k,k-1]**2)*norm[k-1]:k+=1
        else:
            u[:,[k,k-1]]=u[:,[k-1,k]];b=original.astype(L)@u.astype(L);orth,mu,norm=gs(b);operations+=1;k=max(k-1,1)
    return b,u,orth,norm,operations
def babai(b,orth,norm,target):
    residual=target.astype(L).copy();c=np.zeros(b.shape[1],dtype=object)
    for i in range(len(c)-1,-1,-1):
        c[i]=int(np.rint((residual@orth[:,i])/norm[i]));residual-=L(c[i])*b[:,i]
    return c
begin=time.monotonic();calibration=[]
for original,target in [(np.array([[1.,1.],[0.,.001]]),np.array([.2,.0004])),(np.array([[1.,1.,0.],[0.,.001,0.],[0.,0.,.1]]),np.array([.2,.0004,.12]))]:
    b,u,orth,norm,ops=reduce_basis(original,begin);c=babai(b,orth,norm,target);assert np.max(np.abs(b-original@u.astype(L)))<1e-14
    determinant=round(float(np.linalg.det(u.astype(float))));assert abs(determinant)==1
    calibration.append(dict(dimensions=len(c),unimodular_determinant=determinant,operations=ops))
reference=np.fromfile(B/'reference-v.f64',dtype='<f8');v=reference.astype(np.float32).astype(float);history=[];success=False
for attempt in range(3):
    objective,g,p,h,kinds,res=evaluate(v,True)
    ulp=np.array([float(np.nextafter(np.float32(x),np.float32(np.inf)))-x for x in v]);indices=np.where((ulp>1e-30)&np.any(h!=0,axis=0))[0]
    original=h[:,indices]*ulp[indices]
    b,u,orth,norm,operations=reduce_basis(original,begin);c=babai(b,orth,norm,-g)
    offsets=[np.zeros(len(c),dtype=object)]
    for i in range(len(c)):
        for sign in [-2,-1,1,2]:
            d=np.zeros(len(c),dtype=object);d[i]=sign;offsets.append(d)
    best=(res,float(g@g));best_v=v.copy();candidates=[]
    for offset in offsets:
        integers=u@(c+offset);candidate=v.copy();candidate[indices]+=ulp[indices]*np.array(integers,dtype=float)
        grid_exact=bool(np.array_equal(candidate,candidate.astype(np.float32).astype(float)));candidate=candidate.astype(np.float32).astype(float)
        distance=float(np.max(np.abs(candidate-reference)))
        if distance>1e-3:continue
        actual=evaluate(candidate);key=actual[-1],float(actual[1]@actual[1]);candidates.append(dict(scaled_optimality=actual[-1],distance=distance,grid_exact=grid_exact))
        if key<best:best=key;best_v=candidate
        if actual[-1]<=1e-10:success=True;break
    history.append(dict(attempt=attempt,dimensions=len(indices),basis_reductions=operations,original_column_norm_min=float(np.min(np.linalg.norm(original,axis=0))),original_column_norm_max=float(np.max(np.linalg.norm(original,axis=0))),reduced_gram_norm_min=float(np.sqrt(np.min(norm))),reduced_gram_norm_max=float(np.sqrt(np.max(norm))),candidates=candidates,best_scaled_optimality=best[0],projection_branches=kinds))
    if success:v=best_v;break
    if np.array_equal(best_v,v):break
    v=best_v
(B/'lattice-v.f32').write_bytes(v.astype('<f4').tobytes());final=evaluate(v)
report=dict(schema='crow-contact-lattice-search-v1',calibration=calibration,representable_state_found=success,scaled_optimality=final[-1],reference_difference_max=float(np.max(np.abs(v-reference))),seconds=time.monotonic()-begin,history=history,limits='One exposed problem, bounded offline lattice diagnosis and full independent reprojection. No native state feedback or setting changes.',policy_or_profile_promoted=False,goal_complete=False)
(A/'lattice-search-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
