from pathlib import Path
import itertools,json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108'
source=(A/'lattice_certificate.py').read_text().split('report=dict(schema=')[0];scope={};exec(compile(source,'retained-exact-certificate-definitions','exec'),scope)
Q=scope['Q'];n=scope['n'];nv=scope['nv'];indices=scope['indices'];ulp=scope['ulp'];u=scope['u'];H=scope['H'];lower=scope['lower'];diagonal=scope['diagonal'];gradient=scope['gradient'];denominator=scope['denominator_upper'];begin=time.monotonic()
aug=[[Q(int(u[i,j])) for j in range(n)]+[Q(int(i==j)) for j in range(n)] for i in range(n)]
for k in range(n):
    pivot=next(i for i in range(k,n) if aug[i][k]);aug[k],aug[pivot]=aug[pivot],aug[k];p=aug[k][k];aug[k]=[x/p for x in aug[k]]
    for i in range(n):
        if i!=k:
            coefficient=aug[i][k];aug[i]=[x-coefficient*y for x,y in zip(aug[i],aug[k])]
inverse=[row[n:] for row in aug];assert all(x.denominator==1 for row in inverse for x in row)
def solve(rhs):
    z=[]
    for i in range(n):z.append(rhs[i]-sum((lower[i][j]*z[j] for j in range(i)),Q(0)))
    answer=[Q(0)]*n
    for i in range(n-1,-1,-1):answer[i]=z[i]/diagonal[i]-sum((lower[j][i]*answer[j] for j in range(i+1,n)),Q(0))
    return answer
dual=[];phases=[]
for row in inverse:
    rhs=[w/Q(float(ulp[index])) for w,index in zip(row,indices)];short=solve(rhs);y=[Q(0)]*nv
    for i,index in enumerate(indices):y[index]=short[i]
    dual.append(y);phases.append(sum((y[i]*gradient[i] for i in range(nv)),Q(0)))
floating=np.array([[float(x) for x in row] for row in dual]);best=None;tested=0;selected=list(range(max(0,n-6),n))
for support_size in [1,2,3]:
    for support in itertools.combinations(selected,support_size):
        for values in itertools.product([-2,-1,1,2],repeat=support_size):
            phase=sum((phases[k]*c for k,c in zip(support,values)),Q(0));fraction=phase%1;distance=min(fraction,1-fraction)
            y=sum((floating[k]*c for k,c in zip(support,values)),np.zeros(nv));norm1=float(np.sum(np.abs(y)));value=float(distance)/norm1/float(denominator);tested+=1
            if best is None or value>best[0]:best=(value,support,values)
assert best is not None
support,values=best[1:];y=[sum((dual[k][i]*c for k,c in zip(support,values)),Q(0)) for i in range(nv)];integers=[sum((inverse[k][i]*c for k,c in zip(support,values)),Q(0)) for i in range(n)]
for j,index in enumerate(indices):assert sum((y[i]*H[i][index] for i in range(nv)),Q(0))*Q(float(ulp[index]))==integers[j]
phase=sum((y[i]*gradient[i] for i in range(nv)),Q(0));fraction=phase%1;distance=min(fraction,1-fraction);raw_lower=distance/sum(map(abs,y),Q(0));scaled=raw_lower/denominator
report=dict(schema='crow-contact-exact-mixed-dual-certificate-v1',tested_dual_combinations=tested,support=list(support),coefficients=list(values),integer_dual=[int(x) for x in integers],exact_dual=[str(x) for x in y],exact_phase=str(phase),exact_raw_lower_bound=str(raw_lower),raw_lower_bound=float(raw_lower),exact_scaled_lower_bound=str(scaled),scaled_lower_bound=float(scaled),gate_infeasible_in_certified_region=scaled>Q(1,10**10),all_dual_identities_exact=True,seconds=time.monotonic()-begin,scope='Same constant-branch, fixed-excluded-coordinate, base-spacing lattice and 1e-4 denominator box hypotheses as lattice-certificate.json. No global FP32 impossibility claim.',native_settings_changed=False,goal_complete=False)
(A/'mixed-dual-certificate.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('exact_') and k!='integer_dual'}),flush=True)
