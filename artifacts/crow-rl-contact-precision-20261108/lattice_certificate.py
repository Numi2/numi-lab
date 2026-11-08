from pathlib import Path
from fractions import Fraction as Q
import json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
source=(A/'lattice_search.py').read_text().split('begin=time.monotonic();calibration=[]')[0]
ns={};exec(compile(source,'retained-lattice-definitions','exec'),ns)
evaluate=ns['evaluate'];raw=ns['ns'];nv=raw['nv'];reference=np.fromfile(B/'reference-v.f64',dtype='<f8');base=reference.astype(np.float32).astype(float)
_,g,_,h,kinds,_=evaluate(base,True);ulp=np.array([float(np.nextafter(np.float32(x),np.float32(np.inf)))-x for x in base]);indices=np.where((ulp>1e-30)&np.any(h!=0,axis=0))[0]
begin=time.monotonic();basis,u,orth,norm,operations=ns['reduce_basis'](h[:,indices]*ulp[indices],begin)
y_approx=orth[:,-1]/norm[-1];integers=[int(np.rint(x)) for x in ((h[:,indices]*ulp[indices]).astype(np.longdouble).T@y_approx)]
mass=[[Q(float(x)) for x in row] for row in raw['mass']];jac=[[Q(float(x)) for x in row] for row in raw['jac']];bias=[Q(float(x)) for x in raw['bias']];free=[Q(float(x)) for x in raw['free']];v=[Q(float(x)) for x in base]
w=[bias[r]+sum((jac[r][j]*v[j] for j in range(nv)),Q(0)) for r in range(raw['rows'])];impulse=[Q(0)]*raw['rows'];active=[];regions={}
for block in raw['block']:
    start=block['start'];dimension=block['dim'];regularization=[Q(float(x)) for x in block['r']]
    if block['kind']==0:
        lo,hi=[Q(float(x)) for x in block['bounds'][:2]];candidate=-w[start]/regularization[0];impulse[start]=min(max(candidate,lo),hi)
        region='scalar_free' if lo<candidate<hi else 'scalar_bound'
        if region=='scalar_free':active.append((start,regularization[0]))
    else:
        assert block['bounds'][0]==block['bounds'][1]==0
        scale=[Q(float(x)) for x in block['scale']];p=[-w[start+i]/regularization[i] for i in range(dimension)];z=[p[i]/scale[i] for i in range(dimension)]
        if z[0]>=0 and sum((x*x for x in z[1:]),Q(0))<=z[0]*z[0]:
            region='interior'
            for i in range(dimension):impulse[start+i]=p[i];active.append((start+i,regularization[i]))
        else:
            c=[w[start+i]*scale[i] for i in range(dimension)];assert c[0]>=0 and c[0]*c[0]>=sum((x*x for x in c[1:]),Q(0));region='vertex'
    regions[region]=regions.get(region,0)+1
assert regions==kinds
H=[row.copy() for row in mass]
for row,r in active:
    nonzero=[i for i in range(nv) if jac[row][i]]
    for i in nonzero:
        for j in nonzero:H[i][j]+=jac[row][i]*jac[row][j]/r
action=[sum((mass[i][j]*(v[j]-free[j]) for j in range(nv)),Q(0)) for i in range(nv)];constraint=[sum((jac[r][i]*impulse[r] for r in range(raw['rows'])),Q(0)) for i in range(nv)];gradient=[a-c for a,c in zip(action,constraint)]
assert max(abs(float(gradient[i])-g[i]) for i in range(nv))<1e-15
n=len(indices);a=[[H[i][j] for j in indices] for i in indices];rhs=[Q(integer)/Q(float(ulp[index])) for integer,index in zip(integers,indices)]
# Exact LDL^T and triangular solves; no floating inverse is used in the witness.
lower=[[Q(0)]*n for _ in range(n)];diagonal=[]
for k in range(n):
    lower[k][k]=Q(1);d=a[k][k]-sum((lower[k][j]*lower[k][j]*diagonal[j] for j in range(k)),Q(0));assert d>0;diagonal.append(d)
    for i in range(k+1,n):lower[i][k]=(a[i][k]-sum((lower[i][j]*lower[k][j]*diagonal[j] for j in range(k)),Q(0)))/d
z=[]
for i in range(n):z.append(rhs[i]-sum((lower[i][j]*z[j] for j in range(i)),Q(0)))
solution=[Q(0)]*n
for i in range(n-1,-1,-1):solution[i]=z[i]/diagonal[i]-sum((lower[j][i]*solution[j] for j in range(i+1,n)),Q(0))
for i in range(n):assert sum((a[i][j]*solution[j] for j in range(n)),Q(0))==rhs[i]
y=[Q(0)]*nv
for i,index in enumerate(indices):y[index]=solution[i]
for j,index in enumerate(indices):assert sum((y[i]*H[i][index] for i in range(nv)),Q(0))*Q(float(ulp[index]))==integers[j]
phase=sum((y[i]*gradient[i] for i in range(nv)),Q(0));fractional=phase%1;distance=min(fractional,1-fractional);norm1=sum(map(abs,y),Q(0));raw_lower=distance/norm1
bound=Q(1,10000);denominator_upper=1+max(abs(action[i])+bound*sum((abs(mass[i][j]) for j in indices),Q(0)) for i in range(nv))+max(abs(constraint[i])+bound*sum((abs(H[i][j]-mass[i][j]) for j in indices),Q(0)) for i in range(nv));scaled_lower=raw_lower/denominator_upper
report=dict(schema='crow-contact-exact-dual-lattice-certificate-v1',input_capture_sha256='08bcd99d486addeed6f39b9a265ef35e3208ea8cc884c12d5549830be0108d0f',dimensions=n,indices=indices.tolist(),integer_dual=integers,exact_dual=[str(x) for x in y],exact_phase=str(phase),distance_to_integer=str(distance),exact_raw_gradient_infinity_lower_bound=str(raw_lower),raw_gradient_infinity_lower_bound=float(raw_lower),local_velocity_box_radius=float(bound),denominator_upper_bound=float(denominator_upper),exact_scaled_optimality_lower_bound=str(scaled_lower),scaled_optimality_lower_bound=float(scaled_lower),unchanged_gate=1e-10,gate_infeasible_in_certified_region=scaled_lower>Q(1,10**10),exact_projection_regions=regions,all_dual_identities_exact=True,seconds=time.monotonic()-begin,scope='Exact proof for the stated constant projection branches, velocities on the base FP32 spacing lattice, fixed excluded coordinates, and radius 1e-4 for denominator bound. The hypotheses are not proved for every nearby FP32 state; this is not a global representability-floor claim.',native_settings_changed=False,policy_or_profile_promoted=False,goal_complete=False)
(A/'lattice-certificate.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['exact_dual','integer_dual','exact_phase','distance_to_integer','exact_raw_gradient_infinity_lower_bound','exact_scaled_optimality_lower_bound']}),flush=True)
