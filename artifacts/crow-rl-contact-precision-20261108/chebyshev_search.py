from pathlib import Path
from fractions import Fraction
import json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
source=(A/'lattice_search.py').read_text().split('begin=time.monotonic();calibration=[]')[0];ns={};exec(compile(source,'retained-lattice-definitions','exec'),ns)
evaluate=ns['evaluate'];nv=ns['nv'];reference=np.fromfile(B/'reference-v.f64',dtype='<f8');base=reference.astype(np.float32).astype(float);objective,g,p,h,kinds,res=evaluate(base,True)
certificate=json.loads((A/'lattice-certificate.json').read_text());indices=np.array(certificate['indices']);ulp=np.array([float(np.nextafter(np.float32(x),np.float32(np.inf)))-x for x in base]);begin=time.monotonic()
b,u,orth,norm,operations=ns['reduce_basis'](h[:,indices]*ulp[indices],begin)
y=np.array([float(Fraction(x)) for x in certificate['exact_dual']]);phase=Fraction(certificate['exact_phase']);fraction=phase%1;delta=float(fraction if fraction<=Fraction(1,2) else fraction-1);y/=np.max(np.abs(y));delta/=max(abs(float(Fraction(x))) for x in certificate['exact_dual'])
epsilon=1e-10
def project(target):
    # Projection onto the raw-residual box and exact witness hyperplane.
    low=-1.;high=1.
    for _ in range(100):
        middle=(low+high)/2;candidate=np.clip(target-middle*y,-epsilon,epsilon)
        if y@candidate>delta:low=middle
        else:high=middle
    return np.clip(target-(low+high)/2*y,-epsilon,epsilon)
target=project(np.sign(y)*delta/np.sum(np.abs(y)));history=[];best=(res,float(g@g));best_v=base.copy();success=False;seen=set()
for iteration in range(32):
    c=ns['babai'](b,orth,norm,target-g);offsets=[np.zeros(len(c),dtype=object)]
    for i in range(len(c)):
        for sign in [-1,1]:
            offset=np.zeros(len(c),dtype=object);offset[i]=sign;offsets.append(offset)
    iteration_best=None
    for offset in offsets:
        integers=u@(c+offset);v=base.copy();v[indices]+=ulp[indices]*np.array(integers,dtype=float);v=v.astype(np.float32).astype(float)
        if float(np.max(np.abs(v-reference)))>1e-3:continue
        data=v.astype('<f4').tobytes()
        actual=evaluate(v);key=actual[-1],float(actual[1]@actual[1])
        if iteration_best is None or key<iteration_best[0]:iteration_best=(key,v,actual[1])
        if key<best:best=key;best_v=v
        if key[0]<=1e-10:success=True;break
    assert iteration_best is not None
    history.append(dict(iteration=iteration,best_scaled_optimality=best[0],iteration_scaled_optimality=iteration_best[0][0],target_infinity=float(np.max(np.abs(target))),witness_error=float(y@target-delta)))
    if success:break
    fingerprint=iteration_best[1].astype('<f4').tobytes()
    if fingerprint in seen:
        # Bounded deterministic residual dithering within the same witness plane.
        rng=np.random.default_rng(20261108+iteration);target=project(target+rng.uniform(-epsilon/4,epsilon/4,nv))
    else:target=project(target+(target-iteration_best[2])*.75)
    seen.add(fingerprint)
(B/'chebyshev-v.f32').write_bytes(best_v.astype('<f4').tobytes())
report=dict(schema='crow-contact-chebyshev-search-v1',representable_state_found=success,scaled_optimality=best[0],reference_difference_max=float(np.max(np.abs(best_v-reference))),basis_reductions=operations,candidates_seen=len(seen),seconds=time.monotonic()-begin,history=history,scope='Exposed bounded diagnostic, one exact dual plane and independent full reprojection. No native feedback, settings or policy changes.',goal_complete=False)
(A/'chebyshev-search-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
