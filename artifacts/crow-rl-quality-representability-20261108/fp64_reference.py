from pathlib import Path
import hashlib,json,math,struct
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-first-failure-20261108';B=R/'build/crow-quality-first-failure-build-01/repair-01'

def cone(w,r,s):
    # Independent FP64 weighted Lorentz-cone minimization. Boundary KKT is
    # reduced to one monotone scalar root and bracketed by bisection.
    d=r*s*s;c=w*s;z=-c/d
    if np.linalg.norm(z[1:])<=z[0]:return s*z,np.diag(1/r),'interior'
    if c[0]>=np.linalg.norm(c[1:]):return np.zeros(len(w)),np.zeros((len(w),len(w))),'vertex'
    low=max(0.,float(c[0]));high=max(1.,abs(float(c[0])),float(np.linalg.norm(c[1:])))
    def f(tau):
        n=(tau-c[0])/d[0]
        return np.linalg.norm(c[1:]/(d[1:]*n+tau))-1.
    while f(high)>0:high*=2
    for _ in range(160):
        mid=(low+high)/2
        if mid==low or mid==high:break
        if f(mid)>0:low=mid
        else:high=mid
    tau=(low+high)/2;n=(tau-c[0])/d[0]
    z[0]=n;z[1:]=-c[1:]*n/(d[1:]*n+tau)
    u=z[1:]/np.linalg.norm(z[1:]);h=np.diag(d)
    h[1:,1:]+=tau/n*(np.eye(len(w)-1)-np.outer(u,u))
    hi=np.linalg.inv(h);normal=np.r_[-1.,u];ha=hi@normal
    derivative=(hi-np.outer(ha,ha)/(normal@ha))*np.outer(s,s)
    return s*z,derivative,'boundary'

def calibrate():
    vectors=[[-2.,.2,.1],[3.,.2,-.1],[0.,2.,1.],[1.,-2.,3.],[-1.,2.,-.5]]
    for v in vectors:
        w=np.array(v);x=-w;t=np.linalg.norm(x[1:])
        expected=x if t<=x[0] else np.zeros(3) if t<=-x[0] else np.r_[(t+x[0])/2,x[1:]*(t+x[0])/(2*t)]
        p,_,_=cone(w,np.ones(3),np.ones(3));assert np.max(np.abs(p-expected))<2e-14
    w=np.array([.3,-1.,.5]);r=np.array([.8,1.1,2.3]);s=np.array([1.,.4,.7])
    p,derivative,kind=cone(w,r,s);assert kind=='boundary'
    numerical=np.column_stack([(cone(w+np.eye(3)[i]*1e-6,r,s)[0]-cone(w-np.eye(3)[i]*1e-6,r,s)[0])/(-2e-6) for i in range(3)])
    assert np.max(np.abs(derivative-numerical))<2e-9
    return dict(euclidean_closed_form_cases=5,weighted_boundary_derivative_max_error=float(np.max(np.abs(derivative-numerical))),passed=True)

calibration=calibrate()
raw=(B/'capture-one.bin').read_bytes();header=struct.unpack_from('<8I',raw)
assert header[0]==0x51464c31 and header[1]==1 and header[2:5]==(0,0,2) and header[5]==len(raw)
dispatch=struct.unpack_from('<16I8f',raw,32);nv,rows,blocks=dispatch[2],dispatch[3],dispatch[4]
status_i=struct.unpack_from('<12I',raw,128);status_f=struct.unpack_from('<12f',raw,176)
assert status_i[0]==7 and status_i[4]==20 and status_i[5]==77
block=[];covered=np.zeros(rows,dtype=bool);offset=256
for i in range(blocks):
    integers=struct.unpack_from('<8I',raw,offset);values=struct.unpack_from('<20f',raw,offset+32)
    start,dim,kind,flags=integers[:4]
    assert dim>0 and start+dim<=rows and not np.any(covered[start:start+dim]);covered[start:start+dim]=True
    scale=np.array(values[:8])[:dim];regularization=np.array(values[8:16])[:dim]
    bounds=np.array(values[16:20]);assert np.all(scale>0) and np.all(regularization>0)
    block.append(dict(start=start,dim=dim,kind=kind,flags=flags,scale=scale,r=regularization,bounds=bounds,key=integers[4:8]));offset+=128
def take(count):
    global offset
    x=np.frombuffer(raw,dtype='<f4',count=count,offset=offset).astype(np.float64);offset+=4*count;return x
mass=take(nv*nv).reshape(nv,nv);jac=take(rows*nv).reshape(rows,nv);bias=take(rows);free=take(nv);warm=take(nv);warm_p=take(rows);failed=take(nv);impulse=take(rows);row_v=take(rows);gradient=take(nv)
assert offset<=len(raw) and len(raw)-offset<16
assert all(np.isfinite(x).all() for x in [mass,jac,bias,free,warm,warm_p,failed,impulse,row_v,gradient])
padding=np.where(~covered)[0]
assert not padding.size or (np.max(np.abs(jac[padding]))==0 and np.max(np.abs(bias[padding]))==0 and np.max(np.abs(warm_p[padding]))==0)
symmetry=float(np.max(np.abs(mass-mass.T)));assert symmetry==0
eigenvalues=np.linalg.eigvalsh(mass);assert eigenvalues[0]>0

def evaluate(v,hessian=False):
    w=jac@v+bias;p=np.zeros(rows);h=mass.copy() if hessian else None;prox=0.;kinds={}
    for b in block:
        at=slice(b['start'],b['start']+b['dim']);local=w[at];r=b['r']
        if b['kind']==0:
            assert b['dim']==1
            lo,hi=b['bounds'][:2];candidate=-local[0]/r[0]
            lp=np.array([np.clip(candidate,lo,hi)]);derivative=np.array([[1/r[0] if lo<candidate<hi else 0.]])
            kind='scalar_free' if lo<candidate<hi else 'scalar_bound'
        else:
            assert b['kind']==1 and b['bounds'][0]==b['bounds'][1]==0
            lp,derivative,kind=cone(local,r,b['scale'])
        kinds[kind]=kinds.get(kind,0)+1;p[at]=lp
        prox+=.5*float(lp@(r*lp))+float(local@lp)
        if hessian:h+=jac[at].T@derivative@jac[at]
    dv=v-free;action=mass@dv;constraint=jac.T@p;g=action-constraint
    return .5*float(dv@action)-prox,g,p,h,kinds,float(np.max(np.abs(g))/(1+np.max(np.abs(action))+np.max(np.abs(constraint))))

initial=evaluate(failed);captured_action=mass@(failed-free);captured_constraint=jac.T@impulse
captured_gradient_fp64=captured_action-captured_constraint
row_residual=float(np.max(np.abs(jac@failed+bias-row_v)))
v=failed.copy();history=[];scale=1/np.sqrt(np.diag(mass));converged=False
for iteration in range(80):
    objective,g,p,h,kinds,res=evaluate(v,True)
    history.append(dict(iteration=iteration,objective=objective,scaled_optimality=res,gradient_max=float(np.max(np.abs(g))),projection_branches=kinds))
    if res<=1e-13:converged=True;break
    direction=scale*np.linalg.solve(h*np.outer(scale,scale),-g*scale)
    descent=float(g@direction);assert descent<0
    alpha=1.;accepted=False
    for backtrack in range(40):
        candidate=v+alpha*direction;trial=evaluate(candidate)
        if trial[0]<=objective+1e-4*alpha*descent or trial[-1]<res*.8:
            v=candidate;accepted=True;break
        alpha*=.5
    history[-1].update(step_length=alpha,backtracks=backtrack,step_max=float(np.max(np.abs(alpha*direction))))
    if not accepted:break
final=evaluate(v);rounded=v.astype(np.float32).astype(np.float64);rounded_result=evaluate(rounded)
for name,array in [('fp64-reference-v.f64',v),('fp64-reference-impulse.f64',final[2]),('fp64-reference-gradient.f64',final[1])]:
    (B/name).write_bytes(np.asarray(array,dtype='<f8').tobytes())
report=dict(schema='crow-quality-first-failure-fp64-analysis-v1',purpose='exploratory analysis of one previously exposed failure; no native trajectory or numerical setting changed',capture_sha256=hashlib.sha256(raw).hexdigest(),capture_bytes=len(raw),nv=nv,rows=rows,blocks=blocks,calibration=calibration,public_stamp=dict(environment=header[2],control_step=header[3],physics_substep=header[4],quality_code=status_i[0],newton=status_i[4],pcg=status_i[5]),native_certificates0=list(status_f[:4]),native_certificates1=list(status_f[4:8]),native_numerics=list(status_f[8:]),padding_rows=len(padding),padding_jacobian_bias_warm_are_zero=True,padding_impulse_max=float(np.max(np.abs(impulse[padding]))) if len(padding) else 0.,mass_symmetry_error=symmetry,mass_eigen_min=float(eigenvalues[0]),mass_eigen_max=float(eigenvalues[-1]),mass_condition=float(eigenvalues[-1]/eigenvalues[0]),captured_row_velocity_fp64_difference_max=row_residual,captured_impulse_fp64_projection_difference_max=float(np.max(np.abs(impulse[covered]-initial[2][covered]))),captured_gradient_fp64_fixed_impulse_difference_max=float(np.max(np.abs(gradient-captured_gradient_fp64))),fp64_reprojected_failed_scaled_optimality=initial[-1],fp64_reference_converged=converged,fp64_reference_scaled_optimality=final[-1],fp64_reference_iterations=len(history)-1,fp64_rounded_to_fp32_scaled_optimality=rounded_result[-1],failed_to_reference_velocity_max=float(np.max(np.abs(failed-v))),failed_to_reference_impulse_max=float(np.max(np.abs(impulse[covered]-final[2][covered]))),history=history,limitations='One retained problem. FP64 analysis and rounding are diagnostic; they do not prove a global FP32 accuracy floor, identify a causal padding defect, or qualify physical crow behavior.',policy_or_profile_promoted=False,goal_complete=False)
(A/'fp64-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
