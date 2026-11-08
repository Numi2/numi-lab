from pathlib import Path
import hashlib,json,math,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
source=(A/'reference-source.py').read_text().split('initial=evaluate(failed)')[0]
source=source.replace("B=R/'build/crow-quality-representability-build-01'","B=R/'build/crow-contact-precision-build-01'")
ns={};exec(compile(source,'retained-independent-reference','exec'),ns)
evaluate=ns['evaluate'];mass=ns['mass'];jac=ns['jac'];bias=ns['bias'];failed=ns['failed'];free=ns['free'];nv=ns['nv'];rows=ns['rows'];block=ns['block'];cone=ns['cone']
reference=np.fromfile(B/'reference-v.f64',dtype='<f8');rounded=reference.astype(np.float32).astype(float)
F=np.float32
def fma32(a,b,c):return F(float(a)*float(b)+float(c))
def compensated_dot(a,b,initial):
    high=F(initial);low=F(0)
    for x,y in zip(a,b):
        product=F(x*y);product_error=fma32(x,y,-product)
        total=F(high+product);virtual=F(total-high)
        sum_error=F(F(high-F(total-virtual))+F(product-virtual))
        low=F(low+F(product_error+sum_error));high=total
    return F(high+low)
def rows32(v,compensated=False):
    out=[];v=v.astype(np.float32)
    for row in range(rows):
        a=jac[row].astype(np.float32)
        if compensated:value=compensated_dot(a,v,F(bias[row]))
        else:
            value=F(bias[row])
            for x,y in zip(a,v):value=fma32(x,y,value)
        out.append(float(value))
    return np.array(out)
def project(w):
    p=np.zeros(rows)
    for b in block:
        at=slice(b['start'],b['start']+b['dim']);r=b['r']
        if b['kind']==0:p[at]=np.clip(-w[at]/r,b['bounds'][0],b['bounds'][1])
        else:p[at]=cone(w[at],r,b['scale'])[0]
    return p
def summary_rows(v):
    exact=jac@v+bias;naive=rows32(v);accurate=rows32(v,True)
    return dict(naive_error_max=float(np.max(np.abs(naive-exact))),compensated_error_max=float(np.max(np.abs(accurate-exact))),compensated_correctly_rounded_rows=int(np.sum(accurate==exact.astype(np.float32).astype(float))),rows=rows)
calibrations=[]
for a,b,c in [(np.array([2**26,1,-2**26],dtype=np.float32),np.ones(3,dtype=np.float32),0.),(np.array([1.,-1.,.5],dtype=np.float32),np.array([2**24,2**24,2**-16],dtype=np.float32),2**-20)]:
    exact=math.fsum([float(c)]+[float(x)*float(y) for x,y in zip(a,b)]);actual=float(compensated_dot(a,b,c));assert actual==float(F(exact));calibrations.append(dict(exact=exact,compensated=actual))
native_w=ns['row_v'];emulated=rows32(failed);exact_w=jac@failed+bias;compensated_w=rows32(failed,True)
row_bind=bool(np.array_equal(emulated,native_w));assert row_bind,'FMA32 emulation does not reproduce native captured rows'
native_p=ns['impulse'];project_native_w=project(native_w);project_exact_w=project(exact_w);project_compensated_w=project(compensated_w)
def residual(p):
    action=mass@(failed-free);constraint=jac.T@p
    return float(np.max(np.abs(action-constraint))/(1+np.max(np.abs(action))+np.max(np.abs(constraint))))
def rank(v):
    _,g,_,h,_,res=evaluate(v,True);ranked=[]
    for i in np.argsort(np.abs(g))[::-1][:12]:
        ulp=float(np.nextafter(F(v[i]),F(np.inf))-F(v[i]))
        ranked.append(dict(dof=int(i),velocity=float(v[i]),gradient=float(g[i]),positive_ulp=ulp,hessian_diagonal=float(h[i,i]),diagonal_gradient_per_ulp=float(h[i,i]*ulp)))
    return dict(scaled_optimality=res,coordinates=ranked)
def neighbors(start,name):
    v=start.copy();history=[];begin=time.monotonic();reason='iteration_budget'
    for iteration in range(64):
        objective,g,p,h,kinds,res=evaluate(v,True);current=float(np.max(np.abs(g))),float(g@g)
        record=dict(iteration=iteration,scaled_optimality=res,gradient_max=current[0],objective=objective,projection_branches=kinds);history.append(record)
        if res<=1e-10:reason='target_reached';break
        if time.monotonic()-begin>120:reason='time_budget';break
        changes=[];deltas=[];dofs=[]
        for i in range(nv):
            for sign in [-1,1]:
                delta=float(np.nextafter(F(v[i]),F(np.inf*sign)))-v[i]
                if abs(delta)<1e-30:continue
                d=np.zeros(nv);d[i]=delta;deltas.append(d);dofs.append(i);changes.append(h[:,i]*delta)
        shifts=np.array(changes);deltas=np.array(deltas);dofs=np.array(dofs)
        singles=g+shifts;maximum=np.max(np.abs(singles),axis=1);squared=np.sum(singles*singles,axis=1)
        i=int(np.lexsort((squared,maximum))[0]);best=float(maximum[i]),float(squared[i]);candidate=v+deltas[i];chosen=[i]
        pairs=g+shifts[:,None,:]+shifts[None,:,:];maximum=np.max(np.abs(pairs),axis=2);squared=np.sum(pairs*pairs,axis=2)
        invalid=(np.arange(len(shifts))[:,None]>=np.arange(len(shifts))[None,:]) | (dofs[:,None]==dofs[None,:]);maximum[invalid]=np.inf;squared[invalid]=np.inf
        k=int(np.lexsort((squared.ravel(),maximum.ravel()))[0]);i,j=np.unravel_index(k,maximum.shape);key=float(maximum[i,j]),float(squared[i,j])
        if key<best:best=key;candidate=v+deltas[i]+deltas[j];chosen=[int(i),int(j)]
        if best>=current:reason='no_improving_one_or_two_coordinate_neighbor';break
        actual=evaluate(candidate);actual_key=float(np.max(np.abs(actual[1]))),float(actual[1]@actual[1])
        record.update(changed_dofs=[int(dofs[k]) for k in chosen],linearized_gradient_max=best[0],actual_gradient_max=actual_key[0],full_reprojection_accepted=actual_key<current)
        if actual_key>=current:reason='local_linearization_rejected';break
        v=candidate
    final=evaluate(v);(B/(name+'-neighbor-v.f32')).write_bytes(v.astype('<f4').tobytes())
    return dict(start=name,stopping_reason=reason,representable_state_found=final[-1]<=1e-10,scaled_optimality=final[-1],iterations=len(history),seconds=time.monotonic()-begin,history=history,final=rank(v))
top=[]
for i in np.argsort(np.abs(native_w-exact_w))[::-1][:12]:
    term_sum=float(abs(bias[i])+np.abs(jac[i]*failed).sum())
    top.append(dict(row=int(i),native=float(native_w[i]),exact=float(exact_w[i]),error=float(native_w[i]-exact_w[i]),compensated=float(compensated_w[i]),absolute_term_sum=term_sum,cancellation_ratio=term_sum/max(abs(float(exact_w[i])),1e-300)))
report=dict(schema='crow-contact-precision-diagnosis-v1',capture_sha256=hashlib.sha256((B/'substep7-capture.bin').read_bytes()).hexdigest(),reference_calibration=ns['calibration'],arithmetic_calibrations=calibrations,native_row_fma32_emulation_exact=row_bind,failed_row_arithmetic=summary_rows(failed),rounded_reference_row_arithmetic=summary_rows(rounded),projection_native_rows_vs_native_impulse_max=float(np.max(np.abs(project_native_w-native_p))),projection_exact_rows_vs_native_impulse_max=float(np.max(np.abs(project_exact_w-native_p))),fixed_velocity_scaled_residuals=dict(native_reported=float(ns['status_f'][0]),exact_dynamics_native_impulses=residual(native_p),exact_projection_native_rows=residual(project_native_w),exact_projection_exact_rows=residual(project_exact_w),exact_projection_compensated_rows=residual(project_compensated_w)),largest_row_errors=top,failed=rank(failed),rounded_reference=rank(rounded),searches=[neighbors(failed,'failed'),neighbors(rounded,'rounded')],limits='Exposed numerical diagnosis only. Neighbor-search failure is not a proof of a representability floor; emulated compensated arithmetic is not native physics.',native_settings_changed=False,policy_or_profile_promoted=False,goal_complete=False)
(A/'diagnosis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:report[k] for k in ['native_row_fma32_emulation_exact','failed_row_arithmetic','fixed_velocity_scaled_residuals']}))
print(json.dumps([{k:r[k] for k in ['start','stopping_reason','representable_state_found','scaled_optimality','iterations']} for r in report['searches']]),flush=True)
