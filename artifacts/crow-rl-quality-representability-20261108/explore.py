from pathlib import Path
import hashlib,json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-representability-20261108';B=R/'build/crow-quality-representability-build-01'
source=(A/'fp64_reference.py').read_text();source=source[:source.index('initial=evaluate(failed)')]
old="B=R/'build/crow-quality-first-failure-build-01/repair-01'";assert source.count(old)==1
ns={};exec(compile(source.replace(old,"B=R/'build/crow-quality-representability-build-01'"),'retained-fp64-reference','exec'),ns)
evaluate=ns['evaluate'];mass=ns['mass'];failed=ns['failed'];nv=ns['nv'];free=ns['free']
reference=np.fromfile(B/'fp64-reference-v.f64',dtype='<f8');assert len(reference)==nv
rounded=reference.astype(np.float32).astype(np.float64);initial=evaluate(rounded,True)
equilibration=1/np.sqrt(np.diag(mass));eigen=np.linalg.eigvalsh(mass*np.outer(equilibration,equilibration))
def rank(v):
    _,g,_,h,_,res=evaluate(v,True);out=[]
    for i in np.argsort(np.abs(g))[::-1][:12]:
        ulp=float(np.nextafter(np.float32(v[i]),np.float32(np.inf))-np.float32(v[i]))
        out.append(dict(dof=int(i),velocity=float(v[i]),free_velocity=float(free[i]),gradient=float(g[i]),positive_ulp=ulp,hessian_diagonal=float(h[i,i]),one_ulp_diagonal_gradient=float(h[i,i]*ulp)))
    return out
v=rounded.copy();history=[];start=time.monotonic();reason='iteration_budget'
for iteration in range(64):
    objective,g,p,h,kinds,res=evaluate(v,True)
    rec=dict(iteration=iteration,scaled_optimality=res,gradient_max=float(np.max(np.abs(g))),objective=objective,projection_branches=kinds);history.append(rec)
    if res<=5e-11:reason='target_reached';break
    if time.monotonic()-start>120:reason='time_budget';break
    changes=[];deltas=[];dofs=[]
    for i in range(nv):
        for sign in [-1,1]:
            target=float(np.nextafter(np.float32(v[i]),np.float32(np.inf*sign)))
            delta=target-v[i]
            if abs(delta)<1e-30:continue
            d=np.zeros(nv);d[i]=delta;deltas.append(d);dofs.append(i);changes.append(h[:,i]*delta)
    shifts=np.array(changes);deltas=np.array(deltas);dofs=np.array(dofs)
    singles=g+shifts;single_max=np.max(np.abs(singles),axis=1);single_l2=np.sum(singles*singles,axis=1)
    i=int(np.lexsort((single_l2,single_max))[0]);best=(float(single_max[i]),float(single_l2[i]));candidate=v+deltas[i];chosen=[i]
    pairs=g+shifts[:,None,:]+shifts[None,:,:]
    pair_max=np.max(np.abs(pairs),axis=2);pair_l2=np.sum(pairs*pairs,axis=2)
    bad=(np.arange(len(shifts))[:,None]>=np.arange(len(shifts))[None,:]) | (dofs[:,None]==dofs[None,:])
    pair_max[bad]=np.inf;pair_l2[bad]=np.inf
    k=int(np.lexsort((pair_l2.ravel(),pair_max.ravel()))[0]);i,j=np.unravel_index(k,pair_max.shape)
    pair_key=float(pair_max[i,j]),float(pair_l2[i,j])
    if pair_key<best:best=pair_key;candidate=v+deltas[i]+deltas[j];chosen=[int(i),int(j)]
    current=float(np.max(np.abs(g))),float(g@g)
    if best>=current:reason='no_improving_one_or_two_coordinate_neighbor';break
    assert np.array_equal(candidate,candidate.astype(np.float32).astype(np.float64))
    actual=evaluate(candidate);actual_key=float(np.max(np.abs(actual[1]))),float(actual[1]@actual[1])
    rec.update(changed_dofs=[int(dofs[k]) for k in chosen],linearized_gradient_max=best[0],actual_gradient_max=actual_key[0],full_reprojection_accepted=actual_key<current)
    if actual_key>=current:reason='local_linearization_rejected_by_full_projection';break
    v=candidate
final=evaluate(v,True)
(B/'neighbor-v.f32').write_bytes(v.astype('<f4').tobytes())
(B/'neighbor-gradient.f64').write_bytes(final[1].astype('<f8').tobytes())
report=dict(schema='crow-quality-representability-result-v1',verdict='representable_state_found' if final[-1]<=1e-10 else 'search_inconclusive',stopping_reason=reason,capture_sha256=hashlib.sha256((B/'capture-one.bin').read_bytes()).hexdigest(),reference_calibration=ns['calibration'],native_failed_scaled_optimality=float(ns['status_f'][0]),failed_fp64_scaled_optimality=evaluate(failed)[-1],rounded_reference_scaled_optimality=initial[-1],neighbor_scaled_optimality=final[-1],neighbor_fp32_exact=True,neighbor_reference_max_velocity_difference=float(np.max(np.abs(v-reference))),mass_diagonally_equilibrated_condition=float(eigen[-1]/eigen[0]),controlling_coordinates_before=rank(rounded),controlling_coordinates_after=rank(v),iterations=len(history),seconds=time.monotonic()-start,history=history,limitations='One retained problem and bounded neighbor search. A found FP32 state refutes an unavoidable representability floor here; it does not establish native FP32 arithmetic acceptance or crow physical qualification.',policy_or_profile_promoted=False,goal_complete=False)
(A/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['history','controlling_coordinates_before','controlling_coordinates_after']}),flush=True)
