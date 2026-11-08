from pathlib import Path
import json,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01'
source=(A/'reference-source.py').read_text().split('initial=evaluate(failed)')[0].replace("B=R/'build/crow-quality-representability-build-01'","B=R/'build/crow-contact-precision-build-01'")
ns={};exec(compile(source,'retained-independent-reference','exec'),ns);evaluate=ns['evaluate'];nv=ns['nv']
reference=np.fromfile(B/'reference-v.f64',dtype='<f8');report=[]
for start_name,start in [('failed',ns['failed']),('rounded',reference.astype(np.float32).astype(float))]:
    v=start.copy();history=[];begin=time.monotonic();reason='iteration_budget'
    for iteration in range(64):
        objective,g,p,h,kinds,res=evaluate(v,True);current=float(np.max(np.abs(g))),float(g@g)
        rec=dict(iteration=iteration,scaled_optimality=res,gradient_max=current[0],objective=objective,projection_branches=kinds);history.append(rec)
        if res<=1e-10:reason='target_reached';break
        if time.monotonic()-begin>120:reason='time_budget';break
        candidates=[];descriptions=[]
        for i in range(nv):
            column=h[:,i];den=float(column@column)
            if den==0:continue
            step=-float(column@g)/den;target=np.float32(v[i]+step)
            for target in [np.nextafter(target,np.float32(-np.inf)),target,np.nextafter(target,np.float32(np.inf)),np.nextafter(np.float32(v[i]),np.float32(-np.inf)),np.nextafter(np.float32(v[i]),np.float32(np.inf))]:
                d=np.zeros(nv);d[i]=float(target)-v[i]
                if np.max(np.abs(d))<=1e-4:candidates.append(d);descriptions.append([i])
            for j in range(i+1,nv):
                columns=h[:,[i,j]];norms=np.linalg.norm(columns,axis=0)
                if np.any(norms==0):continue
                scaled=columns/norms;coeff=np.linalg.lstsq(scaled,-g,rcond=1e-14)[0]/norms
                target=(v[[i,j]]+coeff).astype(np.float32)
                choices=[[np.nextafter(t,np.float32(-np.inf)),t,np.nextafter(t,np.float32(np.inf))] for t in target]
                for a in choices[0]:
                    for b in choices[1]:
                        d=np.zeros(nv);d[[i,j]]=[float(a)-v[i],float(b)-v[j]]
                        if np.max(np.abs(d))<=1e-4:candidates.append(d);descriptions.append([i,j])
        deltas=np.array(candidates);predicted=g+deltas@h.T;maximum=np.max(np.abs(predicted),axis=1);squared=np.sum(predicted*predicted,axis=1)
        index=int(np.lexsort((squared,maximum))[0]);key=float(maximum[index]),float(squared[index]);candidate=v+deltas[index]
        if key>=current:reason='no_improving_block_candidate';break
        assert np.array_equal(candidate,candidate.astype(np.float32).astype(float))
        actual=evaluate(candidate);actual_key=float(np.max(np.abs(actual[1]))),float(actual[1]@actual[1])
        rec.update(changed_dofs=descriptions[index],delta_max=float(np.max(np.abs(deltas[index]))),linearized_gradient_max=key[0],actual_gradient_max=actual_key[0],full_reprojection_accepted=actual_key<current)
        if actual_key>=current:reason='linearization_rejected';break
        v=candidate
    final=evaluate(v);(B/(start_name+'-block-v.f32')).write_bytes(v.astype('<f4').tobytes())
    result=dict(start=start_name,stopping_reason=reason,representable_state_found=final[-1]<=1e-10,scaled_optimality=final[-1],maximum_difference_from_reference=float(np.max(np.abs(v-reference))),iterations=len(history),seconds=time.monotonic()-begin,history=history);report.append(result)
(A/'block-search-result.json').write_text(json.dumps(dict(schema='crow-contact-block-search-v1',cases=report,native_settings_changed=False,policy_or_profile_promoted=False,goal_complete=False),indent=2,sort_keys=True)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='history'} for r in report]),flush=True)
