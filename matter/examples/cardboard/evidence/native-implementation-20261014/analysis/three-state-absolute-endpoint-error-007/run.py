from pathlib import Path
import hashlib,importlib.util,json
import mpmath as mp
import numpy as np
b=Path('/Users/home/cardboard-implementation-20261014');o=b/'analysis/three-state-absolute-endpoint-error-007';o.mkdir(exist_ok=False)
p=b/'analysis/three-exact-captured-state-matrices-315-005/report.json';refpath=b/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();inputs=[dict(path=str(v),sha256=h(v)) for v in [p,refpath]];epsilons=[5e-11,3.5e-11,1e-12]
(o/'prediction.json').write_text(json.dumps(dict(question='What absolute endpoint stress and centered-difference arithmetic errors occur at the actual binary64 perturbations, including small components and a previously failed step?',inputs=inputs,epsilons=epsilons,precision=90,stopping='All three selected states, all nine columns and every declared step and sign. Preserve failed projections. Absolute observed errors are selected-input envelopes, not a universal or stochastic noise bound.',physical_steps=0),indent=2)+'\n')
spec=importlib.util.spec_from_file_location('reference',refpath);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);mp.mp.dps=90
f32=lambda words:np.asarray([int(x,16) for x in words],dtype='<u4').view('<f4').astype(float)
bits=lambda a:[f'0x{v:016x}' for v in np.asarray(a,dtype='<f8').reshape(-1).view('<u8')]
fmt=lambda a:[mp.nstr(v,80) for v in a]
results=[]
for key,g in json.loads(p.read_text())['unique_states'].items():
 raw=f32(g['input_f32_bits']);f=(raw[:9]+raw[9:18]).reshape(3,3);old=f32(g['accepted_state_f32_bits']);params=np.asarray(g['parameter_f32']);zc,tc=ref.project_fp64_scaled_correction(params,f,old);center=ref.system(params,f.reshape(9),old);zmc,itc,nrc=ref.project(center[4],zc);mc=ref.mp_tangent_matrix(params,f.reshape(9),old,zmc)
 rows=[]
 for eps in epsilons:
  for col,H in enumerate(np.eye(9).reshape(9,3,3)):
   ends=[]
   for sign in [-1,1]:
    ff=f+sign*eps*H;z,t=ref.project_fp64_scaled_correction(params,ff,old)
    original=ref.ref.strain;ref.ref.strain=ref.stable_green_strain
    try:stress64=np.asarray(ref.ref.stress(params,ff,z)).reshape(9)
    finally:ref.ref.strain=original
    sys=ref.system(params,ff.reshape(9),old);zm,it,nr=ref.project(sys[4],z);stressmp=list(sys[5](*zm));error=[ref.mpf(v)-w for v,w in zip(stress64,stressmp)]
    ends.append(dict(sign=sign,F_fp64_bits=bits(ff),state_fp64_bits=bits(z),stress_fp64_bits=bits(stress64),mp_state=fmt(zm),mp_stress=fmt(stressmp),mp_iterations=it,mp_scaled_residual=mp.nstr(nr,30),fp64_iterations=len(t),absolute_stress_error_Pa=fmt(error),stress64=stress64,stressmp=stressmp))
   minus,plus=ends;candidate=(plus['stress64']-minus['stress64'])/(2*eps);exact_fd=[(x-y)/(2*ref.mpf(eps)) for x,y in zip(plus['stressmp'],minus['stressmp'])];actualray=[(ref.mpf(x)-ref.mpf(y))/(2*ref.mpf(eps)) for x,y in zip((f+eps*H).reshape(9),(f-eps*H).reshape(9))];smooth=mc*mp.matrix(actualray);noise=[ref.mpf(x)-y for x,y in zip(candidate,exact_fd)];curvature=[x-y for x,y in zip(exact_fd,smooth)];total=[ref.mpf(x)-y for x,y in zip(candidate,smooth)]
   denom=mp.sqrt(sum(x*x for x in smooth));norm=lambda a:mp.sqrt(sum(x*x for x in a));rows.append(dict(epsilon=eps,column=col,endpoints=[{k:v for k,v in e.items() if k not in ('stress64','stressmp')} for e in ends],candidate_derivative_fp64_bits=bits(candidate),exact_centered_derivative=fmt(exact_fd),actual_ray=fmt(actualray),smooth_derivative=fmt(smooth),fd_arithmetic_error_Pa=fmt(noise),finite_step_curvature_Pa=fmt(curvature),total_error_Pa=fmt(total),relative_total_error=mp.nstr(norm(total)/denom,30),relative_arithmetic_error=mp.nstr(norm(noise)/denom,30),relative_finite_step_curvature=mp.nstr(norm(curvature)/denom,30)))
  print('state',g['tets'],'epsilon',eps,flush=True)
 maxstress=[max(abs(mp.mpf(e['absolute_stress_error_Pa'][i])) for r in rows for e in r['endpoints']) for i in range(9)]
 results.append(dict(tets=g['tets'],unique_state_sha256=key,center_F_fp64_bits=bits(f),center_mp_iterations=itc,center_mp_scaled_residual=mp.nstr(nrc,30),center_mp_matrix=[[mp.nstr(mc[i,j],80) for j in range(9)] for i in range(9)],maximum_observed_endpoint_error_by_component_Pa=fmt(maxstress),rows=rows))
report=dict(schema='numi.cardboard.absolute-endpoint-error.v1',inputs=inputs,results=results,physical_steps=0,limits=['All actual binary64 endpoint operands are bound; reference solves use90-digit arithmetic and scaled residual below1e-60. This is observed numerical disagreement, not a formal interval bound.','Componentwise absolute errors apply only to these exact inputs and this FP64 implementation. They are not a universal native FP32 noise floor.','Every declared small step and endpoint remains retained. No production solver criteria, material parameters or equations changed.'])
(o/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(o/'run.py').write_bytes(Path(__file__).read_bytes());(o/'README.md').write_text('# Selected absolute endpoint error decomposition\n\nThree selected states, nine directions and every declared perturbation are retained. Independent90-digit projections at the actual binary64 operands separate endpoint/FD arithmetic disagreement from finite-step curvature. Componentwise absolute stresses and derivative errors are observations for these exact inputs, not a universal or statistical noise bound. No native GPU or physical timesteps.\n');(o/'manifest.json').write_text(json.dumps(dict(inputs=inputs,artifacts={v.name:dict(bytes=v.stat().st_size,sha256=h(v)) for v in o.iterdir() if v.is_file()}),indent=2)+'\n');print('completed',len(results),flush=True)
