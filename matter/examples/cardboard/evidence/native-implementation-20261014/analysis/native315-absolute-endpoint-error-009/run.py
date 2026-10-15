from pathlib import Path
import hashlib,importlib.util,json
import mpmath as mp
import numpy as np
b=Path('/Users/home/cardboard-implementation-20261014');o=b/'analysis/native315-absolute-endpoint-error-009';o.mkdir(exist_ok=False)
p=b/'analysis/three-exact-captured-state-matrices-315-005/report.json';data=b/'native/finer-controlled-material-sweep-315/output/paper-derivative-sweep-accurate.jsonl';refpath=b/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();inputs=[dict(path=str(v),sha256=h(v)) for v in [p,data,refpath]];epsilons=[1e-7,1e-8,1e-9]
(o/'prediction.json').write_text(json.dumps(dict(question='What componentwise absolute native FP32 stress errors and derivative resolution occur at exact two-word F endpoints from315?',inputs=inputs,epsilons=epsilons,precision=90,selected_tets=[567,1634,1638],stopping='All selected states, all nine directions, all declared signed steps, both direct and tangent stress paths; no pointwise omission. Retain total FD error, endpoint arithmetic disagreement and finite-step curvature separately.',gate='Record <=1e-3 nonzero-vector FD error against the actual realized ray; retain absolute component errors without converting the observed envelope into a universal bound.',physical_steps=0),indent=2)+'\n')
spec=importlib.util.spec_from_file_location('reference',refpath);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);mp.mp.dps=90
f32=lambda words:np.asarray([int(x,16) for x in words],dtype='<u4').view('<f4').astype(float)
fmt=lambda a:[mp.nstr(v,80) for v in a]
samples=[json.loads(x) for x in data.read_text().splitlines()];results=[]
for key,g in json.loads(p.read_text())['unique_states'].items():
 tet=g['tets'][0];old=f32(g['accepted_state_f32_bits']);params=np.asarray(g['parameter_f32']);raw=f32(g['input_f32_bits']);f=[ref.mpf(a)+ref.mpf(z) for a,z in zip(raw[:9],raw[9:18])];center=ref.system(params,f,old);zc,itc,nrc=ref.project(center[4],[mp.mpf(x) for x in g['precisions']['90']['root']]);mc=ref.mp_tangent_matrix(params,f,old,zc)
 rows=[]
 for eps in epsilons:
  for col in range(9):
   ends=[]
   for sign in [-1,1]:
    records=[r for r in samples if r.get('kind')=='basis_perturbation' and r['point']['tet_id']==tet and r['epsilon_s']==eps and r['basis']==col and r['sign']==sign];assert len(records)==1;sample=records[0];assert all(sample[k] for k in ['stress_valid','tangent_valid','projection_valid']);rr=f32(sample['point']['input_f32_bits']);ff=[ref.mpf(a)+ref.mpf(z) for a,z in zip(rr[:9],rr[9:18])];native=f32(sample['output_f32_bits']);guess=[ref.mpf(a)+ref.mpf(z) for a,z in zip(native[35:50],native[50:65])];sys=ref.system(params,ff,old);zm,it,nr=ref.project(sys[4],guess);stressmp=list(sys[5](*zm));paths={name:dict(native_stress_f32_bits=sample['output_f32_bits'][begin:begin+9],absolute_stress_error_Pa=fmt([ref.mpf(v)-w for v,w in zip(native[begin:begin+9],stressmp)]),values=native[begin:begin+9]) for name,begin in [('direct',8),('tangent',17)]}
    ends.append(dict(sign=sign,input_f32_bits=sample['point']['input_f32_bits'],accepted_state_f32_bits=sample['point']['accepted_state_f32_bits'],mp_state=fmt(zm),mp_stress=fmt(stressmp),mp_iterations=it,mp_scaled_residual=mp.nstr(nr,30),paths=paths,F=ff,P=stressmp))
   minus,plus=ends;exactfd=[(x-y)/(2*ref.mpf(eps)) for x,y in zip(plus['P'],minus['P'])];ray=[(x-y)/(2*ref.mpf(eps)) for x,y in zip(plus['F'],minus['F'])];smooth=mc*mp.matrix(ray);denom=mp.sqrt(sum(x*x for x in smooth));norm=lambda a:mp.sqrt(sum(x*x for x in a));paths={}
   for path in ['direct','tangent']:
    # Widen output words before differencing, so subtraction is exact in binary64.
    candidate=(plus['paths'][path]['values']-minus['paths'][path]['values'])/(2*eps);noise=[ref.mpf(x)-y for x,y in zip(candidate,exactfd)];curvature=[x-y for x,y in zip(exactfd,smooth)];total=[ref.mpf(x)-y for x,y in zip(candidate,smooth)];rel=norm(total)/denom
    paths[path]=dict(candidate_derivative_Pa=candidate.tolist(),arithmetic_error_Pa=fmt(noise),finite_step_curvature_Pa=fmt(curvature),total_error_Pa=fmt(total),relative_total_error=mp.nstr(rel,30),relative_arithmetic_error=mp.nstr(norm(noise)/denom,30),relative_finite_step_curvature=mp.nstr(norm(curvature)/denom,30),nonzero_vector_gate=bool(rel<=mp.mpf('1e-3')))
   cleanends=[]
   for end in ends:
    ee={k:v for k,v in end.items() if k not in ('F','P')};ee['paths']={name:{k:v for k,v in vv.items() if k!='values'} for name,vv in ee['paths'].items()};cleanends.append(ee)
   rows.append(dict(epsilon=eps,column=col,endpoints=cleanends,exact_centered_derivative=fmt(exactfd),actual_ray=fmt(ray),smooth_derivative=fmt(smooth),paths=paths))
  print('state',tet,'epsilon',eps,flush=True)
 results.append(dict(tets=g['tets'],unique_state_sha256=key,center_mp_iterations=itc,center_mp_scaled_residual=mp.nstr(nrc,30),rows=rows))
report=dict(schema='numi.cardboard.native-absolute-endpoint-error.v1',inputs=inputs,results=results,physical_steps=0,limits=['All native inputs/outputs come from already source-bound315; no new GPU run.','MP F is the exact sum of two FP32 words, not a rounded binary64 sum. The current source law is rate independent; actual Fdot words are retained but this is not a rate-state derivative qualification.','Absolute component disagreements are observed values for these selected inputs and paths, not formal interval bounds, stochastic noise or universal element/residual noise floors.','Every declared step and both stress paths remain retained including failures.'])
(o/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(o/'run.py').write_bytes(Path(__file__).read_bytes());(o/'README.md').write_text('# Selected native FP32 absolute stress errors\n\nActual two-word F and raw native stress words are compared to90-digit projections at every declared endpoint. Direct and tangent stress paths, finite-step curvature and derivative error remain separate. This is selected material numerical evidence only, with no new native GPU or physical steps.\n');(o/'manifest.json').write_text(json.dumps(dict(inputs=inputs,artifacts={v.name:dict(bytes=v.stat().st_size,sha256=h(v)) for v in o.iterdir() if v.is_file()}),indent=2)+'\n');print('completed',len(results),flush=True)
