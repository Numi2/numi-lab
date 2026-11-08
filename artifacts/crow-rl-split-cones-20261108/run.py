from pathlib import Path
import hashlib,json,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-split-cones-20261108';B=R/'build/crow-split-cones-build-01'
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());runtime=json.loads((A/'runtime.json').read_text())
for name,v in plan['sources'].items():assert sha(A/name)==v['sha256']
for name,v in {**plan['inputs'],**runtime['files']}.items():assert sha(R/name)==v['sha256']
cases=json.loads((A/'cases.json').read_text());assert [c['name'] for c in cases]==plan['order']
# Calibrate independent derivatives before any native observation.
calibration=ref.calibrate();fd=[]
for c in cases:
 if c['group']!='constructed' or c['name']=='seven-regions':continue
 fixture=ref.parse((R/c['path']).read_bytes());v,q,h=ref.reference(fixture);b=fixture['blocks'][0];w=q['w'];eps=1e-5
 D=np.column_stack([(ref.oracle(w+np.eye(len(w))[k]*eps,b['r'],b['s'],*b['bounds'][:2])[0]-ref.oracle(w-np.eye(len(w))[k]*eps,b['r'],b['s'],*b['bounds'][:2])[0])/(-2*eps) for k in range(len(w))])
 error=float(np.max(np.abs(q['W'][0]-D)));assert error<2e-8;fd.append(dict(name=c['name'],error=error))
write(A/'derivative-calibration.json',dict(prior=calibration,cases=fd))
out=B/'observations';out.mkdir();execution=[];started=time.monotonic()
for c in cases:
 for repeat in range(plan['repeats']):
  assert time.monotonic()-started<plan['limits']['total_seconds']
  name=c['name']+f'-{repeat}';cmd=[B/'probe',B/'solver.metallib',R/c['path'],out/name]
  start=time.monotonic()
  with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
   try:code=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr,timeout=plan['limits']['per_command_seconds']).returncode
   except subprocess.TimeoutExpired:code=None
  execution.append(dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start));write(A/'execution.json',execution)
  print(json.dumps(execution[-1]),flush=True);assert code==0
rows=[];allregions=set();repeated=True
for c in cases:
 fixture=ref.parse((R/c['path']).read_bytes());nv,rs,nb=fixture['nv'],fixture['rows'],fixture['nb'];name=c['name']
 dirs=[out/(name+f'-{i}') for i in range(2)]
 repeated &= all((dirs[0]/p).read_bytes()==(dirs[1]/p).read_bytes() for p in ['status.bin','scratch.f32x2'])
 si=struct.unpack('<16I16f',(dirs[0]/'status.bin').read_bytes());data=np.fromfile(dirs[0]/'scratch.f32x2',dtype='<f4').reshape(-1,2).astype(float);ds=data.sum(axis=1)
 v=ds[:nv];p=ds[nv+rs:nv+2*rs];g=ds[nv+2*rs:2*nv+2*rs];W=ds[2*nv+2*rs:2*nv+2*rs+36*nb]
 row=dict(name=name,group=c['group'],code=si[0],newton=si[1],backtracks=si[10],native_residual=si[16],native_feasibility=si[20],region_counts=list(si[2:9]))
 if c['group']=='invalid':
  row['typed_start_unchanged']=si[0]==2 and np.array_equal(v,fixture['start']);rows.append(row);continue
 q=ref.evaluate(fixture,v);reference,rq,history=ref.reference(fixture)
 row.update(independent_residual=q['res'],independent_feasibility=q['feas'],rounded_residual=ref.evaluate(fixture,v.astype('f4').astype(float))['res'],
 reference_residual=rq['res'],reference_iterations=len(history)-1,reference_velocity_abs_error=float(np.max(np.abs(v-reference))),
 impulse_scaled_error=float(np.max(np.abs(p-q['p'])/(1+np.abs(q['p'])))) if rs else 0.,
 gradient_scaled_error=float(np.max(np.abs(g-q['g'])/(1+np.abs(q['g'])))),independent_regions=q['regions'])
 errors=[]
 for bi,D in enumerate(q['W']):
  n=D.shape[0];actual=W[bi*36:bi*36+n*n].reshape(n,n);errors.append(float(np.max(np.abs(actual-D)/(1+np.abs(D)))))
 row['derivative_scaled_error']=max(errors,default=0.)
 if c['group']!='rounding':allregions.update(q['regions'])
 rows.append(row)
valid=[r for r in rows if r['group'] not in ['rounding','invalid']]
observed=dict(valid_split_native_and_independent_residual_max=max(max(r['native_residual'],r['independent_residual']) if r['code']==0 else float(1) for r in valid),
 valid_split_feasibility_max=max(max(r['native_feasibility'],r['independent_feasibility']) for r in valid),
 reference_velocity_abs_error_max=max(r['reference_velocity_abs_error'] for r in valid),
 impulse_scaled_error_max=max(r['impulse_scaled_error'] for r in valid),derivative_scaled_error_max=max(r['derivative_scaled_error'] for r in valid),
 gradient_scaled_error_max=max(r['gradient_scaled_error'] for r in valid),
 all_seven_regions=allregions=={'vertex','interior','boundary','cap-interior','cap-boundary','scalar-free','scalar-bound'},
 repeated_outputs_exact=repeated,rounding_ablation_rejected=all(r['code'] in [6,7] and r['independent_residual']>1e-10 for r in rows if r['group']=='rounding'),
 invalid_r_typed_and_start_unchanged=all(r['typed_start_unchanged'] for r in rows if r['group']=='invalid'))
supported={k:(v==plan['predictions'][k] if isinstance(v,bool) else v<=plan['predictions'][k]) for k,v in observed.items()}
rawbytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert rawbytes<plan['limits']['raw_bytes']
write(A/'case-analysis.json',rows)
report=dict(schema='crow-split-cones-result-v1',observations=observed,predictions_supported=supported,all_predictions_supported=all(supported.values()),
 commands=len(execution),valid_cases=len(valid),cases=len(cases),raw_bytes=rawbytes,seconds=time.monotonic()-started,
 calibration_fd_max_error=max(x['error'] for x in fd),production_changed=False,goal_complete=False,scope=plan['scope'])
write(A/'result.json',report);print(json.dumps(report),flush=True)
