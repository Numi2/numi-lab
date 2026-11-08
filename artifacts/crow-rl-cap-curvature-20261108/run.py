from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-cap-curvature-20261108';B=R/'build/crow-cap-curvature-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());runtime=json.loads((A/'runtime.json').read_text())
for name,f in plan['sources'].items():assert sha(A/name)==f['sha256']
for name,f in {**plan['inputs'],**runtime['files']}.items():assert sha(R/name)==f['sha256']
ref={};exec(compile((A/'independent-reference.py').read_text().split('calibration=calibrate()')[0],'independent-reference','exec'),ref);prior_calibration=ref['calibrate']()
def oracle(w,r,s,adh,cap):
 w=np.asarray(w,float);r=np.asarray(r,float);s=np.asarray(s,float);n=len(w)
 shifted=w.copy();shifted[0]-=r[0]*adh
 p,D,kind=ref['cone'](shifted,r,s);p[0]-=adh
 if cap>0 and p[0]>cap:
  C=cap+adh;z=-w[1:]/(r[1:]*s[1:]);d=r[1:]*s[1:]**2;c=w[1:]*s[1:]
  if np.linalg.norm(z)<=C:return np.r_[cap,s[1:]*z],np.diag(np.r_[0.,1/r[1:]]),'cap-interior'
  lo=0.;hi=1.
  while np.linalg.norm(c/(d+hi))>C:hi*=2
  for _ in range(120):
   mid=(lo+hi)/2
   if mid==lo or mid==hi:break
   if np.linalg.norm(c/(d+mid))>C:lo=mid
   else:hi=mid
  alpha=(lo+hi)/2;y=-c/(d+alpha);inv=1/(d+alpha);a=inv*y;D=np.zeros((n,n));D[1:,1:]=(np.diag(inv)-np.outer(a,a)/(y@a))*np.outer(s[1:],s[1:])
  return np.r_[cap,s[1:]*y],D,'cap-boundary'
 return p,D,kind
cases=json.loads((A/'cases.json').read_text())
calibration=[]
for c in cases:
 if c['group'] not in ['closed','anisotropic']:continue
 w=np.array(c['w']);p,d,kind=oracle(w,c['r'],c['s'],c['adh'],c['cap'])
 eps=1e-5
 numerical=np.column_stack([(oracle(w+np.eye(c['n'])[i]*eps,c['r'],c['s'],c['adh'],c['cap'])[0]-oracle(w-np.eye(c['n'])[i]*eps,c['r'],c['s'],c['adh'],c['cap'])[0])/(-2*eps) for i in range(c['n'])])
 error=float(np.max(np.abs(d-numerical)));assert error<2e-8 and kind=='cap-boundary';calibration.append(dict(name=c['name'],derivative_fd_error=error))
for c in cases:
 if c['group']=='closed':
  p,d,_=oracle(c['w'],c['r'],c['s'],c['adh'],c['cap']);C=c['cap']+c['adh'];u=np.array([.6,.8]);expected=np.zeros((3,3));expected[1:,1:]=C/5*(np.eye(2)-np.outer(u,u))
  assert np.max(np.abs(d-expected))<1e-13 and np.max(np.abs(p-np.r_[c['cap'],-C*u]))<1e-13
out=B/'observations';out.mkdir();execution=[]
for name in plan['order']:
 arm=name.split('-')[0]
 for mode in ['project','closure']:
  cmd=[B/'project-probe',B/(arm+'.metallib'),B/'cases.bin',out/(name+'.f32')] if mode=='project' else [B/'closure-probe',B/(arm+'.metallib'),out/(name+'.jsonl')]
  start=time.monotonic()
  with (out/(name+'-'+mode+'.stdout')).open('xb') as stdout,(out/(name+'-'+mode+'.stderr')).open('xb') as stderr:
   try:code=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr,timeout=60).returncode
   except subprocess.TimeoutExpired:code=None
  rec=dict(name=name,mode=mode,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start);execution.append(rec);write(A/'execution.json',execution);print(json.dumps(rec),flush=True);assert code==0
old=np.fromfile(out/'old-0.f32',dtype='<f4').reshape(-1,48);fixed=np.fromfile(out/'fixed-0.f32',dtype='<f4').reshape(-1,48);assert old.shape==fixed.shape==(33,48) and np.isfinite(old).all() and np.isfinite(fixed).all() and np.all(old[:,45]==1) and np.all(fixed[:,45]==1)
rows=[]
for i,c in enumerate(cases):
 p,d,kind=oracle(c['w'],c['r'],c['s'],c['adh'],c['cap']);n=c['n']
 before=old[i,6:6+n*n].reshape(n,n).astype(float);after=fixed[i,6:6+n*n].reshape(n,n).astype(float)
 rec=dict(name=c['name'],group=c['group'],radius=c['cap']+c['adh'],region=kind,baseline_derivative_abs_error=float(np.max(np.abs(before-d))),candidate_derivative_abs_error=float(np.max(np.abs(after-d))),candidate_impulse_abs_error=float(np.max(np.abs(fixed[i,:n]-p))),candidate_derivative_symmetry_error=float(np.max(np.abs(after-after.T))),candidate_derivative_eigen_min=float(np.linalg.eigvalsh((after+after.T)/2)[0]),impulse_merit_bits_exact=old[i,:6].tobytes()==fixed[i,:6].tobytes() and old[i,42:].tobytes()==fixed[i,42:].tobytes(),derivative_bits_exact=old[i,6:42].tobytes()==fixed[i,6:42].tobytes())
 rows.append(rec)
oldc=[json.loads(x) for x in (out/'old-0.jsonl').read_text().splitlines()];newc=[json.loads(x) for x in (out/'fixed-0.jsonl').read_text().splitlines()];assert len(oldc)==len(newc)==10
for r in oldc+newc:assert r['code'] in [0,7]
repeat=all((out/(arm+'-0'+suffix)).read_bytes()==(out/(arm+'-1'+suffix)).read_bytes() for arm in ['old','fixed'] for suffix in ['.f32','.jsonl'])
observed=dict(candidate_derivative_abs_error_max=max(r['candidate_derivative_abs_error'] for r in rows),baseline_nonunit_closed_derivative_abs_error_min=min(r['baseline_derivative_abs_error'] for r in rows if r['group']=='closed' and r['radius']!=1),unit_control_baseline_abs_error_max=max(r['baseline_derivative_abs_error'] for r in rows if r['group']=='closed' and r['radius']==1),projection_impulses_and_merits_bit_exact=all(r['impulse_merit_bits_exact'] for r in rows),unaffected_control_derivatives_bit_exact=all(r['derivative_bits_exact'] for r in rows if r['group']=='control' or r['radius']==1),candidate_one_newton_closure_passes=sum(r['code']==0 for r in newc),baseline_unit_one_newton_closure_passes=sum(r['code']==0 for r in oldc if r['radius']==1),baseline_nonunit_one_newton_rejections=sum(r['code']==7 for r in oldc if r['radius']!=1),repeated_binary_and_typed_outputs_exact=repeat)
pred=plan['predictions'];passed={k:(v==pred[k] if isinstance(pred[k],bool) or not k.endswith(('_max','_min')) else v>=pred[k] if k.endswith('_min') else v<=pred[k]) for k,v in observed.items()}
write(A/'projection-analysis.json',rows);write(A/'closure-analysis.json',dict(baseline=oldc,candidate=newc))
report=dict(schema='crow-cap-derivative-result-v1',observations=observed,predictions_supported=passed,all_predictions_supported=all(passed.values()),calibration=dict(prior=prior_calibration,weighted_ball_fd_cases=len(calibration),weighted_ball_fd_max_error=max(x['derivative_fd_error'] for x in calibration)),native_commands=len(execution),limits=plan['limits'],production_changed=False,goal_complete=False)
write(A/'result.json',report);print(json.dumps(report),flush=True)
