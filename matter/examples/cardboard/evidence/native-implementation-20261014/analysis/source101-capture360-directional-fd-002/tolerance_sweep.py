#!/usr/bin/env python3
"""Check the directional FP64 finite-difference result against local solve tolerance.
This is a new CPU-only analysis, not a production or native test.
"""
from __future__ import annotations
import hashlib,importlib.util,json,pathlib,sys,time,warnings,types
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
OUT=pathlib.Path(__file__).resolve().parent/'output'
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
TOLS=(2e-10,2e-12,2e-14)
TARGETS=(1e-5,3e-6,1e-6,3e-7,1e-7)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def need(x,m):
 if not x:raise RuntimeError(m)
def load_module(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def load_tol_ref(tol):
 s=REF.read_text()
 s=s.replace('def project(p,f,old,maxiter=80):','def project(p,f,old,maxiter=80,tol=2e-10):')
 s=s.replace('active=norm>2e-10','active=norm>tol')
 need('active=norm>tol' in s,'reference tolerance shim did not apply')
 ns={'__name__':f'ref_tolerance_{tol}'}
 exec(compile(s,str(REF)+f'#tol={tol}','exec'),ns)
 class Wrapper: pass
 w=Wrapper()
 for k,v in ns.items(): setattr(w,k,v)
 orig=w.project
 w.project=lambda p,f,old,maxiter=80:orig(p,f,old,maxiter=maxiter,tol=tol)
 return w

def main():
 need(not OUT.exists(),f'refusing to overwrite {OUT}')
 need(sha(CAP)==CAP_SHA,'capture hash mismatch')
 fd=load_module(FD,'directional_fd')
 ref0=load_module(REF,'base_ref')
 records=[]
 for line in CAP.open():
  if line.strip():records.append(json.loads(line,parse_int=fd.parse_int))
 h=records[0];tets=[r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0 and r.get('material_index') in (0,1)]
 tets.sort(key=lambda x:x['tet_id']);need(len(tets)==1728,'element count mismatch')
 ptab=fd.f32(h['parameters_by_environment'][0]); ids=np.array([int(r['material_index']) for r in tets],np.int32)
 p=np.stack([ptab[k*28:(k+1)*28] for k in ids]);old=np.stack([fd.f32(r['center']['accepted_state']) for r in tets])
 F=np.stack([fd.pair(r['center'],'F_local') for r in tets]); hdir=np.stack([fd.f32(r['center']['F_direction_local_per_s']) for r in tets])
 dt=fd.f32([r['center']['dt_seconds_bulk_modulus_kind_frame_active'][0] for r in tets]); dFbase=dt[:,None,None]*hdir; dnorm=fd.normrows(dFbase)
 need(np.all(dnorm>0),'zero direction')
 reports=[];t0=time.monotonic()
 for tol in TOLS:
  ref=load_tol_ref(tol)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   z0,it0,r0,ok0,fail0=fd.project_resilient(ref,p,F,old)
   P0=np.full((len(p),3,3),np.nan);P0[ok0]=ref.stress(p[ok0],F[ok0],z0[ok0])
   a0=np.full((len(p),3,3),np.nan)
   if ok0.all():a0=ref.tangent(p,F,old,z0,hdir)
   else:
    for i in np.flatnonzero(ok0):a0[i]=ref.tangent(p[i:i+1],F[i:i+1],old[i:i+1],z0[i:i+1],hdir[i:i+1])[0]
   center=ref0.project(p,F,old)
   zbase=center[0]
   dl0=z0[:,12]-old[:,12];branch0=dl0>0
   scale_reports=[]
   for target in TARGETS:
    eps=target/dnorm;Fp=F+eps[:,None,None]*dFbase;Fm=F-eps[:,None,None]*dFbase
    zp,ip,rp,okp,failp=fd.project_resilient(ref,p,Fp,old)
    zm,im,rm,okm,failm=fd.project_resilient(ref,p,Fm,old)
    Pp=np.full_like(P0,np.nan);Pm=np.full_like(P0,np.nan)
    if okp.any():Pp[okp]=ref.stress(p[okp],Fp[okp],zp[okp])
    if okm.any():Pm[okm]=ref.stress(p[okm],Fm[okm],zm[okm])
    fdv=np.full_like(P0,np.nan);valid=okp&okm
    fdv[valid]=(Pp[valid]-Pm[valid])/(2*eps[valid,None,None]*dt[valid,None,None])
    bp=(zp[:,12]-old[:,12])>0;bm=(zm[:,12]-old[:,12])>0
    stable=valid&(bp==branch0)&(bm==branch0)
    by_material={}
    for mat in (0,1):
      mask=ids==mat
      by_material[str(mat)]={
        'all_central':fd.metrics(fdv,a0,valid&mask),
        'branch_stable_central':fd.metrics(fdv,a0,stable&mask),
        'branch_changes_plus':int(np.sum(okp&mask&(bp!=branch0))),
        'branch_changes_minus':int(np.sum(okm&mask&(bm!=branch0)))
      }
    scale_reports.append({'target_delta_F_frobenius_norm':target,
      'epsilon_per_element_min_max':[float(eps.min()),float(eps.max())],
      'plus_success':int(okp.sum()),'minus_success':int(okm.sum()),
      'scaled_residual_max_plus':float(np.nanmax(rp[okp])) if okp.any() else None,
      'scaled_residual_max_minus':float(np.nanmax(rm[okm])) if okm.any() else None,
      'iterations_max_plus':int(ip[okp].max()) if okp.any() else None,
      'iterations_max_minus':int(im[okm].max()) if okm.any() else None,
      'central_all':fd.metrics(fdv,a0,valid),
      'central_branch_stable':fd.metrics(fdv,a0,stable),
      'one_sided_plus_branch_stable':fd.metrics((Pp-P0)/(eps[:,None,None]*dt[:,None,None]),a0,okp&(bp==branch0)),
      'one_sided_minus_branch_stable':fd.metrics((P0-Pm)/(eps[:,None,None]*dt[:,None,None]),a0,okm&(bm==branch0)),
      'branch_stable_count':int(stable.sum()),
      'plus_transition_tets':[int(tets[i]['tet_id']) for i in np.flatnonzero(okp&(bp!=branch0))[:20]],
      'minus_transition_tets':[int(tets[i]['tet_id']) for i in np.flatnonzero(okm&(bm!=branch0))[:20]],
      'per_material':by_material})
   reports.append({'project_tolerance_scaled':tol,'center_success':int(ok0.sum()),
      'center_scaled_residual_max':float(np.nanmax(r0[ok0])) if ok0.any() else None,
      'center_iterations_max':int(it0[ok0].max()) if ok0.any() else None,
      'center_iterations_p50':float(np.median(it0[ok0])) if ok0.any() else None,
      'center_failure_examples':[{'tet_id':int(tets[i]['tet_id']),'error':fail0[i]} for i in np.flatnonzero(~ok0)[:8]],
      'center_stress_vs_reference_tolerance2e10':fd.metrics(P0,ref0.stress(p,F,zbase),ok0),
      'directional_action':{'absolute_l2_max_Pa_per_s':float(np.nanmax(np.linalg.norm((a0-ref0.tangent(p,F,old,zbase,hdir)).reshape(len(p),-1),axis=1))) if ok0.all() else None,
        'global_l2_relative_vs_tolerance2e10':float(np.linalg.norm(a0-ref0.tangent(p,F,old,zbase,hdir))/np.linalg.norm(ref0.tangent(p,F,old,zbase,hdir))) if ok0.all() else None},
      'scales':scale_reports,'warnings':sorted({str(w.message) for w in caught})})
 OUT.mkdir(parents=True,exist_ok=False)
 report={'schema':'numi.cardboard.source101-capture360-fd-projection-tolerance-sweep.v1',
  'binding':{'capture_path':str(CAP),'capture_sha256':sha(CAP),'source_snapshot':str(BASE/'snapshots/source-boundary-reset-order-101'),
    'reference_path':str(REF),'reference_sha256':sha(REF),'directional_fd_script_sha256':sha(FD),'element_count':len(tets),
    'element_ids_sha256':hashlib.sha256(np.asarray([r['tet_id'] for r in tets],dtype='<u4').tobytes()).hexdigest()},
  'perturbation_relation':'F± = F_center ± epsilon_i * dt * captured_local_grad(delta_v); epsilon_i chosen per tet so ||deltaF||F equals listed target; old accepted history and material parameters fixed.',
  'method':'Only the independent FP64 projection stopping threshold is varied; return-map equations, stress law, dt, F words, accepted history, direction, and branch definition are unchanged. Finite differences are checked against a freshly computed implicit tangent at the same center root for each tolerance.',
  'tolerances':reports,
  'noise_reporting':{'projector_tolerance':'explicit scaled residual per center and endpoint',
    'absolute_error':'per-row Pa/s l2 maxima, medians and component maxima in imported metric output',
    'scale_variation':'tolerance/perturbation sweep is numerical sensitivity, not a measured native noise floor',
    'repeatability':'not measured; no noise-floor claim'},
  'limits':['CPU-only independent reference; no native Metal endpoint perturbation.',
    'Rejected Newton 23 capture only; this is not accepted-state or physical qualification.',
    'Branch is classified by exact positive multiplier increment; smoothed complementarity still applies.',
    'Starch interface elements excluded.',
    'No physical operator, history, or solver tolerance changed.'],
  'wall_seconds':time.monotonic()-t0}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'output':str(OUT/'report.json'),'tolerances':[x['project_tolerance_scaled'] for x in reports],'wall_seconds':report['wall_seconds']},indent=2))
if __name__=='__main__':main()
