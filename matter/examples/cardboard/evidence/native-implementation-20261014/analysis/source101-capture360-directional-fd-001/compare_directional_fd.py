#!/usr/bin/env python3
"""Branch-aware independent FP64 replay for exact source101 capture360 inputs.

No Matter/Metal execution. Perturbs the captured local F as
F(e)=F0 + e*dt*grad(delta_v), holds accepted material history and parameters
fixed, and compares independently projected/stressed endpoints with the
independent implicit tangent action. This is diagnostic evidence for the
rejected Newton linearization only.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, pathlib, struct, time, warnings
import numpy as np

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
SOURCE=BASE/'snapshots/source-boundary-reset-order-101'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
CENTER=BASE/'analysis/source101-capture360-independent-fp64-003/output/center_arrays.npz'
OUT=pathlib.Path(__file__).resolve().parent/'output'
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
EPS_TARGETS=np.array([5e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6],dtype=np.float64)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def parse_int(token):
 return -0.0 if token=='-0' else int(token)
def f32(x): return np.asarray(x,dtype=np.float32).astype(np.float64)
def pair(o,k): return f32(o[k])+f32(o.get(k+'_low',np.zeros_like(o[k])))
def load_ref():
 sp=importlib.util.spec_from_file_location('ref15',REF); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def project_resilient(ref,p,F,old,maxiter=80):
 """Batch project, recursively isolate any failed rows while preserving them."""
 n=len(p); z=np.full((n,15),np.nan); it=np.full(n,-1,dtype=np.int32); rn=np.full(n,np.nan); ok=np.zeros(n,dtype=bool); reasons=[None]*n
 def run(ids):
  try:
   zz,ii,rr=ref.project(p[ids],F[ids],old[ids],maxiter=maxiter)
   if np.all(np.isfinite(zz)) and np.all(np.isfinite(rr)):
    z[ids]=zz; it[ids]=ii; rn[ids]=rr; ok[ids]=True; return
   if len(ids)==1: reasons[ids[0]]='nonfinite projection output'; return
  except Exception as e:
   if len(ids)==1: reasons[ids[0]]=f'{type(e).__name__}: {e}'; return
  mid=len(ids)//2
  run(ids[:mid]); run(ids[mid:])
 run(np.arange(n,dtype=np.int64))
 return z,it,rn,ok,reasons

def normrows(x): return np.linalg.norm(np.asarray(x).reshape(len(x),-1),axis=1)
def metrics(actual,reference,mask=None):
 a=np.asarray(actual).reshape(len(actual),-1); r=np.asarray(reference).reshape(len(reference),-1)
 d=a-r
 if mask is None: mask=np.ones(len(a),dtype=bool)
 mask=np.asarray(mask,dtype=bool)&np.all(np.isfinite(a),axis=1)&np.all(np.isfinite(r),axis=1)
 if not np.any(mask): return {'rows':0,'absolute_l2_max':None,'global_l2_relative':None,'per_row_relative_p50':None,'per_row_relative_p95':None,'per_row_relative_max':None}
 dn=np.linalg.norm(d[mask],axis=1); rn=np.linalg.norm(r[mask],axis=1)
 rel=np.divide(dn,rn,out=np.full_like(dn,np.nan),where=rn>0)
 finite=rel[np.isfinite(rel)]
 return {'rows':int(mask.sum()),'absolute_l2_max':float(dn.max()),'absolute_l2_median':float(np.median(dn)),
         'max_component_abs':float(np.max(np.abs(d[mask]))),
         'global_l2_relative':float(np.linalg.norm(d[mask])/np.linalg.norm(r[mask])) if np.linalg.norm(r[mask]) else None,
         'per_row_relative_p50':float(np.nanpercentile(finite,50)) if finite.size else None,
         'per_row_relative_p95':float(np.nanpercentile(finite,95)) if finite.size else None,
         'per_row_relative_max':float(np.nanmax(finite)) if finite.size else None,
         'reference_l2_median':float(np.median(rn))}

def state_direction(ref,p,F,old,z,h):
 # z'=d z / d alpha for F(alpha)=F+alpha*h; full implicit local chain.
 rf=np.imag(ref.residual(p,F.astype(np.complex128)+1e-25j*h,old,z))/1e-25
 j=ref.jacobian(p,F,old,z)
 return np.linalg.solve(j,-rf[...,None])[...,0]

def main():
 need(not OUT.exists(),f'refusing to overwrite {OUT}')
 need(sha(CAP)==CAP_SHA,'capture SHA mismatch')
 records=[]
 with CAP.open() as f:
  for n,line in enumerate(f,1):
   if line.strip():
    try: records.append(json.loads(line,parse_int=parse_int))
    except Exception as e: raise RuntimeError(f'bad capture JSON line {n}: {e}')
 header=records[0]; need(header.get('record')=='header','missing header')
 need((header.get('root'),header.get('newton_iteration'),header.get('world_fingerprint_hex'))==(2,23,'6de3475316b7e8b9'),'selection mismatch')
 need(records[-1].get('record')=='integrity' and records[-1].get('production_inputs_bitwise_unchanged') is True,'capture integrity absent')
 tets=[r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0 and r.get('material_index') in (0,1)]
 tets.sort(key=lambda r:r['tet_id']); need(len(tets)==1728,'paper element count mismatch')
 need(all(r.get('material_state_count')==15 for r in tets),'expected 15 state elements')
 ptab=f32(header['parameters_by_environment'][0])
 ids=np.array([int(r['material_index']) for r in tets],dtype=np.int32)
 p=np.stack([ptab[k*28:(k+1)*28] for k in ids])
 old=np.stack([f32(r['center']['accepted_state']) for r in tets])
 F=np.stack([pair(r['center'],'F_local') for r in tets])
 captured_z=np.stack([pair(r['center'],'projected_state') for r in tets])
 captured_P=np.stack([f32(r['center']['P_local_Pa']) for r in tets])
 native_C=np.stack([f32(r['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9,9) for r in tets])
 h=np.stack([f32(r['center']['F_direction_local_per_s']) for r in tets])
 dt=f32([r['center']['dt_seconds_bulk_modulus_kind_frame_active'][0] for r in tets])
 need(np.all(dt==np.float64(np.float32(header['dt_seconds_by_environment'][0]))),'dt mismatch')
 need(np.all(np.isfinite(F)) and np.all(np.isfinite(old)) and np.all(np.isfinite(h)),'nonfinite capture input')
 ref=load_ref()
 t0=time.monotonic()
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  z0,it0,r0,ok0,fail0=project_resilient(ref,p,F,old)
  need(ok0.all(),'center independent projection failed')
  P0=ref.stress(p,F,z0)
  Cref=np.empty((len(p),9,9))
  for c in range(9):
   b=np.zeros((len(p),3,3)); b[:,c//3,c%3]=1
   Cref[:,:,c]=ref.tangent(p,F,old,z0,b).reshape(len(p),9)
  action_ref=np.einsum('nij,nj->ni',Cref,h.reshape(len(p),9)).reshape(len(p),3,3)
  action_native=np.einsum('nij,nj->ni',native_C,h.reshape(len(p),9)).reshape(len(p),3,3)
  dz_h=state_direction(ref,p,F,old,z0,h)

 dt_h=dt[:,None,None]*h
 dirnorm=normrows(dt_h)
 need(np.all(dirnorm>0),'zero captured material direction')
 dl0=z0[:,12]-old[:,12]
 q0,flow0=ref.hill(p,z0[:,6:12])
 radius0=1+p[:,17]*p[:,19]*p[:,21]*z0[:,13]/p[:,10]+p[:,18]*p[:,20]*p[:,22]*z0[:,14]/p[:,10]
 gap0=radius0-q0
 dgap_h=(p[:,17]*p[:,19]*p[:,21]/p[:,10]*dz_h[:,13]+p[:,18]*p[:,20]*p[:,22]/p[:,10]*dz_h[:,14]-np.sum(flow0*dz_h[:,6:12],axis=1))
 dl_deps=dt*dz_h[:,12]
 gap_deps=dt*dgap_h
 active0=dl0>0
 # Exact return-map event estimates; used only to choose finite probes, never as outcomes.
 eps_cross=np.full(len(p),np.nan); side=np.zeros(len(p),dtype=np.int8); event_kind=[]
 for i in range(len(p)):
  if active0[i] and np.isfinite(dl_deps[i]) and dl_deps[i]!=0:
   x=-dl0[i]/dl_deps[i]; kind='lambda_increment_to_zero'
  elif not active0[i] and np.isfinite(gap_deps[i]) and gap_deps[i]!=0:
   x=-gap0[i]/gap_deps[i]; kind='yield_gap_to_zero'
  else:
   x=np.nan; kind='unresolved'
  if np.isfinite(x): eps_cross[i]=abs(x); side[i]=1 if x>0 else -1
  event_kind.append(kind)

 out_scales=[]; previous_fd=None; endpoint_cache={}
 for target in EPS_TARGETS:
  eps=target/dirnorm
  Fp=F+eps[:,None,None]*dt_h
  Fm=F-eps[:,None,None]*dt_h
  zp,ip,rp,okp,failp=project_resilient(ref,p,Fp,old)
  zm,im,rm,okm,failm=project_resilient(ref,p,Fm,old)
  Pp=np.full_like(P0,np.nan); Pm=np.full_like(P0,np.nan)
  if okp.any(): Pp[okp]=ref.stress(p[okp],Fp[okp],zp[okp])
  if okm.any(): Pm[okm]=ref.stress(p[okm],Fm[okm],zm[okm])
  valid=okp&okm
  fd=np.full_like(P0,np.nan)
  fd[valid]=(Pp[valid]-Pm[valid])/(2*eps[valid,None,None]*dt[valid,None,None])
  fdp=np.full_like(P0,np.nan); fdm=np.full_like(P0,np.nan)
  fdp[okp]=(Pp[okp]-P0[okp])/(eps[okp,None,None]*dt[okp,None,None])
  fdm[okm]=(P0[okm]-Pm[okm])/(eps[okm,None,None]*dt[okm,None,None])
  bp=(zp[:,12]-old[:,12])>0; bm=(zm[:,12]-old[:,12])>0
  transition_plus=okp&(bp!=active0); transition_minus=okm&(bm!=active0)
  record={
   'target_delta_F_frobenius_norm':float(target),
   'epsilon_per_element_range': [float(np.nanmin(eps)),float(np.nanmax(eps))],
   'endpoint_projection':{'plus_success':int(okp.sum()),'minus_success':int(okm.sum()),
     'plus_residual_scaled_max':float(np.nanmax(rp[okp])) if okp.any() else None,
     'minus_residual_scaled_max':float(np.nanmax(rm[okm])) if okm.any() else None,
     'plus_iterations_max':int(np.max(ip[okp])) if okp.any() else None,
     'minus_iterations_max':int(np.max(im[okm])) if okm.any() else None,
     'failure_examples_plus':[{'tet_id':int(tets[i]['tet_id']),'error':failp[i]} for i in np.flatnonzero(~okp)[:8]],
     'failure_examples_minus':[{'tet_id':int(tets[i]['tet_id']),'error':failm[i]} for i in np.flatnonzero(~okm)[:8]]},
   'absolute_perturbation':{'requested_delta_F_norm_median':float(np.median(eps*dirnorm)),
     'realized_plus_delta_F_norm_median':float(np.median(normrows(Fp-F))),
     'realized_minus_delta_F_norm_median':float(np.median(normrows(Fm-F))),
     'max_plus_delta_roundoff_frobenius':float(np.max(normrows((Fp-F)-eps[:,None,None]*dt_h))),
     'max_minus_delta_roundoff_frobenius':float(np.max(normrows((Fm-F)+eps[:,None,None]*dt_h)))},
   'central_derivative_vs_independent_implicit_tangent_Pa_per_s':metrics(fd,action_ref,valid),
   'plus_one_sided_derivative_vs_tangent_Pa_per_s':metrics(fdp,action_ref,okp),
   'minus_one_sided_derivative_vs_tangent_Pa_per_s':metrics(fdm,action_ref,okm),
   'branch_labels':{'center_plastic':int(active0.sum()),'plus_plastic':int((bp&okp).sum()),
     'minus_plastic':int((bm&okm).sum()),'plus_transition_rows':int(transition_plus.sum()),
     'minus_transition_rows':int(transition_minus.sum()),
     'transition_plus_tet_ids':[int(tets[i]['tet_id']) for i in np.flatnonzero(transition_plus)[:40]],
     'transition_minus_tet_ids':[int(tets[i]['tet_id']) for i in np.flatnonzero(transition_minus)[:40]],
     'central_derivative_transition_rows':metrics(fd,action_ref,valid&(transition_plus|transition_minus))},
  }
  if previous_fd is not None:
   common=valid&np.all(np.isfinite(previous_fd),axis=(1,2))
   record['central_derivative_interlevel_change_Pa_per_s']=metrics(fd,previous_fd,common)
  else: record['central_derivative_interlevel_change_Pa_per_s']=None
  previous_fd=fd.copy(); out_scales.append(record)

 # Probe the closest predicted event on each side at 0.5x and 1.25x predicted crossing.
 eligible=np.flatnonzero(np.isfinite(eps_cross)&(eps_cross>1e-10)&(eps_cross<0.1))
 order=eligible[np.argsort(eps_cross[eligible])][:32]
 transition_probes=[]
 for factor in (0.5,1.25):
  if len(order)==0: continue
  eps=factor*eps_cross[order]*side[order]
  Fq=F[order]+eps[:,None,None]*dt_h[order]
  zq,iq,rq,okq,failq=project_resilient(ref,p[order],Fq,old[order])
  Pq=np.full((len(order),3,3),np.nan)
  if okq.any(): Pq[okq]=ref.stress(p[order][okq],Fq[okq],zq[okq])
  dq=(Pq-P0[order])/(eps[:,None,None]*dt[order,None,None])
  branchq=(zq[:,12]-old[order,12])>0
  rows=[]
  for j,i in enumerate(order):
   rows.append({'tet_id':int(tets[i]['tet_id']),'material_index':int(ids[i]),
    'event_kind':event_kind[i],'center_branch':'plastic' if active0[i] else 'elastic',
    'predicted_signed_epsilon':float(eps_cross[i]*side[i]),'factor':factor,
    'actual_signed_epsilon':float(eps[j]),'project_success':bool(okq[j]),
    'project_scaled_residual':float(rq[j]) if okq[j] else None,
    'iterations':int(iq[j]) if okq[j] else None,
    'endpoint_branch':'plastic' if okq[j] and branchq[j] else ('elastic' if okq[j] else 'failed'),
    'branch_changed':bool(okq[j] and branchq[j]!=active0[i]),
    'lambda_increment_center':float(dl0[i]),
    'lambda_increment_endpoint':float(zq[j,12]-old[i,12]) if okq[j] else None,
    'yield_gap_center':float(gap0[i]),
    'yield_gap_endpoint':float((1+p[i,17]*p[i,19]*p[i,21]*zq[j,13]/p[i,10]+p[i,18]*p[i,20]*p[i,22]*zq[j,14]/p[i,10]-ref.hill(p[i:i+1],zq[j:j+1,6:12])[0][0]) if okq[j] else np.nan),
    'one_sided_derivative_error_abs_l2_Pa_per_s':float(np.linalg.norm(dq[j]-action_ref[i])) if okq[j] else None,
    'one_sided_derivative_reference_l2_Pa_per_s':float(np.linalg.norm(action_ref[i]))})
  transition_probes.append({'factor_of_predicted_event':factor,'rows':rows})

 report={
  'schema':'numi.cardboard.source101-capture360-directional-fp64-material-fd.v1',
  'binding':{'capture_path':str(CAP),'capture_sha256':sha(CAP),'source_snapshot':str(SOURCE),
   'reference_path':str(REF),'reference_sha256':sha(REF),'center_array_file_sha256':sha(CENTER),
   'selection':{'environment':0,'root_control_step':2,'newton_iteration':23,'world_fingerprint_hex':header['world_fingerprint_hex']},
   'elements':1728,'materials':{'liner':int(np.sum(ids==0)),'medium':int(np.sum(ids==1))},
   'raw_json_integer_minus_zero_preserved':True},
  'input_relation':{'accepted_history':'exact captured float32 widened values; same immutable old state reused for center and every plus/minus projection',
   'F_center':'captured float32 high plus captured float32 low, promoted and summed in FP64',
   'direction':'captured local grad(delta_v), per second',
   'perturbation':'F(e)=F_center + e*dt*grad(delta_v); therefore delta(Fdot)=e*grad(delta_v); dt and accepted history fixed',
   'material_frame':'captured local material frame; parameters are captured float32 words widened to FP64',
   'scope':'independent equation replay, not a native endpoint perturbation; no source run or solver state modified'},
  'center_reference':{'all_1728_projected':True,'scaled_residual_max':float(np.max(r0)),
   'iterations_max':int(np.max(it0)),'center_state_vs_capture':metrics(captured_z,z0),
   'center_stress_vs_capture_Pa':metrics(captured_P,P0),
   'native_cached_full_9x9_vs_independent_reference_at_center':metrics(native_C,Cref),
   'native_cached_action_vs_independent_reference_action_Pa_per_s':metrics(action_native,action_ref),
   'native_cached_action_vs_independent_max_component_abs_Pa_per_s':float(np.max(np.abs(action_native-action_ref))),
   'zero_reference_action_rows':int(np.sum(normrows(action_ref)==0))},
  'branch_direction':{'reference_center_plastic_rows':int(active0.sum()),'reference_center_elastic_rows':int((~active0).sum()),
    'smallest_positive_lambda_increment':float(np.min(dl0[dl0>0])) if np.any(dl0>0) else None,
    'elastic_yield_gap_min':float(np.min(gap0[~active0])) if np.any(~active0) else None,
    'predicted_event_rows':int(np.isfinite(eps_cross).sum()),
    'event_epsilon_abs_p10_p50_p90':[float(x) for x in np.nanpercentile(eps_cross[np.isfinite(eps_cross)],[10,50,90])] if np.isfinite(eps_cross).any() else [],
    'closest_event_rows':[{'tet_id':int(tets[i]['tet_id']),'material_index':int(ids[i]),'kind':event_kind[i],
       'center_branch':'plastic' if active0[i] else 'elastic','epsilon_abs':float(eps_cross[i]),'signed_side':int(side[i]),
       'dl_center':float(dl0[i]),'gap_center':float(gap0[i]),'dl_per_epsilon':float(dl_deps[i]),'gap_per_epsilon':float(gap_deps[i])}
       for i in order]},
  'finite_difference_scales':out_scales,
  'predicted_transition_one_sided_probes':transition_probes,
  'noise_reporting':{'projection_residuals':'reported per scale; no repeated-run perturbation/noise floor is inferred',
   'absolute_stress_and_tangent_action_errors':'reported in Pa/s for each scale, alongside reference-action norms',
   'interlevel_difference':'reported as scale-to-scale derivative variation, not a measurement-noise estimate',
   'endpoint_roundoff':'reported as FP64 arithmetic realization difference from requested delta-F; does not include native FP32 geometry quantization'},
  'warnings':sorted({str(w.message) for w in caught}),
  'limits':['The capture is a rejected Newton 23 state, not an accepted state or physical result.',
   'The reference is an independent handwritten FP64 transcription, not an executable proof or native implementation.',
   'This finite-difference sweep uses exact captured velocity-gradient direction and fixed dt/history; it is not all nine independent directions at every perturbed state.',
   'No FP32 endpoint quantization/noise-floor result is asserted; no native plus/minus replay is performed.',
   'The 192 starch elements use a different law and are excluded.',
   'No convergence, assembled-operator, calibration, or physical qualification claim is made.'],
  'wall_seconds':time.monotonic()-t0
 }
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'output':str(OUT/'report.json'),'center_native_tangent_l2rel':report['center_reference']['native_cached_full_9x9_vs_independent_reference_at_center']['global_l2_relative'],
  'scales':len(out_scales),'transition_probe_groups':len(transition_probes),'wall_seconds':report['wall_seconds']},indent=2))

if __name__=='__main__': main()
