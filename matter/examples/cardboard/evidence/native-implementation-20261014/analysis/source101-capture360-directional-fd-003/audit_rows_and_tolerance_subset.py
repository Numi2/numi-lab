#!/usr/bin/env python3
"""Row-resolved directional audit and bounded projection-tolerance subset.
CPU-only independent equations; no native execution or production edits.
"""
import hashlib, importlib.util, json, pathlib, warnings, time, csv, math
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
OUT=pathlib.Path(__file__).resolve().parent/'output'
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
TARGET=1.0e-6
TOLS=(2e-10,2e-12,2e-14)
TOL_TARGETS=(1e-6,3e-7,1e-7)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load_module(path,name):
 sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def load_tol_ref(tol):
 s=REF.read_text().replace('def project(p,f,old,maxiter=80):','def project(p,f,old,maxiter=80,tol=2e-10):')
 s=s.replace('active=norm>2e-10','active=norm>tol')
 if 'active=norm>tol' not in s: raise RuntimeError('failed to apply scaled projection tolerance parameter')
 ns={'__name__':f'ref_tol_{tol}'}; exec(compile(s,str(REF)+f'#tol={tol}','exec'),ns)
 class W: pass
 w=W()
 for k,v in ns.items(): setattr(w,k,v)
 orig=w.project
 w.project=lambda p,f,old,maxiter=80:orig(p,f,old,maxiter=maxiter,tol=tol)
 return w

def metric_detail(x,y,ids,mask=None):
 x=np.asarray(x).reshape(len(x),-1); y=np.asarray(y).reshape(len(y),-1)
 d=x-y; ax=np.linalg.norm(d,axis=1); den=np.linalg.norm(y,axis=1)
 good=np.all(np.isfinite(x),axis=1)&np.all(np.isfinite(y),axis=1)
 if mask is not None: good &= np.asarray(mask,bool)
 rows=np.flatnonzero(good); rel=np.full(len(ids),np.nan); nz=good&(den>0); rel[nz]=ax[nz]/den[nz]
 def row(i): return {'tet_id':int(ids[i]),'absolute_error':float(ax[i]),'reference_norm_denominator':float(den[i]),'relative_error':float(rel[i]) if np.isfinite(rel[i]) else None}
 if not len(rows): return {'rows':0}
 order=rows[np.argsort(rel[rows])]
 absorder=rows[np.argsort(ax[rows])]
 denorder=rows[np.argsort(den[rows])]
 return {'rows':int(len(rows)), 'global_l2_relative':float(np.linalg.norm(d[rows])/np.linalg.norm(y[rows])) if np.linalg.norm(y[rows]) else None,
  'absolute_error_l2_max':float(ax[rows].max()),'absolute_error_l2_median':float(np.median(ax[rows])),
  'reference_denominator_l2_min':float(den[rows].min()),'reference_denominator_l2_median':float(np.median(den[rows]),),'reference_denominator_l2_p01':float(np.percentile(den[rows],1)),
  'exact_zero_denominators':int(np.sum(den[rows]==0)), 'near_zero_definition':'Bottom ten nonzero reference-action L2 norms; descriptive order statistic, no pass threshold.',
  'worst_relative_rows':[row(int(i)) for i in order[-5:][::-1]], 'worst_absolute_rows':[row(int(i)) for i in absorder[-5:][::-1]],
  'smallest_nonzero_reference_denominator_rows':[row(int(i)) for i in denorder if den[i]>0][:10]}

def main():
 if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
 if sha(CAP)!=CAP_SHA: raise RuntimeError('capture hash mismatch')
 fd=load_module(FD,'fd_helpers')
 records=[]
 for ln,line in enumerate(CAP.open(),1):
  if line.strip(): records.append(json.loads(line,parse_int=fd.parse_int))
 header=records[0]
 tets=[r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0 and r.get('material_index') in (0,1)]
 tets.sort(key=lambda r:r['tet_id'])
 if len(tets)!=1728: raise RuntimeError('expected 1728 source101 paper tets')
 ids=np.array([int(r['tet_id']) for r in tets],dtype=np.int32)
 mats=np.array([int(r['material_index']) for r in tets],dtype=np.int32)
 ptab=fd.f32(header['parameters_by_environment'][0]); p=np.stack([ptab[k*28:(k+1)*28] for k in mats])
 old=np.stack([fd.f32(r['center']['accepted_state']) for r in tets])
 F=np.stack([fd.pair(r['center'],'F_local') for r in tets]); captured_z=np.stack([fd.pair(r['center'],'projected_state') for r in tets])
 nativeC=np.stack([fd.f32(r['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9,9) for r in tets])
 h=np.stack([fd.f32(r['center']['F_direction_local_per_s']) for r in tets]); dt=fd.f32([r['center']['dt_seconds_bulk_modulus_kind_frame_active'][0] for r in tets])
 dFbase=dt[:,None,None]*h; direction_norm=fd.normrows(dFbase)
 if np.any(direction_norm<=0): raise RuntimeError('captured direction contains zero norm')
 ref0=load_module(REF,'ref_base'); start=time.monotonic()
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  z0,it0,r0,ok0,fail0=fd.project_resilient(ref0,p,F,old)
  if not ok0.all(): raise RuntimeError('reference center failed')
  Cref=np.empty((len(p),9,9))
  for c in range(9):
   b=np.zeros((len(p),3,3)); b[:,c//3,c%3]=1
   Cref[:,:,c]=ref0.tangent(p,F,old,z0,b).reshape(len(p),9)
  action_ref=np.einsum('nij,nj->ni',Cref,h.reshape(len(p),9)).reshape(len(p),3,3)
  action_native=np.einsum('nij,nj->ni',nativeC,h.reshape(len(p),9)).reshape(len(p),3,3)
  eps=TARGET/direction_norm
  Fp=F+eps[:,None,None]*dFbase; Fm=F-eps[:,None,None]*dFbase
  zp,ip,rp,okp,failp=fd.project_resilient(ref0,p,Fp,old)
  zm,im,rm,okm,failm=fd.project_resilient(ref0,p,Fm,old)
  if not(okp.all() and okm.all()): raise RuntimeError('base-tolerance FD projection failed')
  Pp=ref0.stress(p,Fp,zp); Pm=ref0.stress(p,Fm,zm)
  fdcent=(Pp-Pm)/(2*eps[:,None,None]*dt[:,None,None])
  bp=(zp[:,12]-old[:,12])>0; bm=(zm[:,12]-old[:,12])>0; b0=(z0[:,12]-old[:,12])>0
  stable=(bp==b0)&(bm==b0)
  row_rel=np.linalg.norm((fdcent-action_ref).reshape(len(p),-1),axis=1)/np.linalg.norm(action_ref.reshape(len(p),-1),axis=1)
  row_abs=np.linalg.norm((fdcent-action_ref).reshape(len(p),-1),axis=1)
  denom=np.linalg.norm(action_ref.reshape(len(p),-1),axis=1)
  # Carefully bounded, data-selected rows: worst relative/absolute, smallest denominator,
  # closest positive plastic multiplier, nearest elastic yield gap, material-specific controls.
  dl=z0[:,12]-old[:,12]
  from_ref=load_module(REF,'ref_for_select')
  q,flow=from_ref.hill(p,z0[:,6:12]); gap=1+p[:,17]*p[:,19]*p[:,21]*z0[:,13]/p[:,10]+p[:,18]*p[:,20]*p[:,22]*z0[:,14]/p[:,10]-q
  selected=set()
  def add_arg(vals,mask,fn=np.argmin):
   ii=np.flatnonzero(mask)
   if ii.size: selected.add(int(ii[fn(np.asarray(vals)[ii])]))
  for m in (0,1):
   mm=mats==m
   add_arg(row_rel,mm,np.argmax); add_arg(row_abs,mm,np.argmax); add_arg(denom,mm)
   add_arg(dl,mm&(dl>0)); add_arg(gap,mm&(dl<=0))
  order_near=np.argsort(np.abs(dl[dl>0]))
  pos=np.flatnonzero(dl>0)
  selected.update(map(int,pos[order_near[:2]]))
  selected=sorted(selected)
  subset=[]
  for tol in (2e-10,2e-12,2e-14):
   ref=load_tol_ref(tol)
   for i in selected:
    zc,ic,rc,okc,fc=fd.project_resilient(ref,p[i:i+1],F[i:i+1],old[i:i+1])
    entry={'project_tolerance_scaled':tol,'tet_id':int(ids[i]),'material_index':int(mats[i]),'center_ok':bool(okc[0]),
      'center_iterations':int(ic[0]) if okc[0] else None,'center_scaled_residual':float(rc[0]) if okc[0] else None,
      'failure':fc[0] if not okc[0] else None}
    if okc[0]:
     pc=ref.stress(p[i:i+1],F[i:i+1],zc)
     tc=ref.tangent(p[i:i+1],F[i:i+1],old[i:i+1],zc,h[i:i+1])
     entry['center_state_difference_l2_vs_base']=float(np.linalg.norm(zc[0]-z0[i]))
     entry['center_stress_difference_l2_Pa_vs_base']=float(np.linalg.norm(pc[0]-ref0.stress(p[i:i+1],F[i:i+1],z0[i:i+1])[0]))
     entry['center_tangent_action_difference_l2_Pa_per_s_vs_base']=float(np.linalg.norm(tc[0]-action_ref[i]))
     for target in TOL_TARGETS:
      ee=target/direction_norm[i]
      fp=F[i:i+1]+ee*dFbase[i:i+1]; fm=F[i:i+1]-ee*dFbase[i:i+1]
      zp1,ip1,rp1,op1,fp1=fd.project_resilient(ref,p[i:i+1],fp,old[i:i+1])
      zm1,im1,rm1,om1,fm1=fd.project_resilient(ref,p[i:i+1],fm,old[i:i+1])
      ev={'target_deltaF_frobenius':target,'epsilon':float(ee),'plus_ok':bool(op1[0]),'minus_ok':bool(om1[0]),
       'plus_scaled_residual':float(rp1[0]) if op1[0] else None,'minus_scaled_residual':float(rm1[0]) if om1[0] else None,
       'plus_iterations':int(ip1[0]) if op1[0] else None,'minus_iterations':int(im1[0]) if om1[0] else None,
       'actual_plus_deltaF_norm':float(np.linalg.norm(fp[0]-F[i])),'actual_minus_deltaF_norm':float(np.linalg.norm(fm[0]-F[i]))}
      if op1[0] and om1[0]:
       pp=ref.stress(p[i:i+1],fp,zp1)[0]; pm=ref.stress(p[i:i+1],fm,zm1)[0]
       fdact=(pp-pm)/(2*ee*dt[i])
       tref=tc[0]
       err=np.linalg.norm(fdact-tref); deni=np.linalg.norm(tref)
       ev.update({'central_abs_error_l2_Pa_per_s':float(err),'central_reference_action_denominator_l2':float(deni),
        'central_relative_error':float(err/deni) if deni>0 else None,
        'plus_branch':bool(zp1[0,12]-old[i,12]>0),'minus_branch':bool(zm1[0,12]-old[i,12]>0),
        'center_branch':bool(zc[0,12]-old[i,12]>0),
        'plus_one_sided_abs_error':float(np.linalg.norm((pp-ref.stress(p[i:i+1],F[i:i+1],zc)[0])/(ee*dt[i])-tref)),
        'minus_one_sided_abs_error':float(np.linalg.norm((ref.stress(p[i:i+1],F[i:i+1],zc)[0]-pm)/(ee*dt[i])-tref))})
      entry.setdefault('fd',[]).append(ev)
    subset.append(entry)

 OUT.mkdir(parents=True,exist_ok=False)
 metrics_all=metric_detail(fdcent,action_ref,ids)
 metrics_stable=metric_detail(fdcent,action_ref,ids,stable)
 metrics_native=metric_detail(action_native,action_ref,ids)
 # per-material detail with the same explicit denominator treatment
 bymat={}
 for m,name in ((0,'liner'),(1,'medium')):
  bymat[name]={'rows':int(np.sum(mats==m)),'central_fd_all':metric_detail(fdcent,action_ref,ids,mats==m),
   'central_fd_branch_stable':metric_detail(fdcent,action_ref,ids,(mats==m)&stable),
   'native_cached_action':metric_detail(action_native,action_ref,ids,mats==m),
   'branch_changes_plus':int(np.sum((mats==m)&(bp!=b0))),'branch_changes_minus':int(np.sum((mats==m)&(bm!=b0)))}
 # additional gate denominators across 1728, no arbitrary threshold
 rowinfo=[]
 for i in range(len(ids)):
  rowinfo.append({'tet_id':int(ids[i]),'material': 'liner' if mats[i]==0 else 'medium','center_branch':'plastic' if b0[i] else 'elastic',
   'central_fd_branch_stable':bool(stable[i]),'directional_reference_action_norm_Pa_per_s':float(denom[i]),
   'central_fd_abs_error_Pa_per_s':float(row_abs[i]),'central_fd_relative_error':float(row_rel[i]) if denom[i]>0 else None,
   'native_cached_action_abs_error_Pa_per_s':float(np.linalg.norm(action_native[i]-action_ref[i])),
   'native_cached_action_reference_norm_Pa_per_s':float(denom[i]),
   'native_cached_action_relative_error':float(np.linalg.norm(action_native[i]-action_ref[i])/denom[i]) if denom[i]>0 else None,
   'accepted_state_lambda_increment':float(dl[i]),'yield_gap':float(gap[i]),
   'fd_plus_branch':'plastic' if bp[i] else 'elastic','fd_minus_branch':'plastic' if bm[i] else 'elastic',
   'scaled_residual_center':float(r0[i]),'scaled_residual_plus':float(rp[i]),'scaled_residual_minus':float(rm[i])})
 with (OUT/'all_1728_row_metrics.csv').open('w',newline='') as f:
  wri=csv.DictWriter(f,fieldnames=list(rowinfo[0]));wri.writeheader();wri.writerows(rowinfo)
 # Select denominator order stats outside metrics (norms can be tiny but nonzero)
 order=np.argsort(denom)
 report={'schema':'numi.cardboard.source101-capture360-directional-fd-row-audit.v1',
  'binding':{'capture_path':str(CAP),'capture_sha256':sha(CAP),'reference_path':str(REF),'reference_sha256':sha(REF),
   'source_snapshot':str(BASE/'snapshots/source-boundary-reset-order-101'),'source_snapshot_role':'immutable native capture source; this CPU analysis does not rerun it',
   'rows':len(ids),'tet_ids_sha256':hashlib.sha256(ids.astype('<u4').tobytes()).hexdigest(),
   'center_dataset':'source101 capture360 exact center arrays decoded from raw capture, preserving JSON -0 as float -0.0',
   'direction':'captured local grad(delta_v); F±=Fcenter±epsilon*dt*direction; accepted history held fixed'},
  'baseline_directional_fd':{'target_deltaF_frobenius_norm_each_element':TARGET,'plus_and_minus_projection_success':int(np.sum(okp&okm)),
    'branch_stable_rows':int(stable.sum()),'plus_branch_changes':int(np.sum(bp!=b0)),'minus_branch_changes':int(np.sum(bm!=b0)),
    'max_plus_scaled_residual':float(rp.max()),'max_minus_scaled_residual':float(rm.max()),
    'global_l2_relative_vs_independent_tangent':metrics_all['global_l2_relative'],
    'max_per_element_relative':max(x['central_fd_relative_error'] for x in rowinfo if x['central_fd_relative_error'] is not None),
    'max_per_element_absolute_Pa_per_s':max(x['central_fd_abs_error_Pa_per_s'] for x in rowinfo),
    'all_rows':metrics_all,'branch_stable_only':metrics_stable,'per_material':bymat},
  'native_cached_tangent_action_vs_independent_tangent':metrics_native,
  'absolute_denominator_audit':{'denominator_definition':'Per-tet L2 norm of independent FP64 implicit tangent action C_ref * captured grad(delta_v), in Pa/s.',
    'exact_zero_count':int(np.sum(denom==0)),'positive_count':int(np.sum(denom>0)),
    'minimum_positive':float(np.min(denom[denom>0])) if np.any(denom>0) else None,
    'percentiles_l2_Pa_per_s':{str(k):float(np.percentile(denom,k)) for k in (0,0.1,1,5,50,95,99,100)},
    'smallest_20_nonzero_rows':[{'tet_id':int(ids[i]),'material':'liner' if mats[i]==0 else 'medium','action_denominator_l2_Pa_per_s':float(denom[i]),'central_fd_abs_error_Pa_per_s':float(row_abs[i]),'central_fd_relative_error':float(row_rel[i]) if denom[i]>0 else None,'branch_stable':bool(stable[i])} for i in order if denom[i]>0][:20],
    'classification':'Near-zero is reported descriptively as the bottom 20 positive denominators; no numeric threshold is invented and no result is waived.'},
  'worst_row':{'central_fd_relative': next((x for x in sorted(rowinfo,key=lambda x:x['central_fd_relative_error'] or -1,reverse=True)),None),
    'central_fd_absolute':max(rowinfo,key=lambda x:x['central_fd_abs_error_Pa_per_s']),
    'native_cached_action_relative':max((x for x in rowinfo if x['native_cached_action_relative_error'] is not None),key=lambda x:x['native_cached_action_relative_error'])},
  'strict_reference_tolerance_subset':{'selection_method':'Union of per-material worst relative FD, worst absolute FD, smallest tangent-action denominator, closest positive plastic multiplier, and closest elastic yield gap; selected before tolerance sweep and limited to at most ten unique rows.',
    'selected_rows':[{'tet_id':int(ids[i]),'material':'liner' if mats[i]==0 else 'medium','directional_action_denominator_l2_Pa_per_s':float(denom[i]),'fd_relative_error_at_2e-10':float(row_rel[i]) if denom[i]>0 else None,'center_branch':'plastic' if b0[i] else 'elastic','lambda_increment':float(dl[i]),'yield_gap':float(gap[i])} for i in selected],
    'projection_thresholds_scaled':list(TOLS),'target_deltaF_norms':list(TOL_TARGETS),'results':subset,
    'meaning':'Bounded sensitivity test only; threshold changes numerical stopping for the independent CPU reference only and does not modify production or native acceptance.'},
  'solver_context_from_native368':{'source_note':'Parent-provided source-bound native368 result; separate assembled operator replay, not derived from this material audit.',
    'cached_vs_fresh_action_relative':0.00329792,'fresh_action_vs_capture_relative':0.0,
    'cached_b_minus_Ax_norm_Ns':2.161252895e-4,'fresh_b_minus_Ax_norm_Ns':2.157817650e-4,'configured_target_Ns':1.1993e-5,
    'interpretation':'Both captured cached and fresh operator paths leave residual about 18x target; material-only tangent checks cannot explain or close this assembled Krylov residual.'},
  'limits':['Rejected Newton 23 capture only; not an accepted state, full solution, or physical result.','Only environment 0 liner/medium tetrahedra (1,728); starch/interface elements and the second environment are excluded.','The center is an independent FP64 equation replay. Endpoint perturbations are exact captured direction in FP64, not native FP32 perturbed endpoints.','No repeated native noise-floor measurement was made.','Global L2 results do not override per-row worst errors; the stricter derivative gate remains a nonpass where rowwise error exceeds 1e-6.','The material-action denominator audit reports all rows and bottom order statistics without suppressing small-denominator rows.'],
  'warnings':sorted({str(w.message) for w in caught}), 'cpu_seconds':time.monotonic()-start}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(OUT/'report.json'),'rows':len(ids),'strict_subset_rows':len(selected),'cpu_seconds':report['cpu_seconds'],'worst_fd_rel':report['baseline_directional_fd']['max_per_element_relative'],'zero_denominators':report['absolute_denominator_audit']['exact_zero_count']},indent=2))
if __name__=='__main__':main()
