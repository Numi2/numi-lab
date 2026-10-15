from pathlib import Path
import importlib.util,json,hashlib,datetime,csv,sys,time,warnings
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=B/'analysis/source125-fp64-smooth-direction-sweep-006/sweep.py';s=load(S,'sweep_full147')
verifier=s.load_module(s.ADAPTER/'audit_capture.py','verifier147');ref=s.load_module(s.REF_PATH,'ref147');fd=s.load_module(s.FD_PATH,'fd147')
assert not (O/'report.json').exists(),'fresh report required'
run=json.loads((s.RUN/'receipt.json').read_text());assert run['source_inputs_unchanged'] and not run['timed_out']
bound=json.loads((s.AUDIT/'report.json').read_text());assert bound['bindings']['capture_sha256']==sha(s.CAPTURE)
rows=[json.loads(x) for x in s.CAPTURE.read_text().splitlines() if x.strip()];head=rows[0];assert rows[-1]['record']=='integrity'
tets=sorted([r for r in rows if r.get('record')=='tetrahedron' and r.get('environment')==0],key=lambda r:r['tet_id']);assert len(tets)==1920 and all(t['tet_id']==i for i,t in enumerate(tets))
F=np.stack([s.pair(t['center'],'F_local') for t in tets]);mat=np.array([t['material_index'] for t in tets]);old=np.zeros((1920,15));p=np.zeros((1920,28));pt=s.f32(head['parameters_by_environment'][0]);spec={m['index']:m for m in head['materials']}
C=np.zeros((1920,9,9));D=np.zeros_like(C);valid=np.ones(1920,dtype=bool);project={}
for i,t in enumerate(tets):
 mi=int(mat[i]);m=spec[mi];pc=len(m['parameter_names']);assert pc==(28 if mi in (0,1) else 3)
 p[i,:pc]=pt[m['parameter_offset']:m['parameter_offset']+pc]
 if mi in (0,1):old[i]=s.f32(t['center']['accepted_state'])
 cache=t['cached_algorithmic_tangent_local'];assert cache['valid'];C[i]=s.f32(cache['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9,9)
assert bound['input_contract']['rate']=='captured maximum rate exponent must be zero; independent 15-state law is rate independent','existing captured-program rate contract required'
assert bound['bindings']['capture_sha256']==sha(s.CAPTURE)
inputs=[S,s.CAPTURE,s.RUN/'receipt.json',s.RUN/'prediction.json',s.AUDIT/'report.json',s.REF_PATH,s.FD_PATH,s.ADAPTER/'audit_capture.py']
(O/'prediction.json').write_text(json.dumps({'schema':'numi.cardboard.full_tangent_audit_prediction.v1','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'question':'Do all nine native cached tangent columns at419 agree with independently reprojected FP64 implicit derivatives at the same exact F and accepted history?','prediction':'Single-direction comparisons are close, but no complete matrix result is assumed. A failed column/tet is retained. This is analytic derivative-chain comparison, separate from FP32 secant-noise resolution and coupon convergence.','inputs':{str(x):sha(x) for x in inputs}},indent=2)+'\n')
started=time.monotonic();warn=[]
for mi in (0,1):
 ids=np.flatnonzero(mat==mi);pp=p[ids];ff=F[ids];oo=old[ids]
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');base=verifier.project_centers(ref,fd,pp,ff,oo,pp[:,26]);z,it,rn,cn,ok,reasons,extra=s.continue_projection_tight(ref,pp,ff,oo,pp[:,26],base)
 warn += [{'message':str(w.message),'category':w.category.__name__} for w in ww]
 valid[ids]=ok
 project[str(mi)]={'rows':len(ids),'valid':int(ok.sum()),'max_scaled_residual':float(np.nanmax(rn)),'max_dynamic_correction':float(np.nanmax(cn)),'failures':[{'tet':int(ids[j]),'reason':reasons[j]} for j in np.flatnonzero(~ok)]}
 for col in range(9):
  H=np.zeros_like(ff);H.reshape(len(ids),9)[:,col]=1;D[ids,:,col]=ref.tangent(pp,ff,oo,z,H).reshape(len(ids),9)
ids=np.flatnonzero(mat==2)
for col in range(9):
 H=np.zeros((len(ids),3,3));H.reshape(len(ids),9)[:,col]=1;D[ids,:,col]=verifier.glue_tangent_action(F[ids],H,p[ids,1],p[ids,2]).reshape(len(ids),9)
assert np.isfinite(C).all() and np.isfinite(D).all()
err=np.linalg.norm(C-D,axis=1);norm=np.linalg.norm(D,axis=1);rel=np.divide(err,norm,out=np.full_like(err,np.nan),where=norm>0)
ferr=np.linalg.norm(C-D,axis=(1,2));fnorm=np.linalg.norm(D,axis=(1,2));frel=ferr/fnorm
np.savez_compressed(O/'matrices.npz',native_C_Pa=C,independent_C_Pa=D,reference_F=F,material=mat,valid_projection=valid,column_relative_error=rel,column_absolute_l2_Pa=err)
with (O/'per-column.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['tet','material','column','valid_projection','reference_l2_Pa','absolute_l2_Pa','relative_error','relative_gate_1e-3_pass'])
 for i in range(1920):
  for col in range(9):w.writerow([i,int(mat[i]),col,bool(valid[i]),norm[i,col],err[i,col],rel[i,col] if np.isfinite(rel[i,col]) else '',bool(valid[i] and np.isfinite(rel[i,col]) and rel[i,col]<=1e-3)])
report={'schema':'numi.cardboard.full_native_tangent_vs_independent_fp64.v1','scope':{'run':str(s.RUN),'environment':0,'control':1,'newton':48,'tets':1920,'columns_per_tet':9,'rate_state':'rate independent only','physical_transactions':0},'classification':'same-input analytic derivative comparison; independent FP64 projection and implicit differentiation; not native finite-difference noise qualification or complete nonlinear solve','projection':project,'material_groups':{},'warnings':warn,'wall_seconds':time.monotonic()-started,'near_zero_policy':'No measured absolute noise floor supplied. Zero-reference columns are inconclusive and never pass a relative gate. All absolute errors and reference norms retained.','limitations':['One captured center/accepted history, not all solver states.','Native cached matrix versus independent analytic derivative is distinct from unresolved FP32 endpoint/secant comparison.','Full FP64 finite-difference verification for every basis column is not performed here; prior branch-preserving FP64 evidence binds one captured direction.','No damage/interface/rate-dependent law qualification or physical calibration.']}
for mi in (0,1,2):
 ix=np.flatnonzero(mat==mi);r=rel[ix];report['material_groups'][str(mi)]={'tets':len(ix),'valid_projections':int(valid[ix].sum()),'columns':int(r.size),'zero_reference_columns':int(np.sum(norm[ix]==0)),'columns_over_1e-3':int(np.sum(r>1e-3)),'max_column_relative_error':float(np.nanmax(r)),'max_matrix_frobenius_relative_error':float(frel[ix].max()),'max_absolute_column_l2_Pa':float(err[ix].max()),'max_absolute_component_Pa':float(np.abs(C[ix]-D[ix]).max()),'max_error_tet':int(ix[np.unravel_index(np.nanargmax(r),r.shape)[0]]),'max_error_column':int(np.unravel_index(np.nanargmax(r),r.shape)[1])}
report['all_captured_columns_at_relative_gate']=bool(valid.all() and np.isfinite(rel).all() and np.all(rel<=1e-3))
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');files=['audit.py','prediction.json','report.json','matrices.npz','per-column.csv'];(O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.full_tangent_audit_manifest.v1','inputs':{str(x):sha(x) for x in inputs},'files':{x:sha(O/x) for x in files}},indent=2)+'\n');print(json.dumps(report['material_groups'],indent=2));print('all_captured_columns_at_relative_gate',report['all_captured_columns_at_relative_gate'])
