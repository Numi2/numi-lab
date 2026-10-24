#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, time
import numpy as np
from scipy.optimize import linprog
BUILDER=pathlib.Path(__file__).resolve().parent/'build_source_fit_006b.py'
OUT=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001/source-route-frame-owner-394verifyunused/fixed-boundary-source-fit-001/run-006b-134-diagnostic-002')
# Correct the directory spelling from the workspace's pinned evidence path.
OUT=pathlib.Path(str(OUT).replace('visual-verification-2022','visual-verification-20261022'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 t=time.time()
 sp=importlib.util.spec_from_file_location('fit',BUILDER);fit=importlib.util.module_from_spec(sp);sp.loader.exec_module(fit)
 OUT.mkdir(parents=True,exist_ok=False)
 raw=fit.BASE.read_bytes();dec=fit.importlib.util.spec_from_file_location('canon',fit.CANON);canon=importlib.util.module_from_spec(dec);dec.loader.exec_module(canon);payload=canon.decode_nhtiss(raw)
 ident=json.loads(fit.IDENT.read_text());poses={'current191':fit.pose_map(ident,'current'),'rest003':fit.pose_map(ident,'rest')}
 for fn in fit.FK:poses[fn[:-5]]=fit.pose_map(json.loads((fit.POSE_DIR/fn).read_text()),'fk')
 ns=fit.load_emitter();emitted={}
 for name,pmap in poses.items():ns['poses']=pmap;emitted[name]=ns['emit_selected'](fit.BASE)
 sid=134;rec=next(r for r in payload['records'] if int(r[6])==sid);fv,vc,fi,ic=map(int,(rec[2],rec[3],rec[4],rec[5]));ix=payload['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
 _,pfaces,free,bedges=fit.patch_rows(F,fit.SEEDS[sid],3);pv=np.unique(pfaces);mats,offs,_=fit.affine_maps(payload,sid,poses)
 base_local={};A_local={};seedp=fit.canon if hasattr(fit,'canon') else None
 seed_dec=canon.decode_nhtiss(fit.SEED_CAND.read_bytes());seedp=seed_dec['vertices'][fv:fv+vc]['p'].astype(np.float64);basep=payload['vertices'][fv:fv+vc]['p'].astype(np.float64);start=((seedp-basep)*1000.0)[free].reshape(-1)
 for name in poses:
  V=emitted[name][sid][0].astype(np.float64)
  pred=fit.get_world(mats[name],offs[name],payload['vertices'][fv:fv+vc]['p'].astype(np.float64));parity=float(np.max(np.abs(pred-V)))
  if parity>2e-6:raise RuntimeError(f'{name}: native affine parity {parity}m')
  base_local[name]=V[pv];A_local[name]=mats[name][pv]
 local={int(g):i for i,g in enumerate(pv)};pfaces_local=np.asarray([[local[int(g)] for g in face] for face in pfaces],np.int64);free_local=np.asarray([local[int(g)] for g in free],np.int64)
 ev=fit.constraint_evaluator(base_local,A_local,np.arange(len(pv),dtype=np.int64),pfaces_local,free_local)
 c,J=ev(start,True);s=ev.constraint_scales;cs=c/s;Js=J/s[:,None];norm=np.linalg.norm(J,axis=1);snorm=np.linalg.norm(Js,axis=1);nface=len(pfaces);npose=len(poses);nvar=len(start);m=len(c)
 A_ub=np.hstack([-Js,-np.eye(m)]);b_ub=cs.copy();objective=np.r_[np.zeros(nvar),np.ones(m)]
 phase_trials=[]
 for trust in (0.25,0.5,1.0):
  bounds=[(-trust,trust)]*nvar+[(0,None)]*m
  lp=linprog(objective,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')
  slack=np.asarray(lp.x[nvar:]) if lp.success else None
  phase_trials.append({'trust_bound_mm_per_coordinate':trust,'success':bool(lp.success),'message':str(lp.message),'optimal_sum_normalized_slack':float(lp.fun) if lp.success else None,'maximum_normalized_row_slack':float(slack.max()) if slack is not None else None,'positive_slack_rows':np.flatnonzero(slack>1e-9).tolist() if slack is not None else None,'max_displacement_mm':float(np.max(np.abs(lp.x[:nvar]))) if lp.success else None})
 groups=[]
 for pi,name in enumerate(poses):
  lo=pi*2*nface;mid=lo+nface;hi=lo+2*nface
  for kind,a,b in [('altitude',lo,mid),('oriented_double_area',mid,hi)]:
   z=slice(a,b);groups.append({'pose':name,'constraint':kind,'count':b-a,'raw_value_min':float(c[z].min()),'raw_value_max':float(c[z].max()),'raw_jacobian_row_norm_min':float(norm[z].min()),'raw_jacobian_row_norm_median':float(np.median(norm[z])),'raw_jacobian_row_norm_max':float(norm[z].max()),'scaled_value_min':float(cs[z].min()),'scaled_value_max':float(cs[z].max()),'scaled_jacobian_row_norm_min':float(snorm[z].min()),'scaled_jacobian_row_norm_median':float(np.median(snorm[z])),'scaled_jacobian_row_norm_max':float(snorm[z].max()),'violated_at_seed':int(np.count_nonzero(c[z]<-1e-12))})
 fixed=np.flatnonzero((cs< -1e-12)&(snorm<1e-10))
 report={'schema':'numi.human.source-fit-134.scaling-linearized-feasibility-diagnostic.v1','status':'DIAGNOSTIC_ONLY_NO_FIT','baseline_path':str(fit.BASE),'baseline_sha256':sha(fit.BASE),'seed_path':str(fit.SEED_CAND),'seed_sha256':sha(fit.SEED_CAND),'builder_reference_path':str(BUILDER),'builder_reference_sha256':sha(BUILDER),'pose_hashes':{name:sha(fit.IDENT if name in ('current191','rest003') else fit.POSE_DIR/(name+'.json')) for name in poses},'stable_id':sid,'source_faces':fit.SEEDS[sid],'three_ring_face_count':len(pfaces),'free_vertex_local_ids':free.tolist(),'variable_count':nvar,'constraint_count':m,'units':'displacements and altitudes in mm; oriented double-area in mm^2; normalized altitude by 0.001 mm and oriented double-area by each baseline face double-area','start_displacement_from_run003_mm':start.tolist(),'fixed_only_violated_row_indices':fixed.tolist(),'raw_constraint_value_min':float(c.min()),'scaled_constraint_value_min':float(cs.min()),'raw_jacobian_row_norm_quantiles':np.quantile(norm,[0,.1,.5,.9,1]).tolist(),'scaled_jacobian_row_norm_quantiles':np.quantile(snorm,[0,.1,.5,.9,1]).tolist(),'per_pose_constraint_scale':groups,'linearized_phase_i':{'method':'HiGHS LP minimizing sum of nonnegative normalized row slacks; trust sensitivity only, no nonlinear fit','trials':phase_trials},'elapsed_s':time.time()-t}
 path=OUT/'diagnostic.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(path),'sha256':sha(path),'status':report['status'],'phase_i_trials':report['linearized_phase_i']['trials'],'fixed_only':fixed.tolist(),'raw_norm_quantiles':report['raw_jacobian_row_norm_quantiles'],'scaled_norm_quantiles':report['scaled_jacobian_row_norm_quantiles'],'groups':groups,'elapsed_s':report['elapsed_s']},indent=2))
if __name__=='__main__':main()
