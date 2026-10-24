#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
FIT=HERE/'build_source_fit_006b.py'
OUT=HERE/'run-006b-134-robust-normal-003'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def normalized(v):
 n=np.linalg.norm(v,axis=-1,keepdims=True)
 return np.divide(v,n,out=np.zeros_like(v),where=n>0),n[...,0]
def face_metrics(V,F,row):
 tri=V[F[row]]; e=np.stack([tri[1]-tri[0],tri[2]-tri[0],tri[2]-tri[1]])
 cr=np.cross(tri[1]-tri[0],tri[2]-tri[0]); area2=float(np.linalg.norm(cr)); lengths=np.linalg.norm(e,axis=1)
 return tri,cr,area2,float(area2/max(lengths.max(),1e-300)),lengths
def main():
 t=time.time();fit=load('fit006b',FIT);canon=load('canon',fit.CANON)
 base=canon.decode_nhtiss(fit.BASE.read_bytes());seed=canon.decode_nhtiss(fit.SEED_CAND.read_bytes())
 sid=134; rec=next(r for r in base['records'] if int(r[6])==sid); fb,bc,fv,vc,fi,ic,*_=map(int,rec)
 inds=base['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=inds-fv if inds.min()>=fv and inds.max()<fv+vc else inds
 row=63330; triids=F[row].astype(np.int64)
 incident=np.flatnonzero(np.any(np.isin(F,triids),axis=1)); ringverts=np.unique(F[incident].ravel())
 if len(incident)<3 or len(triids)!=3: raise RuntimeError('unexpected microface incidence')
 ident=json.loads(fit.IDENT.read_text()); poses={'current191':fit.pose_map(ident,'current'),'rest003':fit.pose_map(ident,'rest')}
 for fn in fit.FK: poses[fn[:-5]]=fit.pose_map(json.loads((fit.POSE_DIR/fn).read_text()),'fk')
 ns=fit.load_emitter(); emitted_base={}; emitted_seed={}
 for name,pmap in poses.items():
  ns['poses']=pmap;emitted_base[name]=ns['emit_selected'](fit.BASE)[sid][0].astype(np.float64)*1000.0
  emitted_seed[name]=ns['emit_selected'](fit.SEED_CAND)[sid][0].astype(np.float64)*1000.0
 mats,offs,_=fit.affine_maps(base,sid,poses); smats,soffs,_=fit.affine_maps(seed,sid,poses)
 pref=base['vertices'][fv:fv+vc]['p'].astype(np.float64)*1000.0
 # Source oriented area-weighted normal over the face one-ring, plus cancellation quality.
 src_tri=pref[F[incident]];src_cross=np.cross(src_tri[:,1]-src_tri[:,0],src_tri[:,2]-src_tri[:,0]);src_sum=src_cross.sum(axis=0)
 srcn,srcnorm=normalized(src_sum[None,:]);srcn=srcn[0];src_area_sum=float(np.linalg.norm(src_cross,axis=1).sum());src_cancel=float(srcnorm[0]/max(src_area_sum,1e-300))
 reports=[]; robust_pose_normals=[]; raw_pose_normals=[]
 for name in poses:
  Vf=emitted_base[name]; Vs=emitted_seed[name]
  Vd=fit.get_world(mats[name],offs[name],base['vertices'][fv:fv+vc]['p'].astype(np.float64))*1000.0
  _,crf,af,hf,elf=face_metrics(Vf,F,row); _,crd,ad,hd,eld=face_metrics(Vd,F,row)
  nrawf,_=normalized(crf[None,:]); nrawd,_=normalized(crd[None,:]); nrawf=nrawf[0];nrawd=nrawd[0]
  tri_ring=Vf[F[incident]];cross_ring=np.cross(tri_ring[:,1]-tri_ring[:,0],tri_ring[:,2]-tri_ring[:,0]);sum_ring=cross_ring.sum(axis=0)
  tri_ringd=Vd[F[incident]];cross_ringd=np.cross(tri_ringd[:,1]-tri_ringd[:,0],tri_ringd[:,2]-tri_ringd[:,0]);sum_ringd=cross_ringd.sum(axis=0)
  nr,_=normalized(sum_ring[None,:]);nr=nr[0]
  nringd,_=normalized(sum_ringd[None,:]);nringd=nringd[0]
  X=pref[ringverts];Y=Vd[ringverts];xm=X.mean(axis=0);ym=Y.mean(axis=0)
  rowmap=np.linalg.lstsq(X-xm,Y-ym,rcond=None)[0]; A=rowmap.T
  det=float(np.linalg.det(A)); cond=float(np.linalg.cond(A))
  try:cof=det*np.linalg.inv(A).T@srcn;cof=cof/max(float(np.linalg.norm(cof)),1e-300)
  except np.linalg.LinAlgError:cof=np.zeros(3)
  _,crs,ass,hs,els=face_metrics(emitted_seed[name],F,row);nsf,_=normalized(crs[None,:]);nsf=nsf[0]
  spacing=np.abs(np.spacing(Vf[F[row]].astype(np.float32))).max(axis=0)
  source_tri=pref[F[row]];source_cross=np.cross(source_tri[1]-source_tri[0],source_tri[2]-source_tri[0])
  source_alt=float(np.linalg.norm(source_cross)/max(np.linalg.norm(np.stack([source_tri[1]-source_tri[0],source_tri[2]-source_tri[0],source_tri[2]-source_tri[1]]),axis=1).max(),1e-300))
  reports.append({
    'pose':name,'face_record_local_row':row,'face_global_index_row':int(fi//3+row),'vertex_ids_local':triids.tolist(),'incident_one_ring_face_count':int(len(incident)),'one_ring_vertex_count':int(len(ringverts)),
    'face_f32':{'edge_lengths_mm':elf.tolist(),'area2_mm2':af,'minimum_altitude_mm':hf,'normal':nrawf.tolist(),'one_ring_normal':nr.tolist(),'normal_dot_one_ring':float(np.dot(nrawf,nr)),'coordinate_ulp_mm_per_axis_max':spacing.tolist(),'altitude_over_max_axis_ulp':float(hf/max(np.linalg.norm(spacing),1e-300))},
    'face_fp64_affine':{'edge_lengths_mm':eld.tolist(),'area2_mm2':ad,'minimum_altitude_mm':hd,'normal':nrawd.tolist(),'one_ring_normal':nringd.tolist(),'normal_dot_one_ring':float(np.dot(nrawd,nringd)),'normal_f32_fp64_dot':float(np.dot(nrawf,nrawd)),'one_ring_f32_fp64_dot':float(np.dot(nr,nringd))},
    'seed_candidate_f32':{'edge_lengths_mm':els.tolist(),'area2_mm2':ass,'minimum_altitude_mm':hs,'normal':nsf.tolist(),'base_candidate_normal_dot':float(np.dot(nsf,nrawf)),'vertices_vs_base_max_mm':float(np.linalg.norm(emitted_seed[name][F[row]]-Vf[F[row]],axis=1).max())},
    'source_cofactor_transport':{'det_local_fit':det,'condition_number':cond,'cofactor_normal':cof.tolist(),'cofactor_vs_actual_one_ring_fp64_dot':float(np.dot(cof,nringd)),'cofactor_vs_f32_face_dot':float(np.dot(cof,nrawf))},
    'ulp_diagnostic':{'source_reference_altitude_mm':source_alt,'f32_vs_fp64_vertex_max_mm':float(np.linalg.norm(Vf[ringverts]-Vd[ringverts],axis=1).max()),'source_one_ring_area_cancellation_ratio':src_cancel}
  })
  robust_pose_normals.append(nringd);raw_pose_normals.append(nrawd)
 pairwise=[]
 for i in range(len(reports)):
  for j in range(i+1,len(reports)):
   pairwise.append({'pose_a':reports[i]['pose'],'pose_b':reports[j]['pose'],'raw_face_normal_fp64_dot':float(np.dot(raw_pose_normals[i],raw_pose_normals[j])),'robust_one_ring_normal_fp64_dot':float(np.dot(robust_pose_normals[i],robust_pose_normals[j]))})
 diag=json.load(open(HERE/'run-006b-134-row-002/diagnostic.json'))
 free=fit.patch_rows(F,fit.SEEDS[sid],3)[2]
 out={'schema':'numi.human.source-fit-134.microface-robust-normal-audit.v1','status':'DIAGNOSTIC_ONLY_NO_FIT','baseline_path':str(fit.BASE),'baseline_sha256':sha(fit.BASE),'seed_path':str(fit.SEED_CAND),'seed_sha256':sha(fit.SEED_CAND),'fit_builder_path':str(FIT),'fit_builder_sha256':sha(FIT),'emitter_sha256':sha(fit.EMIT),'pose_hashes':{name:sha(fit.IDENT if name in ('current191','rest003') else fit.POSE_DIR/(name+'.json')) for name in poses},'stable_id':sid,'face_identity':{'record_vertex_start':fv,'record_face_start_index_offset':fi,'face_local_row':row,'face_global_index_row':int(fi//3+row),'vertices_local':triids.tolist(),'free_vertices_in_run006b':sorted(set(map(int,triids))&set(map(int,free))),'source_reference_xyz_mm':pref[triids].tolist()},'one_ring_definition':'all source faces incident to any of the three face vertices; oriented area-weighted cross-product sum','source_one_ring_normal':srcn.tolist(),'source_one_ring_area_cancellation_ratio':src_cancel,'poses':reports,'cross_pose_normal_alignment':pairwise,'gradient_cosines_from_phase_i':diag['active_microface_pose_gradient_cosines'],'elapsed_s':time.time()-t,'interpretation':'Diagnostic only. Raw-face orientation under sub-ULP geometry is compared with a one-ring area-weighted normal and source cofactor transport; neither is an admission gate.'}
 path=OUT/'diagnostic.json';path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'path':str(path),'sha256':sha(path),'identity':out['face_identity'],'source_normal':srcn.tolist(),'cancellation_ratio':src_cancel,'poses':[{'pose':p['pose'],'f32_altitude':p['face_f32']['minimum_altitude_mm'],'f32_over_ulp':p['face_f32']['altitude_over_max_axis_ulp'],'f32_face_vs_ring':p['face_f32']['normal_dot_one_ring'],'fp64_face_vs_ring':p['face_fp64_affine']['normal_dot_one_ring'],'f32_vs_fp64_face':p['face_fp64_affine']['normal_f32_fp64_dot'],'cofactor_vs_ring':p['source_cofactor_transport']['cofactor_vs_actual_one_ring_fp64_dot'],'cofactor_vs_face':p['source_cofactor_transport']['cofactor_vs_f32_face_dot'],'seed_altitude':p['seed_candidate_f32']['minimum_altitude_mm']} for p in reports],'cross_pose':pairwise,'elapsed_s':out['elapsed_s']},indent=2))
if __name__=='__main__':main()
