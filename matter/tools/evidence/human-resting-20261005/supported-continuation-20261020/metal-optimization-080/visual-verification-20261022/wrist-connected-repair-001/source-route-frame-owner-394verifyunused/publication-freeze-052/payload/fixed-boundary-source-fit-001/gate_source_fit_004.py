#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused'; FIT=D/'fixed-boundary-source-fit-001'; RUN=FIT/'run-004'
BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
CAND=RUN/'candidate-8874-local-position-fit.nhtissue'; BUILDER=FIT/'build_source_fit_004.py'
IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
POSE_DIR=pathlib.Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
FK=('right-ring-pip-57.json','right-little-pip-61.json','left-ring-pip-95.json','left-little-pip-99.json')
EXACT=E/'exact-plane-filter-owner-264/src'; CANON=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py'
IDS=(131,132,133,134,137,138,141,142);SID=131;SEED=(12677,12679)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pose_map(doc,frame):
 if frame in ('rest','current'):return {int(x['body_index']):x[frame] for x in doc['all_bodies']}
 return {int(x['core_body']):{'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']} for x in doc['source_poses']}
def exact_points(ci,v):return [tuple(__import__('fractions').Fraction(int(k),1<<149) for k in ci.float32_point_lattice_key(row)) for row in np.asarray(v,np.float32)]
def exact_records(ci,points,faces,rows):
 rows=np.asarray(sorted(set(map(int,rows))),np.int64)
 if not len(rows):return []
 rr=ci._records(points,np.asarray(faces,np.int64)[rows].tolist())
 return [(r[0],r[1],r[2],int(row),r[4]) for r,row in zip(rr,rows)]
def exact_pair_ids(ci,first,second,same_surface):
 out=set()
 for a,b in ((first,second),(second,first)) if same_surface else ((first,second),):
  if not a or not b:continue
  rr=ci._audit_pair(a,b,same_surface=same_surface)
  for x,y in rr.get('triangle_pairs',[]):out.add(tuple(sorted((int(x),int(y)))) if same_surface else (int(x),int(y)))
 return out
def box_rows(vertices,faces,lo,hi):
 tri=np.asarray(vertices,np.float32)[np.asarray(faces,np.int64)]
 mask=np.all(tri.max(axis=1)>=np.asarray(lo,np.float32),axis=1)&np.all(tri.min(axis=1)<=np.asarray(hi,np.float32),axis=1)
 return np.flatnonzero(mask).astype(np.int64)
def main():
 t=time.time();src=importlib.util.spec_from_file_location('fit',BUILDER);fit=importlib.util.module_from_spec(src);src.loader.exec_module(fit)
 rpt=json.loads((RUN/'source-fit-report.json').read_text());assert rpt['candidate_sha256']==sha(CAND);assert rpt['baseline_sha256']==sha(BASE)
 if not CAND.is_file():raise RuntimeError('candidate missing')
 spec=importlib.util.spec_from_file_location('canon',CANON);canon=importlib.util.module_from_spec(spec);spec.loader.exec_module(canon)
 base=canon.decode_nhtiss(BASE.read_bytes());cand=canon.decode_nhtiss(CAND.read_bytes())
 ns=fit.load_emitter();identity=json.loads(IDENT.read_text());pose_specs={'current191':fit.pose_map(identity,'current'),'rest003':fit.pose_map(identity,'rest')}
 for fn in FK:pose_specs[fn[:-5]]=fit.pose_map(json.loads((POSE_DIR/fn).read_text()),'fk')
 out=RUN/'exact-local-source-fit-131-001';out.mkdir(exist_ok=False)
 # Face patch remains actual8874 topology; no index/record changes.
 rec=next(r for r in base['records'] if int(r[6])==SID);fv,vc,fi,ic=map(int,(rec[2],rec[3],rec[4],rec[5]));ix=base['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
 patch_rows,pfaces,free,bedges=fit.patch_rows(F,SEED,3)
 quality=[];emitted={}
 for name,poses in pose_specs.items():
  ns['poses']=poses;B=ns['emit_selected'](BASE);ns['poses']=poses;C=ns['emit_selected'](CAND)
  emitted[name]=(B,C)
  bv=B[SID][0].astype(np.float64);cv=C[SID][0].astype(np.float64);bt=bv[pfaces];ct=cv[pfaces]
  bcr=np.cross(bt[:,1]-bt[:,0],bt[:,2]-bt[:,0]);ccr=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]);bn=np.linalg.norm(bcr,axis=1);cn=np.linalg.norm(ccr,axis=1);cos=np.einsum('ij,ij->i',bcr,ccr)/np.maximum(bn*cn,1e-300);proj=np.einsum('ij,ij->i',bcr,ccr)/np.maximum(bn*bn,1e-300)
  edge=np.maximum.reduce([np.linalg.norm(ct[:,1]-ct[:,0],axis=1),np.linalg.norm(ct[:,2]-ct[:,0],axis=1),np.linalg.norm(ct[:,2]-ct[:,1],axis=1)]);alt=np.divide(cn,edge,out=np.zeros_like(cn),where=edge>0)*1000.0;ulp=np.max(np.abs(np.spacing(cv.astype(np.float32)[pfaces])),axis=(1,2))*1000
  quality.append({'pose':name,'minimum_altitude_mm':float(alt.min()),'minimum_altitude_to_vertex_ulp':float(np.min(alt/np.maximum(ulp,1e-30))),'minimum_oriented_area_ratio':float(proj.min()),'minimum_face_normal_cosine':float(cos.min()),'zero_area_face_count':int(np.count_nonzero(cn==0))})
 qpass=all(q['minimum_altitude_to_vertex_ulp']>=4.0 and q['minimum_oriented_area_ratio']>=0.2 and q['zero_area_face_count']==0 for q in quality)
 report={'schema':'numi.human.fixed-boundary-source-fit.exact-local-gate.v1','baseline_sha256':sha(BASE),'candidate_sha256':sha(CAND),'candidate_builder_sha256':sha(BUILDER),'source_fit_report_sha256':sha(RUN/'source-fit-report.json'),'stable_id':SID,'seed_faces':list(SEED),'three_ring_patch_faces':patch_rows.tolist(),'fixed_boundary_edge_count':len(bedges),'free_vertex_local_ids':free.tolist(),'quality_preflight':quality,'quality_preflight_status':'PASS' if qpass else 'FAIL_STOP_BEFORE_EXACT','exact_checks':[],'limitations':['The candidate changes only stable131 position/normal rows; stable134 remains unmodified because its fit failed. Scope is changed three-ring patch versus same-bank and selected-bank surfaces in six poses only. This does not qualify the whole hand or tissue system.']}
 if qpass:
  sys.path.insert(0,str(EXACT));from numilab_human import cardiac_cavity_intersections as ci
  for name,(B,C) in emitted.items():
   for target in IDS:
    rtarget=next(r for r in base['records'] if int(r[6])==target);tfv,tvc,tfi,tic=map(int,(rtarget[2],rtarget[3],rtarget[4],rtarget[5]));tix=base['indices'][tfi:tfi+tic].astype(np.int64).reshape(-1,3);TF=tix-tfv if tix.min()>=tfv and tix.max()<tfv+tvc else tix
    if target==SID:
     rows1=exact_records(ci,exact_points(ci,B[SID][0]),F,patch_rows); rows2=exact_records(ci,exact_points(ci,C[SID][0]),F,patch_rows)
     allb=exact_records(ci,exact_points(ci,B[SID][0]),F,np.arange(len(F)));allc=exact_records(ci,exact_points(ci,C[SID][0]),F,np.arange(len(F)))
     bp=exact_pair_ids(ci,rows1,allb,True);cp=exact_pair_ids(ci,rows2,allc,True)
    else:
     # Narrow targets with exact Float32 triangle AABB overlap to the changed patch.
     pvB=B[SID][0].astype(np.float32);pvC=C[SID][0].astype(np.float32);patch_lo=np.minimum(pvB[pfaces].min(axis=(0,1)),pvC[pfaces].min(axis=(0,1)));patch_hi=np.maximum(pvB[pfaces].max(axis=(0,1)),pvC[pfaces].max(axis=(0,1)))
     rowtarget=box_rows(B[target][0],TF,patch_lo,patch_hi)
     if len(rowtarget)==0:continue
     rows1=exact_records(ci,exact_points(ci,B[SID][0]),F,patch_rows);rows2=exact_records(ci,exact_points(ci,C[SID][0]),F,patch_rows)
     allb=exact_records(ci,exact_points(ci,B[target][0]),TF,rowtarget);allc=exact_records(ci,exact_points(ci,C[target][0]),TF,rowtarget)
     bp=exact_pair_ids(ci,rows1,allb,False);cp=exact_pair_ids(ci,rows2,allc,False)
    new=sorted(cp-bp);resolved=sorted(bp-cp)
    report['exact_checks'].append({'pose':name,'target_stable_id':target,'baseline_scoped_pair_count':len(bp),'candidate_scoped_pair_count':len(cp),'new_pair_count':len(new),'resolved_pair_count':len(resolved),'new_pairs':new[:100],'resolved_pairs':resolved[:100]})
    if new:
     report['status']='FAIL_NEW_EXACT_LOCAL_PAIRS';report['failure_stage']=f'{name}:{target}';(out/'exact-local-gate-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'stage':report['failure_stage'],'quality':quality,'check':report['exact_checks'][-1],'elapsed_s':time.time()-t},indent=2));return
 report['status']='SCOPED_EXACT_NONREGRESSION_PASS' if qpass else 'FAIL_QUALITY'
 (out/'exact-local-gate-report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'quality':quality,'checks':report['exact_checks'],'report':str(out/'exact-local-gate-report.json'),'sha256':sha(out/'exact-local-gate-report.json'),'elapsed_s':time.time()-t},indent=2))
if __name__=='__main__':main()
