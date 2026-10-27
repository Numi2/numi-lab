#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027'
B=W/'combined-d071-b109-b61-001'
OLD=W/'b43-16/attempt-024-witness-constrained-fit-001'
DIRECTION=OLD/'solved-direction.npz'
OUT=Path(__file__).resolve().parent
EXPECTED_BASE='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
EXPECTED_DIRECTION='b1550f73cae5d4b705add5467309ae3843341fec13a1b74a3516ebc308daf504'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
BANKDIR=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight')
E=BANKDIR.parents[0].parents[0].parents[0]
LAYER=B/'skin-fat-layers-541.npz'
TRANSFER=W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
AREA_MIN=0.95
NORMAL_MAX=5.0
MIN_ANGLE_DELTA=-0.05
ASPECT_MAX=1.03
ALPHAS=[0.5,0.25,0.125,0.0625]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot import {p}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def shape_metrics(X0,X1,F,rows):
 a=np.asarray(X0,dtype=np.float64)[F[rows]].astype(np.float64)*1000.
 b=np.asarray(X1,dtype=np.float64)[F[rows]].astype(np.float64)*1000.
 def calc(t):
  e=np.stack([np.linalg.norm(t[:,1]-t[:,0],axis=1),np.linalg.norm(t[:,2]-t[:,1],axis=1),np.linalg.norm(t[:,0]-t[:,2],axis=1)],axis=1)
  c=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); ar=np.linalg.norm(c,axis=1)/2
  ang=[]
  for j in range(3):
   u=t[:,(j+1)%3]-t[:,j];v=t[:,(j+2)%3]-t[:,j];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
   ang.append(np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/np.maximum(den,1e-300),-1,1))))
  asp=np.max(e,axis=1)**2/(4*np.sqrt(3)*np.maximum(ar,1e-300))
  return ar,c,np.stack(ang,axis=1),asp
 ar0,n0,g0,r0=calc(a);ar1,n1,g1,r1=calc(b)
 co=np.sum(n0*n1,axis=1)/(np.maximum(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1),1e-300))
 turn=np.degrees(np.arccos(np.clip(co,-1,1))); ratio=ar1/np.maximum(ar0,1e-300)
 delta=np.min(g1,axis=1)-np.min(g0,axis=1); asp_ratio=r1/np.maximum(r0,1e-300)
 return {'face_count':int(len(rows)),'zero_area_before':int(np.count_nonzero(ar0<=0)),'zero_area_after':int(np.count_nonzero(ar1<=0)),'minimum_area_ratio':float(np.min(ratio,initial=np.inf)),'maximum_normal_turn_deg':float(np.max(turn,initial=0)),'minimum_min_angle_delta_deg':float(np.min(delta,initial=np.inf)),'maximum_aspect_ratio_ratio':float(np.max(asp_ratio,initial=0)),'min_angle_before_deg':float(np.min(g0,initial=np.inf)),'min_angle_after_deg':float(np.min(g1,initial=np.inf))}
def layer_flags(tf,outer,inner,faces,field):
 return tf.face_flags(outer,inner,faces,field)
def layer_check(base_x,trial_x,F,affected,layerz,tf):
 rev=F[affected][:,[0,2,1]]
 O0=np.asarray(layerz['outer_points_f64'],dtype=np.float64);I0=np.asarray(layerz['inner_points_f64'],dtype=np.float64);D0=np.asarray(layerz['deep_points_f64'],dtype=np.float64)
 # Output O must match emitted Float32; I and D carry the same emitted-node displacement.
 delta=np.asarray(trial_x,dtype=np.float64)-np.asarray(base_x,dtype=np.float64)
 O1=np.asarray(trial_x,dtype=np.float32).astype(np.float64)
 I1=I0+delta;D1=D0+delta
 checks=[]
 for name,a0,b0,field in [('skin',O0,I0,layerz['skin_thickness_transferred_m']),('fat',I0,D0,layerz['fat_thickness_transferred_m'])]:
  before=layer_flags(tf,a0,b0,rev,field)
  after=layer_flags(tf,(O1 if name=='skin' else I1),(I1 if name=='skin' else D1),rev,field)
  vb=np.asarray(before['volumes']).reshape(-1);va=np.asarray(after['volumes']).reshape(-1)
  pos=vb>0;nonpos=~pos
  pos_bad=int(np.count_nonzero(va[pos]<0.2*vb[pos]-1e-18))
  nonpos_bad=int(np.count_nonzero(va[nonpos]<vb[nonpos]-1e-18))
  new_auth=int(np.count_nonzero(np.asarray(after['authored_bad'])&~np.asarray(before['authored_bad'])))
  new_nonpos=int(np.count_nonzero(np.asarray(after['nonpositive'])&~np.asarray(before['nonpositive'])))
  new_cond=int(np.count_nonzero(np.asarray(after['conditioned'])&~np.asarray(before['conditioned'])))
  checks.append({'layer':name,'positive_cells':int(pos.sum()),'baseline_nonpositive_cells':int(nonpos.sum()),'positive_floor_failures':pos_bad,'nonpositive_worsening_cells':nonpos_bad,'new_authored_orientation_failures':new_auth,'new_float32_nonpositive_cells':new_nonpos,'new_float32_conditioned_cells':new_cond,'worst_nonpositive_worsening_mm3':float(np.min((va-vb)[nonpos],initial=0)*1e9),'minimum_positive_trial_baseline_ratio':float(np.min(va[pos]/np.maximum(vb[pos],1e-300),initial=np.inf))})
 return {'pass':all(x['positive_floor_failures']==0 and x['nonpositive_worsening_cells']==0 and x['new_authored_orientation_failures']==0 and x['new_float32_nonpositive_cells']==0 and x['new_float32_conditioned_cells']==0 for x in checks),'intervals':checks}
def audit_target(world,faces,changed,bank_T,bank_ids,M,offset,pos,exact,audit):
 local=(world-pos[43])@M[43].T-offset[43]
 skin=local[faces[changed]]
 mask=(bank_ids[:,0]==43)&(bank_ids[:,1]==16)
 bank=bank_T[mask]; bankid=bank_ids[mask,3]
 blo,bhi=bank.min((0,1)),bank.max((0,1));slo,shi=skin.min(1),skin.max(1)
 active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
 if not len(active):return set(),{'skin_aabb_active':0,'bank_aabb_active':0}
 rlo,rhi=slo[active].min(0),shi[active].max(0)
 ba=np.flatnonzero(np.all(bank.max(1)>=rlo,axis=1)&np.all(bank.min(1)<=rhi,axis=1))
 sr=audit.records(skin[active],changed[active],exact);br=audit.records(bank[ba],bankid[ba],exact)
 res=exact._audit_pair(sr,br,same_surface=False)
 pairs={(int(a),int(b)) for a,b in (res.get('triangle_pairs') or [])}
 return pairs,{'skin_aabb_active':int(len(active)),'bank_aabb_active':int(len(ba))}
def map_pairs(raw,indptr,parents):
 result=set()
 for face,tri in raw:
  for parent in parents[int(indptr[face]):int(indptr[face+1])]:result.add((int(parent),int(tri)))
 return result
def main():
 t=time.monotonic()
 base_path=B/'candidate.npz';audit_path=B/'audit-159/report.json';self_path=B/'audit-130/surface-self-pairs.npy'
 if sha(base_path)!=EXPECTED_BASE or sha(DIRECTION)!=EXPECTED_DIRECTION:raise ValueError('pinned base or old direction hash mismatch')
 if sha(LAYER)!='eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57':raise ValueError('806 layer bundle pin mismatch')
 base_doc=json.load(open(B/'audit-130/summary.json'));audit_doc=json.load(open(audit_path))
 if base_doc.get('candidate_sha256')!=EXPECTED_BASE or audit_doc.get('candidate_sha256')!=EXPECTED_BASE or not audit_doc.get('complete'):raise ValueError('pinned 806 audits missing/stale')
 with np.load(base_path,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 with np.load(DIRECTION,allow_pickle=False) as z: support=np.asarray(z['support_nodes'],dtype=np.int64);direction=np.asarray(z['direction_delta_mm'],dtype=np.float64)
 X=np.asarray(base['points_world_m'],dtype=np.float32);F=np.asarray(base['source_O_faces'],dtype=np.int64);ref=base['area_reference_037_points_world_m_f64']
 if len(support)!=373 or direction.shape!=(373,3) or len(np.unique(support))!=len(support) or support.min()<0 or support.max()>=len(X):raise ValueError('old direction support contract mismatch')
 if not np.array_equal(F,np.asarray(base['source_O_faces'],dtype=np.int64)):raise ValueError('current 806 topology pin mismatch')
 # Explicit zero-motion emission control using the current world Float32 points.
 zero=(X[support].astype(np.float64)+np.zeros_like(direction)/1000.).astype(np.float32)
 if not np.array_equal(zero.view(np.uint32),X[support].view(np.uint32)):raise ValueError('zero-motion emission is not bitwise identical')
 changed_faces_all=np.flatnonzero(np.any(np.isin(F,support),axis=1)).astype(np.int64)
 if len(changed_faces_all)==0:raise ValueError('direction has no incident faces on 806 topology')
 exact_audit=mod('rebase_audit159',AUD159);exact=exact_audit.exact_module()
 helper=mod('rebase_bank_helper',BANKHELPER)
 bank_T,bank_ids,bank_gids,bank_bodies,M,offset,pos,bank_keys,bank_pins,_=helper.read_bank_inputs(BANKDIR,E)
 b43=next(x for x in audit_doc['banks'] if (int(x['body']),int(x['stable_bone_id']))==(43,16))
 base_raw={(int(f),int(q)) for f,q in (b43.get('result') or {}).get('triangle_pairs',[])}
 indptr=np.asarray(base['refinement_face_parent_ancestry_indptr'],dtype=np.int64);parents=np.asarray(base['refinement_face_parent_ancestry_face_rows'],dtype=np.int64)
 base_mapped=map_pairs(base_raw,indptr,parents)
 baseline_local,baseline_box=audit_target(X.astype(np.float64),F,changed_faces_all,bank_T,bank_ids,M,offset,pos,exact,exact_audit)
 expected_local={(f,t) for f,t in base_raw if f in set(map(int,changed_faces_all))}
 if baseline_local!=expected_local:raise ValueError(f'806 exact local target baseline mismatch local={len(baseline_local)} stored={len(expected_local)}')
 self0_records,_=exact.float32_surface_records(X,F,source_seam_groups=())
 self0=exact._audit_self_pairs_touching_faces(self0_records,set(map(int,changed_faces_all)))
 if self0.get('triangle_pairs'):raise ValueError('zero-motion 806 self control failed')
 tf=mod('rebase_layer_transfer',TRANSFER)
 with np.load(LAYER,allow_pickle=False) as z:layerz={k:z[k].copy() for k in z.files}
 if not np.array_equal(layerz['outer_faces'],F[:,[0,2,1]]) or not np.array_equal(layerz['outer_points_f32'].view(np.uint32),X.view(np.uint32)):raise ValueError('806 layer face/outer geometry pin mismatch')
 # Cache baseline layer cell flags on all incident faces; trial deltas are applied identically to O/I/D.
 bm=X.astype(np.float64)*1000.; refmm=np.asarray(ref,dtype=np.float64)*1000.
 q=mod('rebase_area_owner',QPATH)
 before_pairs_count=len(base_raw);before_mapped_count=len(base_mapped)
 trial_reports=[];best_artifact=None;selected=None
 for alpha in ALPHAS:
  Y=X.copy();Y[support]=(X[support].astype(np.float64)+(alpha*direction)/1000.0).astype(np.float32)
  actual_changed=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
  changed_faces=np.flatnonzero(np.any(np.isin(F,actual_changed),axis=1)).astype(np.int64)
  outside=np.setdiff1d(np.arange(len(X),dtype=np.int64),support)
  anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
  no_outside_change=bool(np.array_equal(Y[outside].view(np.uint32),X[outside].view(np.uint32)))
  anchors_same=bool(np.array_equal(Y[anchors].view(np.uint32),X[anchors].view(np.uint32)))
  metrics=shape_metrics(X,Y,F,changed_faces)
  area_gate=q.full_f32_area_gate(Y,bm,refmm,F,.1)
  shape_pass=bool(metrics['zero_area_after']==0 and metrics['minimum_area_ratio']>=AREA_MIN and metrics['maximum_normal_turn_deg']<=NORMAL_MAX and metrics['minimum_min_angle_delta_deg']>=MIN_ANGLE_DELTA and metrics['maximum_aspect_ratio_ratio']<=ASPECT_MAX)
  lg=layer_check(X,Y,F,changed_faces,layerz,tf)
  pre={'alpha':alpha,'changed_nodes':int(len(actual_changed)),'changed_faces':int(len(changed_faces)),'outside_bitwise_unchanged':no_outside_change,'anchors_bitwise_unchanged':anchors_same,'shape_metrics':metrics,'shape_pass':shape_pass,'full_area_gate_pass':bool(area_gate['pass']),'layer_gate':lg,'exact_checked':False}
  if not (no_outside_change and anchors_same and shape_pass and area_gate['pass'] and lg['pass']):
   pre['status']='REJECTED_SHAPE_OR_LAYER'
   trial_reports.append(pre);continue
  pre['exact_checked']=True
  post_local,box=audit_target(Y.astype(np.float64),F,changed_faces,bank_T,bank_ids,M,offset,pos,exact,exact_audit)
  after_raw=(base_raw-expected_local)|post_local
  after_mapped=map_pairs(after_raw,indptr,parents)
  rec,_=exact.float32_surface_records(Y,F,source_seam_groups=())
  selfres=exact._audit_self_pairs_touching_faces(rec,set(map(int,changed_faces)))
  selfpairs={(int(a),int(b)) for a,b in (selfres.get('triangle_pairs') or [])}
  new_raw=after_raw-base_raw; removed_raw=base_raw-after_raw
  new_mapped=after_mapped-base_mapped; removed_mapped=base_mapped-after_mapped
  useful=bool(not new_raw and not new_mapped and not selfpairs and len(after_mapped)<len(base_mapped))
  pre.update(status='EXACT_USEFUL' if useful else 'REJECTED_EXACT_NO_NET_IDENTITY_IMPROVEMENT',exact_target={'baseline_raw':len(base_raw),'after_raw':len(after_raw),'new_raw':sorted([list(x) for x in new_raw]),'removed_raw':sorted([list(x) for x in removed_raw]),'baseline_mapped':len(base_mapped),'after_mapped':len(after_mapped),'new_mapped':sorted([list(x) for x in new_mapped]),'removed_mapped':sorted([list(x) for x in removed_mapped]),'local_aabb':box},exact_self={'changed_face_count':len(changed_faces),'pairs':sorted([list(x) for x in selfpairs]),'pair_count':len(selfpairs)})
  trial_reports.append(pre)
  # Preserve the first shape/layer passing geometry as a private failure artifact if exact gates reject it.
  if best_artifact is None:
   best_artifact=OUT/'candidate-exact-rejected-private.npz'
   payload={k:v for k,v in base.items()};payload['points_world_m']=Y
   with best_artifact.open('xb') as f:np.savez_compressed(f,**payload)
  if useful:
   selected=(Y,alpha,pre,actual_changed,changed_faces)
   break
 report={'schema':'numi.human.b43-16.rebased-direction-on-806-screen.v1','status':'EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_POINT_AUDIT' if selected else 'REJECTED_NO_EXACT_USEFUL_BACKTRACK','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'base_candidate_sha256':EXPECTED_BASE,'direction_source_candidate_sha256':'d21241ea6b16f2119faebda48196d79d4312d1316f7af5f7817e3780f145d8bb','direction_sha256':EXPECTED_DIRECTION,'direction_support_nodes':int(len(support)),'direction_moved_nodes_at_full_step':int(np.count_nonzero(np.linalg.norm(direction,axis=1)>0)),'current_806_changed_faces_touched_by_direction':int(np.count_nonzero(np.isin(np.flatnonzero(np.any(F!=F,axis=1)),changed_faces_all))) if False else None,'changed_face_rows_used_current_806':changed_faces_all.tolist(),'baseline_target_raw_pairs':before_pairs_count,'baseline_target_mapped_identities':before_mapped_count,'baseline_local_target_pair_reproduction':True,'zero_motion_bitwise_control':True,'baseline_selected_face_self_pairs':0,'layer_bundle_sha256':sha(LAYER),'audit159_sha256':sha(audit_path),'alphas':ALPHAS,'trials':trial_reports,'rejected_artifact_path':str(best_artifact) if best_artifact else None,'elapsed_s':time.monotonic()-t}
 if selected:
  Y,alpha,pre,changed_nodes,changed_faces=selected; path=OUT/'candidate-exact-backtrack-private.npz'; payload={k:v for k,v in base.items()};payload['points_world_m']=Y
  with path.open('xb') as f:np.savez_compressed(f,**payload)
  report.update(candidate_path=str(path),candidate_sha256=sha(path),selected_alpha=alpha,changed_nodes=changed_nodes.tolist(),changed_face_rows=changed_faces.tolist(),selected_trial=pre)
 elif best_artifact:
  report['rejected_artifact_sha256']=sha(best_artifact)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'baseline_raw':before_pairs_count,'baseline_mapped':before_mapped_count,'trial_summary':[{'alpha':x['alpha'],'status':x['status'],'shape':x['shape_pass'],'layer':x['layer_gate']['pass'],'exact':x['exact_checked'],'after_raw':(x.get('exact_target') or {}).get('after_raw'),'after_mapped':(x.get('exact_target') or {}).get('after_mapped'),'self':(x.get('exact_self') or {}).get('pair_count')} for x in trial_reports],'candidate':report.get('candidate_path'),'sha':report.get('candidate_sha256')},indent=2))
if __name__=='__main__':main()
