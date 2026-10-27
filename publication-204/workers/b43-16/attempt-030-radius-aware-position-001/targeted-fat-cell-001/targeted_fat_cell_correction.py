#!/usr/bin/env python3
from pathlib import Path
import importlib.util,json,sys,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BASE_DIR=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001')
OLD=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001')
SCREEN=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py')
TRANSFER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py')
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
TARGET_FACE=19873; TARGET_INTERVAL=1; TARGET_SLOT=1; ALPHA=0.00390625
EXPECTED_BASE='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
EXPECTED_DIR='b1550f73cae5d4b705add5467309ae3843341fec13a1b74a3516ebc308daf504'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def importmod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def cell_volume_mm3(outer,inner,face_row,faces):
 # Exact same tetra slot 1 construction as face_flags for layer interval 1.
 oriented=np.asarray(faces[face_row][[0,2,1]],dtype=np.int64);a,b,c=sorted(map(int,oriented))
 ids=[(a,outer[a]),(b,outer[b]),(b,inner[b]),(c,inner[c])]
 p=np.asarray([v for _,v in ids],dtype=np.float64)
 v64=float(np.dot(p[1]-p[0],np.cross(p[2]-p[0],p[3]-p[0]))/6.0)
 if v64<0:p[[0,1]]=p[[1,0]]
 q=p.astype(np.float32).astype(np.float64)
 return float(np.dot(q[1]-q[0],np.cross(q[2]-q[0],q[3]-q[0]))/6.0*1e9)
def main():
 t=time.monotonic();base_path=BASE_DIR/'candidate.npz';direction_path=OLD/'solved-direction.npz';layer_path=BASE_DIR/'skin-fat-layers-541.npz'
 if sha(base_path)!=EXPECTED_BASE or sha(direction_path)!=EXPECTED_DIR:raise ValueError(f'pinned baseline/direction mismatch base={sha(base_path)} direction={sha(direction_path)} expected={EXPECTED_BASE}/{EXPECTED_DIR}')
 screen=importmod('target_screen',SCREEN)
 with np.load(base_path,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 with np.load(direction_path,allow_pickle=False) as z:support=np.asarray(z['support_nodes'],dtype=np.int64);direction=np.asarray(z['direction_delta_mm'],dtype=np.float64)
 X=np.asarray(base['points_world_m'],dtype=np.float32);F=np.asarray(base['source_O_faces'],dtype=np.int64)
 if not np.array_equal(F,np.asarray(base['source_O_faces'],dtype=np.int64)):raise ValueError('current topology mismatch')
 loc={int(n):i for i,n in enumerate(support)}
 face_nodes=F[TARGET_FACE]
 missing=[int(n) for n in face_nodes if int(n) not in loc]
 if missing:raise ValueError(f'target fat cell nodes not movable: {missing}')
 with np.load(layer_path,allow_pickle=False) as z:layerz={k:z[k].copy() for k in z.files}
 O0=np.asarray(layerz['outer_points_f64'],dtype=np.float64);I0=np.asarray(layerz['inner_points_f64'],dtype=np.float64);D0=np.asarray(layerz['deep_points_f64'],dtype=np.float64)
 if not np.array_equal(layerz['outer_points_f32'].view(np.uint32),X.view(np.uint32)):raise ValueError('layer outer/baseline mismatch')
 Y=(X[support].astype(np.float64)+(ALPHA*direction)/1000.0).astype(np.float32)
 trial0=X.copy();trial0[support]=Y
 delta0=trial0.astype(np.float64)-X.astype(np.float64)
 Ibase=(I0+delta0);Dbase=(D0+delta0)
 vb=cell_volume_mm3(I0,D0,TARGET_FACE,F)
 vt0=cell_volume_mm3(Ibase,Dbase,TARGET_FACE,F)
 # Search one-node, one-axis local moves, always re-emitting from base+total delta in one Float32 cast.
 steps=[1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2]
 viable=[]
 for node in face_nodes:
  node=int(node)
  for axis in range(3):
   for sign in (-1.,1.):
    for step in steps:
     corr=np.zeros(3);corr[axis]=sign*step
     Ytrial=trial0[support].copy()
     total=ALPHA*direction[loc[node]]+corr
     Ytrial[loc[node]]=(X[node].astype(np.float64)+total/1000.0).astype(np.float32)
     trial=X.copy();trial[support]=Ytrial
     d=trial.astype(np.float64)-X.astype(np.float64)
     vol=cell_volume_mm3(I0+d,D0+d,TARGET_FACE,F)
     if vol>=vb+1e-9:
      viable.append((float(np.linalg.norm(corr)),float(step),int(node),int(axis),float(sign),vol,trial,Ytrial))
 if not viable:
  report={'status':'TARGETED_CELL_CORRECTION_NO_FLOAT32_STEP','base_sha256':EXPECTED_BASE,'direction_sha256':EXPECTED_DIR,'alpha':ALPHA,'target_cell':{'face_row':TARGET_FACE,'interval':TARGET_INTERVAL,'tetra_slot':TARGET_SLOT,'baseline_mm3':vb,'unmodified_trial_mm3':vt0,'worsening_mm3':vt0-vb},'correction_search_steps_mm':steps,'support_node_count':len(support),'missing_cell_nodes':missing,'geometry_admitted':False,'accepted_physical_steps':0}
  (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
 viable.sort(key=lambda q:(q[0],-q[5]))
 norm,step,node,axis,sign,correctedvol,Yfull,Ysup=viable[0]
 changed=np.flatnonzero(np.any(Yfull.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
 changed_faces=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(np.int64)
 affected=np.flatnonzero(np.any(np.isin(F,support),axis=1)).astype(np.int64)
 screen_tf=screen.mod('target_transfer',TRANSFER); q=screen.mod('target_area',QPATH)
 shape=screen.shape_metrics(X,Yfull,F,changed_faces)
 area=q.full_f32_area_gate(Yfull,np.asarray(X,dtype=np.float64)*1000.,np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000.,F,.1)
 layer=screen.layer_check(X,Yfull,F,changed_faces,layerz,screen_tf)
 shape_pass=bool(shape['zero_area_after']==0 and shape['minimum_area_ratio']>=screen.AREA_MIN and shape['maximum_normal_turn_deg']<=screen.NORMAL_MAX and shape['minimum_min_angle_delta_deg']>=screen.MIN_ANGLE_DELTA and shape['maximum_aspect_ratio_ratio']<=screen.ASPECT_MAX)
 zero=bool(np.array_equal((X[support].astype(np.float64)+np.zeros_like(direction)/1000.).astype(np.float32).view(np.uint32),X[support].view(np.uint32)))
 # Scoped exact target and complete self are only run after shape/layer pass.
 report={'schema':'numi.human.b43-16.targeted-fat-witness-correction.v1','status':'REJECTED_SHAPE_OR_LAYER','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'base_sha256':EXPECTED_BASE,'direction_sha256':EXPECTED_DIR,'layer_sha256':sha(layer_path),'alpha':ALPHA,'target_cell':{'face_row':TARGET_FACE,'interval':TARGET_INTERVAL,'tetra_slot':TARGET_SLOT,'baseline_mm3':vb,'unmodified_trial_mm3':vt0,'corrected_trial_mm3':correctedvol,'correction_node':node,'correction_axis':axis,'correction_sign':sign,'correction_mm':step,'correction_max_mm':norm},'support_nodes':int(len(support)),'changed_nodes':changed.tolist(),'changed_face_rows':changed_faces.tolist(),'zero_motion_bitwise_control':zero,'shape_metrics':shape,'shape_pass':shape_pass,'full_area_gate_pass':bool(area['pass']),'layer_gate':layer,'exact_checked':False,'elapsed_s':time.monotonic()-t}
 if shape_pass and area['pass'] and layer['pass'] and zero:
  report['status']='SHAPE_LAYER_PASS_EXACT_PENDING'
  exact_audit=screen.mod('target_audit159',screen.AUD159); exact=exact_audit.exact_module()
  helper=screen.mod('target_bank_helper',screen.BANKHELPER)
  bank_T,bank_ids,bank_gids,bank_bodies,M,offset,pos,bank_keys,bank_pins,_=helper.read_bank_inputs(screen.BANKDIR,screen.E)
  audit_doc=json.load(open(BASE_DIR/'audit-159/report.json')); bb=next(x for x in audit_doc['banks'] if (int(x['body']),int(x['stable_bone_id']))==(43,16))
  raw_base={(int(f),int(q)) for f,q in (bb.get('result') or {}).get('triangle_pairs',[])}
  ip=base['refinement_face_parent_ancestry_indptr']; pa=base['refinement_face_parent_ancestry_face_rows']; mapped_base=screen.map_pairs(raw_base,ip,pa)
  raw_local,box0=screen.audit_target(X.astype(np.float64),F,changed_faces,bank_T,bank_ids,M,offset,pos,exact,exact_audit)
  expected={(f,q) for f,q in raw_base if f in set(map(int,changed_faces))}
  if raw_local!=expected:raise ValueError('zero motion local target pair control differs from audit-159')
  raw_after,box1=screen.audit_target(Yfull.astype(np.float64),F,changed_faces,bank_T,bank_ids,M,offset,pos,exact,exact_audit)
  after=(raw_base-expected)|raw_after; mapped_after=screen.map_pairs(after,ip,pa)
  records,_=exact.float32_surface_records(Yfull,F,source_seam_groups=())
  selfres=exact._audit_self_pairs_touching_faces(records,set(map(int,changed_faces)));selfpairs={(int(a),int(b)) for a,b in (selfres.get('triangle_pairs') or [])}
  new_raw=after-raw_base; removed_raw=raw_base-after; new_map=mapped_after-mapped_base;removed_map=mapped_base-mapped_after
  useful=bool(not new_raw and not new_map and not selfpairs and len(mapped_after)<len(mapped_base))
  report.update(status='EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_POINT_AUDIT' if useful else 'REJECTED_EXACT_NO_NET_IDENTITY_IMPROVEMENT',exact_checked=True,exact_target={'baseline_raw':len(raw_base),'after_raw':len(after),'new_raw':sorted([list(x) for x in new_raw]),'removed_raw':sorted([list(x) for x in removed_raw]),'baseline_mapped':len(mapped_base),'after_mapped':len(mapped_after),'new_mapped':sorted([list(x) for x in new_map]),'removed_mapped':sorted([list(x) for x in removed_map]),'local_aabb_before':box0,'local_aabb_after':box1},exact_self={'pair_count':len(selfpairs),'pairs':sorted([list(x) for x in selfpairs])})
  dest=HERE/('candidate-exact-backtrack-private.npz' if useful else 'candidate-exact-rejected-private.npz')
  payload=dict(base);payload['points_world_m']=Yfull
  with dest.open('xb') as f:np.savez_compressed(f,**payload)
  report.update(candidate_path=str(dest),candidate_sha256=sha(dest))
 else:
  # Keep the targeted correction as a private diagnostic, never as a candidate claim.
  dest=HERE/'candidate-shape-layer-rejected-private.npz';payload=dict(base);payload['points_world_m']=Yfull
  with dest.open('xb') as f:np.savez_compressed(f,**payload)
  report.update(candidate_path=str(dest),candidate_sha256=sha(dest))
 report['elapsed_s']=time.monotonic()-t
 (HERE/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:report.get(k) for k in ['status','target_cell','shape_metrics','shape_pass','full_area_gate_pass','layer_gate','exact_target','exact_self','candidate_path','candidate_sha256','elapsed_s']},indent=2))
if __name__=='__main__':main()
