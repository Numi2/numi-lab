#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027'
B=W/'combined-d071-b109-b61-001'
OUT=Path(__file__).resolve().parent
BASE_SHA='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
BASE=B/'candidate.npz'
DIR=OUT/'contact-feature-direction.npz'
AUD=B/'audit-159/report.json'
SUM=B/'audit-130/summary.json'
PROJ=OUT/'project_contacts.py'
SCREEN=W/'b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
ALPHA=.0625
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 t=time.monotonic()
 if sha(BASE)!=BASE_SHA: raise ValueError('806 candidate SHA mismatch')
 bas=mod('d031_proj',PROJ)
 scr=mod('d031_screen',SCREEN)
 auditmod=mod('d031_aud159',AUD159)
 helper=mod('d031_bankhelper',HELPER)
 with np.load(BASE,allow_pickle=False) as z: p={k:z[k].copy() for k in z.files}
 with np.load(DIR,allow_pickle=False) as z:
  support=np.asarray(z['support_nodes'],dtype=np.int64);d=np.asarray(z['direction_delta_mm'],dtype=np.float64)
 if d.shape!=(len(support),3): raise ValueError('direction shape mismatch')
 X=np.asarray(p['points_world_m'],dtype=np.float32);F=np.asarray(p['source_O_faces'],dtype=np.int64)
 zero=(X[support].astype(np.float64)+np.zeros_like(d)/1000.0).astype(np.float32)
 if not np.array_equal(zero.view(np.uint32),X[support].view(np.uint32)): raise ValueError('zero-motion is not bitwise identity')
 Y=X.copy(); Y[support]=(X[support].astype(np.float64)+(ALPHA*d)/1000.0).astype(np.float32)
 changed=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
 chf=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(np.int64)
 if not len(changed) or not len(chf): raise ValueError('direction emitted no changed geometry')
 base_a=json.loads(AUD.read_text());base130=json.loads(SUM.read_text())
 b=next(r for r in base_a['banks'] if (int(r['body']),int(r['stable_bone_id']))==(43,16))
 raw0={(int(a),int(q)) for a,q in b['result']['triangle_pairs']}
 ip=np.asarray(p['refinement_face_parent_ancestry_indptr'],dtype=np.int64);pa=np.asarray(p['refinement_face_parent_ancestry_face_rows'],dtype=np.int64)
 mapped0=scr.map_pairs(raw0,ip,pa)
 BT,bids,_,_,M,offset,pos,_,_,_=helper.read_bank_inputs(BANKDIR,E)
 exactmod=auditmod.exact_module()
 oldlocal,box0=scr.audit_target(X.astype(np.float64),F,chf,BT,bids,M,offset,pos,exactmod,auditmod)
 expected={(f,q) for f,q in raw0 if f in set(map(int,chf))}
 if oldlocal!=expected: raise ValueError(f'zero-motion changed-face target parity failed: exact {len(oldlocal)} stored {len(expected)}')
 postlocal,box1=scr.audit_target(Y.astype(np.float64),F,chf,BT,bids,M,offset,pos,exactmod,auditmod)
 raw1=(raw0-expected)|postlocal
 mapped1=scr.map_pairs(raw1,ip,pa)
 rawnew=raw1-raw0; rawremoved=raw0-raw1
 mapnew=mapped1-mapped0; mapremoved=mapped0-mapped1
 records,_=exactmod.float32_surface_records(Y,F,source_seam_groups=())
 selfres=exactmod._audit_self_pairs_touching_faces(records,set(map(int,chf)))
 selfpairs={(int(a),int(q)) for a,q in (selfres.get('triangle_pairs') or [])}
 wits=[r for r in base130['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(43,16) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0]
 selected=np.asarray(sorted(int(r['query_row']) for r in wits),dtype=np.int64)
 qdir=OUT/'inside-query-43'
 qpoints,qpairs,qnrows,pnpz=bas.point_query_nodes(Y,selected,qdir)
 if len(qpoints)!=43 or len(qnrows)!=43: raise ValueError('point query cardinality mismatch')
 if any(int(r['total_bone_banks'])!=185 or int(r['ambiguous_banks']) or int(r['invalid_banks']) for r in qnrows): raise ValueError('incomplete/ambiguous 185-bank point query')
 basepairs={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in base130['inside_witnesses'] if int(r['query_row']) in set(map(int,selected))}
 candpairs={(int(selected[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id'])) for r in qpairs if r['classification']=='inside'}
 target={(int(n),43,16) for n in selected}
 targetrows=[r for r in qpairs if (int(selected[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id'])) in target and r['classification']=='inside' and r.get('signed_face_distance_m') not in ('','None')]
 bd=[abs(float(r['signed_face_distance_m']))*1000.0 for r in wits]
 cd=[abs(float(r['signed_face_distance_m']))*1000.0 for r in targetrows]
 projection=json.loads((OUT/'report.json').read_text())
 trial=next(r for r in projection['backtracks'] if float(r['alpha'])==ALPHA)
 report={
  'schema':'numi.human.b43-16.contact-feature-alpha00625-diagnostic.v1',
  'status':'REJECTED_LAYER_GATE_DIAGNOSTIC_ONLY',
  'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
  'baseline_candidate_sha256':BASE_SHA,'direction_sha256':sha(DIR),'projection_report_sha256':sha(OUT/'report.json'),
  'alpha':ALPHA,'emission':'(base_world_float32 promoted to float64 + alpha*delta_mm/1000) cast once to float32',
  'zero_motion_bitwise_identical':True,'changed_nodes':int(len(changed)),'changed_faces':int(len(chf)),
  'changed_face_rows_sha256':hashlib.sha256(chf.astype('<i8').tobytes()).hexdigest(),
  'shape':trial.get('shape'),'shape_pass':trial.get('shape'),
  'layer_gate':trial.get('layer_gate'),'layer_pass':False,
  'exact_43_16':{'baseline_raw':len(raw0),'candidate_raw':len(raw1),'baseline_mapped':len(mapped0),'candidate_mapped':len(mapped1),'new_raw':sorted([list(x) for x in rawnew]),'removed_raw':sorted([list(x) for x in rawremoved]),'new_mapped':sorted([list(x) for x in mapnew]),'removed_mapped':sorted([list(x) for x in mapremoved]),'baseline_changed_face_exact_control_pairs':len(oldlocal),'candidate_changed_face_pairs':len(postlocal),'aabb_baseline':box0,'aabb_candidate':box1},
  'exact_self':{'scope':'all full-surface triangles, pairs touching any changed face','count':len(selfpairs),'pairs':sorted([list(x) for x in selfpairs])},
  'inside_point_query':{'nodes':selected.tolist(),'node_count':len(selected),'banks_per_node':[int(r['total_bone_banks']) for r in qnrows],'invalid_or_ambiguous':0,'baseline_target_inside_count':sum((n,43,16) in basepairs for n in selected),'candidate_target_inside_count':sum(x in candpairs for x in target),'baseline_depth_abs_mm':{'min':float(np.min(bd)),'median':float(np.median(bd)),'max':float(np.max(bd))},'candidate_inside_depth_abs_mm':{'min':float(np.min(cd)) if cd else None,'median':float(np.median(cd)) if cd else None,'max':float(np.max(cd)) if cd else None},'all_bank_baseline_inside_pair_count':len(basepairs),'all_bank_candidate_inside_pair_count':len(candpairs),'new_inside_pairs':sorted([list(x) for x in candpairs-basepairs]),'removed_inside_pairs':sorted([list(x) for x in basepairs-candpairs]),'query_points_npz_sha256':sha(pnpz)},
  'not_run':{'exact_neighbor_bank_triangle_audit':'not required for rejected layer-failing diagnostic; no promotion','complete_185_bank_face_audit':'not run','complete_body_self_audit':'not run beyond exact changed-face-touching scope'},
  'limitations':['alpha 0.0625 fails local skin/fat prism gate; exact checks are diagnostic, not candidate qualification','point-query classification changes at nodes do not establish clearance of the full skin surface','neighbor-bank intersections and full whole-body audits remain required before any proposal','no physical or native admission'],
  'elapsed_s':time.monotonic()-t
 }
 out=OUT/'contact-feature-alpha00625-diagnostic.json'
 out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'raw':[len(raw0),len(raw1)],'mapped':[len(mapped0),len(mapped1)],'new_mapped':len(mapnew),'removed_mapped':len(mapremoved),'self':len(selfpairs),'inside':[report['inside_point_query']['baseline_target_inside_count'],report['inside_point_query']['candidate_target_inside_count']],'depth_before':report['inside_point_query']['baseline_depth_abs_mm'],'depth_after':report['inside_point_query']['candidate_inside_depth_abs_mm']},indent=2))
if __name__=='__main__': main()
