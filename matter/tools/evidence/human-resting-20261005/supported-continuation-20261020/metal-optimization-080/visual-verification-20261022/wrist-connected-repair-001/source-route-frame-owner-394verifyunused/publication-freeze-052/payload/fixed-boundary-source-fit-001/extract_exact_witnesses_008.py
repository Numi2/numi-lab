#!/usr/bin/env python3
import hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
P=pathlib.Path(__file__).resolve().parent
G=P/'gate_source_fit_008_134_allposes.py'; F=P/'build_source_fit_008_robust134_highs.py'
R=P/'run-008-134-robust-normal-highs'; CAND=R/'candidate-8874-local-position-fit.nhtissue'; BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
GATE=R/'exact-local-source-fit-134-allposes-corrected/exact-local-gate-report.json'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=load('gate',G);fit=load('fit',F);canon=load('canon',g.CANON)
base=canon.decode_nhtiss(BASE.read_bytes()); cand=canon.decode_nhtiss(CAND.read_bytes())
rec=next(x for x in base['records'] if int(x[6])==134);fv,vc,fi,ic=map(int,(rec[2],rec[3],rec[4],rec[5]));ix=base['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);faces=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
ids=(131,132,133,134,137,138,141,142); ident=json.loads(g.IDENT.read_text()); pose_specs={'current191':fit.pose_map(ident,'current'),'rest003':fit.pose_map(ident,'rest')}
for fn in g.FK:pose_specs[fn[:-5]]=fit.pose_map(json.loads((g.POSE_DIR/fn).read_text()),'fk')
ns=fit.load_emitter();sys.path.insert(0,str(g.EXACT));from numilab_human import cardiac_cavity_intersections as ci
report=json.loads(GATE.read_text()); pair_ids=sorted({tuple(p) for c in report['exact_checks'] for key in ('new_pairs','resolved_pairs') for p in c[key]})
result={'schema':'numi.human.exact-local-pair-witnesses.v1','baseline_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(CAND.read_bytes()).hexdigest(),'gate_report_sha256':hashlib.sha256(GATE.read_bytes()).hexdigest(),'stable_id':134,'face_pair_ids':pair_ids,'poses':[]}
for name,poses in pose_specs.items():
 ns['poses']=poses;B=ns['emit_selected'](BASE)[134][0];ns['poses']=poses;C=ns['emit_selected'](CAND)[134][0]
 rb={x[3]:x for x in g.exact_records(ci,g.exact_points(ci,B),faces,sorted(set(v for p in pair_ids for v in p)))}
 rc={x[3]:x for x in g.exact_records(ci,g.exact_points(ci,C),faces,sorted(set(v for p in pair_ids for v in p)))}
 poseout={'pose':name,'pairs':[]}
 for a,b in pair_ids:
  shared=sorted(set(map(int,faces[a]))&set(map(int,faces[b])))
  entry={'face_rows':[a,b],'face_vertices_local':[faces[a].tolist(),faces[b].tolist()], 'shared_vertex_local_ids':shared,'shared_vertex_global_ids':[fv+x for x in shared]}
  for label,rs in [('baseline',rb),('candidate',rc)]:
   audit=ci._audit_pair([rs[a]],[rs[b]],same_surface=True);hits=ci.triangle_intersection_points(rs[a][0],rs[b][0]);unique=sorted({tuple(p) for p in hits})
   points=[]
   for p in unique:
    xyz_m=[float(v) for v in p];points.append({'world_m':xyz_m,'world_mm':[v*1000.0 for v in xyz_m]})
   entry[label]={'classified_as_pair':bool(audit['triangle_pairs']),'allowed_shared_feature_count':int(audit['allowed_shared_vertex_or_edge_pairs']),'exact_intersection_point_count_with_shared_points':len(hits),'unique_exact_intersection_points':points}
  poseout['pairs'].append(entry)
 result['poses'].append(poseout)
out=P/'run-008-exact-pair-witnesses.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'pairs':pair_ids,'poses':len(pose_specs)},indent=2))
