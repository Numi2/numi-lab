#!/usr/bin/env python3
"""Construction-only, exact-witness-filtered 2:2 flip proposal for details 8265/8269.
No surface/bone audit or physical admission is performed here.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D'
DEFAULT_BASE=D/'contained-clearance-545/connected-cones-014/15-270-spatial/candidate.npz'
PRIOR_REPORT=D/'contained-clearance-545/topology-017/proposal/report.json'
FAILED_REPORT=D/'contained-clearance-545/topology-017/check/report.json'
FAILED_PAIRS=D/'contained-clearance-545/topology-017/check/audit-130/surface-self-pairs.npy'
OWNER402=D/'local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
REPLAY405=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
QUALITY541=D/'focused-clearance-541/topology/repair_focused_clearance_541.py'
EXPECTED={
 'default_base':'215fa7da9bc9c3fa2a189b853343282e21403ef97c1d691a2a28cab36563ea82',
 'face_topology':'d166551add5128a1c75cd69f19b2a4c761a9595da5613a55cd5d818845e015ab',
 'prior_report':'f3dab3825f6b69f33dd38d6bfb494e667b867ac49cc15f97235461b06a454926',
 'failed_report':'c78c59961549066f882683ebaac198e7cce3f146378e7b4ec5e03f8916e2abed',
 'failed_pairs':'8caa5c03ac230c202b218173fac490696a0aae6a24bf35fd4cc1b1d6bc1a3759',
 'owner402':'f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764',
 'replay405':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'quality541':'26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa',
}
DETAIL_IDS={8265,8269}
PREFERRED_EDGES=[(12579,20182),(19702,20176)]
ALLOWED_CHANGED_KEYS={
 'source_O_faces','area_reference_037_source_O_faces',
 'face_source_ancestry_indptr','face_source_ancestry_face_rows',
 'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',
 'face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows','face_fine_detail_ancestry_semantics',
 'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows','refinement_face_parent_ancestry_semantics',
 'refinement_face_parent_face_row','refinement_face_parent_barycentric','refinement_face_parent_barycentric_semantics',
 'parent_fine_detail_face_row','parent_fine_subface_row','parent_coarse_face_row','parent_source_face_marker'
}

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def csr_rows(ptr,vals,n):
 ptr=np.asarray(ptr);vals=np.asarray(vals)
 if ptr.shape!=(n+1,) or int(ptr[0])!=0 or int(ptr[-1])!=len(vals) or np.any(ptr[1:]<ptr[:-1]):raise ValueError('invalid CSR')
 rows=[list(map(int,vals[int(ptr[i]):int(ptr[i+1])])) for i in range(n)]
 if any(r!=sorted(set(r)) for r in rows):raise ValueError('ancestry rows must be sorted and unique')
 return rows

def csr_rows_for_details(c,patch=None):
 n=len(c['source_O_faces'])
 if 'face_fine_detail_ancestry_indptr' in c:
  return csr_rows(c['face_fine_detail_ancestry_indptr'],c['face_fine_detail_ancestry_rows'],n)
 src=csr_rows(c['face_source_ancestry_indptr'],c['face_source_ancestry_face_rows'],n)
 out=[]
 for i,d in enumerate(c['parent_fine_detail_face_row']):
  out.append([int(d)] if int(d)>=0 else sorted(set(src[i])&DETAIL_IDS))
 return out

def group_of(ds):
 s=set(map(int,ds))
 return 0 if s and s.issubset(DETAIL_IDS) else None

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def identical(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
 return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def cli():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--base',type=Path,default=DEFAULT_BASE)
 ap.add_argument('--base-sha256',required=True,help='expected SHA256 of the complete input candidate NPZ')
 ap.add_argument('--out-dir',type=Path,required=True,help='new, nonexisting proposal directory')
 ap.add_argument('--focus-nodes',type=int,nargs='+',default=[18495,19702,20182])
 ap.add_argument('--max-trials',type=int,default=64)
 ap.add_argument('--max-accept',type=int,default=8)
 return ap.parse_args()

def main():
 args=cli(); base_path=args.base.resolve();outdir=args.out_dir.resolve()
 if outdir.exists():raise FileExistsError(outdir)
 if args.max_trials<1 or args.max_accept<1:raise ValueError('trial and accept limits must be positive')
 if len(set(args.focus_nodes))!=len(args.focus_nodes):raise ValueError('focus nodes must be unique')
 source_pins={'owner402':OWNER402,'replay405':REPLAY405,'quality541':QUALITY541,'prior_report':PRIOR_REPORT,'failed_report':FAILED_REPORT,'failed_pairs':FAILED_PAIRS}
 got={k:sha_file(v) for k,v in source_pins.items()}
 for k,v in got.items():
  if v!=EXPECTED[k]:raise ValueError(f'{k} frozen pin mismatch: {v}')
 base_sha=sha_file(base_path)
 if base_sha.lower()!=args.base_sha256.lower():raise ValueError(f'base candidate hash mismatch: got {base_sha}')
 base=load_npz(base_path)
 if sha_array(base['source_O_faces'])!=EXPECTED['face_topology']:raise ValueError('candidate face topology differs from pinned 017 ancestry basis')
 mod405=loadmod('cross_seam_search_405_topology018',REPLAY405)
 mod405.OWNER402=OWNER402;mod405.DETAIL_GROUPS=(tuple(sorted(DETAIL_IDS)),)
 mod405.detail_csr=csr_rows_for_details;mod405.group_of=group_of
 owner402=loadmod('two_parent_patch_flips_402_topology018',OWNER402)
 owner402.validate_candidate(base)
 if not np.array_equal(base['source_O_faces'],base['area_reference_037_source_O_faces']):raise ValueError('base current/037 topology mismatch')
 for k in ('face_fine_detail_ancestry_indptr','refinement_face_parent_ancestry_indptr'):
  if k not in base:raise ValueError(f'required full CSR ancestry missing: {k}')
 detail=csr_rows_for_details(base)
 if not any(group_of(r)==0 for r in detail):raise ValueError('no selected detail faces')
 anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 if anchors.shape!=(237,) or len(np.unique(anchors))!=237:raise ValueError('expected 237 unique typed anchors')
 if np.any(anchors<0) or np.any(anchors>=len(base['points_world_m'])):raise ValueError('anchor row out of range')
 if any(n<0 or n>=len(base['points_world_m']) for n in args.focus_nodes):raise ValueError('focus node out of range')
 anchor_xyz=base['points_world_m'][anchors].copy();input_points=base['points_world_m'].copy()
 prior=json.loads(PRIOR_REPORT.read_text());fail=json.loads(FAILED_REPORT.read_text())
 pairs=np.load(FAILED_PAIRS,allow_pickle=False)
 prior_candidate=D/'contained-clearance-545/topology-017/proposal/candidate.npz'
 if prior.get('candidate_input_sha256')!=EXPECTED['default_base'] or prior.get('accepted_flip_count')!=8:raise ValueError('pinned 017 proposal does not match expected original base')
 if prior.get('input_faces_sha256')!=EXPECTED['face_topology']:raise ValueError('017 input topology pin mismatch')
 if fail.get('status')!='FAILED' or fail.get('candidate_sha256')!=sha_file(prior_candidate):raise ValueError('failed exact report does not match 017 proposed candidate')
 if pairs.shape!=(10,2):raise ValueError('expected ten exact failed face-pair witnesses')
 if np.any(pairs<0) or np.any(pairs>=len(base['source_O_faces'])):raise ValueError('failed witness face row out of range')
 witness_faces=sorted(set(map(int,pairs.reshape(-1))));witness_set=set(witness_faces)
 prior_ops=prior.get('operations') or [];blocked=[];edge_queue=[]
 focus=set(map(int,args.focus_nodes))
 for edge in PREFERRED_EDGES:
  if focus.intersection(edge):edge_queue.append({'edge':list(edge),'origin':'preferred_focus_edge'})
 for row in prior_ops:
  fr=list(map(int,row['face_rows']))
  if witness_set.intersection(fr):
   blocked.append({'edge':row['edge'],'face_rows':fr,'reason':'touches_exact_failed_self_witness_face'})
  elif focus.intersection(map(int,row['edge'])) or focus.intersection(set(map(int,base['source_O_faces'][fr].reshape(-1)))):
   edge_queue.append({'edge':row['edge'],'origin':'017_local_operation','prior_face_rows':fr})
 # Add one-ring candidate edges around focus nodes and exact witness faces only.
 faces=np.asarray(base['source_O_faces']);em=owner402.edge_map(faces)
 focus_faces=set(map(int,np.flatnonzero(np.any(np.isin(faces,np.asarray(sorted(focus))),axis=1))))
 witness_nodes=set(map(int,faces[np.asarray(witness_faces,dtype=np.int64)].reshape(-1)))
 witness_ring_faces=set(map(int,np.flatnonzero(np.any(np.isin(faces,np.asarray(sorted(witness_nodes))),axis=1))))
 local_rows=focus_faces|witness_ring_faces
 local_edges=set()
 for e,inc in em.items():
  if len(inc)==2 and any(int(row[0]) in local_rows for row in inc):local_edges.add(tuple(sorted(map(int,e))))
 for e in sorted(local_edges):edge_queue.append({'edge':list(e),'origin':'focus_or_witness_one_ring'})
 seen=set();queue=[]
 for item in edge_queue:
  e=tuple(sorted(map(int,item['edge'])))
  if e not in seen:seen.add(e);queue.append((e,item))
 if not queue:raise ValueError('no candidate edges in requested focus/witness neighborhood')
 mod541=loadmod('repair_focused_clearance_541_topology018',QUALITY541)
 points_mm=np.asarray(base['points_world_m'],dtype=np.float64)*1000.0
 ref_mm=np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000.0
 quality=mod541.face_quality(points_mm,ref_mm,faces,base['area_reference_037_source_O_faces'])
 current={k:np.array(v,copy=True) for k,v in base.items()}
 locks0,outer0,between0,_=mod405.boundary_locks(current,detail,{8265,8269})
 accepted=[];attempts=[]
 for edge,item in queue[:args.max_trials]:
  if len(accepted)>=args.max_accept:break
  em=owner402.edge_map(current['source_O_faces']);inc=em.get(edge,[])
  attempt={'edge':list(edge),'origin':item['origin'],'prior_face_rows':item.get('prior_face_rows')}
  if len(inc)!=2:
   attempt.update(status='rejected',reason='edge_not_current_interior_edge');attempts.append(attempt);continue
  candidate_op,reason=mod405.propose_one(current,edge,inc,locks0)
  if candidate_op is None:
   attempt.update(status='rejected',reason=reason);attempts.append(attempt);continue
  fr=list(map(int,candidate_op['face_rows']));attempt['current_face_rows']=fr
  if witness_set.intersection(fr):
   attempt.update(status='rejected',reason='touches_exact_failed_self_witness_face');attempts.append(attempt);continue
  quality_result,reason=mod541.evaluate_op(current,candidate_op,quality,points_mm,ref_mm)
  if quality_result is None:
   attempt.update(status='rejected',reason=f'541_{reason}');attempts.append(attempt);continue
  i,j=fr
  pre_rows={
   'source':csr_rows(current['face_source_ancestry_indptr'],current['face_source_ancestry_face_rows'],len(faces)),
   'subface':csr_rows(current['face_fine_subface_ancestry_indptr'],current['face_fine_subface_ancestry_rows'],len(faces)),
   'detail':csr_rows_for_details(current),
   'parent':csr_rows(current['refinement_face_parent_ancestry_indptr'],current['refinement_face_parent_ancestry_face_rows'],len(faces)),
  }
  expected_union={k:sorted(set(pre_rows[k][i])|set(pre_rows[k][j])) for k in pre_rows}
  candidate_op['quality_541']=quality_result;candidate_op['origin_018']=item['origin']
  current=mod405.apply_op(current,candidate_op)
  if not identical(current['points_world_m'],input_points):raise ValueError('coordinates changed')
  if not np.array_equal(current['source_O_faces'],current['area_reference_037_source_O_faces']):raise ValueError('current/037 face sync lost')
  nowdetail=csr_rows_for_details(current)
  locks1,outer1,between1,_=mod405.boundary_locks(current,nowdetail,{8265,8269})
  if locks1!=locks0 or outer1!=outer0 or between1!=between0:raise ValueError('selected patch boundary/seam changed')
  if not identical(current['points_world_m'][anchors],anchor_xyz):raise ValueError('typed anchor coordinates changed')
  after_rows={
   'source':csr_rows(current['face_source_ancestry_indptr'],current['face_source_ancestry_face_rows'],len(faces)),
   'subface':csr_rows(current['face_fine_subface_ancestry_indptr'],current['face_fine_subface_ancestry_rows'],len(faces)),
   'detail':nowdetail,
   'parent':csr_rows(current['refinement_face_parent_ancestry_indptr'],current['refinement_face_parent_ancestry_face_rows'],len(faces)),
  }
  for k in expected_union:
   if after_rows[k][i]!=expected_union[k] or after_rows[k][j]!=expected_union[k]:raise ValueError(f'{k} ancestry not exact pairwise CSR union')
  if len(expected_union['detail'])>1 and any(int(current['parent_fine_detail_face_row'][r])!=-1 for r in (i,j)):raise ValueError('multi-detail singular ancestry must be -1')
  if len(expected_union['detail'])==1 and any(int(current['parent_fine_detail_face_row'][r])!=expected_union['detail'][0] for r in (i,j)):raise ValueError('single-detail ancestry label mismatch')
  if len(expected_union['subface'])>1 and any(int(current['parent_fine_subface_row'][r])!=-1 for r in (i,j)):raise ValueError('multi-subface singular ancestry must be -1')
  if len(expected_union['subface'])==1 and any(int(current['parent_fine_subface_row'][r])!=expected_union['subface'][0] for r in (i,j)):raise ValueError('single-subface ancestry label mismatch')
  if len(expected_union['parent'])>1 and any(int(current['refinement_face_parent_face_row'][r])!=-1 for r in (i,j)):raise ValueError('multi-parent singular ancestry must be -1')
  op_record={k:v for k,v in candidate_op.items() if k!='new_faces'};op_record['quality_541']=quality_result
  accepted.append(op_record);attempt.update(status='accepted',quality_541=quality_result,new_faces=candidate_op['new_faces']);attempts.append(attempt)
  quality=mod541.face_quality(points_mm,ref_mm,current['source_O_faces'],current['area_reference_037_source_O_faces'])
 # Final construction-only invariants.
 owner402.validate_candidate(current)
 if not accepted:raise RuntimeError('no eligible operations; no candidate emitted')
 if len({tuple(sorted(map(int,t))) for t in current['source_O_faces']})!=len(current['source_O_faces']):raise ValueError('duplicate faces')
 if any(len(v)>2 for v in owner402.edge_map(current['source_O_faces']).values()):raise ValueError('nonmanifold edge introduced')
 changed=sorted({int(r) for op in accepted for r in op['face_rows']})
 if witness_set.intersection(changed):raise ValueError('accepted operation touches exact failed witness face')
 changed_keys=[k for k in base if not identical(base[k],current[k])]
 unexpected=sorted(set(changed_keys)-ALLOWED_CHANGED_KEYS)
 if unexpected:raise ValueError(f'unexpected non-topology arrays changed: {unexpected}')
 outdir.mkdir(parents=True,exist_ok=False);candidate_path=outdir/'candidate.npz'
 np.savez_compressed(candidate_path,**current);candidate_sha=sha_file(candidate_path)
 receipt={
  'status':'CONSTRUCTION_PROPOSAL_UNAUDITED','geometry_admitted':False,'full_exact_surface_and_all_185_bone_checks_required':True,
  'base_path':str(base_path),'base_sha256':base_sha,'base_face_topology_sha256':sha_array(base['source_O_faces']),
  'owner_pins':got,'face_topology_pin':EXPECTED['face_topology'],
  'failure_evidence':{'report_sha256':got['failed_report'],'pairs_sha256':got['failed_pairs'],'pair_count':int(len(pairs)),'blacklisted_face_rows':witness_faces},
  'prior_proposal_report_sha256':got['prior_report'],'adaptation':{'detail_groups':[[8265,8269]],'group_count':1,'cross_detail_union_enabled':True,'405_parameterized_in_memory':True},
  'focus_nodes':list(map(int,args.focus_nodes)),'candidate_edge_sources':['preferred structural edges','eligible 017 operations incident to focus nodes','one-ring edges at focus nodes or exact witness faces'],
  'quality_policy':{'owner':'541 evaluate_op','minimum_face_angle_deg':float(mod541.MIN_PATCH_ANGLE_DEG),'maximum_minimum_angle_loss_deg':float(mod541.MAX_MIN_ANGLE_LOSS_DEG),'minimum_fixed037_area_ratio':float(mod541.MIN_AREA_RATIO)},
  'candidate_path':str(candidate_path),'candidate_sha256':candidate_sha,'candidate_faces_sha256':sha_array(current['source_O_faces']),'candidate_points_sha256':sha_array(current['points_world_m']),
  'input_points_bitwise_unchanged':identical(current['points_world_m'],input_points),'typed_anchor_count':int(len(anchors)),'typed_anchor_coordinates_bitwise_unchanged':identical(current['points_world_m'][anchors],anchor_xyz),
  'current_037_faces_synchronized':bool(np.array_equal(current['source_O_faces'],current['area_reference_037_source_O_faces'])),'selected_patch_boundary_and_outer_edges_preserved':True,
  'changed_npz_keys':changed_keys,'accepted_flip_count':len(accepted),'accepted_operations':accepted,'attempts':attempts,'blocked_prior_operations':blocked,'changed_face_rows':changed,
  'limitations':['Construction-only fixed-coordinate 2:2 flips.','No exact self-intersection, bone-surface, or point-in-volume audit was run.','541 quality screening is not a collision-clearance certificate.']}
 (outdir/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'candidate':str(candidate_path),'candidate_sha256':candidate_sha,'accepted_flip_count':len(accepted),'changed_face_rows':changed,'receipt':str(outdir/'receipt.json')},indent=2))

if __name__=='__main__':main()
