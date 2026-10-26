#!/usr/bin/env python3
"""Replay only selected 531 flips that do not touch root's exact self-pair faces.

This is a topology candidate reducer. It does not audit/admit the result.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/Work/numi-human-connected-surface-069')
BASE=ROOT/'D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz'
ALL_CAND=ROOT/'D/connected-quality-patch-531/run-002/candidate.npz'
ALL_REPORT=ROOT/'D/connected-quality-patch-531/run-002/report.json'
AUDIT=ROOT/'D/connected-quality-patch-531/root-exact-run002/audit-130/summary.json'
PAIRS=ROOT/'D/connected-quality-patch-531/root-exact-run002/audit-130/surface-self-pairs.npy'
REPLAY=ROOT/'D/local-two-parent-retriangulation-405/cross_seam_search_405.py'
BASE_SHA='c1c0b63ee448703ef3240dbcc251e650022d7f76f475bc75db9cae040d1e3a6c'
ALL_CAND_SHA='32ec169a84ce4ed22915c0ffb9ea110cac6d7962c682bad9e6d025b0b03f5e4e'
ALL_REPORT_SHA='b02c863f5409f5f14a72d65931c8e7e2f7427279f13adc35f3badeb2f8bd86b7'
AUDIT_SHA='b2a8713ffc9ca1a893f141e341b771afad98394b36662e38e399937f6064278d'
PAIRS_SHA='129927157f3a00a39d521048cb94d9ff24f4cb5baa87822450f6330f22f9416d'
REPLAY_SHA='e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967'
ALLOWED_CHANGED_KEYS={
 'source_O_faces','area_reference_037_source_O_faces',
 'face_source_ancestry_indptr','face_source_ancestry_face_rows',
 'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',
 'face_fine_subface_ancestry_semantics','face_fine_detail_ancestry_indptr',
 'face_fine_detail_ancestry_rows','face_fine_detail_ancestry_semantics',
 'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows',
 'refinement_face_parent_ancestry_semantics','refinement_face_parent_face_row',
 'refinement_face_parent_barycentric','refinement_face_parent_barycentric_semantics',
 'parent_coarse_face_row','parent_source_face_marker'
}

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def identical(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
 return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def verify_pins(base_path,all_candidate_path,all_report_path,audit_path,pairs_path,replay_path):
 expected=((base_path,BASE_SHA,'baseline'),(all_candidate_path,ALL_CAND_SHA,'531 candidate'),
 (all_report_path,ALL_REPORT_SHA,'531 report'),(audit_path,AUDIT_SHA,'exact audit'),
 (pairs_path,PAIRS_SHA,'self-pair witness'),(replay_path,REPLAY_SHA,'405 replay owner'))
 for p,want,label in expected:
  if not Path(p).is_file():raise ValueError(f'{label} input is missing: {p}')
  got=sha_file(p)
  if got!=want:raise ValueError(f'{label} SHA mismatch: {got} != {want}')
 return {label:{'path':str(p),'sha256':sha_file(p)} for p,want,label in expected}

def validate_operation_manifest(base,full_candidate,full_report):
 operations=full_report.get('selected_flips')
 if not isinstance(operations,list) or len(operations)!=int(full_report.get('selected_flip_count',-1)):
  raise ValueError('selected flip manifest count mismatch')
 if not identical(base['points_world_m'],full_candidate['points_world_m']):raise ValueError('full candidate coordinates differ from baseline')
 seen_faces=set();seen_nodes=set()
 for index,op in enumerate(operations):
  rows=tuple(map(int,op.get('face_rows',())))
  if len(rows)!=2 or len(set(rows))!=2:raise ValueError(f'malformed operation face rows {index}')
  if min(rows)<0 or max(rows)>=len(base['source_O_faces']):raise ValueError(f'operation face row outside topology {index}')
  if set(rows)&seen_faces:raise ValueError(f'operations overlap face rows at {index}')
  old=np.asarray(op.get('old_faces'),dtype=np.int64);new=np.asarray(op.get('new_faces'),dtype=np.int64)
  if old.shape!=(2,3) or new.shape!=(2,3):raise ValueError(f'malformed operation triangles {index}')
  if not np.array_equal(base['source_O_faces'][list(rows)],old):raise ValueError(f'operation baseline faces mismatch {index}')
  if not np.array_equal(full_candidate['source_O_faces'][list(rows)],new):raise ValueError(f'operation emitted faces mismatch {index}')
  nodes=set(map(int,base['source_O_faces'][list(rows)].ravel()))
  if nodes&seen_nodes:raise ValueError(f'operations overlap quad nodes at {index}')
  seen_faces.update(rows);seen_nodes.update(nodes)
 changed=set(map(int,np.flatnonzero(np.any(base['source_O_faces']!=full_candidate['source_O_faces'],axis=1))))
 if changed!=seen_faces:raise ValueError('full candidate changed rows do not exactly match operation manifest')
 return {'operation_count':len(operations),'operation_face_rows':len(seen_faces),'operation_quad_nodes':len(seen_nodes)}

def select_remaining(operations,self_pair_faces):
 bad=set(map(int,self_pair_faces));kept=[];removed=[]
 seen_faces=set()
 for index,op in enumerate(operations):
  rows=tuple(map(int,op['face_rows']))
  if len(rows)!=2 or rows[0]==rows[1]:raise ValueError(f'malformed flip face_rows at op {index}')
  if set(rows)&seen_faces:raise ValueError(f'operations overlap face rows at op {index}')
  seen_faces.update(rows)
  implicated=sorted(set(rows)&bad)
  if implicated:removed.append({'operation_index':index,'face_rows':list(rows),'self_pair_face_rows':implicated})
  else:kept.append((index,op))
 return kept,removed

def apply_ops_batch(base,operations,replay):
 """405 apply_op semantics, one array copy and one CSR rebuild per family."""
 out={k:np.array(v,copy=True) for k,v in base.items()}
 n=len(out['source_O_faces'])
 src=replay.csr_rows(out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows'],n)
 sub=replay.csr_rows(out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows'],n)
 if 'refinement_face_parent_ancestry_indptr' in out:
  par=replay.csr_rows(out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows'],n)
 else:
  par=[[int(v)] if int(v)>=0 else [] for v in out['refinement_face_parent_face_row']]
 det=replay.detail_csr(out,{9570,9577,10226})
 seen_faces=set();seen_nodes=set()
 for op in operations:
  rows=tuple(map(int,op['face_rows']))
  if len(rows)!=2 or set(rows)&seen_faces:raise ValueError('batch operations must be face-disjoint')
  old=np.asarray(op['old_faces'],dtype=np.int64);new=np.asarray(op['new_faces'],dtype=np.int64)
  if old.shape!=(2,3) or new.shape!=(2,3):raise ValueError('malformed flip triangles')
  if not np.array_equal(out['source_O_faces'][list(rows)],old):raise ValueError(f'old face mismatch at {rows}')
  nodes=set(map(int,out['source_O_faces'][list(rows)].ravel()))
  if nodes&seen_nodes:raise ValueError('batch operations must be node-disjoint')
  seen_faces.update(rows);seen_nodes.update(nodes)
  out['source_O_faces'][rows[0]]=new[0];out['source_O_faces'][rows[1]]=new[1]
  out['area_reference_037_source_O_faces'][rows[0]]=new[0];out['area_reference_037_source_O_faces'][rows[1]]=new[1]
  i,j=rows
  src_union=sorted(set(src[i])|set(src[j]));sub_union=sorted(set(sub[i])|set(sub[j]))
  par_union=sorted(set(par[i])|set(par[j]));detail_union=sorted(set(det[i])|set(det[j]))
  for r in rows:src[r]=src_union;sub[r]=sub_union;par[r]=par_union;det[r]=detail_union
  out['refinement_face_parent_face_row'][list(rows)]=par_union[0] if len(par_union)==1 else -1
  if len(par_union)>1:out['refinement_face_parent_barycentric'][list(rows)]=np.nan
  out['parent_fine_detail_face_row'][list(rows)]=detail_union[0] if len(detail_union)==1 else -1
  out['parent_fine_subface_row'][list(rows)]=sub_union[0] if len(sub_union)==1 else -1
  for key in ('parent_coarse_face_row','parent_source_face_marker'):
   if key in out and int(out[key][i])!=int(out[key][j]):out[key][list(rows)]=-1
 out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows']=replay.csr_build(src,np.asarray(base['face_source_ancestry_face_rows']).dtype)
 out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows']=replay.csr_build(sub,np.asarray(base['face_fine_subface_ancestry_rows']).dtype)
 out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows']=replay.csr_build(par,np.int64)
 out['face_fine_detail_ancestry_indptr'],out['face_fine_detail_ancestry_rows']=replay.csr_build(det,np.int64)
 out['refinement_face_parent_ancestry_semantics']=np.asarray(['CSR union of immediate parent face rows; singular row -1 and singular barycentrics NaN when multiple parents apply'])
 out['face_fine_detail_ancestry_semantics']=np.asarray(['CSR union of source fine detail face rows; singular row -1 for cross-detail flips'])
 out['refinement_face_parent_barycentric_semantics']=np.asarray(['single-parent affine map only; NaN explicitly marks multi-parent patch flips'])
 return out

def topology(faces):
 from collections import defaultdict
 edges=defaultdict(int)
 for tri in np.asarray(faces,dtype=np.int64):
  if len(set(map(int,tri)))!=3:raise ValueError('degenerate triangle')
  for a,b in zip(tri,np.roll(tri,-1)):edges[(min(int(a),int(b)),max(int(a),int(b)))]+=1
 return {e for e,n in edges.items() if n==1},{e for e,n in edges.items() if n>2}

def build(base,all_report,self_pairs,replay):
 operations=all_report.get('selected_flips',[])
 pair_arr=np.asarray(self_pairs,dtype=np.int64)
 if pair_arr.ndim!=2 or pair_arr.shape[1]!=2:raise ValueError('self-pair table must have shape [N,2]')
 bad_faces=set(map(int,pair_arr.ravel()))
 kept,removed=select_remaining(operations,bad_faces)
 remaining=[op for _,op in kept]
 out=apply_ops_batch(base,remaining,replay)
 if not identical(out['points_world_m'],base['points_world_m']):raise ValueError('coordinates changed')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise ValueError('fixed-037 connectivity mismatch')
 affected={r for op in remaining for r in map(int,op['face_rows'])}
 changed_face_rows=np.flatnonzero(np.any(out['source_O_faces']!=base['source_O_faces'],axis=1))
 if not set(map(int,changed_face_rows))<=affected:raise ValueError('unexpected face row changed')
 for key in base:
  if key not in ALLOWED_CHANGED_KEYS and (key not in out or not identical(base[key],out[key])):raise ValueError(f'unexpected array change: {key}')
 if set(out)-set(base)-ALLOWED_CHANGED_KEYS:raise ValueError('unexpected candidate key added')
 boundary0,nonmanifold0=topology(base['source_O_faces']);boundary1,nonmanifold1=topology(out['source_O_faces'])
 if nonmanifold0 or nonmanifold1 or boundary0!=boundary1:raise ValueError('boundary/manifold invariant failed')
 if affected & bad_faces:raise ValueError('retained operation still touches a failed self-pair face')
 return out,{'schema':'numi.human.connected-quality-patch-534.v1',
  'status':'LOCAL_CONSTRUCTION_CANDIDATE_PENDING_FULL_AUDIT','geometry_admitted':False,
  'native_admission':False,'accepted_physical_steps':0,
  'exact_self_pair_count':int(len(pair_arr)),'unique_self_pair_face_count':len(bad_faces),
  'selected_operation_count':len(operations),'removed_operation_count':len(removed),
  'remaining_operation_count':len(remaining),'removed_operations':removed,
  'remaining_operation_indices':[int(i) for i,_ in kept],
  'changed_face_rows':list(map(int,changed_face_rows)),
  'coordinates_bitwise_unchanged':True,'matched_037_connectivity_synchronized':True,
  'boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,
  'remaining_selected_operations_do_not_touch_self_pair_faces':True,
  'exact_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT',
  'scope_limit':'Replayed remaining 531 flips only; does not assert self-pair clearance, layer transfer, volume, or admission.'}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--output-dir',required=True)
 ap.add_argument('--base',default=str(BASE));ap.add_argument('--full-candidate',default=str(ALL_CAND))
 ap.add_argument('--full-report',default=str(ALL_REPORT));ap.add_argument('--audit-summary',default=str(AUDIT))
 ap.add_argument('--self-pairs',default=str(PAIRS));ap.add_argument('--replay-owner',default=str(REPLAY))
 a=ap.parse_args();pins=verify_pins(a.base,a.full_candidate,a.full_report,a.audit_summary,a.self_pairs,a.replay_owner)
 with np.load(a.base,allow_pickle=False) as z:base={k:z[k] for k in z.files}
 with np.load(a.full_candidate,allow_pickle=False) as z:full={k:z[k] for k in z.files}
 report=json.loads(Path(a.full_report).read_text());audit=json.loads(Path(a.audit_summary).read_text())
 if report.get('candidate_output_sha256')!=sha_file(a.full_candidate):raise ValueError('531 report candidate output hash mismatch')
 if report.get('input_candidate_sha256')!=pins['baseline']['sha256']:raise ValueError('531 report baseline hash mismatch')
 if audit.get('candidate_sha256')!=pins['531 candidate']['sha256'] or audit.get('complete_surface_audit') is not True:raise ValueError('exact audit does not match complete 531 candidate')
 manifest=validate_operation_manifest(base,full,report)
 pairs=np.load(a.self_pairs,allow_pickle=False)
 replay=loadmod('cross_seam_search_405_for_534',a.replay_owner)
 out,meta=build(base,report,pairs,replay)
 od=Path(a.output_dir)
 if od.exists():raise FileExistsError(od)
 od.mkdir(parents=True)
 dst=od/'candidate.npz';np.savez_compressed(dst,**out)
 meta.update({'candidate_path':str(dst),'candidate_sha256':sha_file(dst),'input_pins':pins,'operation_manifest_validation':manifest})
 rp=od/'report.json';rp.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report_path':str(rp),'report_sha256':sha_file(rp),**meta},indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()

