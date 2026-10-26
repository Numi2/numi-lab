from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
BASE=D/'focused-clearance-541/topology/run-005-filtered/candidate.npz'
FULL=D/'focused-clearance-541/topology/run-006-followup/candidate.npz'
FULL_REPORT=D/'focused-clearance-541/topology/run-006-followup/report.json'
AUDIT=D/'focused-clearance-541/topology/check-006-followup/audit-130/summary.json'
PAIRS=D/'focused-clearance-541/topology/check-006-followup/audit-130/surface-self-pairs.npy'
OUTDIR=D/'focused-clearance-541/topology/run-007-filtered'
REPLAY=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
OWNER=D/'connected-quality-patch-534/replay_surviving_flips_534.py'
PINS={
 'base_sha256':'bd0b103a6c69697e935bf4ab74b2825dec61ccda8487526026337c3456d22c35',
 'full_candidate_sha256':'27611365f7816560a883d0ec7ab15ce4c39c606389228001da9f06306d20a87f',
 'full_report_sha256':'6ed2925fd7c184efd1c94157cdc233b117ec3a2a48a9792c288100baa0b6be03',
 'self_audit_sha256':'7999cf8ab8658cd8da7dcd453c3e7421292c4cce6700e3080cf48658d56346c9',
 'self_pairs_sha256':'9fe8ea0ea319dae7db91b3e5c787b7f9973ae3ca4023e67271ec79c3b791a23f',
 'base_surface_report_sha256':'6a536b9a869f5909dbc521af9fbb8b5065344b24e2972c410f17c9d0bbd738fd',
 'replay405_sha256':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'replay534_sha256':'4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504',
 'builder_followup_sha256':'88536ee1e1297b7249c3f60e02f143937883934d602e10c8e8ddd4a104adb626',
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def same(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
 return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
def filter_batch(base,full,full_report,audit,pairs,replay,mod534,mod541):
 if sha(BASE)!=PINS['base_sha256'] or sha(FULL)!=PINS['full_candidate_sha256'] or sha(FULL_REPORT)!=PINS['full_report_sha256']:raise ValueError('base/full 541 candidate pins changed')
 if sha(AUDIT)!=PINS['self_audit_sha256'] or sha(PAIRS)!=PINS['self_pairs_sha256']:raise ValueError('self-audit witness pins changed')
 if sha(REPLAY)!=PINS['replay405_sha256'] or sha(OWNER)!=PINS['replay534_sha256']:raise ValueError('replay owner pins changed')
 if sha(D/'focused-clearance-541/topology/repair_focused_clearance_followup.py')!=PINS['builder_followup_sha256']:raise ValueError('follow-up builder source pin changed')
 if full_report.get('input_candidate_sha256')!=PINS['base_sha256'] or full_report.get('candidate_output_sha256')!=PINS['full_candidate_sha256']:raise ValueError('follow-up operation report is not bound to run005/run006 candidate')
 if audit.get('candidate_sha256')!=PINS['full_candidate_sha256'] or audit.get('complete_surface_audit') is not True:raise ValueError('self audit is incomplete or wrong candidate')
 pair_arr=np.asarray(pairs,dtype=np.int64)
 if pair_arr.ndim!=2 or pair_arr.shape[1]!=2 or pair_arr.shape[0]!=109:raise ValueError('expected the pinned 109-row exact self-pair witness')
 manifest=mod534.validate_operation_manifest(base,full,full_report)
 bad_faces=set(map(int,pair_arr.ravel()))
 kept,removed=mod534.select_remaining(full_report['selected_flips'],bad_faces)
 remaining=[op for _,op in kept]
 out=mod534.apply_ops_batch(base,remaining,replay)
 if not same(out['points_world_m'],base['points_world_m']):raise ValueError('coordinates changed from 538 base')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise ValueError('037/current face arrays diverged')
 changed=set(map(int,np.flatnonzero(np.any(out['source_O_faces']!=base['source_O_faces'],axis=1))))
 op_rows={int(r) for op in remaining for r in map(int,op['face_rows'])}
 if changed!=op_rows:raise ValueError('remaining operation rows do not match output changes')
 if changed&bad_faces:raise ValueError('retained operation still touches any exact self-pair face')
 base_surface_path=D/'focused-clearance-541/topology/check-005-filtered/audit-159/report.json'
 if sha(base_surface_path)!=PINS['base_surface_report_sha256']:raise ValueError('base exact surface report pin changed')
 base_surface_report=json.loads(base_surface_path.read_text())
 if base_surface_report.get('candidate_sha256')!=PINS['base_sha256'] or base_surface_report.get('complete') is not True or base_surface_report.get('bank_count')!=185:raise ValueError('base exact surface report does not certify the run003 parent')
 focus,_=mod541.focused_crossing_rows(base_surface_report,len(base['source_O_faces']))
 scope=mod541.edge_one_ring(base['source_O_faces'],focus)
 if not changed<=scope:raise ValueError('filtered changes escaped declared 541 scope')
 _,b0,n0=mod541.topology(base['source_O_faces']);_,b1,n1=mod541.topology(out['source_O_faces'])
 if n0 or n1 or b0!=b1:raise ValueError('boundary/manifold invariant failed')
 anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 if anchors.size!=237 or not same(base['points_world_m'][anchors],out['points_world_m'][anchors]):raise ValueError('237 anchor coordinates changed')
 allowed=mod534.ALLOWED_CHANGED_KEYS
 changed_keys=[k for k in base if not same(base[k],out[k])]
 added=sorted(set(out)-set(base))
 if set(changed_keys)-allowed or set(added)-allowed:raise ValueError('undeclared candidate arrays changed')
 outside=np.ones(len(base['source_O_faces']),dtype=bool);outside[list(changed)]=False
 for k,a in base.items():
  b=out[k];aa=np.asarray(a);bb=np.asarray(b)
  if aa.ndim and aa.shape[0]==len(out['source_O_faces']) and bb.shape==aa.shape and not np.array_equal(aa[outside],bb[outside],equal_nan=True):raise ValueError(f'face row changed outside surviving ops: {k}')
 for ptr,val in [('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows'),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows')]:
  if ptr not in base:continue
  for i in np.flatnonzero(outside):
   a0,a1=map(int,base[ptr][i:i+2]);b0,b1=map(int,out[ptr][i:i+2])
   if not np.array_equal(base[val][a0:a1],out[val][b0:b1]):raise ValueError(f'ancestry row changed outside operations: {i}')
 return out,{'schema':'numi.human.focused-clearance-filtered-followup-544.v1','status':'FILTERED_LOCAL_CONSTRUCTION_CANDIDATE_PENDING_ROOT_EXACT_AUDIT','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'input_pins':PINS,'operation_manifest_validation':manifest,'full_batch_selected_operations':len(full_report['selected_flips']),'exact_self_pair_count_in_full_candidate':int(len(pair_arr)),'unique_self_pair_face_count':len(bad_faces),'removed_operation_count':len(removed),'removed_operations':removed,'retained_operation_count':len(remaining),'retained_operation_indices':[int(i) for i,_ in kept],'changed_face_rows':sorted(changed),'changed_candidate_arrays':changed_keys,'added_candidate_arrays':added,'coordinates_bitwise_unchanged_from_538':True,'typed_anchor_count':int(anchors.size),'typed_anchor_coordinates_bitwise_unchanged':True,'all_self_witness_faces_excluded_from_retained_operations':True,'boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,'fixed037_connectivity_synchronized':True,'outside_surviving_operation_rows_bitwise_unchanged':True,'outside_surviving_operation_ancestry_unchanged':True,'area_and_edge_gates':'NOT_RECOMPUTED; requires root arithmetic verification','exact_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT','scope_limit':'Removed every follow-up operation touching any exact self-pair face, replayed remaining operations on the run005 parent coordinates and ancestry. Full exact, area/edge and 185-bank checks remain required.'}
def main():
 t=time.perf_counter()
 for p in (BASE,FULL,FULL_REPORT,AUDIT,PAIRS,REPLAY,OWNER):
  if not p.is_file():raise FileNotFoundError(p)
 base=np.load(BASE,allow_pickle=False);base={k:base[k] for k in base.files}
 fullz=np.load(FULL,allow_pickle=False);full={k:fullz[k] for k in fullz.files}
 report=json.loads(FULL_REPORT.read_text());audit=json.loads(AUDIT.read_text());pairs=np.load(PAIRS,allow_pickle=False)
 mod534=load('replay_534_for_542',OWNER);replay=load('cross_replay_405_for_542',REPLAY)
 mod541=load('builder_followup_for_543',D/'focused-clearance-541/topology/repair_focused_clearance_followup.py')
 out,meta=filter_batch(base,full,report,audit,pairs,replay,mod534,mod541)
 if OUTDIR.exists():raise FileExistsError(OUTDIR)
 OUTDIR.mkdir(parents=True);cp=OUTDIR/'candidate.npz';np.savez_compressed(cp,**out)
 meta.update({'base_candidate_path':str(BASE),'full_batch_candidate_path':str(FULL),'full_batch_report_path':str(FULL_REPORT),'self_audit_summary_path':str(AUDIT),'self_pair_witness_path':str(PAIRS),'candidate_path':str(cp),'candidate_sha256':sha(cp),'filter_source_path':str(Path(__file__)),'filter_source_sha256':sha(__file__),'generation_elapsed_seconds':time.perf_counter()-t})
 rp=OUTDIR/'report.json';rp.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report_path':str(rp),'report_sha256':sha(rp),**meta},indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()

