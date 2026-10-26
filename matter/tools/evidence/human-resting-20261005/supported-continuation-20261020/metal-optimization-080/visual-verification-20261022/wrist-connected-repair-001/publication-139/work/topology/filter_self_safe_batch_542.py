from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
BASE=D/'connected-quality-patch-538/run-001/candidate.npz'
FULL=D/'focused-clearance-541/topology/run-002-full/candidate.npz'
FULL_REPORT=D/'focused-clearance-541/topology/run-002-full/report.json'
AUDIT=D/'focused-clearance-541/topology/check-002-full/audit-130/summary.json'
PAIRS=D/'focused-clearance-541/topology/check-002-full/audit-130/surface-self-pairs.npy'
OUTDIR=D/'focused-clearance-541/topology/run-003-filtered'
REPLAY=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
OWNER=D/'connected-quality-patch-534/replay_surviving_flips_534.py'
PINS={
 'base_sha256':'2f52328e1105dad576717eae872391324a1172ceec3f74f23fae591891078c6e',
 'full_candidate_sha256':'8154d8a1215970deba3e08a772feca133d21ff9de142136b7b1958cd75199eca',
 'full_report_sha256':'64b9b4ec0456d1179c692c4dcf130831929ee61cbd222b1e66033787d37f8847',
 'self_audit_sha256':'3103c1cd460639f019d8352a3025ac70521026494bdd1917e93ddb9c785147b3',
 'self_pairs_sha256':'bdb6c42843179110311afa8072dd439f46fb97bc40d8edad4f82908da0e451a0',
 'replay405_sha256':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'replay534_sha256':'4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504',
 'builder541_sha256':'26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa',
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
 if sha(D/'focused-clearance-541/topology/repair_focused_clearance_541.py')!=PINS['builder541_sha256']:raise ValueError('541 source pin changed')
 if full_report.get('input_candidate_sha256')!=PINS['base_sha256'] or full_report.get('candidate_output_sha256')!=PINS['full_candidate_sha256']:raise ValueError('541 operation report is not bound to 538/full candidate')
 if audit.get('candidate_sha256')!=PINS['full_candidate_sha256'] or audit.get('complete_surface_audit') is not True:raise ValueError('self audit is incomplete or wrong candidate')
 pair_arr=np.asarray(pairs,dtype=np.int64)
 if pair_arr.ndim!=2 or pair_arr.shape[1]!=2 or pair_arr.shape[0]!=72:raise ValueError('expected the pinned 72-row exact self-pair witness')
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
 focus,_=mod541.focused_crossing_rows(json.loads((D/'connected-quality-patch-538/root-exact-run001/audit-159/report.json').read_text()),len(base['source_O_faces']))
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
 return out,{'schema':'numi.human.focused-clearance-filtered-batch-542.v1','status':'FILTERED_LOCAL_CONSTRUCTION_CANDIDATE_PENDING_ROOT_EXACT_AUDIT','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'input_pins':PINS,'operation_manifest_validation':manifest,'full_batch_selected_operations':len(full_report['selected_flips']),'exact_self_pair_count_in_full_candidate':int(len(pair_arr)),'unique_self_pair_face_count':len(bad_faces),'removed_operation_count':len(removed),'removed_operations':removed,'retained_operation_count':len(remaining),'retained_operation_indices':[int(i) for i,_ in kept],'changed_face_rows':sorted(changed),'changed_candidate_arrays':changed_keys,'added_candidate_arrays':added,'coordinates_bitwise_unchanged_from_538':True,'typed_anchor_count':int(anchors.size),'typed_anchor_coordinates_bitwise_unchanged':True,'all_self_witness_faces_excluded_from_retained_operations':True,'boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,'fixed037_connectivity_synchronized':True,'outside_surviving_operation_rows_bitwise_unchanged':True,'outside_surviving_operation_ancestry_unchanged':True,'area_and_edge_gates':'NOT_RECOMPUTED; requires root arithmetic verification','exact_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT','scope_limit':'Removed every 541 operation touching any of the 72 exact self-pair faces, replayed remaining operations on 538 coordinates and ancestry. Full exact, area/edge and 185-bank checks remain required.'}
def main():
 t=time.perf_counter()
 for p in (BASE,FULL,FULL_REPORT,AUDIT,PAIRS,REPLAY,OWNER):
  if not p.is_file():raise FileNotFoundError(p)
 base=np.load(BASE,allow_pickle=False);base={k:base[k] for k in base.files}
 fullz=np.load(FULL,allow_pickle=False);full={k:fullz[k] for k in fullz.files}
 report=json.loads(FULL_REPORT.read_text());audit=json.loads(AUDIT.read_text());pairs=np.load(PAIRS,allow_pickle=False)
 mod534=load('replay_534_for_542',OWNER);replay=load('cross_replay_405_for_542',REPLAY)
 mod541=load('builder_541_for_542',D/'focused-clearance-541/topology/repair_focused_clearance_541.py')
 out,meta=filter_batch(base,full,report,audit,pairs,replay,mod534,mod541)
 if OUTDIR.exists():raise FileExistsError(OUTDIR)
 OUTDIR.mkdir(parents=True);cp=OUTDIR/'candidate.npz';np.savez_compressed(cp,**out)
 meta.update({'base_candidate_path':str(BASE),'full_batch_candidate_path':str(FULL),'full_batch_report_path':str(FULL_REPORT),'self_audit_summary_path':str(AUDIT),'self_pair_witness_path':str(PAIRS),'candidate_path':str(cp),'candidate_sha256':sha(cp),'filter_source_path':str(Path(__file__)),'filter_source_sha256':sha(__file__),'generation_elapsed_seconds':time.perf_counter()-t})
 rp=OUTDIR/'report.json';rp.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report_path':str(rp),'report_sha256':sha(rp),**meta},indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()

