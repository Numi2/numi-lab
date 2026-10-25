#!/usr/bin/env python3
"""Reclip only 085c-003 failed coverage incidence, with pair-local origins."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
BASE = ROOT / 'native-parent-transfer-085c'
RUN003 = BASE / 'run-003'
OUT = BASE / 'local-coverage-precision-004-fix001'
CAND = ROOT / 'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
HUMAN = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
EXPORT = HUMAN / 'connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight'
NODES = EXPORT / 'compiled-skin-fem-nodes.csv'
TETS = EXPORT / 'compiled-skin-fem-tets.csv'
STACKED = ROOT / 'native-parent-transfer-085b/plane_filter_085b.py'
EXACT_SRC = BASE / 'build_native_parent_transfer_085c-003.py'
EXPECTED = {
 CAND:'5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
 NODES:'26c607f9835241ef5d274169e4ac2aeae9c0f6ab7ad94a06b307ca25844862c9',
 TETS:'a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461',
 STACKED:'ae3cbe1e9f79a580b3a0f22b5126a2d91f11293874f9c867f46a83e81d6f3f61',
 EXACT_SRC:'7bdb7756458566176fa4be03f9b6d8e3de13647d746a9207619dc8cafce4dcab',
}
AABB_TOL=2e-10
INSIDE_TOL=2e-10
UNIQUE_TOL=2e-10

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def csvrows(p):
 with Path(p).open(newline='') as f: return list(csv.DictReader(f))

def vol6(x): return abs(float(np.linalg.det(x[1:]-x[0])))/6.

def main():
 if OUT.exists(): raise RuntimeError(f'refusing existing output directory: {OUT}')
 for p,h in EXPECTED.items():
  a=sha(p)
  if a!=h: raise RuntimeError(f'input pin mismatch {p}: {a} != {h}')
 # Reuse the exact-rank classifier frozen in the 003 source and the unchanged 085b clip.
 spec=importlib.util.spec_from_file_location('build003',EXACT_SRC)
 build=importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
 sys.path.insert(0,str(STACKED.parent))
 from plane_filter_085b import tetra_intersection_volume_stacked

 with np.load(CAND,allow_pickle=False) as z:
  points=np.asarray(z['reference_points_m'],dtype=np.float64)
  new_t=np.asarray(z['tetrahedra_zero_based'],dtype=np.int64)
  new_mat=np.asarray(z['material_ids'],dtype=np.int64)
  cell_uids=np.asarray(z['cell_uids'],dtype=np.int64)
 with np.load(RUN003/'geometric-overlap-085c.npz',allow_pickle=False) as z:
  base={k:np.asarray(z[k]).copy() for k in z.files}
 node_rows=csvrows(NODES)
 tet_rows=[r for r in csvrows(TETS) if int(r['identity_active_bit'])==1]
 native_rest=np.asarray([[float(r[f'rest_{a}']) for a in 'xyz'] for r in node_rows],dtype=np.float64)
 global_ids=np.asarray([int(r['global_node']) for r in node_rows],dtype=np.int64)
 global_to_local={int(g):i for i,g in enumerate(global_ids)}
 old_global=np.asarray([[int(r[f'global_node{j}']) for j in range(4)] for r in tet_rows],dtype=np.int64)
 old_t=np.asarray([[global_to_local[int(g)] for g in row] for row in old_global],dtype=np.int64)
 old_mat=np.asarray([int(r['material_index']) for r in tet_rows],dtype=np.int64)
 if len(new_t)!=32796 or len(old_t)!=28632: raise RuntimeError('frozen mesh dimensions changed')
 new_xyz=points[new_t]
 old_xyz=native_rest[old_t]
 new_vol=np.asarray([vol6(x) for x in new_xyz])
 old_vol=np.asarray([vol6(x) for x in old_xyz])
 if np.any(new_vol<=0) or np.any(old_vol<=0): raise RuntimeError('nonpositive input tet volume')
 new_lo,new_hi=new_xyz.min(axis=1),new_xyz.max(axis=1)
 old_lo,old_hi=old_xyz.min(axis=1),old_xyz.max(axis=1)
 bad_children=np.flatnonzero(np.abs(base['candidate_child_reference_volume_m3']-base['child_overlap_coverage_m3'])/base['candidate_child_reference_volume_m3']>1e-7)
 bad_parents=np.flatnonzero(np.abs(base['native_parent_geometry_volume_m3']-base['native_parent_overlap_coverage_m3'])/base['native_parent_geometry_volume_m3']>1e-7)
 expected_children=np.asarray([1561,4451,5816,9448,25669,30856],dtype=np.int64)
 expected_parents=np.asarray([4786,5175,19109],dtype=np.int64)
 if not np.array_equal(bad_children,expected_children) or not np.array_equal(bad_parents,expected_parents):
  raise RuntimeError(f'failed-row inventory changed children={bad_children.tolist()} parents={bad_parents.tolist()}')
 pairs=set()
 for ci in bad_children:
  ids=np.flatnonzero(np.all(old_lo<=new_hi[ci]+AABB_TOL,axis=1)&np.all(old_hi>=new_lo[ci]-AABB_TOL,axis=1))
  pairs.update((int(ci),int(oi)) for oi in ids)
 for oi in bad_parents:
  ids=np.flatnonzero(np.all(new_lo<=old_hi[oi]+AABB_TOL,axis=1)&np.all(new_hi>=old_lo[oi]-AABB_TOL,axis=1))
  pairs.update((int(ci),int(oi)) for ci in ids)
 pair_rows=sorted(pairs)
 if len(pair_rows)!=1025: raise RuntimeError(f'incidence pair count changed: {len(pair_rows)}')
 # Materialize the failed CSR as an immutable map for old pair values.
 ptr=base['child_native_parent_indptr']; pids=base['child_native_parent_rows']; ovs=base['overlap_native_reference_volume_m3']
 old_pair={(ci,int(pids[k])):float(ovs[k]) for ci in range(len(new_t)) for k in range(int(ptr[ci]),int(ptr[ci+1]))}
 if len(old_pair)!=len(ovs): raise RuntimeError('duplicate key in source 003 CSR')
 t0=time.monotonic()
 local_results=[]
 for ci,oi in pair_rows:
  origin=new_xyz[ci,0].copy()
  shifted_child=np.asarray(new_xyz[ci]-origin,dtype=np.float64)
  shifted_parent=np.asarray(old_xyz[oi]-origin,dtype=np.float64)
  raw=float(tetra_intersection_volume_stacked(shifted_child,shifted_parent,INSIDE_TOL,UNIQUE_TOL))
  cutoff=max(min(float(new_vol[ci]),float(old_vol[oi]))*1e-13,1e-25)
  effective=raw if raw>cutoff else 0.0
  exact=None
  target_mat={1:15,2:16}[int(new_mat[ci])]
  if effective and int(old_mat[oi])!=target_mat:
   exact=build.classify_exact_cross_material_pair(new_xyz[ci],old_xyz[oi])
   if exact['affine_rank']<3: effective=0.0
  prior=old_pair.get((ci,oi),0.0)
  local_results.append((ci,oi,prior,raw,effective,effective-prior,cutoff,exact,origin))
 elapsed=time.monotonic()-t0
 # Trial report is written before global gate and before any mass/moment work.
 OUT.mkdir(parents=True)
 np.savez_compressed(OUT/'incident-pair-reclip-004.npz',
  candidate_child_rows=np.asarray([x[0] for x in local_results],np.int32),
  native_parent_rows=np.asarray([x[1] for x in local_results],np.int32),
  old_overlap_m3=np.asarray([x[2] for x in local_results],np.float64),
  translated_raw_overlap_m3=np.asarray([x[3] for x in local_results],np.float64),
  translated_effective_overlap_m3=np.asarray([x[4] for x in local_results],np.float64),
  delta_overlap_m3=np.asarray([x[5] for x in local_results],np.float64),
  positive_cutoff_m3=np.asarray([x[6] for x in local_results],np.float64),
  exact_cross_material_rank=np.asarray([-99 if x[7] is None else int(x[7]['affine_rank']) for x in local_results],np.int8),
  local_origin_candidate_vertex0_m=np.asarray([x[8] for x in local_results],np.float64),
  candidate_sha256=np.asarray(EXPECTED[CAND]),
  baseline_csr_sha256=np.asarray(sha(RUN003/'geometric-overlap-085c.npz')))

 # Copy all old CSR rows, replacing only the 1,025 incident pair values.
 updated={ci:{int(pids[k]):float(ovs[k]) for k in range(int(ptr[ci]),int(ptr[ci+1]))} for ci in range(len(new_t))}
 for ci,oi,prior,raw,effective,delta,cutoff,exact,origin in local_results:
  if effective>0.0: updated[ci][oi]=effective
  else: updated[ci].pop(oi,None)
 new_ptr=np.zeros(len(new_t)+1,dtype=np.int64)
 new_parents=[]; new_overlaps=[]
 for ci in range(len(new_t)):
  row=updated[ci]
  for oi,ov in sorted(row.items()): new_parents.append(oi); new_overlaps.append(ov)
  new_ptr[ci+1]=len(new_parents)
 new_parents=np.asarray(new_parents,dtype=np.int32); new_overlaps=np.asarray(new_overlaps,dtype=np.float64)
 child_coverage=np.bincount(np.repeat(np.arange(len(new_t),dtype=np.int64),np.diff(new_ptr)),weights=new_overlaps,minlength=len(new_t))
 parent_coverage=np.bincount(new_parents,weights=new_overlaps,minlength=len(old_t))
 child_rel=(child_coverage-new_vol)/new_vol
 parent_rel=(parent_coverage-old_vol)/old_vol
 child_bad=np.flatnonzero(np.abs(child_rel)>1e-7)
 parent_bad=np.flatnonzero(np.abs(parent_rel)>1e-7)
 changed=[x for x in local_results if x[5]!=0.0]
 largest=sorted(changed,key=lambda x:abs(x[5]),reverse=True)[:20]
 report={
  'status':'LOCAL_INCIDENCE_RECLIP_COMPLETE',
  'candidate_sha256':EXPECTED[CAND],
  'baseline_003_csr_sha256':sha(RUN003/'geometric-overlap-085c.npz'),
  'clip_source_sha256':EXPECTED[STACKED],
  'exact_rank_classifier_source_sha256':EXPECTED[EXACT_SRC],
  'native_node_csv_sha256':EXPECTED[NODES], 'native_tet_csv_sha256':EXPECTED[TETS],
  'clip_method':'same 085b stacked clip, unchanged inside/unique/AABB tolerances; subtract candidate child vertex 0 from both pair tetrahedra before clip; no separator culling, inverse, rescale, or geometry changes',
  'tolerances_m':{'aabb':AABB_TOL,'inside':INSIDE_TOL,'unique':UNIQUE_TOL},
  'incident_scope':{'failed_children':bad_children.tolist(),'failed_native_parents':bad_parents.tolist(),'unique_aabb_pairs':len(pair_rows),'candidate_bad_child_pairs':[int(sum(1 for x in pair_rows if x[0]==ci)) for ci in bad_children],'native_bad_parent_pairs':[int(sum(1 for x in pair_rows if x[1]==oi)) for oi in bad_parents]},
  'clip_elapsed_seconds':elapsed,
  'pair_delta_summary':{'changed_pair_count':len(changed),'old_positive_pair_count':int(sum(x[2]>0 for x in local_results)),'new_positive_raw_count':int(sum(x[3]>0 for x in local_results)),'new_positive_effective_count':int(sum(x[4]>0 for x in local_results)),'max_abs_delta_m3':float(max((abs(x[5]) for x in changed),default=0.0)),'largest_20':[{'child':int(x[0]),'parent':int(x[1]),'old_m3':x[2],'translated_raw_m3':x[3],'translated_effective_m3':x[4],'delta_m3':x[5],'cross_material_exact':x[7]} for x in largest]},
  'cross_material_exact_rank_counts':{str(k):int(sum(x[7] is not None and x[7]['affine_rank']==k for x in local_results)) for k in (-1,0,1,2,3)},
  'global_coverage':{'child_max_abs_relative_error':float(np.max(np.abs(child_rel))),'parent_max_abs_relative_error':float(np.max(np.abs(parent_rel))),'child_failed_count_over_1e-7':int(len(child_bad)),'parent_failed_count_over_1e-7':int(len(parent_bad)),'failed_children':child_bad.tolist(),'failed_parents':parent_bad.tolist(),'child_worst_rows':[{'row':int(i),'relative_error':float(child_rel[i])} for i in np.argsort(np.abs(child_rel))[-12:][::-1]],'parent_worst_rows':[{'row':int(i),'relative_error':float(parent_rel[i])} for i in np.argsort(np.abs(parent_rel))[-12:][::-1]]},
  'gate':'PASS' if len(child_bad)==0 and len(parent_bad)==0 else 'FAIL_COVERAGE_REMAINS',
  'mass_and_moment_audit':'not_run_unless_both_global_coverage_gates_pass',
 }
 np.savez_compressed(OUT/'geometric-overlap-local-precision-004.npz',
  candidate_cell_uids=cell_uids,
  candidate_child_material_ids=new_mat.astype(np.int8),
  candidate_child_reference_volume_m3=new_vol,
  child_native_parent_indptr=new_ptr,
  child_native_parent_rows=new_parents,
  overlap_native_reference_volume_m3=new_overlaps,
  child_overlap_coverage_m3=child_coverage,
  child_overlap_relative_error=child_rel,
  native_parent_material_ids=old_mat.astype(np.int16),
  native_parent_geometry_volume_m3=old_vol,
  native_parent_overlap_coverage_m3=parent_coverage,
  native_parent_overlap_relative_error=parent_rel,
  candidate_sha256=np.asarray(EXPECTED[CAND]),
  native_tet_csv_sha256=np.asarray(EXPECTED[TETS]),
  baseline_003_csr_sha256=np.asarray(sha(RUN003/'geometric-overlap-085c.npz')),
  locally_reclipped_pair_count=np.asarray(len(pair_rows),np.int32))
 (OUT/'coverage-report-004.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps(report,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
