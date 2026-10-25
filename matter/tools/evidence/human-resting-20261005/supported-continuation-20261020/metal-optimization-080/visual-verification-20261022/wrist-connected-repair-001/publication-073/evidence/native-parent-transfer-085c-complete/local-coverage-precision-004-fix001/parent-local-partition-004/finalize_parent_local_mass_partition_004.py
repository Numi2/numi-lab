#!/usr/bin/env python3
"""Build mass/source-provenance map from the passing local-precision CSR only."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,importlib.util
import numpy as np
ROOT=Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
BASE=ROOT/'native-parent-transfer-085c'
OUT=BASE/'local-coverage-precision-004-fix001'
CAND=ROOT/'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
HUMAN=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
EXPORT=HUMAN/'connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight'
NODES=EXPORT/'compiled-skin-fem-nodes.csv'; TETS=EXPORT/'compiled-skin-fem-tets.csv'; CONTRIB=EXPORT/'premerge-layer-contributions.csv'
PRECISION=OUT/'geometric-overlap-local-precision-004.npz'; PRECISION_REPORT=OUT/'coverage-report-004.json'
PIN={'candidate':('5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',CAND),'nodes':('26c607f9835241ef5d274169e4ac2aeae9c0f6ab7ad94a06b307ca25844862c9',NODES),'tets':('a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461',TETS),'contributions':('75bd67cc6f7aeab9dccbb73da29220fcd35896b217042e8ce9c8181d89ca7c6f',CONTRIB),'precision_csr':('4e3491fab1e4e5c24433eb2106370538231be8892b728c7c973ba72b9eb5c715',PRECISION)}
PARTITION_OUT=BASE/'local-coverage-precision-004-fix001'/'parent-local-partition-004'
MAP=PARTITION_OUT/'native-parent-transfer-085c-parent-local-partition.npz'; REPORT=PARTITION_OUT/'transfer-report-parent-local-partition-004.json'; MANIFEST=PARTITION_OUT/'transfer-manifest-parent-local-partition-004.json'
HELPER_PATH=Path('/Users/n/Work/numi-human-connected-surface-069/publication-073/canonical/matter/tools/human_skin_fat_mass_transfer_geometry.py')
_spec=importlib.util.spec_from_file_location('mass_transfer_geometry_partition',HELPER_PATH)
_geometry=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_geometry)
MATERIAL_MAP={1:15,2:16}; DENSITY_BOUNDS={15:(900.,1100.),16:(850.,950.)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def vol6(x):return abs(float(np.linalg.det(x[1:]-x[0])))/6.
def main():
 for name,(want,p) in PIN.items():
  got=sha(p)
  if got!=want:raise RuntimeError(f'{name} pin mismatch {got} != {want}')
 if MAP.exists() or REPORT.exists() or MANIFEST.exists(): raise RuntimeError('refusing to overwrite existing finalizer outputs')
 cand=np.load(CAND,allow_pickle=False)
 points=np.asarray(cand['reference_points_m'],np.float64); new_t=np.asarray(cand['tetrahedra_zero_based'],np.int64); new_mat=np.asarray(cand['material_ids'],np.int64); cell_uids=np.asarray(cand['cell_uids'],np.int64)
 data=np.load(PRECISION,allow_pickle=False)
 assert np.max(np.abs(data['child_overlap_relative_error']))<=1e-7
 assert np.max(np.abs(data['native_parent_overlap_relative_error']))<=1e-7
 node=rows(NODES); tet=[r for r in rows(TETS) if int(r['identity_active_bit'])==1]; contrib=rows(CONTRIB)
 g=np.asarray([int(r['global_node']) for r in node],np.int64); g2l={int(v):i for i,v in enumerate(g)}
 native_rest=np.asarray([[float(r[f'rest_{a}']) for a in 'xyz'] for r in node],np.float64)
 assert np.array_equal(points[:len(node)],native_rest)
 old_t=np.asarray([[g2l[int(r[f'global_node{j}'])] for j in range(4)] for r in tet],np.int64)
 old_mat=np.asarray([int(r['material_index']) for r in tet],np.int64)
 old_source=np.asarray([int(r['source_tet_row']) for r in tet],np.int64)
 old_compiled_vol=np.asarray([float(r['inverse_rest_w_m3']) for r in tet],np.float64)
 old_density=np.asarray([float(r['compiled_bulk_x']) for r in tet],np.float64)
 old_corner_mass=np.asarray([float(r['bulk_density_corner_mass_kg']) for r in tet],np.float64)
 old_geom_vol=np.asarray([vol6(native_rest[r]) for r in old_t],np.float64)
 child_geom_vol=np.asarray([vol6(points[r]) for r in new_t],np.float64)
 old_cell_mass=old_compiled_vol*old_density
 if not np.array_equal(old_corner_mass,old_cell_mass/4.):raise RuntimeError('native corner mass != compiled parent mass/4')
 if not np.array_equal(old_source,np.arange(len(tet))):raise RuntimeError('source tet row identity changed')
 ptr=np.asarray(data['child_native_parent_indptr'],np.int64); parent=np.asarray(data['child_native_parent_rows'],np.int32); overlap=np.asarray(data['overlap_native_reference_volume_m3'],np.float64)
 new_mat_native=np.asarray([MATERIAL_MAP[int(x)] for x in new_mat],np.int16)
 partition=_geometry.build_parent_local_mass_partition(ptr,parent,overlap,child_geom_vol,old_geom_vol,new_mat_native,old_mat,coverage_tolerance=1e-7)
 parent_mass_fraction=partition['parent_local_mass_fraction_overlap_over_covered_volume']
 # Local precision CSR already had exact rank<3 opposite-material contacts omitted; recheck typed material domains.
 bad_cross=[]
 raw_parent_mass_fraction=overlap/old_geom_vol[parent]
 child_fraction=overlap/np.repeat(child_geom_vol,np.diff(ptr))
 for ci in range(len(new_t)):
  for k in range(int(ptr[ci]),int(ptr[ci+1])):
   oi=int(parent[k])
   if old_mat[oi]!=new_mat_native[ci]:bad_cross.append((ci,oi,float(overlap[k]),int(old_mat[oi]),int(new_mat_native[ci])))
 if bad_cross:raise RuntimeError(f'cross-material overlap remains after exact contact policy: {bad_cross[:10]}')
 child_mass=np.zeros(len(new_t),np.float64)
 for ci in range(len(new_t)):
  a,z=int(ptr[ci]),int(ptr[ci+1]); child_mass[ci]=np.dot(old_cell_mass[parent[a:z]],parent_mass_fraction[a:z])
 child_density=child_mass/child_geom_vol
 density_summary={}; density_bad=[]
 for m,(lo,hi) in DENSITY_BOUNDS.items():
  ids=np.flatnonzero(new_mat_native==m); bad=ids[(child_density[ids]<lo)|(child_density[ids]>hi)]
  density_bad.extend((int(i),int(m),float(child_density[i])) for i in bad)
  density_summary[str(m)]={'child_rows':int(len(ids)),'density_min_kg_m3':float(child_density[ids].min()),'density_max_kg_m3':float(child_density[ids].max()),'bounds_kg_m3':[lo,hi]}
 parent_received=np.zeros(len(tet),np.float64); np.add.at(parent_received,parent,old_cell_mass[parent]*parent_mass_fraction)
 parent_mass_rel=(parent_received-old_cell_mass)/old_cell_mass
 if density_bad: raise RuntimeError(f'density bound failure count={len(density_bad)} examples={density_bad[:10]}')
 if np.max(np.abs(parent_mass_rel))>1e-7: raise RuntimeError(f'parent mass closure fail {np.argmax(np.abs(parent_mass_rel))} {np.max(np.abs(parent_mass_rel))}')
 mat_mass={}
 for m in (15,16):
  oi=old_mat==m; ci=new_mat_native==m
  native=float(old_cell_mass[oi].sum()); child=float(child_mass[ci].sum())
  mat_mass[str(m)]={'native_compiled_mass_kg':native,'candidate_child_mass_kg':child,'relative_delta':(child-native)/native}
 # Source-corner donor provenance matches the 073 owner scheme; cell mass remains cooked compiled parent mass.
 source_by_key=defaultdict(list)
 for ri,r in enumerate(contrib):source_by_key[(int(r['combined_source_global_node_index']),int(r['source_material_index']))].append(ri)
 if any(len(v)!=1 for v in source_by_key.values()):raise RuntimeError('source donor contribution key is not unique')
 corner_src=np.empty((len(tet),4),np.int32); corner_body=np.empty_like(corner_src); corner_local=np.empty((len(tet),4,3),np.float64)
 for oi,r in enumerate(tet):
  m=int(r['material_index'])
  for j in range(4):
   key=(int(r[f'global_node{j}']),m)
   if key not in source_by_key:raise RuntimeError(f'missing donor provenance for parent corner {oi}/{j}/{key}')
   ri=source_by_key[key][0]; s=contrib[ri]
   corner_src[oi,j]=ri; corner_body[oi,j]=int(s['node_frame_owner_body']); corner_local[oi,j]=[float(s[f'donor_local_{a}_m']) for a in 'xyz']
 np.savez_compressed(MAP,
  candidate_cell_uids=cell_uids,candidate_child_material_ids=new_mat.astype(np.int8),candidate_child_native_material_ids=new_mat_native,
  candidate_child_reference_volume_m3=child_geom_vol,child_native_parent_indptr=ptr,child_native_parent_rows=parent,
  overlap_native_reference_volume_m3=overlap,child_parent_fraction_overlap_over_child_volume=child_fraction,
  parent_mass_fraction_overlap_over_native_parent_geometry_volume=raw_parent_mass_fraction,
  parent_mass_fraction_overlap_over_native_parent_covered_volume=parent_mass_fraction,
  parent_local_normalization_factor_geometry_over_covered=partition['parent_local_normalization_factor_geometry_over_covered'],
  parent_local_correction_relative_to_raw=partition['parent_local_correction_relative_to_raw'],
  parent_covered_overlap_volume_m3=partition['parent_covered_volume_m3'],
  child_covered_overlap_volume_m3=partition['child_covered_volume_m3'],
  child_transferred_mass_kg=child_mass,
  child_effective_reference_density_kg_m3=child_density,native_parent_source_tet_rows=old_source.astype(np.int32),
  native_parent_material_ids=old_mat.astype(np.int16),native_parent_compiled_reference_volume_m3=old_compiled_vol,
  native_parent_native_f32_geometry_volume_m3=old_geom_vol,native_parent_compiled_density_bulk_x=old_density,native_parent_compiled_mass_kg=old_cell_mass,
  native_parent_corner_source_contribution_row_ids=corner_src,native_parent_corner_frame_owner_body_ids=corner_body,
  native_parent_corner_donor_local_points_m=corner_local,native_parent_corner_lumped_cell_mass_kg=np.repeat((old_cell_mass/4.)[:,None],4,axis=1))
 report={'status':'PASS_REFERENCE_DOMAIN_OVERLAP_AND_COOKED_MASS_LEDGER_AFTER_LOCAL_PRECISION_RECLIP',
  'admission':{'physical_mass_transfer':False,'current_geometry':False,'native_emission':False,'native_physics':False,'bone_clearance':False},
  'method':'Uses the frozen 003 all-pair geometric CSR with only the 1,025 incident pairs replaced by unchanged 085b locally translated clips. The raw CSR is preserved. After unchanged child and parent geometric coverage gates pass at 1e-7 and same-material identity is checked, cooked parent mass is apportioned by overlap divided by that parent’s math.fsum covered overlap. This is an explicit local roundoff partition; no geometric overlap is changed and no global mass rescale is applied. Rank<3 zero-measure cross-material contacts remain omitted; rank3 cross-material overlaps fail closed.',
  'input_sha256':{k:h for k,(h,p) in PIN.items()},'precision_report_sha256':sha(PRECISION_REPORT),
  'dimensions':{'native_nodes':len(node),'native_active_parents':len(tet),'candidate_nodes':len(points),'candidate_children':len(new_t),'transfer_csr_nnz':len(overlap)},
  'coverage':{'child_max_abs_relative_error':float(np.max(np.abs(data['child_overlap_relative_error']))),'parent_max_abs_relative_error':float(np.max(np.abs(data['native_parent_overlap_relative_error']))),'child_gate':bool(np.max(np.abs(data['child_overlap_relative_error']))<=1e-7),'parent_gate':bool(np.max(np.abs(data['native_parent_overlap_relative_error']))<=1e-7),'cross_material_positive_pairs':len(bad_cross)},
  'raw_geometric_coverage':{'child_max_abs_relative_error':float(np.max(np.abs(data['child_overlap_relative_error']))),'parent_max_abs_relative_error':float(np.max(np.abs(data['native_parent_overlap_relative_error']))),'unchanged_tolerance':1e-7},
  'parent_local_partition':{'formula':'same-material overlap / per-parent math.fsum(covered overlap), applied only after raw geometric coverage passes','parent_count':len(old_geom_vol),'max_abs_local_correction_relative':partition['maximum_absolute_parent_local_correction'],'max_correction_parent_row':int(np.argmax(np.abs(partition['parent_local_correction_relative_to_raw']))),'correction_factor_min':float(np.min(partition['parent_local_normalization_factor_geometry_over_covered'])),'correction_factor_max':float(np.max(partition['parent_local_normalization_factor_geometry_over_covered'])),'correction_factor_mean':float(np.mean(partition['parent_local_normalization_factor_geometry_over_covered'])),'child_coverage_gate_pass':bool(np.max(np.abs(partition['child_coverage_relative_error']))<=1e-7),'parent_coverage_gate_pass':bool(np.max(np.abs(partition['parent_coverage_relative_error']))<=1e-7),'raw_geometric_csr_unchanged':True,'global_mass_rescale':False},
  'mass':{'formula':'compiled inverse_rest_w_m3 * compiled_bulk_x; transfer by same-material overlap / per-parent covered overlap after passing raw 1e-7 geometry coverage','native_total_kg':float(old_cell_mass.sum()),'candidate_total_kg':float(child_mass.sum()),'relative_total_delta':float((child_mass.sum()-old_cell_mass.sum())/old_cell_mass.sum()),'per_material':mat_mass,'max_parent_mass_relative_closure_error':float(np.max(np.abs(parent_mass_rel))),'candidate_density_bounds_kg_m3':DENSITY_BOUNDS,'candidate_density_summary':density_summary,'density_violation_count':len(density_bad)},
  'limitations':['Reference-domain geometry and cooked-mass transfer only. Not current geometry, rendered skin, physical/native mass activation, bone clearance, or physical qualification.','Raw source contribution rows are retained as donor/frame provenance; they are not used as authoritative parent cell mass.']}
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 MANIFEST.write_text(json.dumps({'status':report['status'],'map_sha256':sha(MAP),'report_sha256':sha(REPORT),'finalizer_sha256':sha(Path(__file__)),'helper_sha256':sha(HELPER_PATH),'precision_csr_sha256':sha(PRECISION)},indent=2,sort_keys=True)+'\n')
 print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
