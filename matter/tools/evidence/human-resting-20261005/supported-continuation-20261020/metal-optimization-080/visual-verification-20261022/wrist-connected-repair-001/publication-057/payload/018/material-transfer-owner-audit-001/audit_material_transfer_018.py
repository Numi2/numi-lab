from pathlib import Path
import hashlib,json,sys
import numpy as np
W=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
E=W/'E'
A=W/'connected-skin-cavity-remesh-018'/'run-002-edge-splits-001'
HOLE=W/'connected-skin-cavity-remesh-018'/'run-002-holefix-001'
paths={
 'original_013':E/'connected-skin-mechanical-transfer-013'/'boundary-refinement-002'/'candidate.npz',
 '017_proposal':W/'connected-skin-safe-flow-017'/'run-002'/'connected-rest-proposal.npz',
 'cavity_selection':HOLE/'cavity-selection.npz',
 'tetgen_node':HOLE/'cavity.1.node',
 'tetgen_element':HOLE/'cavity.1.ele',
 '018_edge_geometry':A/'edge-split-geometry.npz',
 'overlap_prefix_report':A/'overlap-first256-001'/'overlap-prefix-256-report.json',
 '018_builder':W/'connected-skin-cavity-remesh-018'/'prepare_local_cavity_018-holefix.py',
 'pullback_guard':A/'material-transfer-owner-audit-001'/'reference_map_gate.py',
 'pullback_guard_tests':A/'material-transfer-owner-audit-001'/'test_reference_map_gate.py',
 'audit_script':A/'material-transfer-owner-audit-001'/'audit_material_transfer_018.py',
 'guard_test_log':A/'material-transfer-owner-audit-001'/'reference-map-tests.log',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
for name,p in paths.items():
 if not p.exists(): raise FileNotFoundError(f'{name}: {p}')
base=np.load(paths['original_013'],allow_pickle=False)
prop=np.load(paths['017_proposal'],allow_pickle=False)
sel=np.load(paths['cavity_selection'],allow_pickle=False)
edge=np.load(paths['018_edge_geometry'],allow_pickle=False)
T0=base['tetrahedra']; X0=base['points_world_m'].astype(np.float64)
T1=prop['tetrahedra']; X1=prop['points_world_m'].astype(np.float64)
assert T0.shape==T1.shape and np.array_equal(T0,T1), '017 topology differs; signed ratio not aligned'
assert X0.shape==X1.shape
rows=sel['selected_input_tet_rows'].astype(np.int64)
def dets(X,T):
 q=X[T]
 return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
d0=dets(X0,T0); d1=dets(X1,T1)
ratio=d1[rows]/d0[rows]
changed=np.flatnonzero(np.any(X0!=X1,axis=1))
nonpos=np.flatnonzero(d1<=0)
inside=np.isin(nonpos,rows)
# Small robust content summary; never serialize the large row arrays.
region=np.unique(edge['region_attributes'],return_counts=True)
edge_keys=set(edge.files)
required_source_map_keys=['node_parent_013_tet_rows','node_parent_013_barycentric','node_reference_points_world_m','child_parent_013_tet_offsets','child_parent_013_tet_rows','child_parent_013_intersection_volume_m3']
report=json.loads(paths['overlap_prefix_report'].read_text())
sel=sel
# Exercise the exact frozen 017 failure through the new typed pullback guard.
sys.path.insert(0,str(A/'material-transfer-owner-audit-001'))
from reference_map_gate import MappingError, validate_reference_parent_map
first_bad=int(np.flatnonzero((d1<=0) & np.isin(np.arange(len(d1)),rows))[0])
q=T1[first_bad]
center=X1[q].mean(axis=0)[None,:]
try:
 validate_reference_parent_map(node_parent_tet_rows=np.asarray([0],np.int64),
  node_barycentric=np.asarray([[.25,.25,.25,.25]]),parent_tetrahedra=q[None,:],
  deformed_parent_points=X1,reference_parent_points=X0,candidate_node_points=center,
  supplied_domain='original_013_tet_rows:'+sha(paths['original_013']),
  expected_domain='original_013_tet_rows:'+sha(paths['original_013']),expected_node_count=1)
 actual_pullback_rejected=False; actual_pullback_rejection='unexpected pass'
except MappingError as exc:
 actual_pullback_rejected=True; actual_pullback_rejection=str(exc)
node_header=paths['tetgen_node'].read_text().splitlines()[0].split()
ele_header=paths['tetgen_element'].read_text().splitlines()[0].split()
script=paths['018_builder'].read_text()
# pin only behavior relevant to transfer; source itself remains immutable
report_out={
 'status':'BLOCKED_NO_VALIDATED_REFERENCE_PULLBACK',
 'inputs':{k:{'path':str(v),'sha256':sha(v),'bytes':v.stat().st_size} for k,v in paths.items()},
 '017_inverse_geometry':{
  'same_tet_topology_as_013':True,'input_node_count':int(len(X0)),'changed_node_count':int(len(changed)),
  'selected_cavity_parent_tets':int(len(rows)),'nonpositive_all_tets':int(len(nonpos)),
  'nonpositive_selected_tets':int(np.count_nonzero(inside)),
  'selected_min_signed_det_013_m3x6':float(d0[rows].min()),
  'selected_min_signed_det_017_m3x6':float(d1[rows].min()),
  'selected_min_signed_det_ratio':float(ratio.min()),
  'selected_positive_det_ratio_min':float(ratio[ratio>0].min()),
  'inverse_map_admissible':bool(np.all(d1[rows]>0)),
  'first_nonpositive_selected_global_tet_row':first_bad,
  'typed_guard_rejected_actual_folded_parent':actual_pullback_rejected,
  'typed_guard_rejection':actual_pullback_rejection,
 },
 '018_remesh':{
  'node_count':int(len(edge['points_world_m'])),'tet_count':int(len(edge['tetrahedra'])),
  'parent_merged_global_tet_rows_count':int(len(edge['parent_merged_global_tet_rows'])),
  'region_attribute_histogram':{str(int(k)):int(v) for k,v in zip(*region)},
  'tetgen_node_header_fields':node_header,'tetgen_element_header_fields':ele_header,
  'node_reference_map_keys_present':[k for k in required_source_map_keys if k in edge_keys],
  'node_reference_map_keys_missing':[k for k in required_source_map_keys if k not in edge_keys],
  'material_interface_face_count_in_input_PLC':int(len(sel['material_interface_faces_global'])),
  'input_region_attribute_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(sel['region_attributes'],return_counts=True))},
  'available_node_lineage_keys':[k for k in edge.files if ('source' in k or 'parent' in k or 'bary' in k) and edge[k].ndim>0],
  'static_source_behavior':{
   'overlap_uses_current_domain': 'local_overlap(x,t[old_rows],old_rows,lx,lt,intersection)' in script,
   'node_composition_uses_013_baseline_geometry': "xyz=base['points_world_m'][old_t].astype(np.float64)" in script,
   'candidate_payload_has_per_node_source_parent_attempt': 'cavity_child_node_parent_013_tet_rows=nodeparent' in script and 'cavity_child_node_parent_013_barycentric=nodebary' in script,
   'new_node_mapping_method_is_point_containment_in_original_013_geometry': "xyz=base['points_world_m'][old_t].astype(np.float64)" in script,
   'mapping_is_explicit_Lagrangian_transport': False,
  },
 },
 'existing_overlap_prefix':{
  'status':report.get('status'), 'children_checked':report.get('domain',{}).get('capacity_children'),
  'children_with_incomplete_coverage':report.get('coverage_prefix_only',{}).get('children_below_1_minus_1e-9'),
  'min_child_coverage_ratio':report.get('coverage_prefix_only',{}).get('child_coverage_ratio_min'),
  'full_parent_csr_available':report.get('completion',{}).get('full_parent_CSR_available'),
  'all_new_cavity_children_complete':report.get('completion',{}).get('all_new_cavity_children_complete'),
  'estimated_full_cavity_seconds':report.get('runtime',{}).get('estimated_full_cavity_s_at_probe_rate'),
 },
 'blocker':'017 has nonpositive selected-cavity parent determinants, so its same-topology piecewise-linear deformation has no valid local inverse on 78 cells. 018 carries child-to-merged-input tet rows and surface-face ancestry, but no per-node original-013 reference simplex/barycentrics in the emitted geometry. The later compose_node_sources routine attempts spatial containment against undeformed 013 coordinates; that is not a Lagrangian material pullback and cannot repair a folded deformation. The PLC has no material-interface faces and one region attribute (1), so it does not partition skin and fat. The 256-child overlap probe is incomplete and found 39 children with uncovered volume; it is raw current-space overlap and not a source-mass certificate.',
 'minimal_next_step':'The next producer increment should construct a material-aware cavity PLC with explicit skin/fat interface facets and region seeds, while transporting source-reference coordinates through a positive-Jacobian local ALE map. Each TetGen node must carry a typed original-013 parent tetrahedron and barycentric coordinate; boundary split nodes use their exact edge/face ancestry, and interior nodes are interpolated from a certified containing reference simplex. Gate positive signed Jacobian and candidate-point reconstruction before composing existing 013->014 conservative CSR. If the current 017 boundary deformation cannot admit a positive ALE map, repair that deformation first or rebuild the cavity in reference space with an orientation-preserving map; do not invert 017.',
 'guard_regression_tests':{'status':'PASS','test_count':7,'source_path':str(paths['pullback_guard_tests']),'log_path':str(paths['guard_test_log']),'log_sha256':sha(paths['guard_test_log'])},
 'qualification':'diagnostic only; no full overlap, no mass conservation claim, no native admission; local signed determinants do not certify global injectivity or non-overlap'
}
out=A/'material-transfer-owner-audit-001'/'material-transfer-blocker-report.json'
out.write_text(json.dumps(report_out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'017':report_out['017_inverse_geometry'],'018':report_out['018_remesh'],'overlap_prefix':report_out['existing_overlap_prefix']},indent=2))
