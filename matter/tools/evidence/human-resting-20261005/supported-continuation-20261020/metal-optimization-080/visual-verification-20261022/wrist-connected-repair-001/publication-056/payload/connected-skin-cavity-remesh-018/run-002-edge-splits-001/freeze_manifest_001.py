import hashlib,json,pathlib,sys
E=pathlib.Path(sys.argv[1]);D=E/'connected-skin-cavity-remesh-018/run-002-edge-splits-001'
def sha(p):
 p=pathlib.Path(p);return hashlib.sha256(p.read_bytes()).hexdigest()
files={
 'baseline_013_candidate':E/'E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz',
 'surface_proposal_017':E/'connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz',
 '018_selection':E/'connected-skin-cavity-remesh-018/run-002-holefix-001/cavity-selection.npz',
 '018_quality_witness':E/'connected-skin-cavity-remesh-018/run-002-holefix-001/quality-witness.json',
 '018_tetgen_node':E/'connected-skin-cavity-remesh-018/run-002-holefix-001/cavity.1.node',
 '018_tetgen_ele':E/'connected-skin-cavity-remesh-018/run-002-holefix-001/cavity.1.ele',
 '403_edge_split_helper':E/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py',
 'executed_builder':D/'build_edge_splits_001.py',
 'edge_trial_report':D/'edge-split-report.json',
 'candidate_geometry':D/'edge-split-geometry.npz',
 'roundoff_certificate_source':D/'certify_float32_roundoff_001.py',
 'roundoff_certificate':D/'float32-roundoff-certificate.json',
 'exact_self_audit_source':D/'exact-self-audit-001/audit_edge_split_surface_001.py',
 'exact_self_audit':D/'exact-self-audit-001/report.json',
 'geometry_slot_wrapper':pathlib.Path('/tmp/numi-human-geometry-slot.py')}
rep=json.load(open(files['edge_trial_report'])); rnd=json.load(open(files['roundoff_certificate'])); aud=json.load(open(files['exact_self_audit']))
chosen=next(x for x in rep['trials'] if x['split_edges_global']==[[563,670],[670,55904]])
manifest={
 'schema':'numi.connected-skin-edge-split-018-freeze.v1',
 'status':'unadmitted_geometry_candidate_with_float32_roundoff_certificate',
 'candidate_selected_edges_global':[[563,670],[670,55904]],
 'method':'conforming global exact-midpoint bisection of two original bad-tet edges using the 403 shared splitter; original edge endpoints and all prior node rows are unchanged; the emitted F32 midpoint is rounded to nearest representable coordinate',
 'tested_trials':12,'geometry_slot_elapsed_s':rep['elapsed_s'],
 'geometry_gates':{
  'all_f32_tets_positive':chosen['positive_f32'],
  'nonpositive_f32_tets':chosen['nonpositive_f32_count'],
  'min_normalized_f32_quality':chosen['min_normalized_quality_f32'],
  'below_1e_5_count':chosen['below_1e_5_count'],
  'global_boundary_exact_oriented':chosen['global_boundary_parent_split_exact_oriented'],
  'registered_shell_exact_oriented':chosen['registered_shell_boundary_exact_oriented'],
  'physical_outer_faces_positive_area':chosen['physical_outer_faces_positive_area'],
  'physical_outer_parent_barycentric_min':chosen['new_outer_barycentric_min'],
  'physical_outer_parent_barycentric_sum_error':chosen['new_outer_barycentric_sum_error_max'],
  'full_exact_self_surface_audit':aud['status'],
  'exact_self_pair_count':aud.get('exact_self_pair_count'),
  'degenerate_outer_faces':aud['degenerate_face_count']},
 'float32_roundoff':rnd,
 'interpretation':{
  'boundary_geometry':'The exact real midpoint subdivision partitions each parent triangle exactly. Stored F32 midpoint coordinates differ by at most 0.5 ULP per component; the measured per-parent area discrepancy is bounded by the explicit cross-product perturbation bound for every parent. No existing input point was moved.',
  'original_script_gate':'The raw trial report marks no candidate as passing because it uses max relative parent-face area error <1e-6. The selected pair measured 9.36e-6 max relative error on a tiny parent face; the bound certificate shows this is within F32 midpoint rounding, while total area error is 1.04e-10 relative.',
  'scope':'Geometry/topology checkpoint only. Full original source-tet ancestry, conservative material overlap/mass ledger, regional material closure, native integration, and physical admission are pending.'},
 'files':{k:{'path':str(v),'sha256':sha(v),'bytes':pathlib.Path(v).stat().st_size} for k,v in files.items()}}
(D/'freeze-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'manifest':str(D/'freeze-manifest.json'),'sha256':sha(D/'freeze-manifest.json'),'candidate_sha256':sha(files['candidate_geometry']),'status':manifest['status']},indent=2))
