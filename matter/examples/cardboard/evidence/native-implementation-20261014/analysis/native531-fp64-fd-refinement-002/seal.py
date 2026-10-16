from pathlib import Path
import hashlib,json,numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
O=Q/'analysis/native531-fp64-fd-refinement-002'
BASE=Q/'analysis/native531-strict-fp64-replay-001/layers'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
S=Q/'snapshots/source-contact-damped-memory-198-001/matter/tools'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=json.loads((BASE/'manifest.json').read_text())
expected={
 'cardboard_bilinear_fp64_reference.py':'a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995',
 'cardboard_compensated_fp64_reference.py':'760ea825449857d565c78e754c755845e4547134b381fa167d02aab2ae63720c',
 'cardboard_compensated_fp64_base.py':'b135c6ff0091f0833c8dad89239abdde209fc6e0eb91babeb72313b29a88bb6c',
 'cardboard_dd_fp64.py':'8edfd60beb2c38d181a08023d9f97bea0fec80b0f6fa04f8690c7dfe234330c2',
 'cardboard_cached_operator_audit.py':'4a7e9c0cbf09eee0ad1f8f5b2b09314a2c0789a908aaab530b56cfff6a1c82ff',
 'cardboard_solver_environment.py':'e3ef5898174dc2081542d952f0c94ea3f0798510508e2a5672e42d877148d0f7'}
for n,h in expected.items(): assert sha(S/n)==h,(n,sha(S/n),h)
with np.load(O/'layers/arrays.npz') as z:
 unresolved=[136,498,499,510,511,650,1012,1362,1363,1523]
 assert np.all(z['fp64_resolved_adjacent_pair'])
 for i in unresolved:
  assert any(z['e%d_smooth'%p][i] and z['e%d_plus_branch_preserving'%p][i] and z['e%d_minus_branch_preserving'%p][i] and z['e%d_relative_error'%p][i]<=1e-6 and z['e%d_smooth'%(p+1)][i] and z['e%d_plus_branch_preserving'%(p+1)][i] and z['e%d_minus_branch_preserving'%(p+1)][i] and z['e%d_relative_error'%(p+1)][i]<=1e-6 for p in range(10,24))
parent_manifest={str(BASE/n):sha(BASE/n) for n in ('manifest.json','report.json','arrays.npz')}
case=Q/'solver/control17-root17-newton34-replay-531'
case_inputs={str(case/n):sha(case/n) for n in ('prediction.json','receipt.json','stderr.log','output/manifest.json')}
capture=Path(json.loads((case/'prediction.json').read_text())['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='manifest.json'}
manifest={
 'schema':'numi.cardboard.native531-fp64-fd-refinement-files.v1',
 'classification':'CPU-only compensated FP64 epsilon-grid refinement; selected captured direction only; not native FP32 noise-floor or physical qualification',
 'parent_output':str(BASE),
 'parent_inputs_sha256':parent_manifest,
 'native_case_inputs_sha256':case_inputs,
 'capture_sha256':sha(capture),
 'source198_reference_tool_sha256':{str(S/n):h for n,h in expected.items()},
 'strict_layer_reader_sha256':sha(PREP/'strict_layers.py'),
 'run_script_sha256':sha(O/'refine.py'),
 'grid_powers':list(range(10,25)),
 'files_sha256':files,
 'qualification':False
}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(O/'manifest.json'),'file_count':len(files),'files':files,'all_ten_adjacent_pass':True},indent=2))
