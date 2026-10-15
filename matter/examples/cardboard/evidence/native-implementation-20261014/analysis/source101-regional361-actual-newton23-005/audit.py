"""CPU comparison of same-Newton native regional factors to retained A; no solve."""
from pathlib import Path
import hashlib, importlib.util, json, shutil
import numpy as np
from scipy import sparse

B = Path('/Users/home/cardboard-implementation-20261014')
R = B / 'solver/reset-augmentation-first-failure-rap-only-361'
C = B / 'solver/reset-augmentation-first-failure-capture-360'
I = B / 'analysis/source101-control2-newton23-audit-004/result-360'
O = B / 'analysis/source101-regional361-actual-newton23-005'
assert O.is_dir() and not (O / "report.json").exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reader = O / 'potential_mask_reader.py'
spec = importlib.util.spec_from_file_location('regional_reader', reader)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
m, a, integrity = module.load_validated_arrays(R / 'operator-capture.regional.json', R / 'operator-capture.regional.bin')
assert (m['root'], m['iteration'], m['environment_count'], m['node_count'], m['mode_count']) == (2,23,2,882,26)
assert m['tangent_cache_used'] and np.all(a['algorithmic_tangent_cache_valid'] == 1)
assert np.all(a['scheduler_exponents'] == 0)
rec = json.loads((R / 'receipt.json').read_text())
assert rec['source_inputs_unchanged'] and not rec['timed_out'] and rec['exit_code'] == 2
def normalize_argv(p):
    argv = p['argv'].copy(); argv[argv.index('--output')+1] = '<output>'; return argv
preds = [json.loads((p / 'prediction.json').read_text()) for p in (R,C)]
assert normalize_argv(preds[0]) == normalize_argv(preds[1])
files = sorted(p.relative_to(R / 'output') for p in (R / 'output').rglob('*') if p.is_file())
assert files == sorted(p.relative_to(C / 'output') for p in (C / 'output').rglob('*') if p.is_file())
assert all(sha(R / 'output' / p) == sha(C / 'output' / p) for p in files)
prefix = 'fem_residual_armijo_iteration='
rows = [[json.loads(s[len(prefix):]) for s in (p/'stderr.log').read_text().splitlines() if s.startswith(prefix)] for p in (R,C)]
assert rows[0] == rows[1] and len(rows[0]) == 64
nodes = np.empty((2,882,5,4),dtype='<f4')
history = np.zeros((2,1920,15),dtype='<f4')
Fhi = np.zeros((2,1920,3,4),dtype='<f4'); Flo = Fhi.copy()
cache = np.empty((2,1920,81),dtype='<f4')
node_seen = set(); tet_seen = set(); topology = np.empty((1920,4),dtype="<u4")
for line in (C / 'operator-capture').open():
    q = json.loads(line)
    if q.get('record') == 'header':
        assert q['world_fingerprint_hex'] == m['source_physics_fingerprint']
        assert q['execution_fingerprint_hex'] == m['device_program_fingerprint']
    elif q.get('record') == 'node':
        e,n = q['environment'],q['node_id']; assert (e,n) not in node_seen; node_seen.add((e,n))
        nodes[e,n] = q['center']['raw_state_float4s']
    elif q.get('record') == 'tetrahedron':
        e,t = q['environment'],q['tet_id']; assert (e,t) not in tet_seen; tet_seen.add((e,t))
        assert len(q['center']['accepted_state']) == q['material_state_count']
        history[e,t,:q['material_state_count']] = q['center']['accepted_state']
        if e == 0: topology[t] = q['nodes']
        Fhi[e,t,:,:3] = np.asarray(q['center']['F_local'],dtype='<f4')
        Flo[e,t,:,:3] = np.asarray(q['center']['F_local_low'],dtype='<f4')
        cache[e,t] = q['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']
assert len(node_seen) == 1764 and len(tet_seen) == 3840
state_matches = {}
for key, value in [('candidate_nodes',nodes.view('u1').reshape(2,882,80)),('material_history',history),('deformation_high',Fhi),('deformation_low',Flo),('algorithmic_tangent_cache',cache)]:
    matched = value.tobytes() == a[key].tobytes(); state_matches[key] = matched
    assert matched,key
potential = (a['graph_node_coverage'] > 0).tolist()
assert hashlib.sha256(bytes(potential)).hexdigest() == m['graph_schwarz']['potential_node_mask_sha256']
topology_bytes = np.asarray([882,1920],dtype='<u4').tobytes() + topology.tobytes()
assert hashlib.sha256(topology_bytes).hexdigest() == m['graph_schwarz']['source_topology_sha256']
rebuilt = module.rebuild_graph_layout(topology,potential)
independent_layout_matches = {key:bool(np.array_equal(rebuilt[key],a['graph_'+key])) for key in ['patches','patch_nodes','rows','node_patch_offsets','node_patch_values','node_coverage']}
A = sparse.load_npz(I/'environment-0-operator-audit.operator.npz').tocsr()
v = np.load(I/'environment-0-operator-audit.vectors.npz')
free = v['free_node_ids']; assert np.array_equal(free,np.flatnonzero(nodes[0,:,2,3] <= .5))
P = a['effective_P'][0][:,free,:3].transpose(1,2,0).reshape(A.shape[0],26).astype(np.float64)
AP = a['AP'][0][:,free,:3].transpose(1,2,0).reshape(A.shape[0],26).astype(np.float64)
expected = A @ P
relative = lambda actual, reference: float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-300))
def reconstructed(lu,pivots):
    n = len(pivots); assert sorted(pivots.tolist()) == list(range(n))
    lower = np.tril(lu,-1); lower[np.diag_indices(n)] = 1
    permuted = lower @ np.triu(lu)
    result = np.empty_like(permuted); result[pivots] = permuted; return result
coarse = P.T @ AP
native_coarse = reconstructed(a['pivoted_RAP'][0].astype(float),a['pivots'][0].astype(int))
node_to_free = np.full(882,-1,dtype=int);node_to_free[free] = np.arange(len(free))
patch_reports = []
for idx,(offset,count,matrix_offset,pivot_offset) in enumerate(a['graph_patches'].astype(int)):
    ndof = 3*count
    patch_nodes = a['graph_patch_nodes'][offset:offset+count].astype(int)
    ids = (3*node_to_free[patch_nodes,None]+np.arange(3)).reshape(-1)
    fixed_local = np.repeat(node_to_free[patch_nodes] < 0,3)
    free_local = np.flatnonzero(~fixed_local)
    expected_patch = np.zeros((ndof,ndof))
    expected_patch[np.ix_(free_local,free_local)] = A[ids[free_local],:][:,ids[free_local]].toarray()
    expected_patch[np.flatnonzero(fixed_local),np.flatnonzero(fixed_local)] = 1.0
    lu = a['graph_patch_matrices_lu'][matrix_offset:matrix_offset+ndof*ndof].reshape(ndof,ndof).astype(float)
    piv = a['graph_pivots'][pivot_offset:pivot_offset+ndof].astype(int)
    reconstructed_patch = reconstructed(lu,piv)
    patch_reports.append(dict(index=idx,nodes=count,dofs=ndof,relative_factor_reconstruction_difference=relative(reconstructed_patch,expected_patch),maximum_absolute_difference=float(np.max(np.abs(reconstructed_patch-expected_patch)))))
inputs = [reader,R/'operator-capture.regional.json',R/'operator-capture.regional.bin',R/'prediction.json',R/'receipt.json',C/'operator-capture',I/'environment-0-operator-audit.operator.npz',I/'environment-0-operator-audit.vectors.npz']
report = dict(schema='numi.cardboard.actual-newton-regional-audit.v1',
    capture_integrity=integrity, potential_mask_and_exact_topology_digest_match=True, older_independent_layout_comparison=independent_layout_matches, same_generation360_state_byte_matches=state_matches,
    ordinary360_361_files_byte_identical=len(files),armijo_rows_identical=len(rows[0]),
    AP_relative_difference=relative(AP,expected), AP_maximum_absolute_difference=float(np.max(np.abs(AP-expected))),
    coarse_LU_reconstruction_relative_difference=relative(native_coarse,coarse),
    patches=patch_reports,maximum_patch_factor_reconstruction_difference=max(q['relative_factor_reconstruction_difference'] for q in patch_reports),
    inputs_sha256={str(p):sha(p) for p in inputs},
    limits=['Zero native/physical transactions executed by this CPU audit.',
    'Older independent graph constructor does not reproduce current patch layout; this report validates captured row maps/checksum and topology/mask digests, not independent layout generation.',
    'A is reconstructed from captured cached tangents; no independent constitutive accuracy qualification.',
    'The capture is before FGMRES correction, with 26 base modes, not the later augmented cycle6 operator.',
    'Potential fixed coordinates are compared to identity rows and columns, as production graph conditioning requires.',
    'LU reconstruction differences include production FP32 factorization and accumulation; no universal noise floor is measured.',
    'Both physical attempts still reject after two of27 controls; no mesh/time convergence or full-box qualification.'])
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
shutil.copy2(__file__,O/'audit.py')
(O/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in O.iterdir() if p.is_file()}),indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['ordinary360_361_files_byte_identical','armijo_rows_identical','AP_relative_difference','coarse_LU_reconstruction_relative_difference','maximum_patch_factor_reconstruction_difference']},indent=2))
