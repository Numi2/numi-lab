#!/usr/bin/env python3
"""Strict CPU adjudication of the source133 native base-Cauchy kernel run."""
from __future__ import annotations
import hashlib, json, math, struct, sys
from pathlib import Path
import numpy as np
from scipy import sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
ANALYSIS = BASE / 'analysis'
NATIVE = ANALYSIS / 'native-base-cauchy-kernel-result-433'
RESULT = NATIVE / 'result'
CHECKER = ANALYSIS / 'native-base-cauchy-kernel-check-133-001'
OVERLAY = ANALYSIS / 'native-base-cauchy-kernel-133-metal-overlay-001'
INPUT = ANALYSIS / 'native-residual-fd-sweep-419-001/replay-input.bin'
REF = ANALYSIS / 'captured-base-state-cauchy-419-001'
FD424 = ANALYSIS / 'native424-residual-sweep-audit-001/report.json'
OUT = Path(__file__).resolve().parent / 'report.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f'duplicate JSON key {k!r}: {path}')
            out[k] = v
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(
                          ValueError(f'non-JSON constant {s!r}: {path}')))


def binding(path: Path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def relerr(actual, reference):
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    denom = math.sqrt(math.fsum(float(x) * float(x) for x in b))
    err = math.sqrt(math.fsum(float(x) * float(x) for x in a - b))
    return {'relative_l2': (err / denom if denom else (0.0 if err == 0 else None)),
            'absolute_l2': err,
            'max_abs': float(np.max(np.abs(a - b), initial=0.0))}


def l2(a):
    x = np.asarray(a, dtype=np.float64).ravel()
    return math.sqrt(math.fsum(float(v) * float(v) for v in x))


def require(ok, what):
    if not ok:
        raise AssertionError(what)


def main():
    # Validate the immutable native run inventory before interpreting any array.
    manifest_path = NATIVE / 'manifest.json'
    native_manifest = strict_json(manifest_path)
    expected_files = native_manifest['files']
    observed_paths = set()
    for rel, rec in expected_files.items():
        p = NATIVE / rel
        require(p.is_file(), f'missing native result file: {rel}')
        require(p.stat().st_size == rec['bytes'], f'byte-count mismatch: {rel}')
        require(sha(p) == rec['sha256'], f'SHA-256 mismatch: {rel}')
        observed_paths.add(rel)
    actual_paths = {str(p.relative_to(NATIVE)) for p in NATIVE.rglob('*') if p.is_file()
                    and p.name != 'manifest.json'}
    require(actual_paths == observed_paths,
            f'native result inventory changed: extra={sorted(actual_paths-observed_paths)} '
            f'missing={sorted(observed_paths-actual_paths)}')

    prediction = strict_json(NATIVE / 'prediction.json')
    receipt = strict_json(NATIVE / 'receipt.json')
    contracts = strict_json(RESULT / 'native-contracts.json')
    require(receipt['exit_code'] == 0 and receipt['timed_out'] is False,
            'native checker did not complete normally')
    require(receipt['protected_inputs_unchanged'] is True and
            receipt['physical_transactions'] == 0 and
            prediction['physical_transactions'] == 0,
            'native run changed protected inputs or performed physical transactions')
    protected = prediction['protected_inputs']
    protected_bindings = []
    for raw, expected in sorted(protected.items()):
        path = Path(raw)
        require(path.is_file(), f'missing protected input: {path}')
        actual = sha(path)
        require(actual == expected, f'protected input hash changed: {path}')
        protected_bindings.append({'path': raw, 'sha256': actual, 'bytes': path.stat().st_size})

    # Validate native source/build artifacts and the independent 419/424 input package.
    overlay_manifest_path = OVERLAY / 'MANIFEST.json'
    overlay_manifest = strict_json(overlay_manifest_path)
    source_bindings = [binding(overlay_manifest_path)]
    overlay_base = Path(overlay_manifest['base_snapshot'])
    for rel, rec in overlay_manifest['candidate_files'].items():
        p = OVERLAY / 'overlay' / rel
        if not p.is_file():
            p = overlay_base / rel
        require(p.is_file(), f'missing overlay/base source: {rel}')
        require(p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256'],
                f'overlay candidate source mismatch: {rel}')
        source_bindings.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    ref_manifest = strict_json(REF / 'manifest.json')
    ref_report = strict_json(REF / 'report.json')
    for rec in ref_manifest['files']:
        p = REF / rec['path']
        require(p.is_file() and p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256'],
                f'independent 419 package file changed: {rec["path"]}')
    capture_hash = sha(INPUT)
    require(capture_hash == prediction['protected_inputs'][str(INPUT)],
            '419 replay input differs from native prediction binding')
    fd424 = strict_json(FD424)
    require(fd424['capture_sha256'] == ref_report['input_bindings']['capture_sha256'],
            '424 sweep was not run on the same 419 native capture as the CPU reference package')
    require(ref_report['source_input_binding_result']['source_snapshot'] ==
            '/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125',
            '419 CPU reference source snapshot changed')
    require(fd424['all_retained_endpoints_bitwise_equal'] is True and
            fd424['derivative_qualified'] is False,
            '424 derivative evidence classification changed or is unexpected')

    # The replay format is packed little-endian; validate exact shape/extent.
    data = INPUT.read_bytes()
    hdr_fmt = '<8s6IQfI'
    hdr_size = struct.calcsize(hdr_fmt)
    magic, version, E, N, T, stride, params, world_fp, dt, reserved = struct.unpack_from(hdr_fmt, data)
    require((magic, version, E, N, T, stride, params) ==
            (b'NMFDSWP1', 1, 2, 882, 1920, 15, 59), 'unexpected 419 replay header')
    require(world_fp == int('75c1264bc96ac2d9', 16) and reserved == 0,
            'replay world fingerprint/reserved word mismatch')
    nodes = np.frombuffer(data, dtype='<f4', count=E*N*20, offset=hdr_size).reshape(E, N, 20)
    off = hdr_size + E*N*20*4
    direction_input = np.frombuffer(data, dtype='<f4', count=E*N*4, offset=off).reshape(E,N,4)
    off += E*N*4*4
    rhs_input = np.frombuffer(data, dtype='<f4', count=E*N*4, offset=off).reshape(E,N,4)
    require(np.isfinite(nodes).all() and np.isfinite(rhs_input).all(), 'nonfinite replay inputs')

    csr = sparse.load_npz(REF / 'A_free_csr.npz').tocsr()
    b_ref = np.fromfile(REF / 'base_rhs.float64le.bin', dtype='<f8')
    require(csr.shape == (2538, 2538) and csr.nnz == 71046 and b_ref.shape == (2538,),
            'independent operator dimensions changed')
    candidate_free = (nodes[0,:,11] == np.float32(0.0)) & (nodes[0,:,3] > 0.0)
    candidate_fixed = (nodes[0,:,11] == np.float32(1.0))
    require(int(candidate_free.sum()) == 846 and int(candidate_fixed.sum()) == 36 and
            bool(np.all(candidate_free | candidate_fixed)), 'constraint mask differs from reference')
    free_ids = np.flatnonzero(candidate_free)
    input_b0 = rhs_input[0, free_ids, :3].astype(np.float64).reshape(-1)
    require(np.array_equal(input_b0, b_ref), 'captured input b differs from 419 reference')
    g_ref = np.asarray(csr.T @ b_ref, dtype=np.float64).reshape(-1)
    ag_ref = np.asarray(csr @ g_ref, dtype=np.float64).reshape(-1)

    # Decode exact production-kernel readbacks. Their layouts are bound by the source checker.
    count = E*N
    def f32(name, shape):
        arr = np.fromfile(RESULT / name, dtype='<f4')
        require(arr.size == math.prod(shape), f'wrong float32 extent for {name}: {arr.size} != {shape}')
        return arr.reshape(shape)
    base_rhs = f32('base-rhs.bin', (2*count,4))
    trans_input = f32('transpose-input.bin', (E,N,4))
    gradient = f32('gradient.bin', (E,N,4))
    forward_input = f32('forward-input.bin', (E,N,4))
    gradient_action = f32('gradient-action.bin', (E,N,4))
    solution = f32('solution.bin', (2*count,4))
    trace = f32('fallback-trace.bin', (E,4))
    metrics = f32('metrics.bin', (E,4))
    line = f32('line-search.bin', (E,4))
    alpha = f32('environment-alpha.bin', (E,))
    states = f32('states.bin', (E,8))

    for env in range(E):
        free = (nodes[env,:,11] == np.float32(0.0)) & (nodes[env,:,3] > 0.0)
        fixed = (nodes[env,:,11] == np.float32(1.0))
        require(int(free.sum()) == 846 and int(fixed.sum()) == 36, f'env{env} constraint count mismatch')
        expected = np.zeros((N,4), dtype=np.float32)
        expected[free,:3] = rhs_input[env,free,:3]
        actual = base_rhs[env*N:(env+1)*N]
        require(np.array_equal(actual, expected), f'preserved base RHS/mask mismatch in env{env}')
        require(np.count_nonzero(trans_input[env]) == 0 if env != 0 else
                np.array_equal(trans_input[env], expected), f'transpose-input mask mismatch in env{env}')
        if env == 1:
            require(np.count_nonzero(gradient[env]) == 0 and np.count_nonzero(gradient_action[env]) == 0,
                    'unselected environment operator output is nonzero')
        require(np.count_nonzero(gradient[env,~free]) == 0 and
                np.count_nonzero(gradient_action[env,~free]) == 0,
                f'fixed rows not zero in env{env}')
        require(np.count_nonzero(gradient[env,:,3]) == 0 and
                np.count_nonzero(gradient_action[env,:,3]) == 0,
                f'reserved w lanes not zero in env{env}')
    require(np.count_nonzero(base_rhs[count:]) == 0,
            'the unsupported field backing RHS is not exactly zero in the admitted replay')
    require(np.array_equal(forward_input, gradient), 'forward input is not the native transpose output')
    require((RESULT/'unselected-invalid-cache.bin').read_bytes() == (RESULT/'gradient.bin').read_bytes(),
            'unselected invalid-cache control changed the selected result')
    require(np.count_nonzero(np.fromfile(RESULT/'latch-control.bin',dtype='<f4')) == 0,
            'completed-FGMRES latch control did not suppress ordinary operator execution')

    g_native = gradient[0,free_ids,:3].astype(np.float64).reshape(-1)
    ag_native = gradient_action[0,free_ids,:3].astype(np.float64).reshape(-1)
    ag_from_native = np.asarray(csr @ g_native, dtype=np.float64).reshape(-1)
    g_err = relerr(g_native,g_ref)
    ag_err_native = relerr(ag_native,ag_from_native)
    ag_err_ref = relerr(ag_native,ag_ref)
    tau_ref = math.fsum(float(x)*float(x) for x in g_ref) / math.fsum(float(x)*float(x) for x in ag_ref)
    tau_from_native = math.fsum(float(x)*float(x) for x in g_native) / math.fsum(float(x)*float(x) for x in ag_from_native)
    tau_gpu = float(trace[0,2])
    p_native = solution[:count].reshape(E,N,4)[0,free_ids,:3].astype(np.float64).reshape(-1)
    p_expected = (np.float32(tau_gpu) * gradient[0,free_ids,:3]).astype(np.float64).reshape(-1)
    p_product = relerr(p_native,p_expected)
    trace_gradient_norm = float(trace[0,0]); trace_action_norm = float(trace[0,1])
    trace_gradient_norm_rel = abs(trace_gradient_norm-l2(g_native))/l2(g_native)
    trace_action_norm_rel = abs(trace_action_norm-l2(ag_native))/l2(ag_native)

    # Builder and status controls: selected env0, unselected env1, fixed rows and field backing.
    mech_solution = solution[:count].reshape(E,N,4)
    field_solution = solution[count:].reshape(E,N,4)
    sentinel = np.array([0.125,-0.25,0.5,0.0],dtype=np.float32)
    require(np.array_equal(mech_solution[0,~candidate_free],np.zeros((36,4),dtype=np.float32)),
            'selected fixed correction rows are not zero')
    require(np.array_equal(mech_solution[1],np.tile(sentinel,(N,1))),
            'unselected mechanical solution changed')
    require(np.array_equal(field_solution[0],np.zeros((N,4),dtype=np.float32)),
            'selected inactive field solution rows are not zeroed')
    require(np.array_equal(field_solution[1],np.tile(sentinel,(N,1))),
            'unselected field solution changed')
    require(np.array_equal(metrics[1],np.array([4.0,5.0,0.75,1.0],dtype=np.float32)) and
            np.array_equal(line[1],np.array([0.25,3.0,0.5,48.0],dtype=np.float32)) and
            alpha[1] == np.float32(0.75),
            'unselected environment metrics/line-search/alpha changed')
    require(metrics[0,0] == np.float32(1.0) and np.isinf(metrics[0,1]) and
            np.array_equal(metrics[0,2:],np.zeros(2,dtype=np.float32)) and
            np.array_equal(line[0],np.array([0.0,np.inf,0.0,48.0],dtype=np.float32)) and
            alpha[0] == np.float32(0.0),
            'selected direction builder did not reset its subsequent line-search state')
    require(np.all(np.isfinite(trace[0])) and trace[0,3] == 1.0 and
            np.array_equal(trace[1],np.zeros(4,dtype=np.float32)),
            'fallback trace or original-FGMRES-latch evidence mismatched')
    require(np.array_equal(states[0,2:3],np.zeros(1,dtype=np.float32)) and states[1,2] == 1.0,
            'builder convergence-latch transitions or environment isolation mismatch')

    def status_codes(name):
        raw=(RESULT/name).read_bytes()
        require(len(raw)==E*48, f'wrong status struct extent {name}')
        return [int.from_bytes(raw[i*48:i*48+4],'little') for i in range(E)]
    st_primary=status_codes('statuses.bin')
    st_trial=status_codes('trial-statuses.bin')
    st_cache=status_codes('selected-invalid-cache-status.bin')
    st_field=status_codes('nonzero-field-status.bin')
    st_nan=status_codes('nonfinite-base-status.bin')
    require(st_primary == [0,0] and st_trial == [0,0], 'healthy kernel path status not clear')
    require(st_cache[0] != 0 and st_cache[1] == 0, 'selected invalid-cache rejection did not fail closed')
    require(st_field[0] != 0 and st_field[1] == 0, 'nonzero field RHS did not fail only selected env')
    require(st_nan[0] != 0 and st_nan[1] == 0, 'nonfinite free RHS did not fail only selected env')
    require(contracts == {'native_commands_complete': True,
        'unselected_environment_unchanged': True, 'immutable_inputs_unchanged': True,
        'strict_cache_mask_control': True, 'physical_transactions': 0},
        'native contract summary differs from expected')

    # Scale gates mirror the authored checker contract, and stay explicitly local.
    require(g_err['relative_l2'] < 1e-3, 'Metal A^T b failed independent FP64 comparison')
    require(ag_err_native['relative_l2'] < 1e-3 and ag_err_ref['relative_l2'] < 1e-3,
            'Metal A(A^T b) failed independent FP64 comparison')
    require(abs(tau_gpu-tau_ref)/tau_ref < 1e-3, 'Cauchy scalar tau differs from independent value')
    require(p_product['relative_l2'] < 1e-6, 'published candidate is not tau times the native gradient')

    input_bindings = [binding(INPUT), binding(REF/'manifest.json'), binding(REF/'report.json'),
                      binding(REF/'A_free_csr.npz'), binding(REF/'base_rhs.float64le.bin'),
                      binding(FD424), binding(manifest_path), binding(NATIVE/'prediction.json'),
                      binding(NATIVE/'receipt.json'), binding(OVERLAY/'MANIFEST.json')]
    for raw in prediction['protected_inputs']:
        p=Path(raw)
        # Keep the complete runtime prediction binding, including immutable binary inputs.
        if p not in [Path(x['path']) for x in input_bindings]:
            input_bindings.append(binding(p))
    report = {
        'schema':'numi.matter.native433_base_cauchy_adjudication.v1',
        'classification':'standalone native Metal kernel contract and captured-block parity only',
        'verdict':'PASS_LOCAL_KERNEL_CONTRACTS; FULL_SOLVER_AND_PHYSICAL_QUALIFICATION_NOT_CLAIMED',
        'native':{
            'result_directory':str(NATIVE), 'result_manifest_sha256':sha(manifest_path),
            'receipt':receipt, 'prediction_sha256':sha(NATIVE/'prediction.json'),
            'source_build_sha256':prediction['protected_inputs'][
                '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-133/source-build.json'],
            'metal_library_sha256':prediction['protected_inputs'][
                '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-133/matter/shaders/NumiMatter.metallib'],
            'checker_source_sha256':sha(CHECKER/'native_cauchy_kernel_check.mm'),
            'checker_binary_sha256':sha(CHECKER/'native_cauchy_kernel_check'),
            'world_package_sha256':prediction['protected_inputs'][
                '/Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack'],
            'physical_transactions':0},
        'captured_reference':{
            'replay_payload_sha256':capture_hash, 'world_fingerprint_hex':f'{world_fp:016x}',
            'environment':0,'root':1,'newton_iteration':48,'nodes_per_environment':N,
            'tetrahedra_per_environment':T,'fixed_nodes':36,'free_nodes':846,
            'operator_shape':list(csr.shape),'operator_nnz':int(csr.nnz),
            'operator':'unsymmetrized free FEM block A=M+dt^2*K, K=-dF_internal/dv; no rigid/contact/cohesive/field blocks',
            'base_rhs_l2_Ns':l2(b_ref),'capture_424':{
                'same_capture':True,'retained_endpoints_bitwise_equal':True,
                'derivative_qualified':False,'report_sha256':sha(FD424)}},
        'native_operator_comparison':{
            'transpose_cache_mode':3,'forward_cache_mode':4,
            'gradient_definition':'g=A^T b', 'forward_definition':'A g',
            'gradient_vs_independent_FP64_CSR':g_err,
            'native_forward_vs_CSR_times_native_gradient':ag_err_native,
            'native_forward_vs_CSR_times_FP64_gradient':ag_err_ref,
            'native_gradient_l2':l2(g_native),'fp64_gradient_l2':l2(g_ref),
            'native_forward_l2':l2(ag_native),'fp64_forward_l2':l2(ag_ref),
            'fixed_gradient_and_action_rows_exact_zero':True,
            'unselected_environment_operator_outputs_exact_zero':True,
            'transpose_input_equals_captured_masked_b':True,
            'forward_input_equals_native_gradient_bitwise':True},
        'cauchy_candidate':{
            'formula':'p=tau*g, tau=(g dot g)/(Ag dot Ag)',
            'tau_fp64_reference':tau_ref,'tau_from_native_vectors_fp64':tau_from_native,
            'tau_metal_float32':tau_gpu,
            'tau_relative_error_vs_fp64':abs(tau_gpu-tau_ref)/tau_ref,
            'metal_compensated_gradient_norm':trace_gradient_norm,
            'metal_compensated_action_norm':trace_action_norm,
            'trace_gradient_norm_relative_error_vs_native_vector':trace_gradient_norm_rel,
            'trace_action_norm_relative_error_vs_native_vector':trace_action_norm_rel,
            'candidate_vs_float32_tau_times_gradient':p_product,
            'candidate_l2_m_per_s':l2(p_native),
            'applied_as_alternative_total_correction_from_captured_center':True,
            'selected_fgmres_latch_seen_before_clear':float(trace[0,3])},
        'controls':{
            'primary_status_codes':st_primary,'trial_status_codes':st_trial,
            'selected_invalid_cache_status_codes':st_cache,
            'nonzero_field_rhs_status_codes':st_field,
            'nonfinite_free_base_rhs_status_codes':st_nan,
            'latch_disabled_operator_output_exact_zero':True,
            'unselected_invalid_cache_selected_output_bitwise_same':True,
            'selected_fixed_solution_rows_zero':True,'selected_inactive_field_solution_rows_zero':True,
            'unselected_solution_rows_unchanged':True,'unselected_metrics_state_line_search_unchanged':True,
            'native_contract_record':contracts},
        'limits':[
            'The source133 checker makes zero physical transactions and exercises isolated production Metal kernels; this is not a run of the production solver fallback path.',
            'The captured 419 state is a rejected Newton candidate, not an accepted physical state.',
            'The checked world has no rigid, contact, cohesive, MPM, or field contribution; the operator comparison is only its free FEM block.',
            'The native direction is a local linear-model candidate. No nonlinear residual replay, Armijo acceptance, full-coupled certificate, cycle admission, or physical qualification is shown.',
            'The source-matched 424 sweep reproduced retained endpoints bitwise but explicitly did not qualify a finite-difference derivative or a numerical noise floor.'
        ],
        'input_bindings':input_bindings,
        'source_overlay':{'manifest_sha256':sha(overlay_manifest_path),
            'kernel_sha256s':{str(rel):rec['sha256'] for rel,rec in overlay_manifest['candidate_files'].items()},
            'metal_compile':'overlay package records PASS at O0 and O3; native133 metallib is separately hash-bound'},
        'checks':{'all_passed':True,'max_gradient_rel_error':g_err['relative_l2'],
            'max_forward_rel_error':max(ag_err_native['relative_l2'],ag_err_ref['relative_l2']),
            'status_controls_passed':True,'immutable_input_and_environment_controls_passed':True}
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'report':str(OUT),'verdict':report['verdict'],
                      'gradient_rel':g_err['relative_l2'],
                      'forward_native_rel':ag_err_native['relative_l2'],
                      'forward_fp64_rel':ag_err_ref['relative_l2'],
                      'tau_rel':abs(tau_gpu-tau_ref)/tau_ref,
                      'candidate_rel':p_product['relative_l2']},sort_keys=True))

if __name__ == '__main__':
    main()
