#!/usr/bin/env python3
"""Independent, read-only adjudication of source142 native CGLS attempt 450."""
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, sys, math
sys.dont_write_bytecode = True
import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'solver/bounded-cgls64-exact-state-450'
OFF = BASE / 'solver/bounded-cgls64-capture-off-parity-452'
SNAP = BASE / 'snapshots/source-bounded-native-cgls-mask-repair-142'
ROOT_AUDIT = BASE / 'analysis/native450-cgls-candidate-audit-001/report.json'
ROOT_ADJ = BASE / 'analysis/native450-cgls-adjudication-001/report.json'
PARITY = BASE / 'analysis/native450-capture-off-parity-001/report.json'
TOOL = BASE / 'analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py'


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def import_at(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_map(root: pathlib.Path):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in root.rglob('*') if p.is_file()}


def verify_prediction_inputs(pred_path: pathlib.Path, snapshot: pathlib.Path):
    pred = load_json(pred_path)
    module_path = snapshot / 'matter/tools/cardboard_cached_operator_audit.py'
    module = import_at(module_path, 'source142_cached_operator_audit')
    runtime = pathlib.Path(pred['runtime_dir']) if pred.get('runtime_dir') else None
    bad = []
    for relative, expected in pred['inputs_sha256'].items():
        path, kind = module._manifest_file(relative, snapshot, runtime, pred)
        if path is None or not path.is_file() or module.sha256(path) != expected:
            bad.append({'relative': relative, 'resolved': str(path) if path else None, 'kind': kind})
    if bad:
        raise AssertionError(f'prediction input mismatch: {bad[:5]}')
    return {'count': len(pred['inputs_sha256']), 'verified': len(pred['inputs_sha256']),
            'inventory_sha256': hashlib.sha256(canonical(pred['inputs_sha256']).encode()).hexdigest(),
            'source_snapshot_matches': pathlib.Path(pred['source_snapshot']).resolve() == snapshot.resolve(),
            'runtime_dir': pred.get('runtime_dir'), 'environment': pred['environment']}


def main():
    on_pred = load_json(RUN / 'prediction.json')
    off_pred = load_json(OFF / 'prediction.json')
    on_receipt = load_json(RUN / 'receipt.json')
    off_receipt = load_json(OFF / 'receipt.json')
    on_result = load_json(RUN / 'output/result.json')
    off_result = load_json(OFF / 'output/result.json')
    root_audit = load_json(ROOT_AUDIT)
    root_adj = load_json(ROOT_ADJ)
    parity = load_json(PARITY)
    snapshot_tool = import_at(TOOL, 'native450_snapshot_parser')
    header, a = snapshot_tool.parse_snapshot(RUN / 'base-cgls.bin')

    # Run and source binding.
    inputs_on = verify_prediction_inputs(RUN / 'prediction.json', SNAP)
    inputs_off = verify_prediction_inputs(OFF / 'prediction.json', SNAP)
    assert on_receipt['execution_started'] and off_receipt['execution_started']
    assert on_receipt['source_inputs_unchanged'] and off_receipt['source_inputs_unchanged']
    assert on_receipt['changed_inputs'] == [] and off_receipt['changed_inputs'] == []
    assert on_receipt['exit_code'] == off_receipt['exit_code'] == 2
    assert on_receipt['timed_out'] is False and off_receipt['timed_out'] is False
    assert on_result['status'] == off_result['status'] == 'failed'
    assert on_result['accepted_steps'] == off_result['accepted_steps'] == 1
    assert on_result['physical_validation'] is False and off_result['physical_validation'] is False
    assert on_result['compiled_world_fingerprint'] == off_result['compiled_world_fingerprint']
    assert sha(RUN/'first-status5.bin') == sha(OFF/'first-status5.bin')
    on_s5 = load_json(RUN/'first-status5.json')
    off_s5 = load_json(OFF/'first-status5.json')
    on_s5['payload']['path'] = '<run>/first-status5.bin'
    off_s5['payload']['path'] = '<run>/first-status5.bin'
    assert canonical(on_s5) == canonical(off_s5)
    assert not (OFF/'base-cgls.bin').exists()
    assert pathlib.Path(on_pred['source_snapshot']).resolve() == SNAP.resolve()
    assert pathlib.Path(off_pred['source_snapshot']).resolve() == SNAP.resolve()
    cap_only_keys = {'NM_FEM_BASE_CGLS_CAPTURE_NEWTON','NM_FEM_BASE_CGLS_CAPTURE_PATH'}
    on_env, off_env = dict(on_pred['environment']), dict(off_pred['environment'])
    assert set(on_env)-set(off_env) == cap_only_keys and not (set(off_env)-set(on_env))
    def norm_path_values(env, root):
        return {k:(str(v).replace(str(root),'<RUN>') if isinstance(v,str) and v.startswith(str(root)) else v)
                for k,v in env.items()}
    assert norm_path_values({k:v for k,v in on_env.items() if k not in cap_only_keys}, RUN) == norm_path_values(off_env, OFF)
    norm_args = lambda d: [('<OUTPUT>' if x == d['argv'][d['argv'].index('--output')+1] else x) for x in d['argv']]
    assert norm_args(on_pred) == norm_args(off_pred)
    assert parity['parity_pass'] is True
    assert parity['history_comparison']['physical_output_files_byte_exact'] == 35
    assert parity['history_comparison']['full_armijo_trace_byte_exact'] is True
    assert parity['history_comparison']['full_true_residual_trace_byte_exact'] is True

    # Native snapshot layout and numerics.
    assert header['environment_count'] == 2 and header['nodes_per_environment'] == 882
    assert header['unknown_count'] == 3528 and header['iteration_cap'] == 64
    assert header['sha256'] == sha(RUN / 'base-cgls.bin')
    cgls = {}
    for name in ('x','r','s','p','Ax','b','mask'):
        arr = a[name]
        assert np.isfinite(arr).all()
        assert np.all(arr.view(np.uint32)[1764:] == 0), f'{name}: field/quiet region not zero'
        assert np.all(arr.view(np.uint32)[882:1764] == 0), f'{name}: quiet mechanical env not zero'
    mask = a['mask']
    free_mask = mask[:882, :3]
    free_nodes = int(np.count_nonzero(np.any(free_mask != 0, axis=1)))
    assert free_nodes == 846
    assert int(np.count_nonzero(free_mask)) == 846*3
    assert np.all(mask[882:1764] == 0)
    assert np.all(mask[1764:] == 0)
    ctrl = a['control']
    assert ctrl[0].tolist() == [2,64,64,1], ctrl[0].tolist()
    # Snapshot selected[] is authoritative; control selected lane is separately present.
    assert int(a['selected'][0]) == 1 and int(a['selected'][1]) == 0
    assert int(a['errors'][0]) == int(a['errors'][1]) == 0
    assert ctrl[1].tolist() == [0,0,64,0], ctrl[1].tolist()
    norm = lambda x: float(np.linalg.norm(np.asarray(x,dtype=np.float64).reshape(-1)))
    native_state = {
        'selected_environment': 0,
        'status_code': int(ctrl[0,0]),
        'status_name': 'exhausted' if int(ctrl[0,0]) == 2 else 'unexpected',
        'iterations': int(ctrl[0,1]), 'cap': int(ctrl[0,2]),
        'error_bits': int(a['errors'][0]), 'free_node_count': free_nodes,
        'b_l2': norm(a['b'][:882]), 'x_l2': norm(a['x'][:882]),
        'r_l2': norm(a['r'][:882]), 'Ax_l2': norm(a['Ax'][:882]),
        's_l2': norm(a['s'][:882]), 'p_l2': norm(a['p'][:882]),
        'quiet_environment_exact_zero': True, 'field_backing_exact_zero': True,
        'finite_all_snapshot_float_arrays': True,
    }
    assert native_state['status_code'] == 2 and native_state['iterations'] == native_state['cap'] == 64
    assert native_state['b_l2'] > 0 and native_state['x_l2'] > 0
    b_mechanical = a['b'][:882,:3].astype(np.float64).reshape(-1)
    ax_mechanical = a['Ax'][:882,:3].astype(np.float64).reshape(-1)
    r_mechanical = a['r'][:882,:3].astype(np.float64).reshape(-1)
    b_dot_Ax = math.fsum(float(q)*float(v) for q,v in zip(b_mechanical, ax_mechanical))
    recurrence_residual_l2 = float(np.linalg.norm(r_mechanical-(b_mechanical-ax_mechanical)))

    # Armijo and true-residual history.
    parse = snapshot_tool.parse_markers
    on_armijo = parse(RUN/'stderr.log', 'fem_residual_armijo_iteration=')
    off_armijo = parse(OFF/'stderr.log', 'fem_residual_armijo_iteration=')
    on_true = parse(RUN/'stderr.log', 'fem_true_residual_cycle=')
    off_true = parse(OFF/'stderr.log', 'fem_true_residual_cycle=')
    cgls_capture_markers = parse(RUN/'stderr.log', 'fem_base_cgls_capture=')
    cgls_direction_markers = parse(RUN/'stderr.log', 'fem_base_cgls_direction=')
    selected_capture = [x for x in cgls_capture_markers if x.get('root') == 1 and x.get('newton_iteration') == 48]
    selected_direction = [x for x in cgls_direction_markers if x.get('root') == 1 and x.get('newton_iteration') == 48]
    assert len(selected_capture) == len(selected_direction) == 1
    assert selected_capture[0].get('complete') is True and selected_capture[0].get('physical_state_changed') is False
    assert selected_capture[0].get('bytes') == selected_capture[0].get('expected_bytes') == (RUN/'base-cgls.bin').stat().st_size
    assert selected_direction[0].get('original_newton_certificate_applies_to_direction') is False
    root_on = [x for x in on_armijo if int(x.get('root',-1)) == 1]
    root_off = [x for x in off_armijo if int(x.get('root',-1)) == 1]
    assert len(root_on) == len(root_off) == 64
    assert [canonical(x) for x in root_on] == [canonical(x) for x in root_off]
    true_on = [x for x in on_true if int(x.get('root',-1)) == 1]
    true_off = [x for x in off_true if int(x.get('root',-1)) == 1]
    assert len(true_on) == len(true_off) == 4096
    assert [canonical(x) for x in true_on] == [canonical(x) for x in true_off]
    records_by_key = {(int(x['newton_iteration']), x['direction_kind']): x for x in root_on}
    assert set(k[0] for k in records_by_key) == set(range(63))
    assert (48,'newton') in records_by_key and (48,'base_cgls') in records_by_key
    assert (63,'newton') not in records_by_key
    cgls_record = records_by_key[(48,'base_cgls')]['environments'][0]
    newton_record = records_by_key[(48,'newton')]['environments'][0]
    assert cgls_record['accepted'] is True and cgls_record['selected_alpha'] == 1
    assert cgls_record['base_health'] == 'healthy_at_armijo_entry'
    assert cgls_record['trial_outcome'] == 'selected_trial_accepted'
    assert cgls_record['last_trial_status_role'] == 'unused_tail_after_selected_accept'
    assert cgls_record['solver_status_code'] == 0
    assert newton_record['accepted'] is False and newton_record['solver_status_code'] == 0
    late = []
    for it in range(49,63):
        row = records_by_key[(it,'newton')]['environments'][0]
        assert row['accepted'] is True and row['selected_alpha'] > 0
        late.append({'newton_iteration':it,'alpha':row['selected_alpha'],
                     'base_l2':row['base_l2'],'trial_l2':row['trial_l2']})
    stops = parse(RUN/'stderr.log','fem_residual_armijo_trace_stopped=')
    assert any(x.get('root')==1 and x.get('reason')=='record_cap' and x.get('maximum_trace_records')==64 for x in stops)
    # True residual includes all 64 Newton indices, not Armijo outcomes.
    per_newton = {}
    for x in true_on:
        per_newton.setdefault(int(x['newton_iteration']),[]).append(x)
    assert set(per_newton) == set(range(64))
    assert all(len(per_newton[i]) == 64 for i in range(64))
    assert per_newton[63][-1].get('final_cycle') is True

    # Exact output tree parity plus independently parsed progress traces.
    on_files, off_files = file_map(RUN/'output'), file_map(OFF/'output')
    assert on_files.keys() == off_files.keys() and on_files == off_files
    assert sha(RUN/'output/compiled.nmatterpack') == sha(OFF/'output/compiled.nmatterpack')
    assert sha(RUN/'output/compiled.nmatterpack') == root_audit['action_comparison']['compiled_pack_sha256']
    assert sha(RUN/'stdout.log') == sha(OFF/'stdout.log')
    assert on_files['observations.csv']
    root_pre_on = [x for x in root_on if int(x['newton_iteration']) < 48]
    baseline_run = BASE/'solver/cgls-disabled-history-parity-445'
    baseline = [x for x in parse(baseline_run/'stderr.log','fem_residual_armijo_iteration=')
                if int(x.get('root',-1))==1 and int(x['newton_iteration']) < 48]
    assert len(root_pre_on) == len(baseline) == 48
    assert [canonical(x) for x in root_pre_on] == [canonical(x) for x in baseline]

    # Independently reconstruct the separate source-bound production tangent capture
    # and compare only the two measured native directions.
    ref_run = BASE/'solver/two-stage-first-failure-rich-fd-419'
    ref_capture = ref_run/'operator-capture'
    ref_module_path = SNAP/'matter/tools/cardboard_cached_operator_audit.py'
    ref_module = import_at(ref_module_path, 'source125_operator_audit_for_450')
    ref_binding = ref_module.bind_inputs(ref_capture, ref_run/'prediction.json',
        ref_run/'receipt.json', ref_run/'stderr.log', 0,
        BASE/'snapshots/source-status5-vtccd-diagnostics-125')
    ref_header, ref_nodes, ref_tets, ref_integrity = ref_module.capture_records(ref_capture)
    _, _, operator, _, opmeta = ref_module.validate_and_assemble(
        ref_header, ref_nodes, ref_tets, ref_integrity, 0)
    assert ref_binding['prediction_selector'] == '1:48'
    assert ref_header['root'] == 1 and ref_header['newton_iteration'] == 48
    assert int(ref_header['world_fingerprint_hex'],16) == int(on_result['compiled_world_fingerprint'])
    assert ref_header['environment_count'] == 2 and ref_header['node_count_per_environment'] == 882
    assert ref_header['tetrahedron_count_per_environment'] == 1920
    free = np.asarray(opmeta['free_nodes'], dtype=np.int64)
    def mech(name): return a[name][free,:3].astype(np.float64).reshape(-1)
    x_native, r_native, ax_native, s_native, b_native = [mech(n) for n in ('x','r','Ax','s','b')]
    ax_ref = operator @ x_native
    atr_ref = operator.transpose() @ r_native
    ax_diff = ax_native - ax_ref
    atr_diff = s_native - atr_ref
    ax_rel = float(np.linalg.norm(ax_diff)/np.linalg.norm(ax_ref))
    atr_rel = float(np.linalg.norm(atr_diff)/np.linalg.norm(atr_ref))
    rhs_ref = np.asarray(opmeta['native_rhs_free_impulse_Ns'], dtype=np.float64).reshape(-1)
    assert np.array_equal(b_native.astype(np.float32).view(np.uint32),
                          rhs_ref.astype(np.float32).view(np.uint32))
    ax_gate = root_audit['action_comparison']['native_Ax_existing_assembled_1_percent_gate']
    atr_gate = root_audit['action_comparison']['native_s_existing_assembled_1_percent_gate']
    assert abs(ax_rel - ax_gate['relative_error']) < 1e-14
    assert abs(atr_rel - atr_gate['relative_error']) < 1e-14

    # Bound the independent operator evidence, keeping it directional only.
    ac = root_audit['action_comparison']
    assert ac['compiled_pack_sha256'] == 'cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c'
    assert ac['world_fingerprint_hex'] == '75c1264bc96ac2d9'
    assert ac['root'] == 1 and ac['newton_iteration'] == 48 and ac['environment'] == 0
    assert ac['free_node_count'] == 846 and ac['operator_shape'] == [2538,2538]
    assert ac['native_Ax_existing_assembled_1_percent_gate']['within_existing_1_percent_gate'] is True
    assert ac['native_s_existing_assembled_1_percent_gate']['within_existing_1_percent_gate'] is True
    assert ac['native_Ax_vs_reference_Ax_on_native_x']['relative_l2'] < 1e-6
    assert ac['native_s_vs_reference_ATr_on_native_r']['relative_l2'] < 1e-6
    assert root_adj['terminal_trace_limit'].find('N63 Armijo outcome is unavailable') >= 0

    # Bind a source-level interpretation to exact source hashes and relevant clauses.
    source_paths = {
        'base_cgls_runtime': SNAP/'matter/src/metal/base_cgls_runtime.metalinc',
        'fgmres': SNAP/'matter/src/metal/fgmres.metalinc',
        'fem': SNAP/'matter/src/metal/fem.metalinc',
        'runtime': SNAP/'matter/src/runtime.mm',
        'base_cgls': SNAP/'matter/src/metal/base_cgls.metalinc',
    }
    source_hashes = {k:sha(v) for k,v in source_paths.items()}
    base_runtime = source_paths['base_cgls_runtime'].read_text()
    fgmres = source_paths['fgmres'].read_text()
    fem = source_paths['fem'].read_text()
    runtime = source_paths['runtime'].read_text()
    assert 'metrics[environment].w == 3.0f' in base_runtime
    assert 'if (state.w != 3.0f)' in fgmres
    assert 'state.w = 3.0f;' in fem and 'metrics[environment] = nextState;' in fem
    assert 'femBaseCauchyFallbackAllNewtonIterations' in runtime
    assert 'maximumTraceRecords' in runtime

    report = {
        'schema':'numi.cardboard.native450_independent_review.v1',
        'verdict':'LIMITED_CGLS_TRIAL_AND_CAPTURE_PARITY_CONFIRMED; FULL_SOLVER_CONVERGENCE_NOT_QUALIFIED',
        'run':{
            'path':str(RUN),'prediction_sha256':sha(RUN/'prediction.json'),
            'receipt_sha256':sha(RUN/'receipt.json'),'stderr_sha256':sha(RUN/'stderr.log'),
            'binary_sha256':sha(RUN/'base-cgls.bin'),'binary_bytes':(RUN/'base-cgls.bin').stat().st_size,
            'exit_code':on_receipt['exit_code'],'timed_out':on_receipt['timed_out'],
            'status':on_result['status'],'accepted_control_steps':on_result['accepted_steps'],
            'physical_validation':on_result['physical_validation'],
            'source_inputs_unchanged':on_receipt['source_inputs_unchanged'],
            'input_hash_audit_on':inputs_on,'input_hash_audit_capture_off':inputs_off,
        },
        'source_snapshot':{'path':str(SNAP),'compiled_world_hex':f"0x{int(on_result['compiled_world_fingerprint']):016x}",
                           'relevant_file_sha256':source_hashes},
        'cgls':{**native_state,'captured_b_dot_Ax_independent_fsum':b_dot_Ax,
                'captured_r_vs_b_minus_Ax_l2':recurrence_residual_l2,
                'capture_marker_complete':True,'capture_marker_physical_state_changed':False,
                'direction_marker_certificate_applies':False},
        'armijo':{
            'root1_records_on':len(root_on),'root1_records_off':len(root_off),
            'exact_record_parity_on_off':True,'true_residual_cycles_on':len(true_on),
            'true_residual_cycles_off':len(true_off),'exact_true_residual_parity_on_off':True,
            'Newton48':{'accepted':newton_record['accepted'],'base_l2':newton_record['base_l2'],
                        'last_tested_alpha':newton_record['trial_alpha'],'solver_status_code':newton_record['solver_status_code']},
            'CGLS48':{'accepted':cgls_record['accepted'],'base_l2':cgls_record['base_l2'],
                      'trial_l2':cgls_record['trial_l2'],'selected_alpha':cgls_record['selected_alpha'],
                      'relative_reduction':1.0-cgls_record['trial_l2']/cgls_record['base_l2'],
                      'base_health':cgls_record['base_health'],'solver_status_code':cgls_record['solver_status_code'],
                      'tail_status_role':cgls_record['last_trial_status_role'],
                      'Newton_certificate_transferred':False},
            'later_primary_newton_49_through_62':late,
            'Armijo_trace_record_cap':64,'Newton63_Armijo_record_present':False,
            'recorded_true_residual_cycle_rows_per_newton':64,'Newton63_true_residual_final_cycle_present':True,
            'N63_armijo_outcome':'UNKNOWN: trace hit its 64-record cap before this decision; do not infer accepted or exhausted',
        },
        'history_and_operator':{
            'capture_on_off_output_file_count':len(on_files),'capture_on_off_outputs_byte_identical':True,
            'pretarget_0_through_47_vs445_record_count':48,
            'pretarget_exact_parity':True,
            'compiled_pack_sha256':ac['compiled_pack_sha256'],'same_root_newton_direction_capture':{'root':1,'newton':48,'environment':0},
            'native_Ax_direction_relative_error':ac['native_Ax_vs_reference_Ax_on_native_x']['relative_l2'],
            'native_ATr_direction_relative_error':ac['native_s_vs_reference_ATr_on_native_r']['relative_l2'],
            'existing_1_percent_directional_gates_pass':True,
            'scope_limit':'two directional FEM actions on a separate same-world/root/Newton capture with bitwise-matching RHS; the CGLS snapshot does not contain candidate nodes, so full candidate-state identity is not established; not whole-matrix identity and not the original Newton solve certificate',
        },
        'selector_all_interpretation':{
            'source_condition':'base-CGLS helper prepare/build/install gates require Armijo metric state w==3; primary accepted steps set w=1, exhaustion may set w=3',
            'observed_N49_to_N62':'all recorded primary trials accepted; fallback direction helper predicate is not active for these accepted states',
            'N63':'unobserved by Armijo trace; a 1:all run could differ there if primary Armijo exhausted, so same-budget repeat is not proven redundant',
        },
        'limitations':[
            'Only one physical control step was accepted; the full run exited 2 and rolled back the subsequent failed step.',
            'CGLS status exhausted at 64 iterations; its direction received ordinary nonlinear Armijo but did not inherit the Newton FGMRES true-residual certificate.',
            'The CGLS direction action and transpose-action are directional comparisons, not full operator or contact qualification.',
            'Armijo telemetry stops at 64 records; although Newton true-residual cycles are logged through iteration63, its Armijo branch remains unknown.',
            'The per-run rich coupled capture was skipped because E=2 is out of its capture scope; the operator reference is from a separate same-world/root/Newton capture.',
            'No complete coupon cycle or physical validation is demonstrated.',
        ],
        'bound_capture_off_parity_report_sha256':sha(PARITY),
        'bound_root_candidate_report_sha256':sha(ROOT_AUDIT),
        'bound_root_adjudication_sha256':sha(ROOT_ADJ),
        'independent_checks':'This review independently re-hashed all prediction inputs for runs450 and452, parsed the CGLS binary, compared selected run traces, and checked source gates; root reports are separately bound corroboration.',
    }
    out = pathlib.Path(__file__).parent/'report.json'
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f'wrote {out}')
    print(json.dumps({'verdict':report['verdict'],'record_count':len(root_on),
                      'late_count':len(late),'outputs':len(on_files),
                      'CGLS_trial':report['armijo']['CGLS48']},indent=2))

if __name__ == '__main__':
    main()
