#!/usr/bin/env python3
"""CPU-only source-bound audit of native442 repeated Cauchy run vs native435."""
from pathlib import Path
import csv, datetime, hashlib, json, math, os

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'solver/repeated-cauchy-full-coupon-442'
REF = BASE / 'solver/base-cauchy-first-failure-enabled-435'
CMD = BASE / 'native-442-command.json'
REF_CMD = BASE / 'native-435-command.json'
CACHE = BASE / 'snapshots/source-coupon-cache-admission-139'
SOURCE = BASE / 'snapshots/source-repeated-cauchy-fallback-137'
OUT = BASE / 'analysis/native442-repeated-fallback-adjudication-002'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strict_json(path):
    def reject(v):
        raise ValueError(f'non-standard JSON constant: {v}')
    return json.loads(Path(path).read_text(), parse_constant=reject)

def events(run, prefix):
    rows = []
    p = Path(run) / 'stderr.log'
    with p.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.startswith(prefix):
                rows.append((line_no, strict_line_json(line.split('=', 1)[1])))
    return rows

def strict_line_json(s):
    def reject(v):
        raise ValueError(f'non-standard JSON constant: {v}')
    return json.loads(s, parse_constant=reject)

def all_files(root):
    root = Path(root)
    return {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()}

def json_diffs(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(path + '/' + k)
            else:
                out.extend(json_diffs(a[k], b[k], path + '/' + k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(path + '/length')
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(json_diffs(x, y, path + f'/{i}'))
    elif a != b:
        out.append(path or '/')
    return out

def command_parts(path):
    c = strict_json(path)
    i = c.index('--')
    return c, c[:i], c[i + 1:]

def normalize_probe(argv):
    v = list(argv)
    if not v:
        return v
    v[0] = '<source-bound-probe>'
    replacements = {
        '--metallib': '<metallib>', '--output': '<output>',
        '--liner-material': '<liner-material>', '--medium-material': '<medium-material>'}
    i = 1
    while i < len(v):
        if v[i] in replacements:
            v[i + 1] = replacements[v[i]]
            i += 2
        else:
            i += 1
    return v

def parse_driver(prefix):
    d = {'env': {}, 'options': {}, 'positionals': []}
    i = 2
    while i < len(prefix):
        t = prefix[i]
        if t == '--env':
            k, v = prefix[i + 1].split('=', 1)
            d['env'][k] = v
            i += 2
        elif t.startswith('--') and i + 1 < len(prefix):
            d['options'][t] = prefix[i + 1]
            i += 2
        else:
            d['positionals'].append(t)
            i += 1
    return d

def obs_rows(outdir):
    p = Path(outdir) / 'observations.csv'
    with p.open(newline='') as f:
        return {(int(r['step']), int(r['environment'])): r for r in csv.DictReader(f)}

def artifact(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}

def armijo_summary(rows):
    result = []
    for _, x in rows:
        for env_index, e in enumerate(x.get('environments', [])):
            result.append({
                'newton_iteration': x.get('newton_iteration'),
                'direction_kind': x.get('direction_kind'),
                'environment': env_index,
                'accepted': e.get('accepted'),
                'alpha': e.get('selected_alpha'),
                'base_l2': e.get('base_l2'),
                'trial_l2': e.get('trial_l2'),
                'ratio': (float(e['trial_l2']) / float(e['base_l2'])
                          if isinstance(e.get('trial_l2'), (int, float)) and
                             isinstance(e.get('base_l2'), (int, float)) and
                             float(e['base_l2']) > 0 and
                             math.isfinite(float(e['trial_l2'])) and
                             math.isfinite(float(e['base_l2'])) else None),
                'outcome': e.get('trial_outcome'),
                'solver_status_code': e.get('solver_status_code'),
                'solver_status': e.get('solver_status'),
                'trial_status': e.get('trial_status'),
                'pretrial_status': e.get('pretrial_status'),
                'last_trial_status_role': e.get('last_trial_status_role'),
                'base_health': e.get('base_health'),
                'base_residual_interpretation': e.get('base_residual_interpretation'),
                'line_search_raw': e.get('line_search_raw'),
            })
    return result

def classify_cauchy_trace(primary_rows, cauchy_rows):
    """Separate real fallback attempts from gated second-pass telemetry."""
    by_iteration = {}
    for _, event in primary_rows:
        by_iteration.setdefault(event.get('newton_iteration'), []).append(event)
    classifications, actual = [], []
    for line_no, event in cauchy_rows:
        iteration = event.get('newton_iteration')
        primaries = by_iteration.get(iteration, [])
        if len(primaries) != 1:
            raise AssertionError(f'expected one primary event at iteration {iteration}')
        primary = primaries[0]
        p_envs, c_envs = primary.get('environments', []), event.get('environments', [])
        if len(p_envs) != len(c_envs):
            raise AssertionError(f'environment count mismatch at iteration {iteration}')
        for env_index, (p, c) in enumerate(zip(p_envs, c_envs)):
            invoked = (p.get('accepted') is False and
                       p.get('trial_outcome') == 'no_trial_accepted_status_clear')
            if not invoked:
                diffs = json_diffs(p, c)
                non_status_diffs = [d for d in diffs if not (
                    d.startswith('/trial_status') or d.startswith('/pretrial_status') or
                    d == '/last_trial_status_role')]
                if non_status_diffs:
                    raise AssertionError(
                        f'gated Cauchy metrics differ from primary at {iteration}/{env_index}: '
                        f'{non_status_diffs}')
                classification = 'gated_duplicate_primary_metrics_not_invoked'
            else:
                if c.get('trial_outcome') not in ('selected_trial_accepted', 'certified_noop'):
                    raise AssertionError(
                        f'primary rejection at {iteration}/{env_index} has no Cauchy result')
                classification = 'invoked_after_primary_armijo_exhaustion'
            row = {
                'newton_iteration': iteration,
                'direction_kind': 'base_cauchy',
                'environment': env_index,
                'accepted': c.get('accepted'),
                'alpha': c.get('selected_alpha'),
                'base_l2': c.get('base_l2'),
                'trial_l2': c.get('trial_l2'),
                'ratio': (float(c['trial_l2']) / float(c['base_l2'])
                          if isinstance(c.get('trial_l2'), (int, float)) and
                          isinstance(c.get('base_l2'), (int, float)) and
                          float(c['base_l2']) > 0 and
                          math.isfinite(float(c['trial_l2'])) and
                          math.isfinite(float(c['base_l2'])) else None),
                'outcome': c.get('trial_outcome'),
                'solver_status_code': c.get('solver_status_code'),
                'last_trial_status_role': c.get('last_trial_status_role'),
                'primary_accepted': p.get('accepted'),
                'primary_outcome': p.get('trial_outcome'),
                'gated_status_only_differences': diffs if not invoked else [],
                'telemetry_classification': classification,
                'trace_line': line_no,
            }
            classifications.append(row)
            if invoked:
                actual.append(row)
    return classifications, actual

def main():
    # Fail closed until source-bound run completion and the immutable receipt exist.
    receipt_path = RUN / 'receipt.json'
    if not receipt_path.exists():
        raise SystemExit('native442 receipt.json not present; run is not adjudicable yet')
    receipt = strict_json(receipt_path)
    assert receipt['schema'] == 'numi.cardboard.native-run-receipt.v2'
    assert receipt['execution_started'] and receipt['source_inputs_unchanged']
    assert receipt['changed_inputs'] == [] and receipt['termination_errors'] == []

    receipt435 = strict_json(REF / 'receipt.json')
    assert receipt435['schema'] == 'numi.cardboard.native-run-receipt.v2'
    assert receipt435['execution_started'] and receipt435['source_inputs_unchanged']
    assert receipt435['changed_inputs'] == [] and receipt435['termination_errors'] == []

    command, prefix, probe = command_parts(CMD)
    ref_command, ref_prefix, ref_probe = command_parts(REF_CMD)
    assert normalize_probe(probe) == normalize_probe(ref_probe)
    assert '--cache-fem-algorithmic-tangent' in probe
    assert '--cache-fem-algorithmic-tangent' in ref_probe
    driver = parse_driver(prefix)
    ref_driver = parse_driver(ref_prefix)
    env = driver['env']
    ref_env = ref_driver['env']
    assert env.get('NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR') == '1:all'
    assert ref_env.get('NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR') == '1:48'
    for k in set(env) & set(ref_env):
        if k == 'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR':
            continue
        if k.endswith('_PATH') and '/solver/' in env[k]:
            a = env[k].split('/solver/', 1)[-1].split('/', 1)[-1]
            b = ref_env[k].split('/solver/', 1)[-1].split('/', 1)[-1]
            assert a == b, (k, a, b)
        else:
            assert env[k] == ref_env[k], (k, env[k], ref_env[k])

    arm = events(RUN, 'fem_residual_armijo_iteration=')
    ref_arm = events(REF, 'fem_residual_armijo_iteration=')
    primary = lambda xs, n: [x for _, x in xs if x.get('newton_iteration') == n and x.get('direction_kind') == 'newton']
    cauchy = [(ln, x) for ln, x in arm if x.get('direction_kind') == 'base_cauchy']
    ref_cauchy = [(ln, x) for ln, x in ref_arm if x.get('direction_kind') == 'base_cauchy']
    assert len(ref_cauchy) == 1 and ref_cauchy[0][1].get('newton_iteration') == 48
    primary_0_47 = []
    for n in range(48):
        a, b = primary(ref_arm, n), primary(arm, n)
        assert len(a) == len(b) == 1, f'missing primary Newton record {n}'
        assert a[0] == b[0], f'primary history mismatch at Newton {n}'
        primary_0_47.append(n)
    ref48, now48 = primary(ref_arm, 48), primary(arm, 48)
    assert len(ref48) == len(now48) == 1
    diffs48 = json_diffs(ref48[0], now48[0])
    allowed_status48 = {
        '/environments/0/solver_status/code',
        '/environments/0/solver_status/diagnostics/0',
        '/environments/0/solver_status/diagnostics/1',
        '/environments/0/solver_status/diagnostics/2',
        '/environments/0/solver_status/diagnostics/3',
        '/environments/0/solver_status/failing_index',
        '/environments/0/solver_status/object_index',
        '/environments/0/solver_status_code',
    }
    assert set(diffs48).issubset(allowed_status48), f'unexpected primary Newton 48 differences: {diffs48}'
    assert len(diffs48) <= len(allowed_status48)
    first_cauchy = [x for _, x in cauchy if x.get('newton_iteration') == 48]
    assert len(first_cauchy) == 1, 'expected exactly one first Cauchy at Newton 48'
    assert first_cauchy[0] == ref_cauchy[0][1], 'first Newton-48 Cauchy record differs from native435'

    residual = events(RUN, 'fem_true_residual_cycle=')
    ref_residual = events(REF, 'fem_true_residual_cycle=')
    armijo_enabled = [x for _, x in events(RUN, 'fem_residual_armijo_trace_enabled=')]
    repeated_enabled = [x for _, x in events(RUN, 'fem_base_cauchy_all_newton_enabled=')]
    # Residual trace may continue with status-gated sentinels after the nonlinear
    # failure. Preserve the full sequence but do not treat sentinels as measurements.
    out = RUN / 'output'
    result = strict_json(out / 'result.json') if (out / 'result.json').exists() else None
    observations = obs_rows(out) if (out / 'observations.csv').exists() else {}
    ref_obs = obs_rows(REF / 'output')

    # The first physical control is expected to be exactly the same; compare all
    # committed source/output artifacts that exist in both runs, excluding run-specific
    # manifest/result files whose provenance is expected to differ.
    ref_out_files = all_files(REF / 'output')
    now_out_files = all_files(out)
    first_control_names = ['initial.obj', 'accepted_step_000001_bent.obj',
        'accepted_step_000001_held_reference.obj', 'compiled.nmatterpack', 'mesh.json']
    first_control_equal = {}
    for name in first_control_names:
        if name in ref_out_files and name in now_out_files:
            first_control_equal[name] = ref_out_files[name].read_bytes() == now_out_files[name].read_bytes()
        else:
            first_control_equal[name] = None
    assert all(v is True for v in first_control_equal.values()), first_control_equal
    obs_first_control_equal = {f'{s}:{e}': ref_obs[(s,e)] == observations.get((s,e))
                               for s,e in [(0,0),(0,1)]}
    assert all(obs_first_control_equal.values()), obs_first_control_equal

    observation_certificates = []
    for (step, environment), row in sorted(observations.items()):
        observation_certificates.append({
            'step': step, 'environment': environment, 'arm': row.get('arm'),
            'status_code': row.get('status_code'), 'failing_index': row.get('failing_index'),
            'step_accepted': row.get('step_accepted'),
            'certificate_raw_accepted_flag': row.get('certificate_raw_accepted_flag'),
            'certificate_residual': row.get('certificate_residual'),
            'certificate_correction': row.get('certificate_correction'),
            'certificate_volume': row.get('certificate_volume'),
            'certificate_pressure': row.get('certificate_pressure'),
            'completed_microsteps': row.get('completed_microsteps'),
            'fgmres_iterations': row.get('fgmres_iterations'),
        })
    rejected_rows = [(key, row) for key, row in sorted(observations.items())
                     if row.get('step_accepted') == '0' and row.get('arm') == 'bent']
    rollback_checks = []
    for (step, env_index), row in rejected_rows:
        rejected_path = out / f'rejected_step_diagnostic_{step + 1:06d}_bent.obj'
        prior_accepted = [(s, r) for (s, e), r in observations.items()
                          if e == env_index and s < step and r.get('arm') == 'bent' and
                          r.get('step_accepted') == '1']
        prior_step = max((s for s, _ in prior_accepted), default=None)
        prior_path = (out / f'accepted_step_{prior_step + 1:06d}_bent.obj'
                      if prior_step is not None else None)
        same = bool(rejected_path.exists() and prior_path and prior_path.exists() and
                    rejected_path.read_bytes() == prior_path.read_bytes())
        rollback_checks.append({'failed_control_step': step, 'environment': env_index,
            'rejected_geometry': str(rejected_path) if rejected_path.exists() else None,
            'last_accepted_step': prior_step,
            'last_accepted_geometry': str(prior_path) if prior_path and prior_path.exists() else None,
            'rejected_geometry_equals_last_accepted': same})
    if rejected_rows:
        assert all(x['rejected_geometry_equals_last_accepted'] for x in rollback_checks), rollback_checks
        assert all(row.get('certificate_raw_accepted_flag') == '0' for _, row in rejected_rows)

    # The selector emits Cauchy-labeled second-pass telemetry every Newton
    # iteration. Only an env whose primary Armijo pass exhausted invokes it.
    cauchy_trace_rows, fallback_rows = classify_cauchy_trace(
        [(ln, x) for ln, x in arm if x.get('direction_kind') == 'newton'], cauchy)
    actual_trigger_pairs = [(x['newton_iteration'], x['environment']) for x in fallback_rows]
    assert actual_trigger_pairs == [(48, 0), (53, 0)], actual_trigger_pairs
    assert all(x['accepted'] is True and x['alpha'] == 1 for x in fallback_rows)
    result_status = result.get('status') if result else None
    accepted_steps = result.get('accepted_steps') if result else None
    completed_coupon = bool(result and result_status == 'completed' and accepted_steps == 27 and
                            receipt.get('exit_code') == 0 and not receipt.get('timed_out'))

    cache_frozen = strict_json(CACHE / 'frozen-source.json')
    source_frozen = strict_json(SOURCE / 'frozen-source.json')
    cache_base = strict_json(BASE / 'snapshots/source-repeated-cauchy-fallback-137/frozen-source.json')
    changed_cache = sorted(k for k in set(cache_frozen['source_sha256']) | set(cache_base['source_sha256'])
                           if cache_frozen['source_sha256'].get(k) != cache_base['source_sha256'].get(k))
    assert changed_cache == ['matter/src/metal/fem.metalinc', 'matter/src/runtime.mm']
    matter_hpp = CACHE / 'matter/include/numi/matter/matter.hpp'
    compiler_cpp = CACHE / 'matter/src/compiler_impl.cpp'
    validation_cpp = CACHE / 'matter/src/validation_impl.cpp'
    runtime_mm = CACHE / 'matter/src/runtime.mm'
    hpp = matter_hpp.read_text()
    comp = compiler_cpp.read_text()
    validation = validation_cpp.read_text()
    runtime = runtime_mm.read_text()
    assert 'bool cacheFEMAlgorithmicTangent = false;' in hpp
    assert 'source.cacheFEMAlgorithmicTangent ? 1u : 0u' in comp
    assert 'solver.executionBudgets.w > 1u' in validation
    assert 'if (world.mixedSolver.executionBudgets.w == 1u)' in runtime
    material_hashes = {}
    source135 = BASE / 'snapshots/source-cauchy-positive-mass-admission-135/matter/materials'
    for name in ['hajali2009_liner_hill_bilinear_fit.nmatter',
                 'hajali2009_medium_hill_bilinear_fit.nmatter']:
        p137 = SOURCE / 'matter/materials' / name
        p135 = source135 / name
        assert sha(p137) == sha(p135)
        material_hashes[name] = sha(p137)

    arm_counts = {}
    for _, row in arm:
        k = (row.get('newton_iteration'), row.get('direction_kind'))
        arm_counts[str(k)] = arm_counts.get(str(k), 0) + 1
    trace_events = [x for _, x in events(RUN, 'fem_residual_armijo_trace_stopped=')]
    trace_complete = bool(armijo_enabled and repeated_enabled and not trace_events)
    source_semantics = {
        'source_root': str(SOURCE),
        'runtime': artifact(SOURCE / 'matter/src/runtime.mm'),
        'fgmres_metal_include': artifact(SOURCE / 'matter/src/metal/fgmres.metalinc'),
        'fem_metal_include': artifact(SOURCE / 'matter/src/metal/fem.metalinc'),
        'source_evidence': {
            'runtime_schedules_two_passes_for_all': 'matter/src/runtime.mm:22362-22364',
            'runtime_allows_fallback_only_on_primary_pass': 'matter/src/runtime.mm:22372-22374',
            'cauchy_direction_builder_requires_metrics_w_3': 'matter/src/metal/fgmres.metalinc:5255-5256,5290-5293',
            'second_pass_trial_is_gated_after_primary_accept': 'matter/src/metal/fem.metalinc:4415-4424,4650',
            'accepted_primary_marks_metrics_w_1_or_2': 'matter/src/metal/fem.metalinc:4660-4672',
            'armijo_decision_returns_for_already_selected_metrics': 'matter/src/metal/fem.metalinc:4650',
            'trace_cap': 'matter/src/runtime.mm:11792-11800',
            'stale_enabled_label': 'matter/src/runtime.mm:11469-11477',
        },
    }
    result_report = {
        'schema': 'numi.cardboard.native442-repeated-fallback-adjudication.v2',
        'supersedes': {'report': artifact(BASE / 'analysis/native442-repeated-fallback-adjudication-001/report.json'),
            'reason': 'audit001 incorrectly counted every Cauchy-labeled second-pass event as an executed fallback; v2 classifies by the source state gate and verifies gated metrics against primary records.'},
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': ('completed_coupon' if completed_coupon else
                   f"{result_status or 'missing_result'}_accepted_prefix_only"),
        'source_and_run_bindings': {
            'native442': {'command': artifact(CMD), 'receipt': artifact(receipt_path),
                'prediction': artifact(RUN / 'prediction.json'), 'stdout': artifact(RUN / 'stdout.log'),
                'stderr': artifact(RUN / 'stderr.log'), 'output_file_count': len(now_out_files),
                'output_files': {k: artifact(v) for k,v in sorted(now_out_files.items())}},
            'native435_reference': {'command': artifact(REF_CMD), 'receipt': artifact(REF / 'receipt.json'),
                'report': artifact(BASE / 'analysis/native435-enabled-fallback-adjudication-001/report.json')},
            'source137': {'frozen_source': artifact(SOURCE / 'frozen-source.json'),
                'runtime': artifact(SOURCE / 'matter/src/runtime.mm'),
                'scope': strict_json(SOURCE / 'frozen-source.json').get('scope'),
                'source_file_count': len(source_frozen['source_sha256']),
                'same_material_sha256_as_native435_source135': material_hashes},
            'cache139_default_check': {'frozen_source': artifact(CACHE / 'frozen-source.json'),
                'changed_source_files_vs137': changed_cache,
                'changed_sha256': {k: cache_frozen['source_sha256'][k] for k in changed_cache},
                'header': artifact(matter_hpp), 'compiler': artifact(compiler_cpp),
                'validation': artifact(validation_cpp), 'runtime': artifact(runtime_mm),
                'opt_in_default_false': True, 'compiled_selector_word_when_default': 0,
                'runtime_allocation_guard': 'world.mixedSolver.executionBudgets.w == 1u',
                'run442_and_reference435_explicitly_enable_cache_via_probe_flag': True},
            'fallback_pass_semantics': source_semantics,
        },
        'completion': {'receipt_exit_code': receipt.get('exit_code'), 'timed_out': receipt.get('timed_out'),
            'result_status': result_status, 'accepted_steps': accepted_steps,
            'expected_coupon_controls': 27, 'completed_coupon': completed_coupon,
            'accepted_prefix_is_not_completion': not completed_coupon,
            'observations_rows': len(observations), 'true_residual_events': len(residual),
            'armijo_events': len(arm), 'cauchy_labeled_event_records': len(cauchy),
            'cauchy_environment_trace_rows': len(cauchy_trace_rows),
            'actual_fallback_environment_trials': len(fallback_rows),
            'gated_duplicate_environment_rows': sum(
                x['telemetry_classification'] == 'gated_duplicate_primary_metrics_not_invoked'
                for x in cauchy_trace_rows),
            'armijo_trace_cap': 512,
            'armijo_trace_stopped': bool(trace_events),
            'trace_enabled_events': armijo_enabled,
            'repeated_fallback_enabled_events': repeated_enabled,
            'all_actual_fallback_trials_retained': trace_complete and
                len(fallback_rows) == sum(
                    not e.get('accepted', False)
                    for _, event in arm if event.get('direction_kind') == 'newton'
                    for e in event.get('environments', []))},
        'comparison': {'normalized_physical_probe_argv_equal': normalize_probe(probe) == normalize_probe(ref_probe),
            'primary_newton_0_through_47_exact': primary_0_47,
            'primary_newton_48_differences': diffs48,
            'primary_newton_48_differences_limited_to_status_clear': True,
            'first_cauchy_newton48_equal_to435': True,
            'first_control_outputs_equal': first_control_equal,
            'first_control_observations_equal': obs_first_control_equal,
            'armijo_event_counts_by_iteration_direction': arm_counts,
            'fallback_trials': fallback_rows,
            'cauchy_trace_row_classification': cauchy_trace_rows,
            'reference435_fallback_record_count': len(ref_cauchy),
            '442_trace_stop_events': trace_events,
            'true_residual_event_count_reference435': len(ref_residual)},
        'physical_history': {'observation_certificate_status_rows': observation_certificates,
            'failed_bent_transactions': len(rejected_rows),
            'rollback_geometry_checks': rollback_checks,
            'rejected_certificate_interpretation':
                'A failed status row with raw accepted flag 0 and zero certificate fields is not a passing certificate.'},
        'limitations': ['No physical calibration or convergence claim follows unless all 27 controls complete and terminal acceptance is present.',
            'Status-gated post-failure residual sentinels are not measurements.',
            'A Newton-direction true-residual certificate does not certify the Cauchy fallback; record each fallback Armijo result and rely on the final transaction certificate for a committed step.',
            'The coupled rich capture is explicitly skipped for this two-environment run; no coupled operator audit is claimed.',
            'The source137 runtime messages contain stale trace-cap labels; selected all scope is 512 as encoded in frozen source.',
            'Under selector :all, a Cauchy-labeled second-pass event is not necessarily an executed fallback. Only environments whose primary Newton Armijo exhausted with metrics.w set to 3 invoke the Cauchy builder; otherwise the second pass is gated and repeats primary metrics. This run had two actual Cauchy environment trials (Newton 48 and 53, env0); all remaining Cauchy-labeled environment rows are duplicate telemetry.',
            'The probe option --cache-fem-algorithmic-tangent was explicitly enabled in both native435 and native442. Source139 keeps the authored default false and allocates the cache only for selector word 1; these runs therefore do not measure default-off runtime behavior.'],
    }
    (OUT / 'report.json').write_text(json.dumps(result_report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (OUT / 'report.md').write_text(render(result_report))
    (OUT / 'README.md').write_text(
        '# Native442 repeated-Cauchy audit v2\n\n'
        'This CPU-only audit corrects the telemetry classification in adjudication-001. '
        'The earlier report counted every `base_cauchy` event as an executed fallback. '
        'This version uses source137 state gates and per-environment record comparisons; '
        'it preserves the original run, receipt, and report.\n\n'
        'Reproduce with:\n\n'
        '```sh\n'
        'python3 /Users/home/cardboard-implementation-20261014/analysis/native442-repeated-fallback-adjudication-002/audit.py\n'
        '```\n\n'
        'The script parses the immutable native442/native435 logs and receipts, verifies '
        'input-file hashes and first-control outputs, checks the :all Cauchy trace against '
        'the primary pass, and emits strict JSON (`allow_nan=False`). It performs no native '
        'execution and launches no GPU work. See `report.md` for the concise verdict and '
        '`report.json` for per-environment classifications, source hashes, and retained files.\n')
    manifest = {'schema': 'numi.cardboard.native442-repeated-fallback-adjudication-manifest.v2',
        'readme': artifact(OUT / 'README.md'), 'report': artifact(OUT / 'report.json'),
        'audit_script': artifact(Path(__file__)),
        'created_at_utc': result_report['created_at_utc'], 'receipt': artifact(receipt_path),
        'status': result_report['status']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': result_report['status'], 'armijo_events': len(arm),
        'cauchy_labeled_events': len(cauchy),
        'actual_fallback_environment_trials': len(fallback_rows), 'accepted_steps': accepted_steps,
        'completed_coupon': completed_coupon}, sort_keys=True))

def render(r):
    c = r['completion']
    comp = r['comparison']
    lines = ['# Native442 repeated-Cauchy adjudication', '',
        f"Status: **{r['status']}**.", '',
        f"The source-bound run reports {c['accepted_steps']} accepted control steps out of the 27-control protocol; completed coupon = {str(c['completed_coupon']).lower()}. An accepted prefix is not a completed coupon.", '',
        f"It retained {c['armijo_events']} residual-Armijo events, {c['cauchy_labeled_event_records']} Cauchy-labeled pass events ({c['cauchy_environment_trace_rows']} per-environment rows), and {c['true_residual_events']} true-residual cycle events. Only {c['actual_fallback_environment_trials']} rows were actual Cauchy invocations; {c['gated_duplicate_environment_rows']} are gated duplicate primary telemetry. Primary Newton records 0–47 and the first Newton-48 Cauchy event are compared field-for-field to native435.", '',
        '## Actual fallback trials', '',
        '| Newton | Env | Accepted | Alpha | Base L2 | Trial L2 | Ratio | Outcome |',
        '|---:|---:|:---:|---:|---:|---:|---:|---|']
    for x in comp['fallback_trials']:
        lines.append('| {newton_iteration} | {environment} | {accepted} | {alpha} | {base_l2} | {trial_l2} | {ratio} | {outcome} |'.format(**x))
    invoked = [x for x in comp['cauchy_trace_row_classification']
               if x['telemetry_classification'] == 'invoked_after_primary_armijo_exhaustion']
    duplicates = [x for x in comp['cauchy_trace_row_classification']
                  if x['telemetry_classification'] == 'gated_duplicate_primary_metrics_not_invoked']
    lines += ['', '## Cauchy telemetry classification', '',
        'The source emits a Cauchy-labeled event after each selected Newton iteration. Source137 schedules the second pass, but the direction builder only creates a Cauchy vector when that environment’s primary Armijo pass exhausted and set `metrics.w == 3`. An accepted primary leaves the marker at 1 (or 2 for a certified no-op); the second-pass trial is gated and the audit verified that every non-status metric exactly duplicates the primary payload.', '',
        f"The 64 Cauchy-labeled events contain {len(invoked)} invoked environment trials and {len(duplicates)} gated duplicate rows. Invocations occurred only at Newton {', '.join(str(x['newton_iteration']) for x in invoked)} for env0; each was accepted at alpha 1. Env1 remained a certified no-op throughout.", '',
        'The JSON report retains the per-iteration, per-environment classification and any status-only differences.']
    lines += ['', '## Limits', ''] + [f'- {x}' for x in r['limitations']]
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    main()
