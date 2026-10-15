#!/usr/bin/env python3
"""Read-only, fail-closed adjudication of source-bound coupon run 409."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

DEFAULT_RUN = Path('/Users/home/cardboard-implementation-20261014/solver/two-stage-certified-coupon-409')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_prefixed_json(path: Path, prefix: str):
    records = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            at = line.find(prefix)
            if at < 0:
                continue
            payload = line[at + len(prefix):].strip()
            try:
                records.append((line_no, json.loads(payload)))
            except Exception as e:
                raise RuntimeError(f'{path}:{line_no}: malformed {prefix} JSON: {e}') from e
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=DEFAULT_RUN)
    ap.add_argument('--output', type=Path, help='write audit JSON here; destination must not exist')
    args = ap.parse_args()
    run = args.run.resolve()
    require(run.is_dir(), f'missing run directory: {run}')
    pred = read_json(run / 'prediction.json')
    receipt = read_json(run / 'receipt.json')
    source_build_path = run / 'inputs/external/source-build.json'
    build = read_json(source_build_path)
    frozen_path = Path(build['source_snapshot']) / 'frozen-source.json'
    frozen = read_json(frozen_path)
    manifest = read_json(run / 'output/manifest.json')
    result = read_json(run / 'output/result.json')
    reject_manifest_path = run / 'output/rejected-conditioning-000002/manifest.json'
    reject_manifest = read_json(reject_manifest_path)

    # Bind the exact prediction, run receipt and source/build closure.
    require(receipt['schema'] == 'numi.cardboard.native-run-receipt.v2', 'unexpected receipt schema')
    require(receipt['execution_started'] is True and receipt['source_inputs_unchanged'] is True,
            'receipt does not confirm execution with unchanged source inputs')
    require(receipt['exit_code'] == 2 and receipt['timed_out'] is False,
            'receipt is not the expected completed failure')
    require(pred['source_snapshot'] == build['source_snapshot'], 'prediction/source-build snapshot mismatch')
    require(build['success'] is True and not build.get('changed_sources'), 'source build failed or changed sources')
    require(build['schema'] == 'numi.cardboard.source-build.v1', 'unexpected source-build schema')
    require(frozen['schema'] == 'numi.cardboard.frozen-source.v1', 'unexpected frozen-source schema')
    require(frozen['source_sha256'] == build['source_sha256'], 'source-build hashes differ from frozen manifest')

    verified = {}
    failures = []
    snapshot = Path(build['source_snapshot'])
    for rel, expected in build['source_sha256'].items():
        p = snapshot / rel
        actual = sha256(p) if p.is_file() else None
        if actual != expected:
            failures.append(f'source:{rel}')
        verified[f'source/{rel}'] = {'expected': expected, 'actual': actual, 'match': actual == expected}
    for path_text, expected in build['artifacts_sha256'].items():
        p = Path(path_text)
        actual = sha256(p) if p.is_file() else None
        if actual != expected:
            failures.append(f'artifact:{path_text}')
        verified[f'artifact:{path_text}'] = {'expected': expected, 'actual': actual, 'match': actual == expected}
    build_dir = Path(build['build_dir'])
    for rel, expected in build.get('build_metadata_sha256', {}).items():
        p = build_dir / rel
        actual = sha256(p) if p.is_file() else None
        if actual != expected:
            failures.append(f'build_metadata:{rel}')
        verified[f'build_metadata/{rel}'] = {'expected': expected, 'actual': actual, 'match': actual == expected}
    # The prediction's complete input map must match the source-build manifest.
    expected_prediction_inputs = {f'source/{k}': v for k, v in build['source_sha256'].items()}
    runtime_inputs = {f'runtime/{Path(k).relative_to(Path(build["build_dir"])).as_posix()}': v
                      for k, v in build['artifacts_sha256'].items()}
    prediction_inputs = pred['inputs_sha256']
    require({k: v for k, v in prediction_inputs.items() if k.startswith('source/')} ==
            expected_prediction_inputs, 'prediction source-file hashes differ from source-build inputs')
    for key, value in prediction_inputs.items():
        if key.startswith('runtime/'):
            require(runtime_inputs.get(key) == value, f'prediction runtime artifact not bound by source build: {key}')
    require(prediction_inputs.get('external/source-build.json') == sha256(source_build_path),
            'prediction does not bind its external source-build manifest')
    require(set(prediction_inputs) == set(expected_prediction_inputs) |
            {k for k in prediction_inputs if k.startswith('runtime/')} | {'external/source-build.json'},
            'prediction contains an unexpected input category')
    require(not failures, 'source/build hash mismatches: ' + ', '.join(failures[:12]))

    # Independently bind materials and run outputs.
    materials = manifest['materials']
    material_hashes = {}
    for label, path_key, hash_key in (('liner', 'liner_path', 'liner_sha256'),
                                      ('medium', 'medium_path', 'medium_sha256'),
                                      ('glue', 'glue_path', 'glue_sha256')):
        p = Path(materials[path_key])
        actual = sha256(p)
        require(actual == materials[hash_key], f'{label} material hash mismatch')
        material_hashes[label] = {'path': str(p), 'sha256': actual}
    require(result['schema'] == 'numi.cardboard.probe-result.v1' and result['status'] == 'failed',
            'result is not failed probe output')
    require(result['accepted_steps'] == 1 and result['physical_validation'] is False,
            'unexpected accepted-step or physical-validation result')
    require(result['compiled_world_fingerprint'] == 8485105279886344921,
            'world fingerprint differs from captured run')
    require(reject_manifest['accepted_physical_state'] is False and reject_manifest['control_step'] == 1,
            'rejected conditioning capture is not correctly marked')
    rejected_files = {}
    for rec in reject_manifest['files']:
        p = run / 'output/rejected-conditioning-000002' / rec['name']
        actual = sha256(p)
        require(actual == rec['sha256'], f'rejected-conditioning hash mismatch: {rec["name"]}')
        require(p.stat().st_size == rec['bytes'], f'rejected-conditioning byte-count mismatch: {rec["name"]}')
        rejected_files[rec['name']] = {'sha256': actual, 'bytes': rec['bytes']}

    # The protocol requested 27 controls; do not infer completed stages from the plan.
    argv = pred['argv']
    argmap = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith('--') and i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            argmap[argv[i]] = argv[i + 1]
            i += 2
        else:
            argmap[argv[i]] = True
            i += 1
    planned = {k: int(argmap[k]) for k in ('--steps', '--hold-steps', '--unload-steps', '--relax-steps', '--release-steps')}
    planned_count = sum(planned.values())
    require(planned_count == 27, f'expected 27 planned macro controls, found {planned_count}')
    require(argmap['--nx'] == '8' and argmap['--ny'] == '2' and argmap['--thickness-slices'] == '1',
            'unexpected mesh resolution')
    require(argmap['--dt'] == '.0001' and argmap['--newton-iterations'] == '64' and
            argmap['--line-search-steps'] == '24', 'unexpected timestep or Newton/Armijo budgets')
    require(argmap['--without-self-contact'] is True, 'unexpected self-contact configuration')
    require(manifest['tooling']['enabled'] is False and manifest['solver']['deformable_self_contact'] is False,
            'run scope differs: tooling/contact unexpectedly enabled')

    with (run / 'output/observations.csv').open(newline='', encoding='utf-8') as f:
        observations = list(csv.DictReader(f))
    require(len(observations) == 4, f'expected four step/environment rows, found {len(observations)}')
    obs = {(int(r['step']), int(r['environment'])): r for r in observations}
    require(set(obs) == {(0, 0), (0, 1), (1, 0), (1, 1)}, 'unexpected step/environment row set')
    require(all(r['phase'] == 'loading' for r in observations), 'observations contain non-loading phase')
    require(obs[(0, 0)]['step_accepted'] == '1' and obs[(0, 0)]['status_code'] == '0', 'bent control 0 not accepted')
    require(obs[(1, 0)]['step_accepted'] == '0' and obs[(1, 0)]['status_code'] == '10', 'bent control 1 not failed with code 10')
    require(obs[(0, 1)]['step_accepted'] == '1' and obs[(1, 1)]['step_accepted'] == '1',
            'held reference controls not accepted')

    armijo = [d for _, d in parse_prefixed_json(run / 'stderr.log', 'fem_residual_armijo_iteration=')]
    cycles = [d for _, d in parse_prefixed_json(run / 'stderr.log', 'fem_true_residual_cycle=')]
    require(len(armijo) == 64 and len(cycles) == 4096, 'unexpected Armijo or residual-cycle record count')
    by_newton = {d['newton_iteration']: d for d in armijo}
    require(set(by_newton) == set(range(64)), 'Newton records are incomplete or duplicated')
    env0 = {n: by_newton[n]['environments'][0] for n in range(64)}
    require(all(env0[n]['accepted'] and env0[n]['base_health'] == 'healthy_at_armijo_entry'
                for n in range(48)), 'one or more pre-failure Newton steps did not select a healthy trial')
    fail = env0[48]
    require(fail['accepted'] is False and fail['trial_outcome'] == 'no_trial_accepted_status_clear',
            'Newton 48 is not the first recorded line-search failure')
    require(fail['pretrial_status']['code'] == 0 and fail['trial_status']['code'] == 0,
            'Newton 48 base/last-trial status is not clear')
    require(fail['solver_status_code'] == 10 and fail['solver_status']['failing_index'] == 48,
            'Newton 48 did not latch line-search failure code 10/index 48')
    require(all(env0[n]['base_health'] == 'invalid_pretrial_solver_status' and
                env0[n]['pretrial_status']['code'] == 10 and
                env0[n]['trial_outcome'] == 'blocked_by_pretrial_solver_status'
                for n in range(49, 64)), 'Newton 49–63 are not all blocked by the prior failure')
    require(all(by_newton[n]['environments'][1]['certified_noop'] for n in range(64)),
            'held-reference environment is not a certified no-op through recorded Newton loop')

    n48_cycles = sorted([d for d in cycles if d.get('newton_iteration') == 48], key=lambda d: d['restart_cycle'])
    require(len(n48_cycles) == 64 and [d['restart_cycle'] for d in n48_cycles] == list(range(64)),
            'Newton 48 true-residual cycles missing/duplicated')
    ecycles = [d['environments'][0] for d in n48_cycles]
    active = [e for e in ecycles if e['diagnostics'][3] > 0]
    require(len(active) == 3 and all(e['solver_status_code'] == 0 and e['operator_status_code'] == 0 for e in active),
            'Newton 48 FGMRES did not complete exactly three healthy cycles')
    fgmres_iters = sum(int(e['diagnostics'][3]) for e in active)
    require(fgmres_iters == 44 and active[-1]['diagnostics'][2] == 1,
            'Newton 48 active FGMRES iteration/convergence evidence differs')
    rhs = active[-1]['rhs_l2']
    true = active[-1]['true_l2']
    linear_ratio = true / rhs
    require(all(d['environments'][0]['solver_status_code'] == 10 and
                d['environments'][0]['operator_status_code'] == 4
                for d in cycles if d['newton_iteration'] in range(49, 64)),
            'later residual-cycle rows are not failure-gated code10/operator4')

    base = fail['base_l2']
    trial = fail['trial_l2']
    final_delta = trial - base
    final_relative = final_delta / base
    alpha = fail['trial_alpha']
    require(alpha > 0 and math.isfinite(alpha), 'missing last attempted Armijo alpha')
    require(fail['line_search_raw'][0] == 0 and fail['line_search_raw'][2] == 0,
            'line-search failure vector does not show zero selected alpha/count')

    # Exact rollback evidence: the rejected diagnostic surface is identical to accepted bent state.
    accepted_obj = run / 'output/accepted_step_000001_bent.obj'
    rejected_obj = run / 'output/rejected_step_diagnostic_000002_bent.obj'
    accepted_hash, rejected_hash = sha256(accepted_obj), sha256(rejected_obj)
    require(accepted_hash == rejected_hash, 'rejected bent OBJ differs from accepted prior state')
    held0 = sha256(run / 'output/accepted_step_000001_held_reference.obj')
    held1 = sha256(run / 'output/accepted_step_000002_held_reference.obj')
    require(held0 == held1, 'held-reference state changed between controls')
    material_state = read_json(run / 'output/material_state_000002_bent.json')
    require(material_state['status_code'] == 10 and material_state['solver_certificate_raw_accepted_flag'] == 0,
            'post-failure material-state record lacks failed certificate/status')

    observed_phase_names = sorted(set(r['phase'] for r in observations))
    audit = {
        'schema': 'numi.cardboard.two-stage-coupon409-adjudication.v1',
        'verdict': 'failed_incomplete_not_converged',
        'claim_limit': 'Source-bound native failure adjudication only; not material or physical qualification.',
        'run_dir': str(run),
        'source_binding': {
            'snapshot': build['source_snapshot'],
            'frozen_source_sha256': sha256(frozen_path),
            'source_file_count_verified': len(build['source_sha256']),
            'source_build_manifest_sha256': sha256(source_build_path),
            'build_artifact_count_verified': len(build['artifacts_sha256']),
            'build_metadata_count_verified': len(build.get('build_metadata_sha256', {})),
            'executable_sha256': build['artifacts_sha256'].get(str(Path(build['build_dir']) / 'matter/numi-matter-cardboard-probe')),
            'metallib_sha256': build['artifacts_sha256'].get(str(Path(build['build_dir']) / 'matter/shaders/NumiMatter.metallib')),
            'source_build_success': build['success'],
            'source_inputs_unchanged': receipt['source_inputs_unchanged'],
            'changed_source_files': build.get('changed_sources', []),
            'platform_limitation': build.get('limitation'),
        },
        'run_binding': {
            'command_argv': pred['argv'],
            'environment': pred['environment'],
            'prediction_sha256': sha256(run / 'prediction.json'),
            'receipt_sha256': sha256(run / 'receipt.json'),
            'exit_code': receipt['exit_code'],
            'timed_out': receipt['timed_out'],
            'wall_seconds': receipt['wall_seconds'],
            'finished_at_utc': receipt['finished_at_utc'],
            'world_fingerprint': result['compiled_world_fingerprint'],
            'materials': material_hashes,
            'result_sha256': sha256(run / 'output/result.json'),
            'output_manifest_sha256': sha256(run / 'output/manifest.json'),
            'observations_sha256': sha256(run / 'output/observations.csv'),
            'stderr_sha256': sha256(run / 'stderr.log'),
            'stdout_sha256': sha256(run / 'stdout.log'),
            'rejected_conditioning_manifest_sha256': sha256(reject_manifest_path),
            'rejected_conditioning_files': rejected_files,
        },
        'protocol': {
            'planned_macro_controls': planned,
            'planned_macro_control_count': planned_count,
            'observed_macro_step_indices': [0, 1],
            'observed_rows': len(observations),
            'observed_phases': observed_phase_names,
            'accepted_bent_controls': 1,
            'failed_bent_control': 1,
            'held_reference_controls_accepted_as_noop': 2,
            'completed_later_hold_unload_relax_release_controls': 0,
            'physical_validation': result['physical_validation'],
        },
        'failure_localization': {
            'first_line_search_failure_newton_iteration_zero_based': 48,
            'preceding_selected_newton_updates': 48,
            'failed_newton_pretrial_status_code': fail['pretrial_status']['code'],
            'failed_newton_last_trial_status_code': fail['trial_status']['code'],
            'failed_newton_result_status_code': fail['solver_status_code'],
            'failed_newton_failing_index': fail['solver_status']['failing_index'],
            'trial_outcome': fail['trial_outcome'],
            'candidate_count_exhausted': int(fail['solver_status']['diagnostics'][3]),
            'last_tested_alpha': alpha,
            'selected_alpha': fail['selected_alpha'],
            'minimum_J': fail['minimum_J'],
            'base_raw_residual_impulse_norm': base,
            'last_tested_raw_residual_impulse_norm': trial,
            'last_minus_base_raw_residual': final_delta,
            'last_minus_base_relative': final_relative,
            'solver_terminal_is_local_material_invalid': False,
            'local_material_scope_note': ('Pretrial solver and final recorded trial statuses are clear; the run does not contain every per-tetrahedron projection for every line-search candidate, so it cannot establish that every candidate projection was valid. No recorded terminal status identifies a local material failure.'),
            'later_gated_newton_records': 15,
            'later_gated_env0_true_residual_cycle_slots': 15 * 64,
            'later_gated_status': {'solver_status_code': 10, 'operator_status_code': 4},
        },
        'linear_solve': {
            'newton_iteration': 48,
            'active_fgmres_cycles': 3,
            'active_fgmres_iterations': fgmres_iters,
            'true_residual_norm_impulse': true,
            'rhs_norm_impulse': rhs,
            'true_residual_over_rhs': linear_ratio,
            'all_active_cycles_solver_status_code': 0,
            'all_active_cycles_operator_status_code': 0,
            'converged_flag_on_last_active_cycle': active[-1]['diagnostics'][2],
            'interpretation': 'Inner FGMRES returned success under its forcing condition; this does not satisfy or certify the outer nonlinear equilibrium criterion.',
        },
        'rollback': {
            'accepted_bent_obj': str(accepted_obj),
            'rejected_diagnostic_bent_obj': str(rejected_obj),
            'accepted_bent_obj_sha256': accepted_hash,
            'rejected_diagnostic_bent_obj_sha256': rejected_hash,
            'byte_identical': True,
            'held_reference_obj_hashes_equal': held0 == held1,
            'post_failure_material_status_code': material_state['status_code'],
            'post_failure_certificate_raw_accepted_flag': material_state['solver_certificate_raw_accepted_flag'],
        },
        'recommended_bounded_rerun_capture': {
            'scope': 'same source123 physics and command; fresh unique capture paths only',
            'operator_selector': 'NM_FEM_OPERATOR_FD_SELECTOR=1:48',
            'operator_capture_path': 'NM_FEM_OPERATOR_FD_CAPTURE_PATH=<fresh unique JSONL path>',
            'retain': {
                'NM_FEM_RESIDUAL_ARMIJO_ROOT': 'all',
                'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT': '1',
                'NM_FEM_TRUE_RESIDUAL_ROOT': '1',
                'NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR': '48',
                'NM_FEM_TRUE_RESIDUAL_CERTIFY': '1',
                'NM_FEM_TRUE_RESIDUAL_RESTART': '0',
            },
            'reason': 'The rich selected-Newton operator capture records center/per-tet local projection and paired operator data immediately before the Newton correction; selecting step 1, Newton 48 targets the pretrial state at the first failure. The current trace does not identify all 24 candidate-local outcomes.',
            'limits': 'Diagnostic-on/off parity must be checked; selectors do not make the failed run converged. This run has no self-contact or cohesive interaction, so contact CCD tracing is not the relevant observer.',
        },
        'event_counts': {
            'armijo_records': len(armijo),
            'fgmres_true_residual_records': len(cycles),
            'newton48_active_cycle_restarts': [0, 1, 2],
            'newton48_static_tail_cycles': 61,
            'newton49_to63_gated_cycle_rows': 15 * 64,
        },
        'evidence_hashes': {
            'run_receipt': sha256(run / 'receipt.json'),
            'prediction': sha256(run / 'prediction.json'),
            'source_build': sha256(source_build_path),
            'frozen_source': sha256(frozen_path),
            'result': sha256(run / 'output/result.json'),
            'output_manifest': sha256(run / 'output/manifest.json'),
            'observations': sha256(run / 'output/observations.csv'),
            'material_state': sha256(run / 'output/material_state_000002_bent.json'),
            'stderr': sha256(run / 'stderr.log'),
            'stdout': sha256(run / 'stdout.log'),
            'compiled_nmatterpack': sha256(run / 'output/compiled.nmatterpack'),
            'mesh_json': sha256(run / 'output/mesh.json'),
        },
    }
    out_text = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        out = args.output.resolve()
        require(not out.exists(), f'output destination already exists: {out}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_text, encoding='utf-8')
    else:
        sys.stdout.write(out_text)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'audit failed: {e}', file=sys.stderr)
        raise SystemExit(2)
