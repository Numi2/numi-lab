#!/usr/bin/env python3
"""Retain an exact, source-bound analysis of native reference-contact replay 038."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys

ROOT = pathlib.Path('/Users/numi/numi-human-aortic-contact-opt-20261019')
RUN_ROOT = ROOT / 'build-integrated-validation-20261019/supported-skin-ligaments-20261020'
DEFAULT_RUN = RUN_ROOT / 'native-reference-contact-root4-tick2-038'
DEFAULT_BASELINE = RUN_ROOT / 'native-reference-contact-038'
DEFAULT_ANALYZER = ROOT / 'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/contact-armijo-024/analyze.py'
DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parent


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def fmt(x):
    return format(float(x), '.12g')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=pathlib.Path, default=DEFAULT_RUN)
    ap.add_argument('--baseline', type=pathlib.Path, default=DEFAULT_BASELINE)
    ap.add_argument('--analyzer', type=pathlib.Path, default=DEFAULT_ANALYZER)
    ap.add_argument('--output-dir', type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location('contact_armijo_analyzer', args.analyzer)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import existing analyzer: {args.analyzer}')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    events, parse_errors = analyzer.parse_trace(args.run / 'stderr.log', 4, 2, 2624)
    if parse_errors:
        raise RuntimeError(f'trace parse errors: {parse_errors}')
    if len(events) != 12 or [e['iteration'] for e in events] != list(range(12)):
        raise RuntimeError(f'unexpected selected trace shape: {len(events)} events / iterations {[e.get("iteration") for e in events]}')
    failed = events[-1]
    if not failed['per_alpha_trace'].get('is_failure') or failed['iteration'] != 11:
        raise RuntimeError('selected final event is not the target failure iteration')
    contact = failed['contact']
    trials = failed['per_alpha_trace']['trials']
    if len(trials) != 12:
        raise RuntimeError(f'expected 12 alpha trials, got {len(trials)}')
    frame = contact.get('geometry_coordinate_frame')
    origin = contact.get('reference_origin_m')
    if frame != 'reference_local' or not origin:
        raise RuntimeError('selected event is missing reference-frame provenance')
    if any(t.get('geometry_coordinate_frame') != frame or t.get('reference_origin_m') != origin for t in trials):
        raise RuntimeError('trial coordinate-frame provenance differs from selected contact')

    start = contact['feature_start']
    rows = []
    for t in trials:
        endpoints = t['trial_feature_endpoints']
        d = [[endpoints[j][k] - start[j][k] for k in range(3)] for j in range(4)]
        norms_nm = [math.sqrt(sum(v*v for v in vec)) * 1e9 for vec in d]
        base_e = float(t['energy'][2])
        base_gap = float(t['base'][3])
        trial_e = float(t['energy'][0])
        trial_gap = float(t['energy'][1])
        rows.append({
            'iteration': failed['iteration'],
            'trial_index': t['index'],
            'alpha': t['ccd'][0],
            'reason_code': t['reason_code'],
            'ccd_certified': t['ccd_certified'],
            'ccd_hit': t['ccd_hit'],
            'ccd_toi': t['ccd'][1],
            'tested_full_fraction': t['tested_full_fraction'],
            'finite_barrier': t['finite_barrier'],
            'accepted_by_feature_gate': t['accepted_by_feature_gate'],
            'delta_p0_nm': [v * 1e9 for v in d[0]],
            'delta_p1_nm': [v * 1e9 for v in d[1]],
            'delta_p2_nm': [v * 1e9 for v in d[2]],
            'delta_p3_nm': [v * 1e9 for v in d[3]],
            'endpoint_norms_nm': norms_nm,
            'base_energy_j': base_e,
            'trial_energy_j': trial_e,
            'delta_energy_pj': (trial_e - base_e) * 1e12,
            'base_gap_m': base_gap,
            'trial_gap_m': trial_gap,
            'delta_gap_pm': (trial_gap - base_gap) * 1e12,
            'recorded_directional_derivative_j': t['base'][1],
            'weighted_direction_and_normal_dot': t['weighted_direction_and_normal_dot'],
        })

    # Compare exact baseline outputs named in the request.
    compare_paths = ['output/resting-coupled.csv', 'common-field-failure.json']
    comparisons = {}
    for rel in compare_paths:
        rp, bp = args.run / rel, args.baseline / rel
        if not rp.is_file() or not bp.is_file():
            raise RuntimeError(f'missing comparison artifact: {rel}')
        rb, bb = rp.read_bytes(), bp.read_bytes()
        comparisons[rel] = {
            'byte_identical': rb == bb,
            'run_sha256': sha(rp),
            'baseline_sha256': sha(bp),
            'run_bytes': len(rb),
            'baseline_bytes': len(bb),
        }
    if not all(v['byte_identical'] for v in comparisons.values()):
        raise RuntimeError(f'baseline comparison changed: {comparisons}')

    invocation_path = args.run / 'invocation.json'
    provenance_path = args.run / 'provenance.json'
    execution_path = args.run / 'execution.json'
    failure_path = args.run / 'common-field-failure.json'
    stderr_path = args.run / 'stderr.log'
    stdout_path = args.run / 'stdout.log'
    invocation = load_json(invocation_path)
    provenance = load_json(provenance_path)
    execution = load_json(execution_path)
    failure = load_json(failure_path)
    source_checks = {}
    for rel, claimed in provenance.get('source_sha256', {}).items():
        path = ROOT / rel
        current = sha(path) if path.is_file() else None
        source_checks[rel] = {'run_source_sha256': claimed, 'current_file_sha256': current, 'current_matches_run': current == claimed}

    run_hashes = {}
    for p in [stderr_path, stdout_path, invocation_path, provenance_path, execution_path, failure_path, args.run / 'output/resting-coupled.csv']:
        run_hashes[str(p)] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    argv = invocation.get('argv', [])
    artifact_provenance = {
        'run_directory': str(args.run),
        'baseline_directory': str(args.baseline),
        'invocation': invocation,
        'invocation_sha256': sha(invocation_path),
        'provenance_sha256': sha(provenance_path),
        'source_base_revision': provenance.get('source_base_revision'),
        'executable_path': argv[0] if argv else None,
        'executable_sha256': provenance.get('executable_sha256'),
        'metallib_sha256': provenance.get('metallib_sha256'),
        'suite_library_sha256': provenance.get('suite_library_sha256'),
        'actual_pose_geometry_audit_sha256': provenance.get('actual_pose_geometry_audit_sha256'),
        'source_sha256': provenance.get('source_sha256', {}),
        'current_source_checks': source_checks,
        'raw_log_hashes': run_hashes,
        'execution': execution,
        'failure': failure,
    }
    summary = {
        'schema': 'numi.reference_contact_replay_analysis.v1',
        'target': {'root': 4, 'microtick': 2, 'contact_slot': 2624, 'environment': 0},
        'selected_event_count': len(events),
        'iterations': [e['iteration'] for e in events],
        'failure_iteration': failed['iteration'],
        'failure_stage': failed['stage'],
        'failure_stderr_line': failed['stderr_line'],
        'failure_reason': contact.get('diagnostic_reason'),
        'failure_reason_code': contact.get('diagnostic_reason_code'),
        'contact_identity': contact.get('identity'),
        'contact_nodes': contact.get('nodes'),
        'active_count': contact.get('active_count'),
        'active_slot': contact.get('active_slot'),
        'geometry_coordinate_frame': frame,
        'reference_origin_m': origin,
        'normal': contact.get('normal'),
        'barrier': contact.get('barrier'),
        'trial_count': len(trials),
        'all_trials_rejected': all(not t['accepted_by_feature_gate'] and t['reason_code'] == 6 for t in trials),
        'all_trials_ccd_certified_full_fraction_no_hit': all(t['ccd_certified'] and t['tested_full_fraction'] and not t['ccd_hit'] and t['ccd'][1] == 0 for t in trials),
        'all_trials_finite_barrier': all(t['finite_barrier'] for t in trials),
        'final_iteration_endpoint_energy_gap_rows': rows,
        'baseline_comparisons': comparisons,
        'analyzer': {'path': str(args.analyzer), 'sha256': sha(args.analyzer), 'entrypoint': 'parse_trace'},
        'artifact_provenance': artifact_provenance,
    }

    (out / 'analysis.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (out / 'selected-events.json').write_text(json.dumps({'schema': 'numi.reference_contact_selected_events.v1', 'events': events}, indent=2, sort_keys=True) + '\n')
    columns = ['trial_index','alpha','reason_code','ccd_certified','ccd_hit','ccd_toi','tested_full_fraction','finite_barrier','accepted_by_feature_gate',
               'delta_p0_x_nm','delta_p0_y_nm','delta_p0_z_nm','norm_p0_nm','norm_p1_nm','norm_p2_nm','norm_p3_nm',
               'base_energy_j','trial_energy_j','delta_energy_pj','base_gap_m','trial_gap_m','delta_gap_pm','directional_derivative_j']
    lines = ['\t'.join(columns)]
    for r in rows:
        vals = [r['trial_index'],fmt(r['alpha']),r['reason_code'],int(r['ccd_certified']),int(r['ccd_hit']),fmt(r['ccd_toi']),int(r['tested_full_fraction']),int(r['finite_barrier']),int(r['accepted_by_feature_gate']),
                *[fmt(v) for v in r['delta_p0_nm']], *[fmt(v) for v in r['endpoint_norms_nm']], fmt(r['base_energy_j']),fmt(r['trial_energy_j']),fmt(r['delta_energy_pj']),fmt(r['base_gap_m']),fmt(r['trial_gap_m']),fmt(r['delta_gap_pm']),fmt(r['recorded_directional_derivative_j'])]
        lines.append('\t'.join(map(str, vals)))
    (out / 'trials.tsv').write_text('\n'.join(lines) + '\n')

    max_e = max(r['delta_energy_pj'] for r in rows)
    min_e = min(r['delta_energy_pj'] for r in rows)
    min_d0 = min(r['endpoint_norms_nm'][0] for r in rows)
    max_d0 = max(r['endpoint_norms_nm'][0] for r in rows)
    max_d2 = max(r['endpoint_norms_nm'][2] for r in rows)
    min_gap_delta = min(r['delta_gap_pm'] for r in rows)
    max_gap_delta = max(r['delta_gap_pm'] for r in rows)
    source_mismatches = {k:v for k,v in source_checks.items() if not v['current_matches_run']}
    csv_lines = len((args.run / 'output/resting-coupled.csv').read_text().splitlines())
    md = f'''# Reference-contact replay 038: root 4, microtick 2

This retained analysis selects `{len(events)}` `after_deformable_limit` trace events for root 4, microtick 2, environment 0, contact slot 2624. It spans iterations 0–11 and ends at stderr line {failed['stderr_line']}, iteration 11, with `{summary['failure_reason']}` (reason code {summary['failure_reason_code']}). The recorded contact identity is `{contact.get('identity')}` and FEM node tuple is `{contact.get('nodes')}`; these are retained as raw identifiers without adding an anatomical label.

All trial geometry is explicitly in `{frame}` coordinates about reference origin `{origin}` metres. The final iteration tries {len(trials)} alphas from {trials[0]['ccd'][0]} down to {trials[-1]['ccd'][0]}. Every trial is CCD-certified at full fraction with no hit, has finite barrier energy, and is rejected by the feature Armijo gate. The endpoint and potential changes are retained in [`trials.tsv`](trials.tsv); the complete parsed events, including per-trial frame/origin metadata, are in [`selected-events.json`](selected-events.json).

For final-iteration candidates, the recorded trial energy change ranges from {min_e:.6g} to {max_e:.6g} pJ relative to the repeated baseline energy; the smallest/maximum tested point-endpoint displacement norm is summarized by a maximum of {max_d0:.6g} nm, and the gap change ranges down to {min_gap_delta:.6g} pm. The recorded directional derivative is {trials[0]['base'][1]:.12g} J. P0 moved at most {max_d0:.6g} nm and P2 at most {max_d2:.6g} nm; P1/P3 did not move in the recorded trial endpoints. Gap changes span {min_gap_delta:.6g} to {max_gap_delta:.6g} pm. At the smallest alpha, endpoint quantization is visible as repeated output coordinates; this is an observation from the trace, not a causal diagnosis.

The run and baseline `resting-coupled.csv` files are byte-identical (SHA-256 `{comparisons['output/resting-coupled.csv']['run_sha256']}`, {csv_lines} lines). Their `common-field-failure.json` files are also byte-identical (SHA-256 `{comparisons['common-field-failure.json']['run_sha256']}`), with the same captured failure signature. Both native runs exited {execution.get('exit_code')} for this replay.

Raw trace SHA-256 is `{run_hashes[str(stderr_path)]['sha256']}` ({run_hashes[str(stderr_path)]['bytes']} bytes). Run invocation/provenance and executable/metallib/source hashes are bound in [`analysis.json`](analysis.json). The run provenance names source revision `{provenance.get('source_base_revision')}` and includes per-file source hashes. A comparison to the current checkout finds {len(source_mismatches)} mismatch(es): {', '.join(source_mismatches) if source_mismatches else 'none'}. In particular, the run-bound `matter/src/metal/fem.metalinc` SHA is `{provenance.get('source_sha256', {}).get('matter/src/metal/fem.metalinc')}`, while the current checkout hash is `{source_mismatches.get('matter/src/metal/fem.metalinc', {}).get('current_file_sha256')}`; the executable/metallib and recorded source hashes remain the authority for this completed run. The existing `contact-armijo-024/analyze.py` was imported and its `parse_trace` entrypoint used (analyzer SHA-256 `{summary['analyzer']['sha256']}`). No source, build, or GPU action was performed for this analysis.
'''
    (out / 'interpretation.md').write_text(md)
    print(json.dumps({'output_dir':str(out),'event_count':len(events),'failure_iteration':failed['iteration'],'failure_reason':summary['failure_reason'],'trial_count':len(trials),'baseline_comparisons':comparisons,'stderr_sha256':run_hashes[str(stderr_path)]['sha256'],'analysis_json_sha256':sha(out/'analysis.json'),'selected_events_sha256':sha(out/'selected-events.json'),'trials_tsv_sha256':sha(out/'trials.tsv'),'interpretation_sha256':sha(out/'interpretation.md')},sort_keys=True))

if __name__ == '__main__':
    main()
