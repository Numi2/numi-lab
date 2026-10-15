#!/usr/bin/env python3
"""Source-bound comparison of the four retained tiled-cohesive diagnostic arms."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

RUN = Path('/Users/home/cardboard-implementation-20261014/native/ordered-cohesive-pages-stage-diagnostics-287')
OUT = Path('/Users/home/cardboard-implementation-20261014/analysis/tiled-cohesive-stage-diagnostics-287')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path):
    return json.loads(path.read_text())

def require(ok: bool, message: str):
    if not ok:
        raise ValueError(message)

def persistent_state_equal(a, b):
    return a['accepted_state_vector_sha256'] == b['accepted_state_vector_sha256']

def summarize_trace(path: Path):
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith('fem_residual_armijo_iteration='):
            continue
        records.append(json.loads(line.split('=', 1)[1]))
    return records

def compare_arm(face_count: int):
    dirs = {layout: RUN/'diagnostic-state'/f'face-count-{face_count}'/layout
            for layout in ('legacy', 'tiled')}
    results = {layout: load(path/'arm-result.json') for layout, path in dirs.items()}
    require(results['legacy']['face_count'] == face_count and
            results['tiled']['face_count'] == face_count, 'face count binding mismatch')
    semantic_keys = ('status', 'last_stage', 'native_transactions_executed', 'error')
    require(all(results['legacy'][k] == results['tiled'][k] for k in semantic_keys),
            f'legacy/tiled arm outcomes differ at face count {face_count}')
    tx_reports = []
    for tx in range(results['legacy']['native_transactions_executed']):
        paths = {layout: dirs[layout]/f'transaction-{tx:06d}.json'
                 for layout in ('legacy', 'tiled')}
        records = {layout: load(path) for layout, path in paths.items()}
        lj, tj = records['legacy'], records['tiled']
        require(lj == tj,
                'complete transaction metadata differs between layouts')
        before = {layout: dirs[layout]/records[layout]['before_file']
                  for layout in dirs}
        after = {layout: dirs[layout]/records[layout]['after_file']
                 for layout in dirs}
        cross_before_equal = before['legacy'].read_bytes() == before['tiled'].read_bytes()
        cross_after_equal = after['legacy'].read_bytes() == after['tiled'].read_bytes()
        require(cross_before_equal and cross_after_equal,
                f'legacy/tiled raw snapshot bytes differ, face count {face_count}, tx {tx}')
        before_after_rollback = persistent_state_equal(lj['before'], lj['after'])
        tiled_rollback = persistent_state_equal(tj['before'], tj['after'])
        require(before_after_rollback == tiled_rollback,
                'accepted-state rollback comparison differs between layouts')
        if not lj['accepted']:
            require(before_after_rollback and tiled_rollback,
                    'rejected transaction changed accepted-state vector digest')
        require(persistent_state_equal(lj['before'], tj['before']) and
                persistent_state_equal(lj['after'], tj['after']),
                'accepted-state vector hashes differ between layouts')
        tx_reports.append({
            'attempt_index': tx,
            'requested_control_step': lj['requested_control_step'],
            'accepted': lj['accepted'],
            'status_before': lj['before']['statuses'][0]['code'],
            'status_after': lj['after']['statuses'][0]['code'],
            'failure_index_after': lj['after']['statuses'][0]['failing_index'],
            'diagnostics_after': lj['after']['statuses'][0]['diagnostics'],
            'raw_before_sha256': sha(before['legacy']),
            'raw_after_sha256': sha(after['legacy']),
            'raw_snapshot_bytes_equal_across_layouts': True,
            'accepted_state_hashes_equal_across_layouts': True,
            'persistent_accepted_state_unchanged_on_rejection':
                (not lj['accepted']) and before_after_rollback,
            'accepted_state_hashes_before_equal_after': before_after_rollback,
            'raw_before_after_bytes_equal': before['legacy'].read_bytes() == after['legacy'].read_bytes(),
        })
    logs = {}
    for stage in ('activation', 'closure'):
        lp = dirs['legacy']/f'fixture-{stage}.log'
        tp = dirs['tiled']/f'fixture-{stage}.log'
        if lp.exists() != tp.exists():
            raise ValueError(f'layout lacks paired {stage} trace')
        if not lp.exists():
            continue
        lr, tr = summarize_trace(lp), summarize_trace(tp)
        require(lr == tr, f'full-contact Armijo trace differs in {stage}, face count {face_count}')
        failed = [r for r in lr if any(not e['accepted'] or e['solver_status_code'] != 0
                                      for e in r['environments'])]
        logs[stage] = {'legacy_sha256': sha(lp), 'tiled_sha256': sha(tp),
                       'armijo_record_count': len(lr),
                       'armijo_records_equal': True,
                       'first_failed_or_statused_iteration': (
                           {'newton_iteration': failed[0]['newton_iteration'],
                            'environment': failed[0]['environments'][0]}
                           if failed else None)}
    return {'face_count': face_count,
            'status': results['legacy']['status'],
            'last_stage': results['legacy']['last_stage'],
            'transactions': tx_reports,
            'trace_parity': logs,
            'arm_result_sha256': {k: sha(dirs[k]/'arm-result.json') for k in dirs},
            'semantic_outcome_equal_across_layouts': True}

def main():
    receipt = load(RUN/'receipt.json')
    source_build = load(RUN/'inputs/external/source-build.json')
    prediction = load(RUN/'prediction.json')
    require(receipt['schema'] == 'numi.cardboard.native-run-receipt.v2', 'receipt schema mismatch')
    require(receipt['source_inputs_unchanged'] is True and not receipt['timed_out'],
            'run source integrity/termination was not clean')
    require(receipt['exit_code'] == 1, 'unexpected run exit status; expected retained failure')
    require(prediction.get('executed_build_verified') is True, 'prediction lacks verified build')
    require(source_build['source_snapshot'].endswith('source-captured-point-and-contact-scope-086'),
            'source snapshot mismatch')
    source_root = RUN/'inputs/source'
    source_mismatches = []
    for relative, expected in source_build['source_sha256'].items():
        path = source_root/relative
        if not path.is_file() or sha(path) != expected:
            source_mismatches.append(relative)
    require(not source_mismatches, f'{len(source_mismatches)} retained source files differ from build manifest')
    build_root = Path(source_build['build_dir'])
    runtime_root = RUN/'inputs/runtime'
    runtime_checks = {}
    used_paths = [
        Path(source_build['build_dir'])/'matter/numi-matter-cardboard-full-contact-armijo-check',
        Path(source_build['build_dir'])/'lib/libmetalrobo.dylib',
        Path(source_build['build_dir'])/'matter/shaders/NumiMatter.metallib',
        Path(source_build['build_dir'])/'matter/shaders/NumiMatterPhysicalStateDigest.metallib',
    ]
    for original in used_paths:
        expected = source_build['artifacts_sha256'].get(str(original))
        require(expected is not None, f'build manifest omits executed artifact {original}')
        copied = runtime_root/original.relative_to(build_root)
        require(copied.is_file() and sha(copied) == expected,
                f'executed copied artifact does not match build record: {copied}')
        runtime_checks[str(copied)] = {'sha256': sha(copied), 'matches_build_manifest': True}
    material_rel = 'matter/materials/hajali2009_liner_hill_ideal.nmatter'
    require(material_rel in source_build['source_sha256'], 'build manifest omits selected material')
    arms = [compare_arm(1), compare_arm(3)]
    zero = (RUN/'diagnostic-state/zero-face-control.txt').read_text()
    require('status=failed_closed_before_transaction' in zero,
            'zero-face negative control did not fail closed')
    tile_summary = load(RUN/'diagnostic-state/tiled-parity-result.json')
    require(tile_summary['comparison_status'] == 'failed_prefix_partial_comparison',
            'run summary is not the expected retained partial comparison')
    report = {
        'schema': 'numi.cardboard.tiled_cohesive_stage_diagnostics_audit.v1',
        'status': 'layout_parity_on_retained_prefix; four-arm comparison incomplete',
        'run_path': str(RUN),
        'receipt': receipt,
        'source_snapshot': source_build['source_snapshot'],
        'source_build_source_sha256': source_build['source_sha256'],
        'prediction': prediction,
        'zero_face_control': 'failed_closed_before_transaction; expected negative control',
        'arms': arms,
        'interpretation': {
            'expected_negative_control_first': 'zero-face control is intentionally run before the four arms and fails closed before a transaction',
            'arm_execution_order': [
                {'face_count': 1, 'layout': 'legacy'},
                {'face_count': 1, 'layout': 'tiled'},
                {'face_count': 3, 'layout': 'legacy'},
                {'face_count': 3, 'layout': 'tiled'}
            ],
            'earliest_unexpected_failed_action': 'face_count=1 legacy closure_validation, requested control step 2: code 10 after its activation step 1 was accepted; matching tiled closure fails identically',
            'three_face_failed_action': 'face_count=3 activation_validation, requested control step 1: code 6, failing index 78, no accepted control step; legacy and tiled raw before/after snapshots are byte-identical',
            'one_face_failure': 'activation control step 1 is accepted identically in both layouts; closure control step 2 then rejects code 10 after Newton 31 with residual 1.69753184e-4 and certificate invalid',
            'rollback': 'rejected transactions preserve every reported accepted-state vector digest; raw after snapshots differ from before because transaction status/certificate diagnostics are part of the serialized snapshot',
            'layout_result': 'all paired raw transaction snapshots and all paired Armijo records are identical; this supports observational parity only on the prefixes actually exercised',
            'limits': ['no complete four-arm parity result: all tested arms end as failed prefixes', 'three-face failure diagnostic is retained verbatim and not decoded into an unsupported cause', 'the run does not establish closure success, full solver success, or physical validation', 'no repair to tiled layout is justified by these data because legacy exhibits byte-identical failures']
        },
        'build_binding': {
            'source_build_manifest_sha256': sha(RUN/'inputs/external/source-build.json'),
            'source_file_count': len(source_build['source_sha256']),
            'all_retained_source_files_match_build_manifest': True,
            'executed_artifacts_match_build_manifest': runtime_checks,
            'selected_material_source_sha256': source_build['source_sha256'][material_rel],
            'prediction_argv': prediction['argv'],
            'prediction_external_inputs': prediction['external_inputs'],
            'compiler_and_os_limitation': prediction['source_binding_limitation']
        },
        'artifact_sha256': {}
    }
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_paths = [RUN/'receipt.json', RUN/'prediction.json', RUN/'stdout.log',
                      RUN/'stderr.log', RUN/'inputs/external/source-build.json',
                      RUN/'diagnostic-state/tiled-parity-result.json',
                      RUN/'diagnostic-state/zero-face-control.txt']
    artifact_paths.extend(sorted((RUN/'diagnostic-state').glob('face-count-*/*/*')))
    report['artifact_sha256'] = {str(path.relative_to(RUN)): sha(path)
                                 for path in artifact_paths if path.is_file()}
    report_path = OUT/'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    manifest = {
        'schema': 'numi.cardboard.tiled_cohesive_stage_diagnostics_manifest.v1',
        'run_receipt_sha256': sha(RUN/'receipt.json'),
        'prediction_sha256': sha(RUN/'prediction.json'),
        'source_build_manifest_sha256': sha(RUN/'inputs/external/source-build.json'),
        'executed_artifacts_sha256': runtime_checks,
        'all_772_retained_sources_match_build_manifest': True,
        'analysis': {
            'report_sha256': sha(report_path),
            'analyzer_sha256': sha(Path(__file__)),
            'readme_sha256': sha(OUT/'README.md'),
            'status': report['status']
        }
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': report['status'],
                      'arms': [{'face_count': x['face_count'], 'last_stage': x['last_stage'],
                                'transactions': len(x['transactions'])} for x in arms],
                      'report': str(report_path)}, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'audit failed closed: {exc}', file=sys.stderr)
        raise
