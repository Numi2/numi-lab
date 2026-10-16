#!/usr/bin/env python3
"""Independent byte-level audit of publication030; reads inputs, writes only report.json."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
OUT = B / 'analysis/publication030-independent-byte-audit-001'
SUMMARY = D / 'summary.json'
OLD_SUMMARY = D / 'summary-reference029.json'
BINDING = D / 'publication-binding-reference030.json'
OLD_BINDING = D / 'publication-binding-reference029.json'
OLD_PATHSPEC = B / 'publication-reference029-pathspec.nul'
PATHSPEC = B / 'publication-reference030-pathspec.nul'
FROZEN_ROOT = B / 'snapshots/source-checkpoint-warm-breakdown-173'
FROZEN_MANIFEST = FROZEN_ROOT / 'frozen-source.json'
BUILD_META = R / 'build-cardboard-provenance-173/source-build.json'


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def count_file(path: Path) -> int:
    size = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            size += len(block)
    return size


def safe_rel(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() or not p.parts or any(part in ('..', '') for part in p.parts):
        raise ValueError(f'unsafe relative path: {raw!r}')
    return p


def compare_streams(left: BinaryIO, right: BinaryIO) -> bool:
    while True:
        a = left.read(1024 * 1024)
        b = right.read(1024 * 1024)
        if a != b:
            return False
        if not a:
            return True


def pathspec_entries(path: Path) -> tuple[bytes, list[str]]:
    raw = path.read_bytes()
    if not raw.endswith(b'\0'):
        raise ValueError(f'{path}: missing trailing NUL')
    parts = raw[:-1].split(b'\0')
    if any(not p for p in parts):
        raise ValueError(f'{path}: empty path entry')
    decoded = [p.decode('utf-8', errors='strict') for p in parts]
    return raw, decoded


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    assert not (OUT / 'report.json').exists(), 'refuse to overwrite an existing audit report'
    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    checks: dict[str, object] = {}
    input_hashes = {}

    def check(name: str, ok: bool, detail=None):
        checks[name] = {'passed': bool(ok), 'detail': detail}
        if not ok:
            failures.append(name)

    for label, path in (
        ('summary', SUMMARY), ('historical_summary', OLD_SUMMARY), ('binding', BINDING),
        ('old_binding', OLD_BINDING), ('old_pathspec', OLD_PATHSPEC), ('pathspec', PATHSPEC),
        ('frozen_manifest', FROZEN_MANIFEST), ('build_metadata', BUILD_META),
        ('archive_script', B / 'archive_reference030.py'),
        ('payload_preflight', B / 'analysis/publication030-payload-preflight-001/report.json'),
    ):
        input_hashes[label] = {'path': str(path), 'exists': path.is_file(),
                               'sha256': sha_file(path) if path.is_file() else None,
                               'bytes': path.stat().st_size if path.is_file() else None}

    binding = load_json(BINDING)
    summary = load_json(SUMMARY)
    old_summary = load_json(OLD_SUMMARY)
    old_binding = load_json(OLD_BINDING)
    frozen = load_json(FROZEN_MANIFEST)
    build = load_json(BUILD_META)
    preflight = load_json(B / 'analysis/publication030-payload-preflight-001/report.json')

    old_spec_bytes, old_spec = pathspec_entries(OLD_PATHSPEC)
    spec_bytes, spec = pathspec_entries(PATHSPEC)
    old_summary_bytes = OLD_SUMMARY.read_bytes()
    summary_bytes = SUMMARY.read_bytes()
    binding_bytes = BINDING.read_bytes()

    # Bind the historical prefix and prior publication exactly.
    check('historical_summary_hash', sha_bytes(old_summary_bytes) == binding['historical_prefix']['sha256'],
          {'expected': binding['historical_prefix']['sha256'], 'observed': sha_bytes(old_summary_bytes)})
    check('historical_summary_record_count', len(old_summary.get('records', [])) == 1524,
          {'observed': len(old_summary.get('records', []))})
    check('summary_historical_prefix_exact', summary['records'][:1524] == old_summary['records'],
          {'current_records': len(summary.get('records', [])), 'prefix_records': 1524})
    check('prior_binding_digest', sha_file(OLD_BINDING) == binding['prior_binding_sha256'],
          {'expected': binding['prior_binding_sha256'], 'observed': sha_file(OLD_BINDING)})
    check('prior_pathspec_digest', sha_bytes(old_spec_bytes) == binding['prior_pathspec_sha256'],
          {'expected': binding['prior_pathspec_sha256'], 'observed': sha_bytes(old_spec_bytes)})
    check('prior_pathspec_entry_count', len(old_spec) == binding['prior_pathspec_entries'] == 8376,
          {'observed': len(old_spec), 'binding': binding['prior_pathspec_entries']})
    check('summary_historical_snapshot_pointer',
          summary.get('historical_summary_snapshots', {}).get('reference029') ==
          {'path': 'summary-reference029.json', 'sha256': sha_bytes(old_summary_bytes)},
          summary.get('historical_summary_snapshots', {}).get('reference029'))

    cases = binding['cases']
    new_records = summary['records'][1524:]
    check('new_record_count_and_order', len(cases) == 43 and len(new_records) == 43 and
          [r.get('case') for r in new_records] == cases,
          {'binding_cases': len(cases), 'summary_new_records': len(new_records)})
    check('summary_total_records', len(summary['records']) == 1567,
          {'observed': len(summary['records'])})
    check('case_uniqueness', len(set(cases)) == len(cases), {'unique': len(set(cases)), 'total': len(cases)})
    forbidden_case_tokens = [c for c in cases if '476' in c or '477' in c]
    check('active_476_477_case_exclusion', not forbidden_case_tokens and
          binding.get('new_host476477_results_excluded') is True,
          {'forbidden_case_entries': forbidden_case_tokens,
           'binding_exclusion_flag': binding.get('new_host476477_results_excluded')})

    # Validate every case payload against its external source and repository copy.
    observed_case_files = observed_raw = observed_gzip = 0
    observed_raw_bytes = observed_gzip_bytes = 0
    expected_payload_paths: set[str] = set()
    payload_errors: list[dict[str, str]] = []
    for record in new_records:
        case = record['case']
        case_rel = safe_rel(case)
        case_root = (B / case_rel).resolve(strict=True)
        if not case_root.is_dir():
            payload_errors.append({'case': case, 'error': 'external case root is not a directory'})
            continue
        check(f'physical_qualification_false:{case}', record.get('physical_box_qualification') is False,
              record.get('physical_box_qualification'))
        for row in record.get('files', []):
            observed_case_files += 1
            name = row.get('name', '')
            try:
                rel = safe_rel(name)
                src_expected = case_root / rel
                src_declared = Path(row['external_path'])
                if src_declared != src_expected:
                    raise ValueError('external_path does not equal case-root/name')
                if not src_expected.is_file() or src_expected.is_symlink():
                    raise ValueError('external source missing, nonregular, or symlink')
                if sha_file(src_expected) != row['sha256'] or count_file(src_expected) != row['bytes']:
                    raise ValueError('external source hash or byte count mismatch')

                raw_copy = row.get('repository_copy')
                compressed_copy = row.get('repository_compressed_copy')
                if bool(raw_copy) == bool(compressed_copy):
                    raise ValueError('record must name exactly one raw or compressed repository copy')
                if raw_copy:
                    target_rel = safe_rel(raw_copy)
                    target = D / target_rel
                    expected_payload_paths.add(PREFIX + target_rel.as_posix())
                    if not target.is_file() or target.is_symlink():
                        raise ValueError('raw repository copy missing, nonregular, or symlink')
                    if target.stat().st_size != row['bytes'] or sha_file(target) != row['sha256']:
                        raise ValueError('raw repository copy size/hash mismatch')
                    with src_expected.open('rb') as a, target.open('rb') as b:
                        if not compare_streams(a, b):
                            raise ValueError('raw repository copy differs bytewise from external source')
                    observed_raw += 1
                    observed_raw_bytes += row['bytes']
                else:
                    target_rel = safe_rel(compressed_copy)
                    target = D / target_rel
                    expected_payload_paths.add(PREFIX + target_rel.as_posix())
                    if not target.is_file() or target.is_symlink():
                        raise ValueError('gzip repository copy missing, nonregular, or symlink')
                    compressed_size = target.stat().st_size
                    compressed_hash = sha_file(target)
                    if compressed_size != row['compressed_bytes'] or compressed_hash != row['compressed_sha256']:
                        raise ValueError('gzip repository copy size/hash mismatch')
                    import gzip as gzip_module
                    with gzip_module.open(target, 'rb') as restored, src_expected.open('rb') as original:
                        if not compare_streams(restored, original):
                            raise ValueError('gzip-restored bytes differ from external source')
                    with gzip_module.open(target, 'rb') as restored:
                        h = hashlib.sha256()
                        n = 0
                        for block in iter(lambda: restored.read(1024 * 1024), b''):
                            n += len(block)
                            h.update(block)
                    if n != row['bytes'] or h.hexdigest() != row['sha256']:
                        raise ValueError('gzip-restored size/hash mismatch')
                    observed_gzip += 1
                    observed_gzip_bytes += compressed_size
            except Exception as exc:  # record all independent source/payload failures
                payload_errors.append({'case': case, 'name': name, 'error': f'{type(exc).__name__}: {exc}'})

    check('payload_sources_and_copies', not payload_errors, payload_errors[:20])
    check('payload_file_counts', observed_case_files == binding['copied_files'] == 2278 and
          observed_raw == binding['raw_files'] == 2202 and observed_gzip == binding['gzip_files'] == 76,
          {'observed_total': observed_case_files, 'observed_raw': observed_raw, 'observed_gzip': observed_gzip,
           'binding': [binding['copied_files'], binding['raw_files'], binding['gzip_files']]})
    check('preflight_matches_binding', preflight.get('cases') == 43 and preflight.get('payloads') == 2278 and
          preflight.get('raw') == 2202 and preflight.get('gzip') == 76 and
          preflight.get('summary_sha256') == sha_bytes(summary_bytes) and
          preflight.get('binding_sha256') == sha_bytes(binding_bytes) and
          preflight.get('pathspec_sha256') == sha_bytes(spec_bytes) and
          preflight.get('pathspec_entries') == 10656,
          {'preflight': preflight})

    # Publication pathspec must preserve exact previous bytes and enumerate only new payloads plus two refs.
    prefix_ok = spec_bytes.startswith(old_spec_bytes)
    appended = spec[len(old_spec):] if prefix_ok else []
    expected_added = expected_payload_paths | {
        PREFIX + 'summary-reference029.json',
        PREFIX + 'publication-binding-reference030.json',
    }
    check('pathspec_prior_bytes_prefix', prefix_ok,
          {'old_bytes': len(old_spec_bytes), 'new_bytes': len(spec_bytes)})
    check('pathspec_unique_entries', len(spec) == len(set(spec)), {'entries': len(spec), 'unique': len(set(spec))})
    check('pathspec_entries_exact', len(spec) == 10656 and len(appended) == 2280 and
          set(appended) == expected_added and len(expected_added) == 2280,
          {'total': len(spec), 'old': len(old_spec), 'appended': len(appended),
           'expected_appended': len(expected_added),
           'missing': sorted(expected_added - set(appended))[:20],
           'unexpected': sorted(set(appended) - expected_added)[:20]})
    check('pathspec_excludes_476_477_native_results',
          not any(('native476' in p or 'native477' in p or 'native-476' in p or 'native-477' in p) for p in appended),
          [p for p in appended if ('native476' in p or 'native477' in p or 'native-476' in p or 'native-477' in p)][:10])

    # Independently check all 804 selected source hashes in snapshot, live checkout, and build provenance.
    source_hashes = frozen.get('source_sha256', {})
    build_source_hashes = build.get('source_sha256', {})
    snapshot_errors: list[dict[str, str]] = []
    checkout_errors: list[dict[str, str]] = []
    for raw_path, expected in source_hashes.items():
        try:
            rel = safe_rel(raw_path)
            snapshot_file = FROZEN_ROOT / rel
            if not snapshot_file.is_file() or snapshot_file.is_symlink() or sha_file(snapshot_file) != expected:
                snapshot_errors.append({'path': raw_path, 'expected': expected,
                                        'observed': sha_file(snapshot_file) if snapshot_file.is_file() else 'MISSING'})
            checkout_file = R / rel
            if not checkout_file.is_file() or checkout_file.is_symlink() or sha_file(checkout_file) != expected:
                checkout_errors.append({'path': raw_path, 'expected': expected,
                                        'observed': sha_file(checkout_file) if checkout_file.is_file() else 'MISSING'})
        except Exception as exc:
            snapshot_errors.append({'path': raw_path, 'expected': expected, 'observed': f'{type(exc).__name__}: {exc}'})
    check('source173_manifest_has_804_hashes', len(source_hashes) == 804,
          {'observed': len(source_hashes)})
    check('source173_snapshot_files_match', not snapshot_errors, snapshot_errors[:20])
    check('source173_current_checkout_files_match', not checkout_errors, checkout_errors[:20])
    check('source173_build_source_map_matches', build_source_hashes == source_hashes,
          {'build_count': len(build_source_hashes), 'frozen_count': len(source_hashes)})
    check('source173_build_success_and_snapshot', build.get('success') is True and
          Path(build.get('source_snapshot', '')).resolve() == FROZEN_ROOT.resolve(),
          {'success': build.get('success'), 'source_snapshot': build.get('source_snapshot')})
    check('source173_snapshot_digest', sha_file(FROZEN_MANIFEST) == binding['source173_frozen_sha256'],
          {'expected': binding['source173_frozen_sha256'], 'observed': sha_file(FROZEN_MANIFEST)})
    check('source173_build_metadata_digest', sha_file(BUILD_META) == binding['source173_original_host_build_sha256'],
          {'expected': binding['source173_original_host_build_sha256'], 'observed': sha_file(BUILD_META)})
    artifact_errors = []
    for raw_path, expected in build.get('artifacts_sha256', {}).items():
        artifact = Path(raw_path)
        if not artifact.is_file() or artifact.is_symlink() or sha_file(artifact) != expected:
            artifact_errors.append({'path': raw_path, 'expected': expected,
                                    'observed': sha_file(artifact) if artifact.is_file() else 'MISSING'})
    check('source173_build_artifacts_match', not artifact_errors, artifact_errors[:20])

    # Check the publication's evidence boundary rather than infer qualification from helper/build checks.
    expected_axis_statuses = {
        'solver_correctness': 'partial_component_evidence; complete_nonlinear_solve_unqualified',
        'numerical_convergence': 'unqualified',
        'component_calibration': 'partial_source_data_only',
        'analogue_validation': 'unqualified_for_requested_full_sequence',
        'physical_assembly_in_simulation': 'unqualified',
        'crush_impact_prediction': 'unexecuted',
    }
    actual_axes = binding.get('six_axis_verdicts', {})
    axis_statuses = {k: (v or {}).get('status') for k, v in actual_axes.items()}
    check('six_axis_verdict_boundaries', axis_statuses == expected_axis_statuses and
          summary.get('verdicts') == actual_axes and binding.get('physical_validation') is False,
          {'axis_statuses': axis_statuses, 'physical_validation': binding.get('physical_validation')})
    check('no_complete_coupon_or_assembly_claim',
          'complete_nonlinear_solve_unqualified' in axis_statuses.get('solver_correctness', '') and
          axis_statuses.get('numerical_convergence') == 'unqualified' and
          axis_statuses.get('physical_assembly_in_simulation') == 'unqualified' and
          axis_statuses.get('crush_impact_prediction') == 'unexecuted', axis_statuses)

    report = {
        'schema': 'numi.cardboard.publication030-independent-byte-audit.v1',
        'verdict': 'PASS' if not failures else 'FAIL',
        'scope': 'Read-only independent verification of publication030 summary history, payload bytes, pathspec extension, selected source173 hashes, and stated evidence boundaries. No source or archive files were modified.',
        'binding_sha256': sha_file(BINDING),
        'summary_sha256': sha_file(SUMMARY),
        'pathspec_sha256': sha_file(PATHSPEC),
        'source173_frozen_manifest_sha256': sha_file(FROZEN_MANIFEST),
        'source173_build_metadata_sha256': sha_file(BUILD_META),
        'historical_prefix': {'records': 1524, 'summary_sha256': sha_file(OLD_SUMMARY),
                              'old_pathspec_entries': len(old_spec), 'current_summary_prefix_exact': checks['summary_historical_prefix_exact']['passed'],
                              'prior_pathspec_bytes_exact_prefix': checks['pathspec_prior_bytes_prefix']['passed']},
        'new_publication_payloads': {'cases': len(cases), 'files': observed_case_files,
                                     'raw_files': observed_raw, 'gzip_files': observed_gzip,
                                     'raw_bytes': observed_raw_bytes, 'compressed_payload_bytes': observed_gzip_bytes,
                                     'pathspec_entries': len(spec), 'appended_pathspec_entries': len(appended)},
        'source173_selected_hashes': {'expected_files': len(source_hashes),
                                      'snapshot_mismatches': len(snapshot_errors),
                                      'checkout_mismatches': len(checkout_errors),
                                      'build_source_map_matches': build_source_hashes == source_hashes,
                                      'build_artifact_mismatches': len(artifact_errors)},
        'six_axis_verdicts': axis_statuses,
        'active_476_477_results_included': bool(forbidden_case_tokens),
        'checks': checks,
        'failed_checks': failures,
        'input_hashes': input_hashes,
    }
    out = OUT / 'report.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'verdict': report['verdict'], 'failed_checks': failures,
                      'payloads': report['new_publication_payloads'],
                      'source173': report['source173_selected_hashes'],
                      'report_sha256': sha_file(out), 'report_path': str(out)}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
