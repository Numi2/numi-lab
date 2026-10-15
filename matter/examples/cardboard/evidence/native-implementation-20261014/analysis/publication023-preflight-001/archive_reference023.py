#!/usr/bin/env python3
"""Append publication reference 023, preserving the exact 022 summary prefix."""
from __future__ import annotations
import gzip, hashlib, json, shutil
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
PREFLIGHT = B / 'analysis/publication023-preflight-001'
CASES = json.loads((B / 'publication023-cases-prepared.json').read_text())['cases']
EXCLUDE_PARTS = {'inputs', 'build', 'source', 'candidate', '__pycache__'}
RAW_LIMIT = 262144

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

summary_path = D / 'summary.json'
old_summary_bytes = summary_path.read_bytes()
old_summary = json.loads(old_summary_bytes)
old_binding_path = D / 'publication-binding-reference022.json'
old_binding = json.loads(old_binding_path.read_text())
old_pathspec = B / 'publication-reference022-pathspec.nul'
old_pathspec_bytes = old_pathspec.read_bytes()
assert old_binding['summary_sha256'] == sha_bytes(old_summary_bytes)
assert len(old_summary['records']) == 1282
assert old_binding['physical_validation'] is False
assert len(old_binding['cases']) == 57
assert len(old_pathspec_bytes.split(b'\0')) - 1 == 1187
assert len(set(x for x in old_pathspec_bytes.split(b'\0') if x)) == 1187
assert not (D / 'summary-reference022.json').exists()
assert not (D / 'publication-binding-reference023.json').exists()
assert len(set(CASES)) == len(CASES)
records = {row['case']: row for row in old_summary['records']}
assert len(records) == len(old_summary['records'])
for case in CASES:
    assert case not in records, case
    assert (B / case).is_dir(), case

# Seal the exact 022 state before extending the current summary.
prefix_path = D / 'summary-reference022.json'
prefix_path.write_bytes(old_summary_bytes)
assert prefix_path.read_bytes() == old_summary_bytes
prefix_sha = sha_bytes(old_summary_bytes)

new_rows: list[dict] = []
new_archive_paths: list[str] = []
for case in CASES:
    root = B / case
    if case == 'analysis/publication023-preflight-001':
        assert (root / 'README.md').is_file()
    if case == 'solver/repeated-cauchy-full-coupon-438':
        receipt = json.loads((root / 'receipt.json').read_text())
        assert receipt['execution_started'] is False
        assert receipt['stage'] == 'input-seal'
        assert receipt['reason'] == 'duplicate external input basename'
    if case == 'solver/repeated-cauchy-full-coupon-442':
        receipt = json.loads((root / 'receipt.json').read_text())
        assert receipt['execution_started'] is True and receipt['exit_code'] == 2
    if case == 'analysis/native442-repeated-fallback-adjudication-001':
        assert json.loads((root / 'manifest.json').read_text())['status'] == 'failed_accepted_prefix_only'
    if case == 'analysis/native442-repeated-fallback-adjudication-002':
        assert json.loads((root / 'manifest.json').read_text())['schema'].endswith('.v2')

    rows = []
    for src in sorted(root.rglob('*')):
        rel = src.relative_to(root)
        if not src.is_file():
            continue
        if any(part in EXCLUDE_PARTS or part.startswith('CMakeFiles') for part in rel.parts):
            continue
        assert not src.is_symlink(), str(src)
        raw = src.read_bytes()
        source_sha = sha_bytes(raw)
        target = D / case / rel
        assert not target.exists(), str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {'name': str(rel), 'bytes': len(raw), 'sha256': source_sha,
               'external_path': str(src)}
        if len(raw) <= RAW_LIMIT or src.suffix == '.gz':
            target.write_bytes(raw)
            assert target.stat().st_size == len(raw) and sha_file(target) == source_sha
            row['repository_copy'] = str(target.relative_to(D))
        else:
            packed = gzip.compress(raw, compresslevel=6, mtime=0)
            packed_target = target.with_name(target.name + '.gz')
            packed_target.write_bytes(packed)
            assert sha_bytes(gzip.decompress(packed)) == source_sha
            assert sha_file(packed_target) == sha_bytes(packed)
            row.update(repository_compressed_copy=str(packed_target.relative_to(D)),
                       compressed_bytes=len(packed), compressed_sha256=sha_bytes(packed),
                       restore='gzip decompression recovers original bytes; verify original sha256')
        rows.append(row)
        new_archive_paths.append('matter/examples/cardboard/evidence/native-implementation-20261014/' +
                                 (row.get('repository_copy') or row['repository_compressed_copy']))
    record = {'case': case, 'external_root': str(root), 'physical_box_qualification': False,
              'files': rows}
    for key in ('receipt', 'prediction'):
        p = root / f'{key}.json'
        if p.is_file():
            record[key] = json.loads(p.read_text())
    new_rows.append(record)

# Update only the current index; preserve the entire 022 record list as an exact prefix.
summary = json.loads(old_summary_bytes)
summary.setdefault('historical_summary_snapshots', {})['reference022'] = {
    'path': 'summary-reference022.json', 'sha256': prefix_sha,
}
summary['records'].extend(new_rows)
assert summary['records'][:len(old_summary['records'])] == old_summary['records']
summary['verdicts'] = {
    'solver_correctness': 'component-qualified; complete-contact-cycle open',
    'numerical_convergence': 'unqualified',
    'component_calibration': old_summary['verdicts']['component_calibration'],
    'analogue_validation': old_summary['verdicts']['analogue_validation'],
    'physical_assembly_in_simulation': 'unexecuted',
    'crush_impact_prediction': 'unexecuted',
}
summary['milestone'] = (
    'Source139 cache admission preserves the source133 cache-off full attempted history: '
    'the exact attempted-history parity audit passes, and both runs fail at Newton48. '
    'The standalone 439 direct-operator ABI suite passes its five compiled Metal checks. '
    'Repeated Cauchy fallback reaches the 64-Newton budget in coupon442 but does not complete '
    'the coupon; coupon438 is retained as an input-seal preflight failure and did not execute. '
    'There is no completed cycle, convergence qualification, shipping assembly, or crush/drop result.'
)
summary['qualification']['solver_correctness'] = (
    'Source139 cache-on versus source133 cache-off attempted-history parity passes at the '
    'recorded scope; both trajectories stop at Newton48. Native439 passes five direct operator '
    'ABI regressions. Native442 exercises repeated Cauchy fallback through the authored '
    '64-Newton budget, then fails; this is not a complete coupon. Native438 is preserved as an '
    'execution_started=false input-seal failure. These results do not qualify a complete '
    'contact-cycle solve or physical assembly.'
)
summary['qualification']['numerical_convergence'] = (
    'No completed spatial, timestep, or slower-loading series. Cache reuse has exact attempted '
    'history parity in the recorded source139/source133 comparison, but both stop at Newton48. '
    'The repeated fallback run442 stops unsuccessfully at the 64-Newton limit. No coupon '
    'convergence, performance, shipping allocation, or scalability qualification.'
)
write_json(summary_path, summary)
summary_sha = sha_file(summary_path)

new_files = sum(len(r['files']) for r in new_rows)
new_paths = set(new_archive_paths)
new_paths.add('matter/examples/cardboard/evidence/native-implementation-20261014/summary-reference022.json')
new_paths.add('matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference023.json')
old_paths = {x.decode('utf-8') for x in old_pathspec_bytes.split(b'\0') if x}
assert not (new_paths & old_paths), sorted(new_paths & old_paths)
new_binding = {
    'schema': 'numi.cardboard.reference023-publication-binding.v1',
    'prior_binding_sha256': sha_file(old_binding_path),
    'prior_pathspec_sha256': sha_bytes(old_pathspec_bytes),
    'prior_pathspec_entries': len(old_paths),
    'historical_prefix': {'path': 'summary-reference022.json', 'records': len(old_summary['records']),
                          'sha256': prefix_sha},
    'cases': CASES,
    'summary_sha256': summary_sha,
    'previous_prefix_records': len(old_summary['records']),
    'new_case_records': len(new_rows),
    'copied_files': new_files,
    'all_archived_bytes_verified': True,
    'physical_validation': False,
    'execution_started_false_case': 'solver/repeated-cauchy-full-coupon-438',
    'superseded_adjudication': 'analysis/native442-repeated-fallback-adjudication-001',
    'authoritative_adjudication': 'analysis/native442-repeated-fallback-adjudication-002',
    'six_axis_verdicts': summary['verdicts'],
}
write_json(D / 'publication-binding-reference023.json', new_binding)

# Cumulative pathspec: preserve every byte of the prior NUL list, then append sorted new paths.
extra_paths = sorted(new_paths)
pathspec_path = B / 'publication-reference023-pathspec.nul'
assert not pathspec_path.exists()
pathspec_path.write_bytes(old_pathspec_bytes + b''.join(p.encode('utf-8') + b'\0' for p in extra_paths))

print(json.dumps({'cases': len(new_rows), 'new_files': new_files,
                  'previous_records': len(old_summary['records']),
                  'new_records': len(summary['records']),
                  'summary_sha256': summary_sha,
                  'prefix_sha256': prefix_sha,
                  'binding_sha256': sha_file(D/'publication-binding-reference023.json'),
                  'pathspec_entries': len(pathspec_path.read_bytes().split(b'\0'))-1,
                  'pathspec_sha256': sha_file(pathspec_path)}, indent=2))
