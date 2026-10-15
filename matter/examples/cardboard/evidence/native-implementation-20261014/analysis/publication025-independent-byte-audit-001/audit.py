#!/usr/bin/env python3
"""Independent read-only verifier for publication025 payload bytes and prefixes."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
EVIDENCE = Path('/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014')
OUT = BASE / 'analysis/publication025-independent-byte-audit-001'
BINDING = EVIDENCE / 'publication-binding-reference025.json'
SUMMARY = EVIDENCE / 'summary.json'
SUMMARY_REF = EVIDENCE / 'summary-reference024.json'
BINDING_PREV = EVIDENCE / 'publication-binding-reference024.json'
PATHSPEC_PREV = BASE / 'publication-reference024-pathspec.nul'
PATHSPEC = BASE / 'publication-reference025-pathspec.nul'
ARCHIVE_PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
EXPECTED_BINDING = '908a12e88c411c8240d4ab4c0f220976f6e4cff7d0d939b3bb83bf81406c66fd'
EXPECTED_PREVIOUS_BINDING = '635ec94166fa02e415dfaa8df7be54a7fb78e0abc4c4661a912d449c58cd2d29'
EXPECTED_PREVIOUS_SUMMARY = '24e0b44039397159eb0fb0d8018bdff25321acee3452acce5959dc5a3631213b'
EXPECTED_PREVIOUS_PATHSPEC = '7ac9978a0f21c6480b49d6d9ac22952aa7587cf0bfd54e86e0b3a0afa7b9ab36'

def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def hash_gzip_payload(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with gzip.open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            n += len(block)
            h.update(block)
    return n, h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text())

binding = read_json(BINDING)
summary = read_json(SUMMARY)
previous_summary = read_json(SUMMARY_REF)
previous_binding = read_json(BINDING_PREV)

# Exact published anchors and summary-prefix preservation.
summary_sha = sha_file(SUMMARY)
previous_summary_sha = sha_file(SUMMARY_REF)
binding_sha = sha_file(BINDING)
previous_binding_sha = sha_file(BINDING_PREV)
check(binding['schema'] == 'numi.cardboard.reference025-publication-binding.v1', 'unexpected binding schema')
check(binding_sha == EXPECTED_BINDING, 'publication025 binding digest changed from the audited input')
check(previous_binding_sha == EXPECTED_PREVIOUS_BINDING, 'publication024 binding changed')
check(binding['prior_binding_sha256'] == previous_binding_sha, '025 prior-binding link mismatch')
check(previous_summary_sha == EXPECTED_PREVIOUS_SUMMARY, 'summary-reference024 digest mismatch')
check(binding['historical_prefix'] == {'path': 'summary-reference024.json', 'records': 1342, 'sha256': EXPECTED_PREVIOUS_SUMMARY}, 'historical-prefix metadata mismatch')
check(previous_binding['summary_sha256'] == previous_summary_sha, 'summary-reference024 is not the summary bound by 024')
check(binding['summary_sha256'] == summary_sha, 'current summary does not match 025 binding')
check(len(previous_summary['records']) == 1342, 'summary-reference024 record count mismatch')
check(len(summary['records']) == 1398, 'current summary record count is not 1398')
check(summary['records'][:1342] == previous_summary['records'], 'current summary does not preserve all 1342 previous records exactly')
new_records = summary['records'][1342:]
case_names = [r['case'] for r in new_records]
check(len(new_records) == 56 and len(set(case_names)) == 56, 'new summary suffix is not 56 unique case records')
check(case_names == binding['cases'], 'binding case list/order differs from appended summary records')

# Verify each new case file against its original source and archived copy.
raw_count = gzip_count = file_count = original_bytes = compressed_bytes = 0
archived_paths: set[str] = set()
source_checks = 0
for record in new_records:
    case = record['case']
    check(record.get('physical_box_qualification') is False, f'{case}: physical-qualification flag is not false')
    files = record.get('files', [])
    check(bool(files), f'{case}: no files recorded')
    names = set()
    for row in files:
        name = row['name']
        check(name not in names, f'{case}: duplicate filename {name}')
        names.add(name)
        check(row['bytes'] >= 0 and len(row['sha256']) == 64, f'{case}/{name}: malformed original metadata')
        source_path = Path(row['external_path'])
        check(source_path == BASE / case / name, f'{case}/{name}: source path does not match case/name')
        check(source_path.is_file() and not source_path.is_symlink(), f'{case}/{name}: source file missing or symlinked')
        check(source_path.stat().st_size == row['bytes'] and sha_file(source_path) == row['sha256'], f'{case}/{name}: original source bytes mismatch')
        source_checks += 1
        if 'repository_copy' in row:
            check('repository_compressed_copy' not in row, f'{case}/{name}: ambiguous archive copy type')
            rel = row['repository_copy']
            dest = EVIDENCE / rel
            check(dest.is_file() and not dest.is_symlink(), f'{case}/{name}: archived raw payload missing or symlinked')
            check(dest.stat().st_size == row['bytes'] and sha_file(dest) == row['sha256'], f'{case}/{name}: archived raw bytes mismatch')
            raw_count += 1
            compressed_bytes += row['bytes']
        elif 'repository_compressed_copy' in row:
            rel = row['repository_compressed_copy']
            dest = EVIDENCE / rel
            check(dest.is_file() and not dest.is_symlink(), f'{case}/{name}: archived gzip payload missing or symlinked')
            check(dest.stat().st_size == row['compressed_bytes'], f'{case}/{name}: compressed size mismatch')
            check(sha_file(dest) == row['compressed_sha256'], f'{case}/{name}: compressed SHA mismatch')
            restored_size, restored_sha = hash_gzip_payload(dest)
            check(restored_size == row['bytes'] and restored_sha == row['sha256'], f'{case}/{name}: decompressed content mismatch')
            gzip_count += 1
            compressed_bytes += row['compressed_bytes']
        else:
            raise AssertionError(f'{case}/{name}: no repository archive path')
        check(rel not in archived_paths, f'duplicate archived destination: {rel}')
        archived_paths.add(rel)
        file_count += 1
        original_bytes += row['bytes']

check(file_count == binding['copied_files'] == 1756, f'new copied-file count mismatch: {file_count}')
check(raw_count == binding['raw_files'] == 1679, f'new raw-file count mismatch: {raw_count}')
check(gzip_count == binding['gzip_files'] == 77, f'new gzip-file count mismatch: {gzip_count}')
check(raw_count + gzip_count == file_count, 'raw/gzip count does not equal file count')
check(binding['all_archived_bytes_verified'] is True, 'binding does not assert archived-byte verification')
check(binding['physical_validation'] is False, 'publication incorrectly claims physical validation')

# Exact prior pathspec byte-prefix and suffix membership.
old_pathbytes = PATHSPEC_PREV.read_bytes()
new_pathbytes = PATHSPEC.read_bytes()
check(sha_bytes(old_pathbytes) == EXPECTED_PREVIOUS_PATHSPEC, 'publication024 pathspec changed')
check(binding['prior_pathspec_sha256'] == sha_bytes(old_pathbytes), '025 prior pathspec hash link mismatch')
old_paths = [x.decode('utf-8') for x in old_pathbytes.split(b'\0') if x]
new_paths = [x.decode('utf-8') for x in new_pathbytes.split(b'\0') if x]
check(old_pathbytes.endswith(b'\0') and len(old_paths) == 2312 and len(set(old_paths)) == 2312, 'prior pathspec is not a unique 2312-entry NUL list')
check(new_pathbytes.startswith(old_pathbytes), '025 pathspec does not preserve exact prior bytes')
check(binding['prior_pathspec_entries'] == 2312, 'binding prior path count mismatch')
suffix_bytes = new_pathbytes[len(old_pathbytes):]
suffix_paths = [x.decode('utf-8') for x in suffix_bytes.split(b'\0') if x]
expected_suffix = sorted([ARCHIVE_PREFIX + p for p in archived_paths] + [ARCHIVE_PREFIX + 'summary-reference024.json', ARCHIVE_PREFIX + 'publication-binding-reference025.json'])
check(suffix_paths == expected_suffix, '025 pathspec suffix is not exactly new payloads plus two reference controls')
check(new_pathbytes.endswith(b'\0') and len(new_paths) == 4070 and len(set(new_paths)) == 4070, '025 cumulative pathspec is not a unique 4070-entry NUL list')

report = {
    'schema': 'numi.cardboard.publication025_independent_byte_audit.v1',
    'classification': 'read-only independent archive-byte and prefix audit; no repository or archive writes',
    'result': 'pass',
    'audit_script_sha256': sha_file(Path(__file__)),
    'binding': {'path': str(BINDING), 'sha256': binding_sha},
    'current_summary': {'path': str(SUMMARY), 'sha256': summary_sha, 'record_count': len(summary['records'])},
    'historical_summary': {'path': str(SUMMARY_REF), 'sha256': previous_summary_sha, 'record_count': len(previous_summary['records']), 'exact_prefix_records_preserved': 1342},
    'new_cases': {'case_count': len(new_records), 'case_names_sha256': sha_bytes(('\n'.join(case_names) + '\n').encode()), 'file_count': file_count, 'source_files_verified': source_checks, 'raw_files': raw_count, 'gzip_files': gzip_count, 'original_bytes': original_bytes, 'stored_bytes': compressed_bytes, 'raw_and_decompressed_hashes_match_summary': True},
    'pathspec': {'prior_pathspec_sha256': sha_bytes(old_pathbytes), 'prior_entries': len(old_paths), 'exact_prior_byte_prefix_preserved': True, 'suffix_entries': len(suffix_paths), 'suffix_is_exact_expected_payload_and_reference_set': True, 'cumulative_entries': len(new_paths), 'current_pathspec_sha256': sha_bytes(new_pathbytes)},
    'qualification_boundary': {'binding_physical_validation': binding['physical_validation'], 'note': 'Archive integrity and preservation only; no physical, solver, calibration, or convergence qualification is inferred.'},
    'inputs': {
        str(BINDING): {'sha256': binding_sha, 'bytes': BINDING.stat().st_size},
        str(SUMMARY): {'sha256': summary_sha, 'bytes': SUMMARY.stat().st_size},
        str(SUMMARY_REF): {'sha256': previous_summary_sha, 'bytes': SUMMARY_REF.stat().st_size},
        str(BINDING_PREV): {'sha256': previous_binding_sha, 'bytes': BINDING_PREV.stat().st_size},
        str(PATHSPEC_PREV): {'sha256': sha_bytes(old_pathbytes), 'bytes': len(old_pathbytes)},
        str(PATHSPEC): {'sha256': sha_bytes(new_pathbytes), 'bytes': len(new_pathbytes)},
    },
}
(OUT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
