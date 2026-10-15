#!/usr/bin/env python3
"""Independent read-only byte audit for publication024 against prepared inventory."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
D = Path('/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014')
OUT = B / 'analysis/publication024-independent-byte-audit-001'
PREP = B / 'publication024-cases-prepared-v2.json'
PATH23 = B / 'publication-reference023-pathspec.nul'
PATH24 = B / 'publication-reference024-pathspec.nul'
EXPECTED_BINDING24 = '635ec94166fa02e415dfaa8df7be54a7fb78e0abc4c4661a912d449c58cd2d29'
ARCHIVE_PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
FORBIDDEN_PARTS = {'inputs', 'build', 'source', 'base', 'candidate', '__pycache__', 'archives'}

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())

def load(path: Path):
    return json.loads(path.read_text())

def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)

prepared = load(PREP)
summary = load(D / 'summary.json')
summary23 = load(D / 'summary-reference023.json')
binding24 = load(D / 'publication-binding-reference024.json')
binding23 = load(D / 'publication-binding-reference023.json')
binding22 = load(D / 'publication-binding-reference022.json')
binding21 = load(D / 'publication-binding-reference021.json')

# Freeze publication anchor and exact old summary bytes.
summary23_bytes = (D / 'summary-reference023.json').read_bytes()
check(sha_bytes(summary23_bytes) == '38da763f3d87b50d78eca8b081ef0c6581e4f1db1a90c1512ddbb4a553e27288', 'summary-reference023 bytes do not match the 023 binding')
check(sha_bytes(summary23_bytes) == binding23['summary_sha256'], 'summary-reference023 does not match publication023 summary_sha256')
check(len(summary23['records']) == 1300, '023 reference does not have exactly 1300 records')
check(summary['records'][:1300] == summary23['records'], 'current first 1300 records differ from the exact 023 reference records')
check(len(summary['records']) == 1342, 'current summary record count is not 1342')
check(sha(D / 'summary.json') == binding24['summary_sha256'], 'current summary.json differs from 024 binding')
check(binding24['previous_prefix_records'] == 1300 and binding24['new_case_records'] == 42, 'binding record counts mismatch')
check(sha(D / 'publication-binding-reference024.json') == EXPECTED_BINDING24, 'publication024 binding hash mismatch')
check(binding24['prior_binding_sha256'] == sha(D / 'publication-binding-reference023.json'), '024 prior binding link mismatch')
check(binding24['historical_prefix']['sha256'] == sha_bytes(summary23_bytes), '024 historical summary link mismatch')
check(binding24['historical_prefix']['records'] == 1300, '024 historical prefix count mismatch')

# Validate the prepared case sequence and each new record against the v2 inventory.
case_paths = prepared['cases']
case_rows = prepared['case_inventory']
check(len(case_paths) == len(set(case_paths)) == 42, 'prepared append case list must contain 42 unique paths')
check(case_paths == [r['case'] for r in case_rows], 'prepared case strings and detailed rows differ')
new_records = summary['records'][1300:]
check([r['case'] for r in new_records] == case_paths, 'summary append case order differs from v2 prepared list')
check(binding24['cases'] == case_paths, '024 binding case list differs from prepared list')
check(len(new_records) == 42, 'summary does not contain exactly 42 new records')

excluded_components = []
new_payloads = {}
case_file_total = 0
original_byte_total = 0
for inv in case_rows:
    case = inv['case']
    record = next((r for r in new_records if r['case'] == case), None)
    check(record is not None, f'missing new summary record {case}')
    check(record.get('physical_box_qualification') is False, f'physical qualification flag is not false for {case}')
    inv_files = inv['files']
    summary_files = record.get('files', [])
    check(len(inv_files) == len(summary_files), f'file count differs for {case}')
    by_name = {f['path']: f for f in inv_files}
    check(len(by_name) == len(inv_files), f'duplicate prepared file path under {case}')
    for row in summary_files:
        name = row['name']
        check(name in by_name, f'{case}/{name} missing from prepared inventory')
        f = by_name[name]
        check(f['bytes'] == row['bytes'] and f['sha256'] == row['sha256'], f'{case}/{name} source digest differs from summary')
        rel = Path(name)
        check(not any(x in FORBIDDEN_PARTS or x.startswith(('CMakeFiles', '.air')) for x in rel.parts), f'forbidden path archived under {case}/{name}')
        source = B / case / rel
        check(source.is_file() and not source.is_symlink(), f'missing or symlinked external source {source}')
        check(source.stat().st_size == row['bytes'] and sha(source) == row['sha256'], f'external source changed for {case}/{name}')
        if 'repository_copy' in row:
            dest_rel = row['repository_copy']
            check('repository_compressed_copy' not in row, f'copy type ambiguous for {case}/{name}')
            dest = D / dest_rel
            actual = dest.read_bytes()
            check(len(actual) == row['bytes'] and sha_bytes(actual) == row['sha256'], f'raw archived bytes mismatch {dest_rel}')
            kind = 'raw'
        else:
            dest_rel = row['repository_compressed_copy']
            dest = D / dest_rel
            packed = dest.read_bytes()
            check(len(packed) == row['compressed_bytes'], f'compressed byte count mismatch {dest_rel}')
            check(sha_bytes(packed) == row['compressed_sha256'], f'compressed SHA mismatch {dest_rel}')
            actual = gzip.decompress(packed)
            check(len(actual) == row['bytes'] and sha_bytes(actual) == row['sha256'], f'decompressed payload mismatch {dest_rel}')
            kind = 'gzip'
        check(dest_rel not in new_payloads, f'duplicate archived destination {dest_rel}')
        new_payloads[dest_rel] = {'kind': kind, 'compressed_bytes': row.get('compressed_bytes'), 'compressed_sha256': row.get('compressed_sha256'), 'original_bytes': row['bytes'], 'original_sha256': row['sha256']}
        original_byte_total += row['bytes']
        case_file_total += 1

check(case_file_total == 591, f'expected 591 new payloads, got {case_file_total}')
raw_count = sum(x['kind'] == 'raw' for x in new_payloads.values())
gzip_count = sum(x['kind'] == 'gzip' for x in new_payloads.values())
check(raw_count == 499 and gzip_count == 92, f'raw/gzip split mismatch: {raw_count}/{gzip_count}')
check(binding24['copied_files'] == 591 and binding24['raw_files'] == raw_count and binding24['gzip_files'] == gzip_count, 'binding payload counters differ')
check(binding24['all_archived_bytes_verified'] is True and prepared['physical_validation'] is False, 'archive verification or physical claim flag is inconsistent')

# Recover all 1714 pre-024 payloads (1184 from 022 plus 530 from 023) and verify exact bytes.
old_pathbytes = PATH23.read_bytes()
old_paths = [x.decode() for x in old_pathbytes.split(b'\0') if x]
check(len(old_paths) == len(set(old_paths)) == 1719 and old_pathbytes.endswith(b'\0'), '023 pathspec size/uniqueness mismatch')
check(sha(old_pathbytes and PATH23) == binding24['prior_pathspec_sha256'], '024 prior pathspec SHA mismatch')
check(binding24['prior_pathspec_entries'] == 1719, '024 prior pathspec entry count mismatch')
check(binding23['prior_pathspec_entries'] == 1187, '023 prior pathspec count mismatch')
path22 = (B / 'publication-reference022-pathspec.nul').read_bytes()
check(len([x for x in path22.split(b'\0') if x]) == 1187, '022 pathspec entry count mismatch')
check(sha_bytes(path22) == binding23['prior_pathspec_sha256'], '023 binding prior pathspec hash mismatch')

# Build unambiguous hash expectations for archive payloads from the exact 1300-record summary.
old_expected = {}
for rec in summary23['records']:
    for row in rec.get('files', []):
        if 'repository_copy' in row:
            rel = ARCHIVE_PREFIX + row['repository_copy']
            val = ('raw', row.get('bytes'), row.get('sha256'), None, None)
        elif 'repository_compressed_copy' in row:
            rel = ARCHIVE_PREFIX + row['repository_compressed_copy']
            val = ('gzip', row.get('bytes'), row.get('sha256'), row.get('compressed_bytes'), row.get('compressed_sha256'))
        else:
            continue
        if rel in old_expected:
            check(old_expected[rel] == val, f'conflicting prior payload metadata for {rel}')
        old_expected[rel] = val
old_payload_paths = [p for p in old_paths if p in old_expected]
old_metadata_paths = [p for p in old_paths if p not in old_expected]
check(len(old_payload_paths) == 1714, f'023 prior payload count derived from pathspec is {len(old_payload_paths)}, expected 1714')
check(len(old_metadata_paths) == 5, f'023 pathspec has unexpected non-payload path count {len(old_metadata_paths)}')
prior_payload_raw = prior_payload_gzip = prior_payload_bytes = 0
for rel in old_payload_paths:
    kind, nbytes, rawsha, packedbytes, packedsha = old_expected[rel]
    p = D / rel.removeprefix(ARCHIVE_PREFIX)
    data = p.read_bytes()
    if kind == 'raw':
        check(len(data) == nbytes and sha_bytes(data) == rawsha, f'prior raw payload changed: {rel}')
        prior_payload_raw += 1
    else:
        check(len(data) == packedbytes and sha_bytes(data) == packedsha, f'prior compressed archive bytes changed: {rel}')
        restored = gzip.decompress(data)
        check(len(restored) == nbytes and sha_bytes(restored) == rawsha, f'prior decompressed payload changed: {rel}')
        prior_payload_gzip += 1
    prior_payload_bytes += nbytes
check(prior_payload_raw + prior_payload_gzip == 1714, 'prior payload totals mismatch')

# The five non-payload control paths are verified through the binding chain; summary.json is
# mutable by design, with its old bytes preserved by summary-reference023.json.
metadata_rel = {p.removeprefix(ARCHIVE_PREFIX) for p in old_metadata_paths}
check(metadata_rel == {'summary.json', 'summary-reference021.json', 'summary-reference022.json', 'publication-binding-reference022.json', 'publication-binding-reference023.json'}, 'unexpected pre-024 control paths')
check(sha(D / 'summary-reference021.json') == binding21['summary_sha256'], 'summary-reference021 preservation failed')
check(sha(D / 'summary-reference022.json') == binding23['historical_prefix']['sha256'], 'summary-reference022 preservation failed')
check(sha(D / 'publication-binding-reference022.json') == binding23['prior_binding_sha256'], 'publication-binding-reference022 preservation failed')
check(sha(D / 'publication-binding-reference023.json') == binding24['prior_binding_sha256'], 'publication-binding-reference023 preservation failed')
check(sha(D / 'summary.json') != sha_bytes(summary23_bytes), 'summary.json unexpectedly remained the old summary rather than the new 1342-record summary')

# Cumulative NUL pathspec must preserve the exact prior bytes and append 591 payloads plus two controls.
new_pathbytes = PATH24.read_bytes()
check(new_pathbytes.startswith(old_pathbytes), '024 pathspec does not preserve the exact 023 byte prefix')
suffix_bytes = new_pathbytes[len(old_pathbytes):]
suffix_paths = [x.decode() for x in suffix_bytes.split(b'\0') if x]
expected_new_paths = sorted([ARCHIVE_PREFIX + p for p in new_payloads] + [ARCHIVE_PREFIX + 'summary-reference023.json', ARCHIVE_PREFIX + 'publication-binding-reference024.json'])
check(suffix_paths == expected_new_paths, '024 pathspec suffix does not equal 591 payload paths plus two control paths')
allpaths = [x.decode() for x in new_pathbytes.split(b'\0') if x]
check(len(allpaths) == len(set(allpaths)) == 2312 and new_pathbytes.endswith(b'\0'), '024 cumulative pathspec count/uniqueness mismatch')
check(len(suffix_paths) == 593, '024 suffix entry count mismatch')
check(sha(D / 'publication-binding-reference024.json') == EXPECTED_BINDING24, 'binding changed during audit')

# No forbidden source/base/cache copies were added by 024. Two exact base/ paths are
# historical 023 prefix payloads and are recorded, not rewritten or hidden.
def forbidden(path):
 return any(part in FORBIDDEN_PARTS or part.startswith(('CMakeFiles', '.air')) for part in Path(path).parts)
new_exclusions = [p for p in suffix_paths if forbidden(p)]
check(not new_exclusions, f'024 suffix contains forbidden paths: {new_exclusions[:5]}')
historical_exceptions = [p for p in old_paths if forbidden(p)]
check(historical_exceptions == [
 ARCHIVE_PREFIX + 'analysis/cache-projection-reuse-admission-overlay-138-001/base/matter/src/metal/fem.metalinc',
 ARCHIVE_PREFIX + 'analysis/cache-projection-reuse-admission-overlay-138-001/base/matter/src/runtime.mm.gz',
], f'unexpected forbidden historical-prefix paths: {historical_exceptions}')

# Source142 integration binding and six-axis limits.
source142 = load(B / 'analysis/source142-root-integration-001/report.json')
check(source142['all_787_selected_source_hashes_match'] is True, 'source142 selected-source hash audit failed')
check(source142['source_snapshot'].endswith('/snapshots/source-bounded-native-cgls-mask-repair-142'), 'source142 identity mismatch')
check(source142['complete_coupon_qualified'] is False and source142['physical_validation'] is False, 'source142 qualification flags overstate evidence')
check(not any('143' in c for c in case_paths), 'source143 case unexpectedly included')
check(binding24['six_axis_verdicts'] == summary['qualification'], 'binding six-axis verdicts differ from summary')
limits = binding24['six_axis_verdicts']
check(limits['numerical_convergence'] == 'unqualified', 'numerical convergence is not explicitly unqualified')
check(limits['physical_assembly_in_simulation'] == 'unexecuted', 'physical assembly is not explicitly unexecuted')
check(limits['crush_impact_prediction'] == 'unexecuted', 'crush/impact is not explicitly unexecuted')
check(binding24['physical_validation'] is False, 'binding physical_validation must remain false')

# Existing prefix payload count composition: publication022 copied 1184 + publication023 copied 530.
check(binding22['copied_files'] == 1184 and binding23['copied_files'] == 530, '023 prefix component counts differ')
check(binding22['copied_files'] + binding23['copied_files'] == 1714, 'derived prior payload count differs')

report = {
 'schema': 'numi.cardboard.publication024_independent_byte_audit.v1',
 'classification': 'independent read-only published-byte and reference-prefix verification; no archive writes',
 'binding': {'path': str(D / 'publication-binding-reference024.json'), 'sha256': sha(D / 'publication-binding-reference024.json'), 'expected_sha256': EXPECTED_BINDING24},
 'summary': {'records': len(summary['records']), 'prior_records': len(summary23['records']), 'new_records': len(new_records), 'summary_reference023_sha256': sha_bytes(summary23_bytes), 'prior_record_prefix_exact': True, 'current_summary_sha256': sha(D / 'summary.json')},
 'new_payloads': {'case_count': len(case_paths), 'files': case_file_total, 'raw': raw_count, 'gzip': gzip_count, 'original_bytes': original_byte_total, 'all_source_and_archived_hashes_match': True},
 'prior_payloads': {'pathspec_entries': len(old_paths), 'payload_files_derived_from_022_and_023_bindings': binding22['copied_files'] + binding23['copied_files'], 'raw_payloads': prior_payload_raw, 'gzip_payloads': prior_payload_gzip, 'control_paths': len(old_metadata_paths), 'original_bytes_verified': prior_payload_bytes, 'all_raw_and_decompressed_hashes_match': True},
 'pathspec': {'prior_entries': len(old_paths), 'suffix_entries': len(suffix_paths), 'new_payload_plus_control_entries': len(suffix_paths), 'cumulative_entries': len(allpaths), 'exact_prior_byte_prefix_preserved': True, 'suffix_is_exact_expected_set': True, 'sha256': sha(new_pathbytes)},
 'historical_prefix_exceptions': {'preserved_without_rewrite': historical_exceptions, 'count': len(historical_exceptions), 'scope': 'two prior publication023 base/ overlay artifacts; publication024 suffix has zero excluded source/base/build/cache paths'},
 'source142': {'snapshot': source142['source_snapshot'], 'selected_hashes_match': source142['all_787_selected_source_hashes_match'], 'selected_hash_count': 787, 'complete_coupon_qualified': source142['complete_coupon_qualified']},
 'six_axis_verdicts': limits,
 'qualification_limits_verified': True,
 'root_summary_path': str(D / 'summary.json'),
 'prepared_cases_path': str(PREP),
 'prepared_cases_sha256': sha(PREP),
 'audit_script_sha256': sha(Path(__file__)),
 'audit_pass': True,
}
(OUT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, indent=2, sort_keys=True))
