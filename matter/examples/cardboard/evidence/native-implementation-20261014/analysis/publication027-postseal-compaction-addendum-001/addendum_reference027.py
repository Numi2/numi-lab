from pathlib import Path
import gzip
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
O = B / 'analysis/publication027-postseal-compaction-addendum-001'
EXCLUDE = {'inputs', 'build', 'base', 'candidate', '__pycache__', 'archives', 'restore-check'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, v: p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')


def eligible(rel):
    return not any(p in EXCLUDE or p.startswith(('CMakeFiles', '.air')) for p in rel.parts)


def copy_payload(case, src, rel):
    assert not src.is_symlink()
    raw = src.read_bytes()
    target = D / case / rel
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {'name': str(rel), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'external_path': str(src)}
    if len(raw) <= 262144 or src.suffix == '.gz':
        target.write_bytes(raw)
        row['repository_copy'] = str(target.relative_to(D))
        assert sha(target) == row['sha256']
    else:
        target = target.with_name(target.name + '.gz')
        assert not target.exists(), target
        target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == row['sha256']
        row.update(repository_compressed_copy=str(target.relative_to(D)), compressed_bytes=target.stat().st_size,
                   compressed_sha256=sha(target), restore='gzip decompression recovers original bytes; verify original sha256')
    return row, 'matter/examples/cardboard/evidence/native-implementation-20261014/' + str(target.relative_to(D))


oldsummary = (D / 'summary.json').read_bytes()
oldbinding = (D / 'publication-binding-reference027.json').read_bytes()
oldpaths = (B / 'publication-reference027-pathspec.nul').read_bytes()
summary = json.loads(oldsummary)
binding = json.loads(oldbinding)
assert len(summary['records']) == 1457
assert hashlib.sha256(oldbinding).hexdigest() == 'f4650f209c282f535594200875d0f5f4815a1de3d6053204f1137980b556fe01'
assert binding['summary_sha256'] == hashlib.sha256(oldsummary).hexdigest()
assert len({p for p in oldpaths.split(b'\0') if p}) == 4811
prefix = json.loads((D / 'summary-reference026.json').read_text())['records']
assert len(prefix) == 1424 and summary['records'][:1424] == prefix
extra_cases = [
    'analysis/closed-solver-input-compaction-024',
    'analysis/mortar-union-four-patch-buildhost-compaction-020',
    'analysis/publication027-postseal-compaction-addendum-001',
]
assert not set(extra_cases).intersection(binding['cases'])
O.mkdir(exist_ok=False)
(O / 'prospective-summary-before-compaction-addendum.json.gz').write_bytes(gzip.compress(oldsummary, compresslevel=6, mtime=0))
(O / 'prospective-binding-before-compaction-addendum.json').write_bytes(oldbinding)
(O / 'prospective-pathspec-before-compaction-addendum.nul').write_bytes(oldpaths)
(O / 'addendum_reference027.py').write_bytes(Path(__file__).read_bytes())
write(O / 'case-list-revision002.json', {'cases': binding['cases'] + extra_cases})
write(O / 'manifest.json', {
    'schema': 'numi.cardboard.publication027.postseal_compaction_addendum.v1',
    'reason': 'Terminal native464/466/467 duplicate inputs were losslessly compacted after prospective027 archiving. The resulting three recovery pointers must be included; first strict independent inventory audit refused the missing464 pointer.',
    'prior_prospective_binding_sha256': hashlib.sha256(oldbinding).hexdigest(),
    'prior_prospective_summary_sha256': hashlib.sha256(oldsummary).hexdigest(),
    'prior_prospective_pathspec_sha256': hashlib.sha256(oldpaths).hexdigest(),
    'prior_prospective_pathspec_entries': 4811, 'historical_prefix_records_unchanged': 1424,
    'added_cases': extra_cases,
    'no_existing_payload_or_native_history_rewritten': True,
    'archive_policy': 'Same published exclusion rules; sole external recovery archives retained. Restore proofs and pointers included. Existing two command input-alias exceptions unchanged.',
})
newpaths = set()
for case in ['solver/cgls-regional-state-capture-464', 'solver/cgls-newton128-default64-parity-466',
             'solver/cgls-complete-outer128-allcontrols-467']:
    rec = next(row for row in summary['records'][1424:] if row['case'] == case)
    assert not any(row['name'] == 'inputs.compaction.json' for row in rec['files'])
    src = B / case / 'inputs.compaction.json'
    pointer = json.loads(src.read_text())
    assert sha(Path(pointer['archive_path'])) == pointer['archive_sha256']
    assert sha(Path(pointer['before_manifest_path'])) == pointer['before_manifest_sha256']
    row, path = copy_payload(case, src, Path('inputs.compaction.json'))
    rec['files'].append(row)
    newpaths.add(path)
for case in extra_cases:
    root = B / case
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and eligible(p.relative_to(root))]
    assert len(files) < 80 and sum(p.stat().st_size for p in files) < 25 * 1024**2, (case, len(files))
    rec = {'case': case, 'external_root': str(root), 'physical_box_qualification': False, 'files': []}
    for src in files:
        row, path = copy_payload(case, src, src.relative_to(root))
        rec['files'].append(row)
        newpaths.add(path)
    assert rec['files']
    summary['records'].append(rec)
assert summary['records'][:1424] == prefix
assert not {p.decode() for p in oldpaths.split(b'\0') if p}.intersection(newpaths)
write(D / 'summary.json', summary)
binding['cases'].extend(extra_cases)
binding['new_case_records'] = len(summary['records']) - 1424
rows = [row for rec in summary['records'][1424:] for row in rec['files']]
binding['copied_files'] = len(rows)
binding['raw_files'] = sum('repository_copy' in row for row in rows)
binding['gzip_files'] = sum('repository_compressed_copy' in row for row in rows)
binding['summary_sha256'] = sha(D / 'summary.json')
binding['postseal_addendum'] = {
    'case': str(O.relative_to(B)), 'prior_prospective_binding_sha256': hashlib.sha256(oldbinding).hexdigest(),
    'new_recovery_pointers': 3, 'new_compaction_cases': 2, 'prior_native_histories_unchanged': True,
}
binding['external_recovery_archives'] = 'Compaction019/020/023/024 and mortar buildhost020 recovery archives remain external; keep these sole archives. Metadata, exact pointers and full restore proofs are published.'
write(D / 'publication-binding-reference027.json', binding)
(B / 'publication-reference027-pathspec.nul').write_bytes(oldpaths + b''.join(p.encode() + b'\0' for p in sorted(newpaths)))
result = {'cases': binding['new_case_records'], 'files': binding['copied_files'], 'raw': binding['raw_files'],
          'gzip': binding['gzip_files'], 'records': len(summary['records']), 'new_addendum_paths': len(newpaths),
          'pathspec_entries': 4811 + len(newpaths), 'binding_sha256': sha(D / 'publication-binding-reference027.json')}
write(B / 'publication027-addendum-result.json', result)
print(json.dumps(result, indent=2))
