#!/usr/bin/env python3
"""Independent strict byte/prefix/scope/index audit for publication027 revision002."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
PKG = B / 'analysis/publication027-independent-byte-audit-002'
EXCLUDE_PARTS = {'inputs', 'build', 'base', 'candidate', '__pycache__', 'archives', 'restore-check'}
ALIAS_CASES = {
    'analysis/native466-source154-default64-parity-command-002',
    'analysis/native467-source154-allcontrol-outer128-command-002',
}
POINTER_CASES = {
    'solver/cgls-regional-state-capture-464',
    'solver/cgls-newton128-default64-parity-466',
    'solver/cgls-complete-outer128-allcontrols-467',
}
OLD_PROSPECTIVE_CASES = 33
PRIOR_027_PATH_ENTRIES = 4811
EXPECTED_HEAD = '67ba06be7f48981dda142ea35eedd792d7cd2ca5'
EXPECTED_BINDING_SHA = 'e2f1751e204cf108e811cf16b9635e9db706827429d084765e888260f4329f54'
EXPECTED_STAGE_SHA = '3d95a0c12a627e7f1035d1921398e64c9b1025155ce4a8f0464f69ab758158aa'
EXPECTED_UNTRACKED = {
    'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/CMakeOBJCXXCompilerId.mm',
    'matter/tools/evidence/cohesive-scratch-alias/source-audit.json',
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest_stream(stream):
    h = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(1 << 20), b''):
        count += len(block)
        h.update(block)
    return count, h.hexdigest()


def allowed(case, rel):
    parts = rel.parts
    if case in ALIAS_CASES and len(parts) == 2 and parts[0] == 'inputs':
        return True
    if case in POINTER_CASES and rel.as_posix() == 'inputs.compaction.json':
        return True
    return not any(p in EXCLUDE_PARTS or p.startswith(('CMakeFiles', '.air')) for p in parts)


def git(*args):
    return subprocess.run(['git', *args], cwd=R, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def verify_record_file(case, row, repo_copies):
    rel = Path(row['name'])
    assert not rel.is_absolute() and '..' not in rel.parts
    assert allowed(case, rel), f'excluded path archived: {case}/{rel}'
    root = B / case
    src = root / rel
    assert src.resolve().is_relative_to(root.resolve())
    assert src.is_file() and not src.is_symlink(), f'missing or symlink external file: {src}'
    source_bytes, source_sha = src.stat().st_size, sha_file(src)
    assert source_bytes == row['bytes'] and source_sha == row['sha256'], f'external bytes mismatch: {src}'
    if 'repository_copy' in row:
        copy_rel = row['repository_copy']
        assert copy_rel not in repo_copies
        repo_copies.add(copy_rel)
        copy = D / copy_rel
        assert copy.is_file() and not copy.is_symlink()
        count, digest = copy.stat().st_size, sha_file(copy)
        assert count == row['bytes'] and digest == source_sha, f'repository raw mismatch: {copy}'
        assert row.get('restore') is None
        return 'raw', 'matter/examples/cardboard/evidence/native-implementation-20261014/' + copy_rel
    assert 'repository_compressed_copy' in row
    copy_rel = row['repository_compressed_copy']
    assert copy_rel not in repo_copies
    repo_copies.add(copy_rel)
    copy = D / copy_rel
    assert copy.is_file() and not copy.is_symlink()
    packed_count, packed_sha = copy.stat().st_size, sha_file(copy)
    assert packed_count == row['compressed_bytes'] and packed_sha == row['compressed_sha256']
    with gzip.open(copy, 'rb') as stream:
        count, digest = digest_stream(stream)
    assert count == row['bytes'] and digest == source_sha, f'repository gzip restoration mismatch: {copy}'
    return 'gzip', 'matter/examples/cardboard/evidence/native-implementation-20261014/' + copy_rel


def enumerate_allowed(case):
    root = B / case
    assert root.is_dir() and not root.is_symlink()
    found = set()
    for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        kept_dirs = []
        for name in dirs:
            child = base_path / name
            rel = child.relative_to(root)
            alias_input_dir = case in ALIAS_CASES and rel.as_posix() == 'inputs'
            if not alias_input_dir and any(p in EXCLUDE_PARTS or p.startswith(('CMakeFiles', '.air')) for p in rel.parts):
                continue
            assert not child.is_symlink(), f'unexpected included-scope symlink: {child}'
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in files:
            path = base_path / name
            rel = path.relative_to(root)
            if not allowed(case, rel):
                continue
            assert path.is_file() and not path.is_symlink(), f'included path is not regular: {path}'
            found.add(rel.as_posix())
    return found


def main():
    prepared = json.loads((B / 'analysis/publication027-postseal-compaction-addendum-001/case-list-revision002.json').read_text())
    cases = prepared['cases']
    assert len(cases) == 36 and len(set(cases)) == 36
    binding_path = D / 'publication-binding-reference027.json'
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    assert sha_bytes(binding_bytes) == EXPECTED_BINDING_SHA
    assert binding['schema'] == 'numi.cardboard.reference027-publication-binding.v1'
    assert binding['cases'] == cases
    assert binding['previous_prefix_records'] == 1424
    assert binding['new_case_records'] == 36
    assert binding['copied_files'] == 503 and binding['raw_files'] == 469 and binding['gzip_files'] == 34
    assert binding['all_archived_bytes_verified'] is True
    assert binding['physical_validation'] is False

    summary_path = D / 'summary.json'
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    assert sha_bytes(summary_bytes) == binding['summary_sha256']
    assert len(summary['records']) == 1460
    old_summary_path = D / 'summary-reference026.json'
    old_summary_bytes = old_summary_path.read_bytes()
    old_summary = json.loads(old_summary_bytes)
    assert len(old_summary['records']) == 1424
    assert sha_bytes(old_summary_bytes) == binding['historical_prefix']['sha256']
    assert binding['historical_prefix'] == {'path': 'summary-reference026.json', 'records': 1424, 'sha256': sha_bytes(old_summary_bytes)}
    assert summary['records'][:1424] == old_summary['records']
    assert len(summary['records'][1424:]) == 36
    assert [r['case'] for r in summary['records'][1424:]] == cases

    old_binding_bytes = (D / 'publication-binding-reference026.json').read_bytes()
    assert sha_bytes(old_binding_bytes) == binding['prior_binding_sha256']
    prior_path = B / 'publication-reference026-pathspec.nul'
    prior_path_bytes = prior_path.read_bytes()
    assert sha_bytes(prior_path_bytes) == binding['prior_pathspec_sha256']
    assert binding['prior_pathspec_entries'] == 4344
    prior_paths = [x.decode() for x in prior_path_bytes.split(b'\0') if x]
    assert len(prior_paths) == 4344 and len(set(prior_paths)) == 4344

    addendum_dir = B / 'analysis/publication027-postseal-compaction-addendum-001'
    addendum_manifest_path = addendum_dir / 'manifest.json'
    addendum_manifest = json.loads(addendum_manifest_path.read_text())
    assert addendum_manifest['historical_prefix_records_unchanged'] == 1424
    assert addendum_manifest['no_existing_payload_or_native_history_rewritten'] is True
    assert addendum_manifest['prior_prospective_binding_sha256'] == 'f4650f209c282f535594200875d0f5f4815a1de3d6053204f1137980b556fe01'
    assert addendum_manifest['prior_prospective_pathspec_entries'] == PRIOR_027_PATH_ENTRIES
    expected_added_cases = [
        'analysis/closed-solver-input-compaction-024',
        'analysis/mortar-union-four-patch-buildhost-compaction-020',
        'analysis/publication027-postseal-compaction-addendum-001',
    ]
    assert addendum_manifest['added_cases'] == expected_added_cases
    prospective_binding = (addendum_dir / 'prospective-binding-before-compaction-addendum.json').read_bytes()
    assert sha_bytes(prospective_binding) == addendum_manifest['prior_prospective_binding_sha256']
    old_prospective = json.loads(prospective_binding)
    assert len(old_prospective['cases']) == OLD_PROSPECTIVE_CASES
    assert binding['cases'][:OLD_PROSPECTIVE_CASES] == old_prospective['cases']
    assert binding['postseal_addendum']['prior_prospective_binding_sha256'] == sha_bytes(prospective_binding)
    assert binding['postseal_addendum']['new_recovery_pointers'] == 3
    assert binding['postseal_addendum']['new_compaction_cases'] == 2
    addendum_result = json.loads((B / 'publication027-addendum-result.json').read_text())
    assert addendum_result == {
        'cases': 36, 'files': 503, 'raw': 469, 'gzip': 34,
        'records': 1460, 'new_addendum_paths': 38, 'pathspec_entries': 4849,
        'binding_sha256': EXPECTED_BINDING_SHA,
    }
    prospective_summary_gz = addendum_dir / 'prospective-summary-before-compaction-addendum.json.gz'
    with gzip.open(prospective_summary_gz, 'rb') as stream:
        prior_summary_bytes, prior_summary_sha = digest_stream(stream)
    assert prior_summary_sha == addendum_manifest['prior_prospective_summary_sha256']
    with gzip.open(prospective_summary_gz, 'rb') as stream:
        prior_prospective_summary = json.load(stream)
    assert len(prior_prospective_summary['records']) == 1457
    # The addendum intentionally appends one recovery-pointer row to each of
    # three already-published case records. All other prior records stay exact.
    pointer_case_names = {'solver/cgls-regional-state-capture-464',
                          'solver/cgls-newton128-default64-parity-466',
                          'solver/cgls-complete-outer128-allcontrols-467'}
    changed_prior_records = []
    for old_record, current_record in zip(prior_prospective_summary['records'], summary['records'][:1457]):
        if old_record == current_record:
            continue
        assert old_record['case'] in pointer_case_names
        filtered = dict(current_record)
        filtered['files'] = [row for row in current_record['files'] if row['name'] != 'inputs.compaction.json']
        assert filtered == old_record, old_record['case']
        assert sum(row['name'] == 'inputs.compaction.json' for row in current_record['files']) == 1
        changed_prior_records.append(old_record['case'])
    assert set(changed_prior_records) == pointer_case_names
    prospective_paths = addendum_dir / 'prospective-pathspec-before-compaction-addendum.nul'
    prospective_path_bytes = prospective_paths.read_bytes()
    prospective_path_sha = sha_bytes(prospective_path_bytes)
    assert prospective_path_sha == addendum_manifest['prior_prospective_pathspec_sha256']
    assert len([x for x in prospective_path_bytes.split(b'\0') if x]) == PRIOR_027_PATH_ENTRIES

    pathspec_path = B / 'publication-reference027-pathspec.nul'
    pathspec_bytes = pathspec_path.read_bytes()
    path_entries = [x.decode() for x in pathspec_bytes.split(b'\0') if x]
    assert len(path_entries) == 4849 and len(set(path_entries)) == 4849
    assert pathspec_bytes.startswith(prior_path_bytes)
    assert pathspec_bytes.startswith(prospective_path_bytes)
    assert path_entries[:4344] == prior_paths
    new_path_set = set(path_entries[4344:])
    assert len(new_path_set) == 505
    assert len([x for x in path_entries[4344:] if x]) == 505

    archive_rows = raw = compressed = 0
    case_file_counts = {}
    alias_audits = {}
    pointer_audits = {}
    archive_paths = set()
    repo_copies = set()
    summary_record_by_case = {r['case']: r for r in summary['records'][1424:]}
    assert len(summary_record_by_case) == 36
    for case in cases:
        record = summary_record_by_case[case]
        extroot = B / case
        assert record['external_root'] == str(extroot)
        assert record['physical_box_qualification'] is False
        rows = record['files']
        names = [row['name'] for row in rows]
        assert rows and len(names) == len(set(names))
        listed = set()
        for row in rows:
            rel = Path(row['name'])
            kind, repo_path = verify_record_file(case, row, repo_copies)
            if kind == 'raw':
                raw += 1
            else:
                compressed += 1
            archive_paths.add(repo_path)
            listed.add(rel.as_posix())
            archive_rows += 1
            if case in ALIAS_CASES and len(rel.parts) == 2 and rel.parts[0] == 'inputs':
                repair_path = extroot / 'repair.json'
                repair = json.loads(repair_path.read_text())
                bind = next((x for x in repair['inputs'] if Path(x['copied']).name == rel.name), None)
                assert bind is not None
                original = Path(bind['original'])
                assert original.is_file() and not original.is_symlink()
                assert sha_file(original) == row['sha256'] == bind['sha256']
                assert original.stat().st_size == row['bytes'] == bind['bytes']
                alias_audits.setdefault(case, []).append({'alias': rel.as_posix(), 'original': str(original), 'sha256': row['sha256'], 'bytes': row['bytes']})
            if case in POINTER_CASES and rel.as_posix() == 'inputs.compaction.json':
                pointer = json.loads((extroot / rel).read_text())
                assert pointer['schema'] == 'numi.cardboard.closed-solver-input-compaction.pointer.v1'
                assert pointer['case'] == str(extroot)
                assert pointer['restore_required_to_recreate_prediction_inputs'] is True
                archive = Path(pointer['archive_path'])
                before = Path(pointer['before_manifest_path'])
                assert archive.is_file() and not archive.is_symlink()
                assert archive.stat().st_size == pointer['archive_bytes']
                assert sha_file(archive) == pointer['archive_sha256']
                assert before.is_file() and not before.is_symlink()
                assert sha_file(before) == pointer['before_manifest_sha256']
                pointer_audits[case] = {'pointer_sha256': row['sha256'], 'archive_sha256': pointer['archive_sha256'], 'before_manifest_sha256': pointer['before_manifest_sha256']}
        actual = enumerate_allowed(case)
        assert actual == listed, (case, len(actual), len(listed), sorted(actual-listed)[:5], sorted(listed-actual)[:5])
        case_file_counts[case] = len(rows)
    assert set(alias_audits) == ALIAS_CASES
    assert set(pointer_audits) == POINTER_CASES
    assert archive_rows == 503 and raw == 469 and compressed == 34
    assert (raw, compressed) == (binding['raw_files'], binding['gzip_files'])

    archive_paths.update({
        'matter/examples/cardboard/evidence/native-implementation-20261014/summary-reference026.json',
        'matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference027.json',
    })
    assert archive_paths == new_path_set, (len(archive_paths), len(new_path_set), sorted(archive_paths-new_path_set)[:5], sorted(new_path_set-archive_paths)[:5])

    # Independently confirm the two compaction packages expose the claimed sole recovery archives.
    c024 = B / 'analysis/closed-solver-input-compaction-024'
    selected024 = json.loads((c024 / 'selected.json').read_text())
    assert selected024['cases'] == ['cgls-regional-state-capture-464', 'cgls-newton128-default64-parity-466', 'cgls-complete-outer128-allcontrols-467']
    c020 = B / 'analysis/mortar-union-four-patch-buildhost-compaction-020'
    c020_manifest = json.loads((c020 / 'archive-manifest.json').read_text())
    archive_checks = {}
    for key, expected in c020_manifest['targets'].items():
        archive = c020 / expected['archive']
        assert archive.is_file() and archive.stat().st_size == expected['archive_bytes']
        assert sha_file(archive) == expected['archive_sha256']
        archive_checks[key] = {'bytes': archive.stat().st_size, 'sha256': expected['archive_sha256']}

    # Git staging is read-only: exact HEAD, addition-only index, staged path list, and blob bytes.
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == EXPECTED_HEAD
    staged_raw = git('diff', '--cached', '--name-only', '-z')
    staged = [x.decode() for x in staged_raw.split(b'\0') if x]
    expected_stage_bytes = (B / 'publication027-staged-paths-revision002.nul').read_bytes()
    expected_stage = [x.decode() for x in expected_stage_bytes.split(b'\0') if x]
    assert sha_bytes(expected_stage_bytes) == EXPECTED_STAGE_SHA
    assert len(staged) == len(expected_stage) == 511 and staged == expected_stage
    assert len(set(staged)) == 511
    added = [x.decode() for x in git('diff', '--cached', '--diff-filter=A', '--name-only', '-z').split(b'\0') if x]
    modified = [x.decode() for x in git('diff', '--cached', '--diff-filter=M', '--name-only', '-z').split(b'\0') if x]
    deleted = [x.decode() for x in git('diff', '--cached', '--diff-filter=D', '--name-only', '-z').split(b'\0') if x]
    assert len(added) == 506 and len(modified) == 5 and not deleted
    assert set(added) | set(modified) == set(staged)
    assert not [x for x in git('diff', '--name-only', '-z').split(b'\0') if x]
    untracked = [x.decode() for x in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if x]
    assert set(untracked) == EXPECTED_UNTRACKED
    nonarchive = sorted(set(staged) - new_path_set)
    assert len(nonarchive) == 6
    assert set(modified).issubset(nonarchive)
    assert len(set(added) - new_path_set) == 1
    assert not set(untracked).intersection(staged)

    index = {}
    for item in git('ls-files', '--stage', '-z').split(b'\0'):
        if not item:
            continue
        meta, path = item.split(b'\t', 1)
        mode, oid, stage_number = meta.decode().split(' ')
        assert stage_number == '0'
        index[path.decode()] = (mode, oid)
    assert set(staged).issubset(index)
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=R, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    blob_checks = 0
    for path in staged:
        mode, oid = index[path]
        proc.stdin.write(oid.encode() + b'\n')
        proc.stdin.flush()
        header = proc.stdout.readline().decode().strip().split()
        assert len(header) == 3 and header[0] == oid and header[1] == 'blob'
        size = int(header[2])
        blob = proc.stdout.read(size)
        assert len(blob) == size and proc.stdout.read(1) == b'\n'
        worktree = R / path
        assert worktree.is_file() and not worktree.is_symlink()
        assert len(blob) == worktree.stat().st_size and sha_bytes(blob) == sha_file(worktree), f'index/worktree mismatch: {path}'
        assert mode in ('100644', '100755')
        blob_checks += 1
    proc.stdin.close()
    assert proc.wait() == 0 and blob_checks == 511

    # Bind the narrow source/native evidence claims without upgrading physical qualification.
    integration = json.loads((B / 'analysis/source154-root-integration-001/report.json').read_text())
    assert integration['source_files_verified'] == 787
    assert integration['all_selected_root_source_hashes_equal_snapshot154'] is True
    parity = json.loads((B / 'analysis/source154-default64-native466-parity-001/report.json').read_text())
    assert parity['parity_pass'] is True and parity['source_files_verified_per_snapshot'] == 787
    guard = json.loads((B / 'analysis/native467-launch-guard-001/receipt.json').read_text())
    assert guard['space_abort'] is True
    interrupted = json.loads((B / 'analysis/native467-capacity-interruption-audit-001/report.json').read_text())
    assert 'capacity' in json.dumps(interrupted).lower() or 'space' in json.dumps(interrupted).lower()
    assert set(binding['six_axis_verdicts']) == {
        'solver_correctness', 'numerical_convergence', 'component_calibration',
        'analogue_validation', 'physical_assembly_in_simulation', 'crush_impact_prediction'
    }
    assert all(isinstance(v, dict) and 'status' in v for v in binding['six_axis_verdicts'].values())
    assert binding['physical_validation'] is False

    old_audit = B / 'analysis/publication027-independent-byte-audit-001/audit.py'
    assert old_audit.is_file()
    report = {
        'schema': 'numi.cardboard.publication027-independent-audit.v2',
        'result': 'PASS_BYTE_PREFIX_SCOPE_INDEX_AUDIT',
        'revision': 2,
        'head': head,
        'binding_sha256': sha_bytes(binding_bytes),
        'summary_sha256': sha_bytes(summary_bytes),
        'staged_pathlist_sha256': sha_bytes(expected_stage_bytes),
        'staged_paths': len(staged),
        'staged_scope_exact': True,
        'staged_status_counts': {'added': len(added), 'modified': len(modified), 'deleted': len(deleted)},
        'worktree_matches_index_blob_count': blob_checks,
        'unstaged_diff_paths': 0,
        'untracked_preserved': untracked,
        'new_archive_path_count': len(new_path_set),
        'new_nonarchive_paths': nonarchive,
        'prior_026_pathspec_entries': len(prior_paths),
        'prior_027_pathspec_entries': PRIOR_027_PATH_ENTRIES,
        'current_027_pathspec_entries': len(path_entries),
        'prior_027_pathspec_exact_byte_prefix': True,
        'historical_summary_record_count': 1424,
        'historical_summary_prefix_exact': True,
        'pre_addendum_summary_record_count': 1457,
        'pre_addendum_summary_unchanged_except_three_pointer_rows': changed_prior_records,
        'new_summary_record_count': len(summary['records']),
        'postseal_addendum_case_names': expected_added_cases,
        'case_count': len(cases),
        'case_file_counts': case_file_counts,
        'payload_files': archive_rows,
        'raw_files': raw,
        'gzip_files': compressed,
        'input_alias_verification': alias_audits,
        'postseal_compaction_pointer_verification': pointer_audits,
        'buildhost_compaction_archive_verification': archive_checks,
        'source154_selected_hash_count': integration['source_files_verified'],
        'native466_full_attempt_parity': parity['parity_pass'],
        'native467_capacity_interruption_preserved': guard['space_abort'],
        'six_axis_verdicts': binding['six_axis_verdicts'],
        'physical_validation': False,
        'prior_v1_audit_failure_preserved': {
            'path': 'analysis/publication027-independent-byte-audit-001/audit.py',
            'sha256': sha_file(old_audit),
            'observed_failure': "AssertionError for solver/cgls-regional-state-capture-464: external allowed-file inventory had 47 entries while the pre-addendum row listed 46; the additional path was inputs.compaction.json.",
            'disposition': 'Preserved as a valid refusal of the stale prospective inventory; revision002 binds all three recovery pointers and both compaction cases.'
        }
    }
    report_path = PKG / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    stdout = {
        'result': report['result'], 'cases': len(cases), 'payloads': archive_rows,
        'raw': raw, 'gzip': compressed, 'pathspec_entries': len(path_entries),
        'staged_paths': len(staged), 'index_blob_checks': blob_checks,
        'postseal_pointers': len(pointer_audits), 'nonarchive_stage_paths': nonarchive,
    }
    (PKG / 'audit.stdout.json').write_text(json.dumps(stdout, indent=2, sort_keys=True) + '\n')
    print(json.dumps(stdout, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
