#!/usr/bin/env python3
"""Independent read-only publication029 audit: staged blobs, source payloads, and append-only indices."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from collections import Counter

REPO = Path('/Users/home/numi-cardboard-20261006')
BASE = Path('/Users/home/cardboard-implementation-20261014')
EVIDENCE = REPO / 'matter/examples/cardboard/evidence/native-implementation-20261014'
SNAPSHOT = BASE / 'snapshots/source-reference-jacobian-168'
OUT = BASE / 'analysis/publication029-independent-byte-audit-001'
EXPECTED_HEAD = 'a5ccffa52ffe368b49c877094a5081b7ee246ebf'
STAGE_PATHLIST = BASE / 'publication029-staged-paths.nul'
PREVIOUS_PATHSPEC = BASE / 'publication-reference028-pathspec.nul'
CURRENT_PATHSPEC = BASE / 'publication-reference029-pathspec.nul'


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with path.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            n += len(block)
            h.update(block)
    return n, h.hexdigest()


def require(test: bool, message: str) -> None:
    if not test:
        raise RuntimeError(message)


def nul_entries(data: bytes) -> list[bytes]:
    if data and not data.endswith(b'\0'):
        raise RuntimeError('NUL path list missing terminal delimiter')
    return [x for x in data.split(b'\0') if x]


def git(*args: str, binary: bool = True) -> bytes:
    return subprocess.run(['git', '-C', str(REPO), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def worktree_content(path: Path, mode: str) -> tuple[int, str, int]:
    st = path.lstat()
    if mode == '120000':
        require(stat.S_ISLNK(st.st_mode), f'expected symlink: {path}')
        payload = os.fsencode(os.readlink(path))
        return len(payload), sha_bytes(payload), st.st_mode
    require(stat.S_ISREG(st.st_mode), f'expected regular file: {path}')
    return (*sha_file(path), st.st_mode)


def cat_file_hashes(entries: list[tuple[str, str, str]]) -> dict[str, tuple[int, str]]:
    """Stream each index blob through git cat-file --batch and verify its object bytes."""
    unique = {}
    for _mode, oid, _path in entries:
        unique.setdefault(oid, None)
    proc = subprocess.Popen(['git', '-C', str(REPO), 'cat-file', '--batch'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    result: dict[str, tuple[int, str]] = {}
    try:
        for idx, oid in enumerate(unique, 1):
            proc.stdin.write((oid + '\n').encode('ascii'))
            proc.stdin.flush()
            header = proc.stdout.readline().decode('ascii').strip().split()
            require(len(header) == 3 and header[0] == oid and header[1] == 'blob',
                    f'bad cat-file header for {oid}: {header!r}')
            size = int(header[2])
            h = hashlib.sha256()
            left = size
            while left:
                block = proc.stdout.read(min(1024 * 1024, left))
                require(bool(block), f'truncated git object {oid}, {left} bytes remain')
                left -= len(block)
                h.update(block)
            require(proc.stdout.read(1) == b'\n', f'missing cat-file delimiter after {oid}')
            result[oid] = (size, h.hexdigest())
            if idx % 500 == 0:
                print(f'streamed {idx}/{len(unique)} distinct staged blobs', file=sys.stderr, flush=True)
        proc.stdin.close()
        stderr = proc.stderr.read().decode('utf-8', 'replace') if proc.stderr else ''
        rc = proc.wait()
        require(rc == 0, f'git cat-file --batch exited {rc}: {stderr}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    binding_path = EVIDENCE / 'publication-binding-reference029.json'
    summary_path = EVIDENCE / 'summary.json'
    prior_summary_path = EVIDENCE / 'summary-reference028.json'
    prior_binding_path = EVIDENCE / 'publication-binding-reference028.json'
    docs = [REPO / 'matter/examples/cardboard/implementation-20261014.md',
            REPO / 'matter/examples/cardboard/solver-progress-20261014.md',
            REPO / 'matter/examples/cardboard/solver-milestone-source167.md']

    binding_bytes = binding_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    prior_summary_bytes = prior_summary_path.read_bytes()
    prior_binding_bytes = prior_binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    summary = json.loads(summary_bytes)
    previous = json.loads(prior_summary_bytes)
    require(sha_bytes(summary_bytes) == binding['summary_sha256'], 'current summary hash does not match publication binding')
    require(binding.get('copied_files') == 2808 and binding.get('raw_files') == 2750 and binding.get('gzip_files') == 58,
            'publication binding raw/gzip payload counters mismatch')

    head = git('rev-parse', 'HEAD').decode().strip()
    require(head == EXPECTED_HEAD, f'HEAD changed: {head}')
    stage_bytes = STAGE_PATHLIST.read_bytes()
    index_delta = git('diff', '--cached', '--name-only', '-z')
    require(stage_bytes == index_delta, 'independent staged path list differs from the publication pathlist')
    staged_paths_bytes = nul_entries(stage_bytes)
    staged_paths = [os.fsdecode(p) for p in staged_paths_bytes]
    require(len(staged_paths) == 2841 and len(set(staged_paths)) == len(staged_paths),
            f'staged path count/uniqueness mismatch: {len(staged_paths)}')
    require(sha_bytes(stage_bytes) == 'afbf36e4dd40647b30ff583f5bd9f22433ba05d355e3b254e5a6d6e27a62482d',
            'stage path-list SHA mismatch')

    index_raw = git('ls-files', '--stage', '-z')
    index_entries: dict[str, tuple[str, str, str]] = {}
    for rec in nul_entries(index_raw):
        meta, sep, raw_path = rec.partition(b'\t')
        require(bool(sep), 'malformed git ls-files --stage entry')
        mode, oid, stage = meta.decode('ascii').split()
        path = os.fsdecode(raw_path)
        if stage == '0':
            index_entries[path] = (mode, oid, stage)
    changed: list[tuple[str, str, str]] = []
    errors = []
    for path in staged_paths:
        ent = index_entries.get(path)
        if ent is None:
            errors.append(f'missing stage-0 index entry: {path}')
            continue
        changed.append((ent[0], ent[1], path))
    require(not errors, '; '.join(errors[:10]))
    objects = cat_file_hashes(changed)
    blob_mismatches = []
    mode_mismatches = []
    total_blob_bytes = 0
    for mode, oid, path in changed:
        obj_size, obj_sha = objects[oid]
        total_blob_bytes += obj_size
        fs_path = REPO / path
        try:
            fs_size, fs_sha, fs_mode = worktree_content(fs_path, mode)
        except Exception as exc:
            blob_mismatches.append({'path': path, 'error': str(exc)})
            continue
        if (fs_size, fs_sha) != (obj_size, obj_sha):
            blob_mismatches.append({'path': path, 'git_size': obj_size, 'git_sha256': obj_sha,
                                    'worktree_size': fs_size, 'worktree_sha256': fs_sha})
        expected_exec = mode == '100755'
        actual_exec = bool(fs_mode & 0o111) if mode != '120000' else False
        if mode in ('100644', '100755') and expected_exec != actual_exec:
            mode_mismatches.append({'path': path, 'index_mode': mode, 'worktree_mode': oct(fs_mode)})
    require(not blob_mismatches, f'{len(blob_mismatches)} staged-object/worktree byte mismatches')
    require(not mode_mismatches, f'{len(mode_mismatches)} staged mode mismatches')

    # Bind and validate the exact 802-file source168 frozen set.
    frozen_path = SNAPSHOT / 'frozen-source.json'
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    source_map = frozen['source_sha256']
    require(len(source_map) == 802, f'expected 802 frozen source paths, got {len(source_map)}')
    require(sha_bytes(frozen_bytes) == binding['source168_frozen_sha256'], 'source168 frozen manifest digest mismatch')
    source_mismatches = []
    for rel, expected_sha in source_map.items():
        snap_file = SNAPSHOT / rel
        repo_file = REPO / rel
        if not snap_file.is_file() or not repo_file.is_file():
            source_mismatches.append({'path': rel, 'snapshot_exists': snap_file.is_file(), 'repo_exists': repo_file.is_file()})
            continue
        snap_size, snap_sha = sha_file(snap_file)
        repo_size, repo_sha = sha_file(repo_file)
        if snap_sha != expected_sha or repo_sha != expected_sha or snap_size != repo_size:
            source_mismatches.append({'path': rel, 'expected_sha256': expected_sha,
                                      'snapshot_sha256': snap_sha, 'repo_sha256': repo_sha,
                                      'snapshot_bytes': snap_size, 'repo_bytes': repo_size})
    require(not source_mismatches, f'{len(source_mismatches)} source168 file mismatches')

    # Append-only summary and pathspec proof.
    require(sha_bytes(prior_binding_bytes) == binding['prior_binding_sha256'], 'prior binding hash mismatch')
    require(sha_bytes(prior_summary_bytes) == binding['historical_prefix']['sha256'], 'prior summary hash mismatch')
    require(len(previous['records']) == 1480 and len(summary['records']) == 1524,
            'historical/new summary record counts differ')
    require(summary['records'][:1480] == previous['records'], 'historical summary prefix changed')
    require(binding['new_case_records'] == 44, 'binding expected 44 newly closed cases')
    expected_cases = binding['cases']
    new_records = summary['records'][1480:]
    require(len(set(expected_cases)) == 44 and {r['case'] for r in new_records} == set(expected_cases),
            'new case binding and summary append do not match exactly')
    require(all(r.get('physical_box_qualification') is False for r in new_records),
            'a new case claims physical box qualification')

    old_spec = PREVIOUS_PATHSPEC.read_bytes()
    new_spec = CURRENT_PATHSPEC.read_bytes()
    old_names = nul_entries(old_spec)
    new_names = nul_entries(new_spec)
    require(len(old_names) == 5566 and len(new_names) == 8376, 'prior/current pathspec counts mismatch')
    require(sha_bytes(old_spec) == binding['prior_pathspec_sha256'], 'prior pathspec SHA mismatch')
    require(new_spec.startswith(old_spec), 'prior pathspec is not a byte-identical prefix')
    delta_names = new_names[len(old_names):]
    require(len(delta_names) == 2810 and len(set(new_names)) == len(new_names), 'pathspec append length/uniqueness mismatch')
    delta_set = {os.fsdecode(x) for x in delta_names}
    stage_set = set(staged_paths)
    require(delta_set.issubset(stage_set), 'one or more newly added pathspec entries are not staged')
    pathspec_untracked = sorted(stage_set - {os.fsdecode(x) for x in new_names})

    # Verify every new raw/compressed payload against both its external original and staged copy.
    records_by_case = {r['case']: r for r in new_records}
    payload_rows = []
    representation_counts = Counter()
    payload_paths = set()
    payload_failures = []
    for case in expected_cases:
        rec = records_by_case[case]
        ext_root = Path(rec['external_root'])
        require(ext_root == BASE / case, f'external root mismatch for {case}: {ext_root}')
        for item in rec['files']:
            src = Path(item['external_path'])
            require(src.is_relative_to(ext_root), f'payload source escapes case root: {src}')
            raw_path = item.get('repository_copy')
            gz_path = item.get('repository_compressed_copy')
            require(bool(raw_path) ^ bool(gz_path), f'payload must have exactly one repository copy: {case}/{item.get("name")}')
            copy_rel = raw_path or gz_path
            require(copy_rel == f"{case}/{item['name']}" if raw_path else copy_rel == f"{case}/{item['name']}.gz",
                    f'unexpected published copy path: {copy_rel}')
            git_copy_rel = f'matter/examples/cardboard/evidence/native-implementation-20261014/{copy_rel}'
            require(git_copy_rel in stage_set, f'payload copy is not in the staged path set: {git_copy_rel}')
            dst = EVIDENCE / copy_rel
            if not src.is_file() or not dst.is_file():
                payload_failures.append({'case': case, 'name': item['name'], 'source_exists': src.is_file(), 'copy_exists': dst.is_file()})
                continue
            src_n, src_sha = sha_file(src)
            copy_n, copy_sha = sha_file(dst)
            if (src_n, src_sha) != (item['bytes'], item['sha256']):
                payload_failures.append({'case': case, 'name': item['name'], 'kind': 'external_original_hash',
                                         'expected': [item['bytes'], item['sha256']], 'actual': [src_n, src_sha]})
            if raw_path:
                representation_counts['raw'] += 1
                ok = (copy_n, copy_sha) == (item['bytes'], item['sha256']) and dst.read_bytes() == src.read_bytes()
                if not ok:
                    payload_failures.append({'case': case, 'name': item['name'], 'kind': 'raw_copy_mismatch'})
            else:
                representation_counts['gzip'] += 1
                if (copy_n, copy_sha) != (item['compressed_bytes'], item['compressed_sha256']):
                    payload_failures.append({'case': case, 'name': item['name'], 'kind': 'compressed_hash_mismatch',
                                             'expected': [item['compressed_bytes'], item['compressed_sha256']],
                                             'actual': [copy_n, copy_sha]})
                h = hashlib.sha256()
                decomp_n = 0
                with gzip.open(dst, 'rb') as f:
                    while True:
                        block = f.read(1024 * 1024)
                        if not block:
                            break
                        h.update(block)
                        decomp_n += len(block)
                if (decomp_n, h.hexdigest()) != (item['bytes'], item['sha256']):
                    payload_failures.append({'case': case, 'name': item['name'], 'kind': 'gzip_restore_mismatch',
                                             'expected': [item['bytes'], item['sha256']],
                                             'actual': [decomp_n, h.hexdigest()]})
            payload_paths.add(copy_rel)
            payload_rows.append({'case': case, 'name': item['name'], 'external_path': str(src),
                                 'external_bytes': item['bytes'], 'external_sha256': item['sha256'],
                                 'repository_copy': copy_rel, 'representation': 'raw' if raw_path else 'gzip',
                                 'repository_bytes': copy_n, 'repository_sha256': copy_sha,
                                 'verified': not any(x.get('case') == case and x.get('name') == item['name'] for x in payload_failures)})
    require(not payload_failures, f'{len(payload_failures)} payload original/copy failures')
    require(representation_counts == Counter({'raw': 2750, 'gzip': 58}), f'payload counts mismatch: {representation_counts}')
    require(len(payload_paths) == 2808, f'payload path uniqueness mismatch: {len(payload_paths)}')

    # Open axes and excluded live run checks.
    verdicts = summary['verdicts']
    axis_names = ['solver_correctness', 'numerical_convergence', 'component_calibration',
                  'analogue_validation', 'physical_assembly_in_simulation', 'crush_impact_prediction']
    require(set(verdicts) == set(axis_names), 'six-axis verdict set mismatch')
    require(set(binding['six_axis_verdicts']) == set(axis_names), 'publication binding six-axis verdict set mismatch')
    require(all(verdicts[x].get('status') not in ('qualified', 'passed', 'complete') for x in axis_names),
            'one qualification axis is marked closed')
    require(all(binding['six_axis_verdicts'][x].get('status') not in ('qualified', 'passed', 'complete') for x in axis_names),
            'publication binding marks one qualification axis closed')
    require(binding.get('physical_validation') is False, 'publication binding physical_validation is not false')
    active_case = 'solver/cgls-dt-half-cycle-470'
    require(active_case not in expected_cases and not any(p.startswith('matter/examples/cardboard/evidence/native-implementation-20261014/' + active_case + '/') for p in staged_paths),
            'active native470 case appears in staged publication')
    require(not any('native470' in p or '/470/' in p for p in staged_paths), 'active470 artifact path appears staged')
    require(binding.get('active_native470_excluded') is True, 'binding does not explicitly exclude active470')

    # Ensure the two currently-untracked work areas remain outside the index.
    untracked_raw = git('ls-files', '--others', '--exclude-standard', '-z')
    untracked = sorted(os.fsdecode(x) for x in nul_entries(untracked_raw))
    untracked_expected = [
        'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/CMakeOBJCXXCompilerId.mm',
        'matter/tools/evidence/cohesive-scratch-alias/source-audit.json',
    ]
    untracked_preserved = untracked == sorted(untracked_expected)
    require(untracked_preserved, 'one or both unrelated untracked areas are missing/unexpectedly staged')
    require(not any(p in stage_set for p in untracked), 'untracked work area is staged')

    warm_record = records_by_case['analysis/warm-start-native-kernel-check-001']
    warm_root = Path(warm_record['external_root'])
    warm_receipt = json.loads((warm_root / 'receipt.json').read_text())
    warm_stderr = (warm_root / 'stderr.log').read_text(errors='replace')
    require(warm_receipt.get('package_manifest_sha256') == '29043b640b6fde5fc66c864ff84d80400cb09771fb5e455368afdd7ae76c7f8d',
            'published warm-start check does not bind the executed v2 package')
    require(warm_receipt.get('returncode') == 1 and warm_receipt.get('gpu_execution') is True and
            warm_receipt.get('physical_transactions') == 0 and 'declined refinement cleared original certificate' in warm_stderr,
            'published warm-start failed-check evidence does not match the stated failed execution')

    doc_records = []
    for p in docs:
        n, h = sha_file(p)
        doc_records.append({'path': str(p), 'bytes': n, 'sha256': h})
    docs_text = {p.name: p.read_text(errors='replace') for p in docs}
    content_checks = {
        'implementation_explicitly_unqualified': 'No complete coupon, resolution study, calibrated corrugated failure, released shipping assembly or crush/drop result is qualified.' in docs_text['implementation-20261014.md'],
        'progress_reports_468_469_failure': 'Native468/469 reject control20' in docs_text['solver-progress-20261014.md'],
        'milestone_lists_six_open_axes': 'All six qualification axes remain distinct.' in docs_text['solver-milestone-source167.md'] and 'still open' in docs_text['solver-milestone-source167.md'],
        'milestone_no_incomplete_prefix_qualification': 'No tolerance was relaxed and no incomplete prefix qualifies the requested shipping sequence.' in docs_text['solver-milestone-source167.md'],
    }
    require(all(content_checks.values()), f'document qualification caveat missing: {content_checks}')

    payload_jsonl = OUT / 'payload-audit.jsonl'
    with payload_jsonl.open('w') as f:
        for row in payload_rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    report = {
        'schema': 'numi.cardboard.publication029_independent_byte_audit.v1',
        'verdict': 'pass',
        'read_only': True,
        'head': head,
        'staged_path_count': len(staged_paths),
        'staged_pathlist_sha256': sha_bytes(stage_bytes),
        'staged_pathlist_matches_git_index_delta': True,
        'git_cat_file_unique_blob_count': len(objects),
        'git_cat_file_blob_bytes_sum_with_duplicates': total_blob_bytes,
        'all_staged_blob_bytes_match_worktree': True,
        'staged_blob_mismatch_count': 0,
        'staged_mode_mismatch_count': 0,
        'source168': {'frozen_source_sha256': sha_bytes(frozen_bytes), 'source_file_count': len(source_map),
                      'all_snapshot_and_repo_files_match_declared_hashes': True, 'mismatch_count': 0},
        'append_only_summary': {'prior_records': len(previous['records']), 'current_records': len(summary['records']),
                                'historical_prefix_exact': True, 'appended_cases': len(expected_cases),
                                'new_case_list_exact': True, 'current_summary_sha256_matches_binding': True},
        'pathspec': {'prior_entries': len(old_names), 'current_entries': len(new_names),
                     'prior_sha256': sha_bytes(old_spec), 'prior_bytes_preserved_as_prefix': True,
                     'current_sha256': sha_bytes(new_spec), 'new_suffix_entries': len(delta_names),
                     'new_suffix_all_staged': True},
        'payloads': {'record_count': len(payload_rows), 'raw_count': representation_counts['raw'],
                     'gzip_count': representation_counts['gzip'], 'all_external_originals_verified': True,
                     'all_staged_restore_bytes_verified': True, 'all_copy_paths_are_staged': True,
                     'unique_published_copy_count': len(payload_paths),
                     'audit_jsonl': str(payload_jsonl)},
        'open_axes': {k: verdicts[k]['status'] for k in axis_names},
        'physical_validation': binding['physical_validation'],
        'native470': {'excluded_from_new_case_list_and_staged_paths': True,
                      'binding_exclusion_flag': binding['active_native470_excluded']},
        'untracked_areas_preserved_unstaged': untracked_preserved,
        'untracked_paths': untracked,
        'published_warm_start_failure': {'executed_package': '159-002', 'returncode': warm_receipt['returncode'],
                                         'physical_transactions': warm_receipt['physical_transactions'],
                                         'failure': 'declined refinement cleared original certificate',
                                         'receipt_sha256': sha_bytes((warm_root/'receipt.json').read_bytes())},
        'reviewed_document_hashes': doc_records,
        'documentation_caveat_checks': content_checks,
        'documentation_overclaim_review': 'latest summaries state component evidence separately and leave complete nonlinear convergence, exact-grade calibration, analogue validation, shipping assembly, and impact open; no physical qualification claimed',
        'staged_source_check_report_input_sha256': sha_bytes((BASE/'analysis/publication029-staged-source-check-001/report.json').read_bytes()),
        'binding_sha256': sha_bytes(binding_bytes),
        'summary_sha256': sha_bytes(summary_bytes),
        'prior_binding_sha256': sha_bytes(prior_binding_bytes),
        'prior_summary_sha256': sha_bytes(prior_summary_bytes),
        'prior_pathspec_sha256': sha_bytes(old_spec),
        'stage_pathlist_expected_sha256': 'afbf36e4dd40647b30ff583f5bd9f22433ba05d355e3b254e5a6d6e27a62482d',
    }
    report_path = OUT / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'verdict': report['verdict'], 'staged_paths': len(staged_paths),
                      'blobs': len(objects), 'blob_bytes': total_blob_bytes,
                      'payloads': len(payload_rows), 'report': str(report_path)}, sort_keys=True))

if __name__ == '__main__':
    main()
