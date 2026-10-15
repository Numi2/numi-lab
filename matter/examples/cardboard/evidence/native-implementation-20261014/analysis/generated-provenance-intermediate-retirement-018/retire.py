#!/usr/bin/env python3
"""Verified retirement of generated compiler intermediates in closed provenance builds."""
from pathlib import Path
import hashlib, json, os, re, subprocess, sys, time

ROOT = Path('/Users/home/numi-cardboard-20261006')
BASE = Path('/Users/home/cardboard-implementation-20261014')
OUT = BASE / 'analysis/generated-provenance-intermediate-retirement-018'
ALLOWED_SUFFIXES = {'.o', '.d', '.air'}
MAX_BUILD = 132
TARGET_FREE = 2 * 1024**3


def sha_file(path):
    h = hashlib.sha256()
    n = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def map_sha(mapping):
    raw = json.dumps(mapping, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def fs_available():
    return os.statvfs(BASE).f_bavail * os.statvfs(BASE).f_frsize


def generated_files(build):
    rows = []
    for path in sorted(build.rglob('*')):
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        try:
            st = path.lstat()
        except FileNotFoundError:
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'not a regular generated file: {path}')
        digest, size = sha_file(path)
        rows.append({
            'path': str(path), 'relative_path': str(path.relative_to(build)),
            'suffix': path.suffix, 'bytes': size, 'allocated_bytes': st.st_blocks * 512,
            'mode': st.st_mode, 'mtime_ns': st.st_mtime_ns, 'link_count': st.st_nlink,
            'sha256': digest,
        })
    return rows


def lsof_build(build):
    result = subprocess.run(['lsof', '-nP', '+D', str(build)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr}


def validate_build(build, rec, record_sha):
    failures = []
    snap = Path(rec.get('source_snapshot', ''))
    if not snap.is_dir():
        failures.append('source_snapshot_missing')
    source_map = rec.get('source_sha256', {})
    for relative, expected in source_map.items():
        path = snap / relative
        if not path.is_file() or path.is_symlink():
            failures.append(f'source_missing_or_symlink:{relative}')
            break
        actual, _ = sha_file(path)
        if actual != expected:
            failures.append(f'source_hash_mismatch:{relative}')
            break
    metadata = rec.get('build_metadata_sha256', {})
    for relative, expected in metadata.items():
        path = build / relative
        if not path.is_file() or path.is_symlink():
            failures.append(f'build_metadata_missing_or_symlink:{relative}')
            break
        actual, _ = sha_file(path)
        if actual != expected:
            failures.append(f'build_metadata_hash_mismatch:{relative}')
            break
    artifacts = rec.get('artifacts_sha256', {})
    for raw_path, expected in artifacts.items():
        path = Path(raw_path)
        try:
            path.resolve(strict=True).relative_to(build.resolve(strict=True))
        except Exception:
            failures.append(f'artifact_outside_build:{raw_path}')
            break
        if not path.is_file():
            failures.append(f'artifact_missing:{raw_path}')
            break
        actual, _ = sha_file(path)
        if actual != expected:
            failures.append(f'artifact_hash_mismatch:{raw_path}')
            break
    record_path = build / 'source-build.json'
    actual_record_sha, _ = sha_file(record_path)
    if actual_record_sha != record_sha:
        failures.append('source_build_record_changed')
    return {
        'ok': not failures, 'failures': failures,
        'source_count': len(source_map), 'source_map_sha256': map_sha(source_map),
        'metadata_count': len(metadata), 'metadata_map_sha256': map_sha(metadata),
        'artifact_count': len(artifacts), 'artifact_map_sha256': map_sha(artifacts),
    }


def candidate_builds():
    result = []
    for build in sorted(ROOT.glob('build-cardboard-provenance-*')):
        if not build.is_dir():
            continue
        match = re.fullmatch(r'build-cardboard-provenance-(\d+)(.*)', build.name)
        if not match or int(match.group(1)) > MAX_BUILD:
            continue
        record_path = build / 'source-build.json'
        if not record_path.is_file():
            continue
        generated = generated_files(build)
        if generated:
            result.append((build, match, record_path, generated))
    return result


def prepare():
    builds = []
    for build, match, record_path, generated in candidate_builds():
        raw_record = record_path.read_bytes()
        rec = json.loads(raw_record)
        record_sha = hashlib.sha256(raw_record).hexdigest()
        verification = validate_build(build, rec, record_sha)
        open_check = lsof_build(build)
        if open_check['stdout'].strip() or open_check['stderr'].strip():
            verification['failures'].append('open_handles_or_lsof_error')
            verification['ok'] = False
        if open_check['returncode'] not in (0, 1):
            verification['failures'].append('lsof_unexpected_exit')
            verification['ok'] = False
        builds.append({
            'build': str(build), 'build_number': int(match.group(1)),
            'suffix': match.group(2), 'source_build_record_sha256': record_sha,
            'source_snapshot': rec.get('source_snapshot'),
            'source_build_success': rec.get('success'),
            'verification': verification, 'lsof': open_check,
            'generated_files': generated,
            'generated_logical_bytes': sum(row['bytes'] for row in generated),
            'generated_allocated_bytes': sum(row['allocated_bytes'] for row in generated),
        })
    manifest = {
        'schema': 'numi.cardboard.generated-provenance-intermediate-retirement.v1',
        'created_utc_epoch': time.time(), 'scope': {
            'build_root': str(ROOT), 'maximum_numeric_build': MAX_BUILD,
            'allowed_generated_suffixes': sorted(ALLOWED_SUFFIXES),
            'deleted_directories': False,
            'preserved': ['binaries', 'dylibs', 'metallibs', 'CMake metadata',
                          'source snapshots', 'receipts', 'histories'],
        },
        'filesystem_available_before_bytes': fs_available(),
        'build_count': len(builds), 'file_count': sum(len(b['generated_files']) for b in builds),
        'allocated_bytes': sum(b['generated_allocated_bytes'] for b in builds),
        'logical_bytes': sum(b['generated_logical_bytes'] for b in builds),
        'all_verified': all(b['verification']['ok'] for b in builds),
        'builds': builds,
    }
    if not manifest['all_verified']:
        raise RuntimeError('preflight verification failed; see errors')
    raw = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    path = OUT / 'before-inventory.json'
    if path.exists():
        raise RuntimeError(f'refusing overwrite of {path}')
    path.write_bytes(raw)
    seal = {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    sealpath = OUT / 'before-inventory.seal.json'
    if sealpath.exists():
        raise RuntimeError(f'refusing overwrite of {sealpath}')
    sealpath.write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('filesystem_available_before_bytes','build_count','file_count','allocated_bytes','logical_bytes','all_verified')}))
    print('before_sha256', seal['sha256'])


def execute():
    inventory_path = OUT / 'before-inventory.json'
    seal_path = OUT / 'before-inventory.seal.json'
    inv_raw = inventory_path.read_bytes()
    inv = json.loads(inv_raw)
    seal = json.loads(seal_path.read_text())
    if hashlib.sha256(inv_raw).hexdigest() != seal['sha256']:
        raise RuntimeError('before-inventory seal mismatch')
    start_available = fs_available()
    deleted = []
    skipped = []
    for build_rec in inv['builds']:
        build = Path(build_rec['build'])
        if not build.is_dir():
            skipped.append({'build': str(build), 'reason': 'build_missing'})
            continue
        record_path = build / 'source-build.json'
        if not record_path.is_file():
            skipped.append({'build': str(build), 'reason': 'record_missing'})
            continue
        raw_record = record_path.read_bytes()
        if hashlib.sha256(raw_record).hexdigest() != build_rec['source_build_record_sha256']:
            skipped.append({'build': str(build), 'reason': 'record_changed'})
            continue
        rec = json.loads(raw_record)
        verification = validate_build(build, rec, build_rec['source_build_record_sha256'])
        if not verification['ok']:
            skipped.append({'build': str(build), 'reason': 'provenance_or_artifact_mismatch', 'failures': verification['failures']})
            continue
        current = generated_files(build)
        expected = build_rec['generated_files']
        if current != expected:
            skipped.append({'build': str(build), 'reason': 'generated_file_set_changed'})
            continue
        open_check = lsof_build(build)
        if open_check['stdout'].strip() or open_check['stderr'].strip() or open_check['returncode'] not in (0, 1):
            skipped.append({'build': str(build), 'reason': 'open_handle_or_lsof_error', 'lsof': open_check})
            continue
        # Hardlinks can make deletion affect an out-of-scope path; keep those files.
        unlink_rows = [row for row in expected if row['link_count'] == 1]
        retained_link_rows = [row for row in expected if row['link_count'] != 1]
        for row in unlink_rows:
            path = Path(row['path'])
            actual, size = sha_file(path)
            st = path.lstat()
            if actual != row['sha256'] or size != row['bytes'] or st.st_blocks * 512 != row['allocated_bytes']:
                raise RuntimeError(f'last-moment file mismatch: {path}')
            path.unlink()
            deleted.append({'path': str(path), 'sha256': actual, 'bytes': size,
                            'allocated_bytes_before': st.st_blocks * 512, 'build': str(build)})
            if fs_available() >= TARGET_FREE:
                break
        if retained_link_rows:
            skipped.append({'build': str(build), 'reason': 'hardlinked_files_retained', 'paths': [r['path'] for r in retained_link_rows]})
        if fs_available() >= TARGET_FREE:
            break
    end_available = fs_available()
    post_builds = []
    for build_rec in inv['builds']:
        build = Path(build_rec['build'])
        rec = json.loads((build/'source-build.json').read_text())
        verification = validate_build(build, rec, build_rec['source_build_record_sha256'])
        post_builds.append({'build': str(build), 'verification': verification})
    after = {
        'schema': 'numi.cardboard.generated-provenance-intermediate-retirement.after.v1',
        'before_inventory_sha256': seal['sha256'],
        'started_available_bytes': start_available, 'finished_available_bytes': end_available,
        'filesystem_delta_bytes': end_available-start_available,
        'target_available_bytes': TARGET_FREE, 'target_reached': end_available >= TARGET_FREE,
        'deleted_file_count': len(deleted), 'deleted_logical_bytes': sum(r['bytes'] for r in deleted),
        'deleted_allocated_bytes': sum(r['allocated_bytes_before'] for r in deleted),
        'remaining_candidates_exhausted': all(not generated_files(Path(b['build'])) for b in inv['builds']),
        'deleted': deleted, 'skipped': skipped, 'post_build_verification': post_builds,
    }
    outpath = OUT / 'after-report.json'
    if outpath.exists():
        raise RuntimeError(f'refusing overwrite of {outpath}')
    outpath.write_text(json.dumps(after, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k:after[k] for k in ('started_available_bytes','finished_available_bytes','filesystem_delta_bytes','deleted_file_count','deleted_logical_bytes','deleted_allocated_bytes','target_reached','remaining_candidates_exhausted')}))
    print('after_sha256', sha_file(outpath)[0])
    print('skipped_builds',json.dumps(skipped))
    if any(not x['verification']['ok'] for x in post_builds):
        raise RuntimeError('post-delete final artifact/source verification failed')

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'prepare','execute'}:
        raise SystemExit('usage: retire.py prepare|execute')
    {'prepare': prepare, 'execute': execute}[sys.argv[1]]()
