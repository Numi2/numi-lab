#!/usr/bin/env python3
"""Losslessly archive and verify one closed solver inputs directory before compaction."""
from pathlib import Path, PurePosixPath
import argparse, gzip, hashlib, json, os, re, shutil, stat, subprocess, tarfile, time

BASE = Path('/Users/home/cardboard-implementation-20261014')
SOLVER = BASE / 'solver'
OUT = BASE / 'analysis/closed-solver-input-compaction-024'
ARCHIVES = OUT / 'archives'


def sha_stream(stream):
    h = hashlib.sha256(); count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block); count += len(block)
    return h.hexdigest(), count


def sha_file(path):
    with path.open('rb') as stream:
        return sha_stream(stream)


def fs_available():
    st = os.statvfs(BASE)
    return st.f_bavail * st.f_frsize


def lsof_case(case):
    p = subprocess.run(['lsof', '-nP', '+D', str(case)], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def parse_case_number(name):
    numbers = re.findall(r'(?<!\d)(\d{3})(?!\d)', name)
    return int(numbers[-1]) if numbers else None


def inventory(case):
    inp = case / 'inputs'
    if inp.is_symlink() or not inp.is_dir():
        raise RuntimeError('input directory missing or symlink')
    dirs = [inp] + sorted((p for p in inp.rglob('*') if p.is_dir()),
                           key=lambda p: (len(p.relative_to(inp).parts), p.relative_to(inp).as_posix()))
    files = sorted((p for p in inp.rglob('*') if p.is_file()), key=lambda p: p.relative_to(inp).as_posix())
    if any(p.is_symlink() for p in inp.rglob('*')):
        raise RuntimeError('input contains symlink; refusing compaction')
    dir_rows = []
    for p in dirs:
        st = p.stat()
        dir_rows.append({'path': p.relative_to(inp).as_posix() if p != inp else '',
                         'mode': stat.S_IMODE(st.st_mode), 'mtime_ns': st.st_mtime_ns})
    file_rows = []
    for p in files:
        st = p.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise RuntimeError(f'input is nonregular or hardlinked: {p}')
        digest, size = sha_file(p)
        file_rows.append({'path': p.relative_to(inp).as_posix(), 'bytes': size,
                          'sha256': digest, 'mode': stat.S_IMODE(st.st_mode),
                          'mtime_ns': st.st_mtime_ns, 'allocated_bytes': st.st_blocks * 512,
                          'link_count': st.st_nlink})
    return dir_rows, file_rows


def validate_case(case, pred, receipt, dirs, files):
    number = parse_case_number(case.name)
    if case.name not in ['cgls-regional-state-capture-464', 'cgls-newton128-default64-parity-466', 'cgls-complete-outer128-allcontrols-467']:
        raise RuntimeError('case outside explicit task-owned closed-case manifest')
    if number is None or number > 467:
        raise RuntimeError(f'case outside permitted numeric scope: {number}')
    if any(token in case.name.lower() for token in ('review', 'active')):
        raise RuntimeError('case name marks active/review scope')
    if receipt.get('exit_code') is None or not receipt.get('finished_at_utc') or receipt.get('timed_out'):
        raise RuntimeError('case receipt is active, incomplete, or timed out')
    if not receipt.get('source_inputs_unchanged'):
        raise RuntimeError('closed receipt does not confirm unchanged source inputs')
    if (case/'inputs.compaction.json').exists():
        raise RuntimeError('already compacted; refusing repeated mutation')
    snapshot = Path(pred.get('source_snapshot', ''))
    if not snapshot.exists():
        raise RuntimeError('prediction source snapshot is unavailable')
    expected = pred.get('inputs_sha256')
    if not isinstance(expected, dict):
        raise RuntimeError('prediction inputs_sha256 missing')
    actual_paths = {row['path'] for row in files}
    if actual_paths != set(expected):
        raise RuntimeError('input file set differs from prediction inputs_sha256')
    for row in files:
        if expected[row['path']] != row['sha256']:
            raise RuntimeError(f'input hash mismatch: {row["path"]}')


def tar_info(name, row, is_dir):
    ti = tarfile.TarInfo(name)
    ti.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    ti.mode = row['mode']
    ti.size = 0 if is_dir else row['bytes']
    ti.mtime = row['mtime_ns'] / 1e9
    ti.uid = 0; ti.gid = 0; ti.uname = ''; ti.gname = ''
    ti.pax_headers = {'NUMI.mtime_ns': str(row['mtime_ns'])}
    return ti


def write_archive(case, dirs, files, archive_path):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                    for row in dirs:
                        name = 'inputs' + ('/' + row['path'] if row['path'] else '')
                        tar.addfile(tar_info(name, row, True))
                    inp = case / 'inputs'
                    for row in files:
                        name = 'inputs/' + row['path']
                        with (inp / row['path']).open('rb') as stream:
                            tar.addfile(tar_info(name, row, False), stream)
                zipped.flush()
            raw.flush()
            os.fsync(raw.fileno())
    except Exception:
        try: archive_path.unlink()
        except FileNotFoundError: pass
        raise
    dirfd = os.open(archive_path.parent, os.O_RDONLY)
    try: os.fsync(dirfd)
    finally: os.close(dirfd)


def verify_and_stream_restore(case, archive_path, dirs, files, restore_dir):
    if restore_dir.exists():
        raise RuntimeError(f'refusing existing restoration path {restore_dir}')
    restore_dir.mkdir(parents=True)
    expected_dirs = {row['path']: row for row in dirs}
    expected_files = {row['path']: row for row in files}
    seen_dirs = set(); seen_files = set()
    try:
        with tarfile.open(archive_path, mode='r|gz') as tar:
            for member in tar:
                pp = PurePosixPath(member.name)
                if pp.is_absolute() or '..' in pp.parts or not pp.parts or pp.parts[0] != 'inputs':
                    raise RuntimeError(f'unsafe archive member: {member.name}')
                rel_parts = pp.parts[1:]
                rel = '/'.join(rel_parts)
                if member.isdir():
                    if rel not in expected_dirs or rel in seen_dirs:
                        raise RuntimeError(f'unexpected/duplicate archive directory: {member.name}')
                    row = expected_dirs[rel]
                    if member.mode != row['mode']:
                        raise RuntimeError(f'directory mode mismatch: {member.name}')
                    dest = restore_dir / rel if rel else restore_dir
                    dest.mkdir(parents=True, exist_ok=True)
                    seen_dirs.add(rel)
                    continue
                if not member.isfile() or rel not in expected_files or rel in seen_files:
                    raise RuntimeError(f'unexpected/duplicate archive file: {member.name}')
                row = expected_files[rel]
                if member.size != row['bytes'] or member.mode != row['mode']:
                    raise RuntimeError(f'file extent/mode mismatch: {member.name}')
                dest = restore_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None: raise RuntimeError(f'cannot read archive member {member.name}')
                h = hashlib.sha256(); count = 0
                with src, dest.open('xb') as out:
                    for block in iter(lambda:src.read(1024*1024),b''):
                        h.update(block); count += len(block); out.write(block)
                    out.flush(); os.fsync(out.fileno())
                if count != row['bytes'] or h.hexdigest() != row['sha256']:
                    raise RuntimeError(f'file hash mismatch during stream restore: {member.name}')
                os.chmod(dest, row['mode'])
                seen_files.add(rel)
        if seen_dirs != set(expected_dirs) or seen_files != set(expected_files):
            raise RuntimeError('archive member set incomplete')
        # Verify restored paths, content, and modes; preserve original nanosecond mtimes.
        for row in files:
            p = restore_dir / row['path']
            digest, size = sha_file(p)
            if digest != row['sha256'] or size != row['bytes'] or stat.S_IMODE(p.stat().st_mode) != row['mode']:
                raise RuntimeError(f'restored file verification failed: {row["path"]}')
            os.utime(p, ns=(row['mtime_ns'], row['mtime_ns']))
        for row in sorted(dirs, key=lambda r: len(PurePosixPath(r['path']).parts), reverse=True):
            p = restore_dir / row['path'] if row['path'] else restore_dir
            os.chmod(p, row['mode'])
            os.utime(p, ns=(row['mtime_ns'], row['mtime_ns']))
        return {'verified_dirs':len(seen_dirs),'verified_files':len(seen_files),
                'verified_bytes':sum(row['bytes'] for row in files)}
    except Exception:
        shutil.rmtree(restore_dir, ignore_errors=True)
        raise


def compact(case_name):
    case = SOLVER / case_name
    if not case.is_dir() or case.resolve().parent != SOLVER.resolve():
        raise RuntimeError('case path is not a direct solver child')
    pred_path = case / 'prediction.json'; receipt_path = case / 'receipt.json'
    if not pred_path.is_file() or not receipt_path.is_file():
        raise RuntimeError('prediction/receipt missing')
    pred_raw = pred_path.read_bytes(); receipt_raw = receipt_path.read_bytes()
    pred = json.loads(pred_raw); receipt = json.loads(receipt_raw)
    dirs, files = inventory(case)
    validate_case(case, pred, receipt, dirs, files)
    pre_lsof = lsof_case(case)
    if pre_lsof['stdout'].strip() or pre_lsof['stderr'].strip() or pre_lsof['returncode'] not in (0,1):
        raise RuntimeError('case has open handle or lsof error before archive')
    case_dir = OUT / 'cases' / case.name
    case_dir.mkdir(parents=True, exist_ok=False)
    archive_path = ARCHIVES / case.name / 'inputs.tar.gz'
    if archive_path.exists(): raise RuntimeError('archive path already exists; refusing overwrite')
    before_available = fs_available()
    before = {
        'schema':'numi.cardboard.closed-solver-input-compaction.case-before.v1',
        'case':str(case), 'case_number':parse_case_number(case.name),
        'created_utc_epoch':time.time(), 'prediction_path':str(pred_path),
        'prediction_sha256':hashlib.sha256(pred_raw).hexdigest(),
        'receipt_path':str(receipt_path),'receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),
        'receipt':receipt,'source_snapshot':pred['source_snapshot'],
        'source_snapshot_exists':True,'inputs_sha256':pred['inputs_sha256'],
        'inputs_directory_mode':stat.S_IMODE((case/'inputs').stat().st_mode),
        'directories':dirs,'files':files,'file_count':len(files),
        'logical_bytes':sum(row['bytes'] for row in files),
        'allocated_bytes':sum(row['allocated_bytes'] for row in files),
        'filesystem_available_before_bytes':before_available,
        'lsof_before_archive':pre_lsof,
        'archive_path':str(archive_path),
        'archive_policy':{'gzip_mtime':0,'gzip_compresslevel':6,'tar_format':'PAX','member_root':'inputs/',
                          'uid_gid_normalized':0,'members_sorted_directories_then_files':True},
    }
    before_path = case_dir/'before-inventory.json'
    before_raw = (json.dumps(before,sort_keys=True,indent=2)+'\n').encode()
    before_path.write_bytes(before_raw)
    with before_path.open('rb') as stream: os.fsync(stream.fileno())
    before_seal = {'file':before_path.name,'sha256':hashlib.sha256(before_raw).hexdigest(),'bytes':len(before_raw)}
    (case_dir/'before-inventory.seal.json').write_text(json.dumps(before_seal,indent=2)+'\n')
    write_archive(case,dirs,files,archive_path)
    archive_sha,archive_size=sha_file(archive_path)
    restore_dir=OUT/'restore-check'/case.name/'inputs'
    restore_stats=verify_and_stream_restore(case,archive_path,dirs,files,restore_dir)
    shutil.rmtree(restore_dir.parent)
    # Ensure the source did not change while archiving and the full input map still binds it.
    dirs_after,files_after=inventory(case)
    if dirs_after!=dirs or files_after!=files:
        raise RuntimeError('original inputs changed during archive verification')
    pred_after=json.loads(pred_path.read_text())
    if pred_after.get('inputs_sha256')!=pred.get('inputs_sha256'):
        raise RuntimeError('prediction input hash map changed during archive verification')
    pre_delete_lsof=lsof_case(case)
    if pre_delete_lsof['stdout'].strip() or pre_delete_lsof['stderr'].strip() or pre_delete_lsof['returncode'] not in (0,1):
        raise RuntimeError('case has open handle or lsof error before unlink')
    for row in files:
        p=case/'inputs'/row['path']
        digest,size=sha_file(p)
        st=p.stat()
        if digest!=row['sha256'] or size!=row['bytes'] or st.st_nlink!=1:
            raise RuntimeError(f'last-moment original mismatch or hardlink: {row["path"]}')
    for row in files:
        (case/'inputs'/row['path']).unlink()
    pointer={
        'schema':'numi.cardboard.closed-solver-input-compaction.pointer.v1',
        'case':str(case),'archive_path':str(archive_path),'archive_sha256':archive_sha,
        'archive_bytes':archive_size,'before_manifest_path':str(before_path),
        'before_manifest_sha256':before_seal['sha256'],
        'restore_script':str(OUT/'restore_inputs.py'),
        'restore_required_to_recreate_prediction_inputs':True,
    }
    pointer_path=case/'inputs.compaction.json'
    if pointer_path.exists(): raise RuntimeError('existing compaction pointer; refusing overwrite')
    pointer_raw=(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode()
    pointer_path.write_bytes(pointer_raw)
    with pointer_path.open('rb') as stream: os.fsync(stream.fileno())
    after_available=fs_available()
    remaining_files=[p for p in (case/'inputs').rglob('*') if p.is_file()]
    if remaining_files: raise RuntimeError('input files remain after compaction')
    after={
        'schema':'numi.cardboard.closed-solver-input-compaction.case-after.v1',
        'case':str(case),'before_manifest_sha256':before_seal['sha256'],
        'pointer_path':str(pointer_path),'pointer_sha256':hashlib.sha256(pointer_raw).hexdigest(),
        'archive_path':str(archive_path),'archive_sha256':archive_sha,'archive_bytes':archive_size,
        'restoration_verification':restore_stats,'filesystem_available_before_bytes':before_available,
        'filesystem_available_after_bytes':after_available,'filesystem_delta_bytes':after_available-before_available,
        'original_logical_bytes':sum(row['bytes'] for row in files),
        'original_allocated_bytes':sum(row['allocated_bytes'] for row in files),
        'case_input_directories_preserved':True,'original_input_files_removed':len(files),
        'prediction_and_receipt_preserved':True,
        'physical_or_solver_qualification':False,
    }
    after_path=case_dir/'after-report.json'
    after_raw=(json.dumps(after,sort_keys=True,indent=2)+'\n').encode()
    after_path.write_bytes(after_raw)
    with after_path.open('rb') as stream: os.fsync(stream.fileno())
    print(json.dumps(after,sort_keys=True,separators=(',',':')))
    print('before_sha256',before_seal['sha256'])
    print('after_sha256',hashlib.sha256(after_raw).hexdigest())
    print('archive_sha256',archive_sha)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case')
    args=parser.parse_args()
    compact(args.case)
