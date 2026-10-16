from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'publication029-payload-staging'
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
O = B / 'analysis/publication029-duplicate-staging-retirement-001'
O.mkdir(exist_ok=False)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=R)
def free():
    v = os.statvfs(B)
    return v.f_bavail * v.f_frsize
head = git('rev-parse', 'HEAD').decode().strip()
assert head == '22b927a002b8006a7bf5026e5bf3873a382259dd'
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert git('diff', '--name-only', '-z') == b''
assert git('diff', '--cached', '--name-only', '-z') == b''
entries = {}
for raw in git('ls-tree', '-r', '-z', 'HEAD', '--', str(D.relative_to(R))).split(b'\0'):
    if raw:
        meta, path = raw.split(b'\t', 1)
        mode, kind, oid = meta.split()
        assert kind == b'blob'
        entries[path.decode()] = (mode.decode(), oid.decode())
rows = []
allocated = 0
for directory, dirs, files in os.walk(S):
    assert not any((Path(directory) / name).is_symlink() for name in dirs)
    for name in sorted(files):
        p = Path(directory) / name
        assert p.is_file() and not p.is_symlink()
        target = D / p.relative_to(S)
        assert target.is_file() and not target.is_symlink()
        data = p.read_bytes()
        assert target.read_bytes() == data
        st, other = p.stat(), target.stat()
        assert (st.st_dev, st.st_ino) != (other.st_dev, other.st_ino)
        key = str(target.relative_to(R))
        mode, oid = entries[key]
        digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert mode == '100644' and oid == digest, key
        allocated += st.st_blocks * 512
        rows.append({'duplicate': str(p), 'retained': str(target), 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': oid,
                     'allocated_bytes_before': st.st_blocks * 512})
assert len(rows) == 2808
handles = subprocess.run(['lsof', '-nP', '+D', str(S)], capture_output=True)
(O / 'lsof.stdout').write_bytes(handles.stdout)
(O / 'lsof.stderr').write_bytes(handles.stderr)
assert handles.returncode == 1 and handles.stdout == b'' and handles.stderr == b''
(O / 'verified-duplicates.json').write_text(json.dumps(rows, indent=2) + '\n')
before = free()
for row in rows:
    p = Path(row['duplicate'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    assert hashlib.sha256(Path(row['retained']).read_bytes()).hexdigest() == row['sha256']
    p.unlink()
for directory, dirs, files in os.walk(S, topdown=False):
    assert not files
    Path(directory).rmdir()
after = free()
receipt = {'schema': 'numi.cardboard.verified-duplicate-retirement.v1', 'commit': head,
           'remote_commit_verified': True, 'duplicate_files_removed': len(rows),
           'sole_inputs_removed': False, 'retained_external_originals': True,
           'retained_repository_and_remote_git_copies': True,
           'allocated_bytes_in_removed_duplicate_files': allocated,
           'available_before_bytes': before, 'available_after_bytes': after,
           'measured_available_difference_bytes': after - before,
           'active_native_outputs_touched': False,
           'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(O / 'retire.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt))
