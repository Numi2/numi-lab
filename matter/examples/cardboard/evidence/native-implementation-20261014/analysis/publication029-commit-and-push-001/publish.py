from pathlib import Path
import datetime
import hashlib
import json
import subprocess

R = Path('/Users/home/numi-cardboard-20261006')
B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/publication029-commit-and-push-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def run(args, **kw):
    return subprocess.run(args, cwd=R, capture_output=True, **kw)
def checked(args, **kw):
    result = run(args, **kw)
    assert result.returncode == 0, (args, result.stderr.decode())
    return result.stdout

audit_path = B / 'analysis/publication029-independent-byte-audit-001/report.json'
assert sha(audit_path) == 'a14f142091035afaf4b65697606da71e3b51bc7c5a972a3c1b71941a37499896'
audit = json.loads(audit_path.read_text())
assert audit['verdict'] == 'pass'
assert audit['all_staged_blob_bytes_match_worktree'] is True
assert audit['payloads']['all_external_originals_verified'] is True
assert audit['payloads']['all_staged_restore_bytes_verified'] is True
assert audit['payloads']['record_count'] == 2808
assert audit['source168']['source_file_count'] == 802
assert audit['source168']['mismatch_count'] == 0
assert audit['native470']['excluded_from_new_case_list_and_staged_paths'] is True
assert audit['physical_validation'] is False
parent = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
assert parent == audit['head'] == 'a5ccffa52ffe368b49c877094a5081b7ee246ebf'
assert checked(['git', 'branch', '--show-current']).decode().strip() == 'main'
remote_before = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert remote_before == parent
pathlist = B / 'publication029-staged-paths.nul'
assert sha(pathlist) == audit['staged_pathlist_sha256']
expected = set(pathlist.read_bytes().split(b'\0')) - {b''}
actual = set(checked(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')) - {b''}
assert expected == actual and len(actual) == 2841
assert checked(['git', 'diff', '--name-only', '-z']) == b''
assert checked(['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z']) == b''
request = b''.join(b':' + name + b'\n' for name in sorted(expected))
blobs = checked(['git', 'cat-file', '--batch'], input=request)
offset = 0
for name in sorted(expected):
    end = blobs.index(b'\n', offset)
    fields = blobs[offset:end].split()
    assert len(fields) == 3 and fields[1] == b'blob', fields
    size = int(fields[2])
    offset = end + 1
    assert blobs[offset:offset + size] == (R / name.decode()).read_bytes(), name
    offset += size
    assert blobs[offset:offset + 1] == b'\n'
    offset += 1
assert offset == len(blobs)
summary = R / 'matter/examples/cardboard/evidence/native-implementation-20261014/summary.json'
assert sha(summary) == audit['summary_sha256']
frozen = B / 'snapshots/source-reference-jacobian-168/frozen-source.json'
assert sha(frozen) == audit['source168']['frozen_source_sha256']
sources = json.loads(frozen.read_text())['source_sha256']
assert len(sources) == 802
for name, digest in sources.items():
    assert sha(R / name) == digest, name
preflight = {'parent': parent, 'remote_main_before': remote_before,
             'audit_sha256': sha(audit_path), 'staged_pathlist_sha256': sha(pathlist),
             'staged_paths': len(expected), 'source_hashes_verified': len(sources),
             'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
commit = run(['git', '-c', 'gc.auto=0', 'commit', '-m',
              'Add accepted cardboard checkpoints and repair independent tangent reference'])
(O / 'commit.stdout').write_bytes(commit.stdout)
(O / 'commit.stderr').write_bytes(commit.stderr)
assert commit.returncode == 0, commit.stderr.decode()
head = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
push = run(['git', '-c', 'gc.auto=0', 'push', 'origin', 'main'])
(O / 'push.stdout').write_bytes(push.stdout)
(O / 'push.stderr').write_bytes(push.stderr)
assert push.returncode == 0, push.stderr.decode()
published = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert published == head
(O / 'publish.py').write_bytes(Path(__file__).read_bytes())
receipt = {'schema': 'numi.cardboard.publication029_commit_push.v1', **preflight,
           'commit': head, 'remote_main': published, 'remote_verified': True,
           'commit_exit_code': commit.returncode, 'push_exit_code': push.returncode,
           'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'qualification': 'Accepted-checkpoint infrastructure and independent-reference correction milestone; all six qualification axes remain open.'}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'commit': head, 'remote_verified': True}))
