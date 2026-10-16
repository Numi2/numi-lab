from pathlib import Path
import datetime
import hashlib
import json
import subprocess

R = Path('/Users/home/numi-cardboard-20261006')
B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/publication028-commit-and-push-001'
O.mkdir(exist_ok=False)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):
    return subprocess.run(args, cwd=R, capture_output=True)
def checked(args):
    result = run(args)
    assert result.returncode == 0, (args, result.stderr.decode())
    return result.stdout

audit_path = B / 'analysis/publication028-independent-byte-audit-002/report.json'
assert sha(audit_path) == '2ff60ecac3cfde338f0c68e64a708447673950f4459ccb6988df8a0817b88ea1'
audit = json.loads(audit_path.read_text())
assert audit['verdict'] == 'PASS' and not audit['failures']
assert all(check['pass'] is True for check in audit['checks'].values())
assert audit['new_cases'] == 20 and audit['payloads']['total'] == 715
assert audit['source157']['source_files_verified'] == 787
assert audit['staging']['paths'] == 722 and audit['staging']['worktree_byte_equal'] is True
assert audit['staging']['unstaged_tracked'] == 0
parent = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
assert parent == audit['head'] == '2298e1a1d49dd09fa149c57f65ccbfaab3985c29'
assert checked(['git', 'branch', '--show-current']).decode().strip() == 'main'
remote_before = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert remote_before == parent
pathlist = B / 'publication028-staged-paths.nul'
assert sha(pathlist) == audit['staging']['stage_list_sha256']
expected = set(pathlist.read_bytes().split(b'\0')) - {b''}
actual = set(checked(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')) - {b''}
assert expected == actual and len(actual) == 722
assert checked(['git', 'diff', '--name-only', '-z']) == b''
assert checked(['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z']) == b''
for path in sorted(expected):
    name = path.decode()
    assert checked(['git', 'show', ':' + name]) == (R / name).read_bytes(), name
summary = R / 'matter/examples/cardboard/evidence/native-implementation-20261014/summary.json'
assert sha(summary) == audit['summary_sha256']
source_record = json.loads((B / 'analysis/source157-telemetry-native-build-001/build-metadata/source-build.json').read_text())
source_map = source_record['source_sha256']
assert len(source_map) == 787
for name, digest in source_map.items():
    assert sha(R / name) == digest, name
preflight = {'parent': parent, 'remote_main_before': remote_before,
             'audit_sha256': sha(audit_path), 'staged_pathlist_sha256': sha(pathlist),
             'staged_paths': len(expected), 'source_hashes_verified': len(source_map),
             'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
commit = run(['git', '-c', 'gc.auto=0', 'commit', '-m',
              'Correct cardboard grip telemetry and bind retained evidence'])
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
receipt = {'schema': 'numi.cardboard.publication028_commit_push.v1', **preflight,
           'commit': head, 'remote_main': published, 'remote_verified': True,
           'commit_exit_code': commit.returncode, 'push_exit_code': push.returncode,
           'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'qualification': 'Telemetry, source retention and bounded evidence milestone; all six physical qualification axes remain open.'}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'commit': head, 'remote_verified': True}))
