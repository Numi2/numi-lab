from pathlib import Path
import datetime
import hashlib
import json
import subprocess

R = Path('/Users/home/numi-cardboard-20261006')
B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/publication027-commit-and-push-001'
O.mkdir(exist_ok=False)
def run(args):
    return subprocess.run(args, cwd=R, capture_output=True)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(args):
    r = run(args)
    assert r.returncode == 0, (args, r.stderr.decode())
    return r.stdout

ap = B / 'analysis/publication027-independent-byte-audit-002/report.json'
assert sha(ap) == '60408130429486d76a6cf4821c977710c88af913d01eb56f9df0eba2e01b5c80'
a = json.loads(ap.read_text())
assert a['result'] == 'PASS_BYTE_PREFIX_SCOPE_INDEX_AUDIT'
for k in ['historical_summary_prefix_exact', 'prior_027_pathspec_exact_byte_prefix',
          'native466_full_attempt_parity', 'native467_capacity_interruption_preserved',
          'staged_scope_exact']:
    assert a[k] is True, k
assert a['physical_validation'] is False
assert a['staged_paths'] == a['worktree_matches_index_blob_count'] == 511
assert a['unstaged_diff_paths'] == 0
assert a['source154_selected_hash_count'] == 787
assert a['payload_files'] == 503 and a['case_count'] == 36
head = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
assert head == a['head'] == '67ba06be7f48981dda142ea35eedd792d7cd2ca5'
assert checked(['git', 'branch', '--show-current']).decode().strip() == 'main'
remote = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert remote == head
pathlist = B / 'publication027-staged-paths-revision002.nul'
assert sha(pathlist) == a['staged_pathlist_sha256']
expected = set(pathlist.read_bytes().split(b'\0')) - {b''}
actual = set(checked(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')) - {b''}
assert actual == expected
assert checked(['git', 'diff', '--name-only', '-z']) == b''
for p in sorted(expected):
    name = p.decode()
    assert checked(['git', 'show', ':' + name]) == (R / name).read_bytes(), name
assert sha(R / 'matter/examples/cardboard/evidence/native-implementation-20261014/summary.json') == a['summary_sha256']
preflight = {'parent': head, 'remote_main': remote, 'audit_sha256': sha(ap),
             'stage_pathlist_sha256': sha(pathlist), 'staged_paths': len(expected),
             'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
commit = run(['git', '-c', 'gc.auto=0', 'commit', '-m',
              'Extend bounded solver controls and preserve native failure evidence'])
(O / 'commit.stdout').write_bytes(commit.stdout)
(O / 'commit.stderr').write_bytes(commit.stderr)
assert commit.returncode == 0, commit.stderr.decode()
after = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
push = run(['git', '-c', 'gc.auto=0', 'push', 'origin', 'main'])
(O / 'push.stdout').write_bytes(push.stdout)
(O / 'push.stderr').write_bytes(push.stderr)
assert push.returncode == 0, push.stderr.decode()
published = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert published == after
(O / 'publish.py').write_bytes(Path(__file__).read_bytes())
receipt = {'schema': 'numi.cardboard.publication027_commit_push.v1',
           **preflight, 'commit': after, 'remote_main': published, 'remote_verified': True,
           'commit_exit_code': commit.returncode, 'push_exit_code': push.returncode,
           'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'qualification': 'Bounded solver-control and failure-preservation milestone; all six physical qualification axes remain open.'}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'commit': after, 'remote_verified': True}))
