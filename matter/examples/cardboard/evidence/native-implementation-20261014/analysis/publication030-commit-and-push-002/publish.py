from pathlib import Path
import datetime
import hashlib
import json
import subprocess

R = Path('/Users/home/numi-cardboard-20261006')
B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/publication030-commit-and-push-002'
PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def checked(args, **kw):
    p = subprocess.run(args, cwd=R, capture_output=True, **kw)
    assert p.returncode == 0, (args, p.stderr.decode())
    return p.stdout
def emit(name, value):
    (O / name).write_text(json.dumps(value, indent=2) + '\n')

assert not O.exists()
audit_path = B / 'analysis/publication030-independent-byte-audit-001/report.json'
assert sha(audit_path) == '5ee1e426247f8178844f9a962aed0ea226a7386472e005dcef79cc86c4998817'
audit = json.loads(audit_path.read_text())
assert audit['verdict'] == 'PASS' and not audit['failed_checks']
assert audit['new_publication_payloads']['files'] == 2278
assert audit['historical_prefix']['current_summary_prefix_exact']
assert audit['source173_selected_hashes']['checkout_mismatches'] == 0
assert not audit['active_476_477_results_included']
parent = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
assert parent == '22b927a002b8006a7bf5026e5bf3873a382259dd'
assert checked(['git', 'branch', '--show-current']).decode().strip() == 'main'
assert checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == parent
# Prior attempt staged the allowed non-ignored subset; verify exact subset below.
oldlist = (B / 'publication-reference029-pathspec.nul').read_bytes()
newlist = (B / 'publication-reference030-pathspec.nul').read_bytes()
assert sha(B / 'publication-reference030-pathspec.nul') == audit['pathspec_sha256']
assert newlist.startswith(oldlist)
newpaths = set(newlist[len(oldlist):].split(b'\0')) - {b''}
assert len(newpaths) == 2280
integration = json.loads((B / 'analysis/source173-repository-integration-001/report.json').read_text())
source_paths = {name.encode() for name in integration['changed_files']}
assert len(source_paths) == 10
docs = {('matter/examples/cardboard/' + name).encode() for name in (
    'implementation-20261014.md', 'solver-progress-20261014.md', 'solver-milestone-source173.md')}
expected = newpaths | source_paths | docs | {(PREFIX + 'summary.json').encode()}
assert len(expected) == 2294
tracked = set(checked(['git', 'diff', '--name-only', '-z']).split(b'\0')) - {b''}
assert tracked <= expected
frozen = B / 'snapshots/source-checkpoint-warm-breakdown-173/frozen-source.json'
assert sha(frozen) == audit['source173_frozen_manifest_sha256']
sources = json.loads(frozen.read_text())['source_sha256']
assert len(sources) == 804
for name, digest in sources.items():
    assert sha(R / name) == digest, name
assert sha(R / (PREFIX + 'summary.json')) == audit['summary_sha256']
assert sha(R / (PREFIX + 'publication-binding-reference030.json')) == audit['binding_sha256']
assert checked(['git', 'diff', '--check']) == b''
O.mkdir()
pathlist = B / 'publication030-staged-paths.nul'
assert pathlist.read_bytes() == b''.join(name + b'\0' for name in sorted(expected))
partial = set(checked(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')) - {b''}
assert partial <= expected and len(partial) == 2291
checked(['git', 'add', '-f', '--pathspec-from-file=' + str(pathlist), '--pathspec-file-nul'])
actual = set(checked(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')) - {b''}
assert actual == expected
assert checked(['git', 'diff', '--name-only', '-z']) == b''
assert checked(['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z']) == b''
request = b''.join(b':' + name + b'\n' for name in sorted(expected))
blobs = checked(['git', 'cat-file', '--batch'], input=request)
offset = 0
for name in sorted(expected):
    end = blobs.index(b'\n', offset)
    fields = blobs[offset:end].split()
    assert len(fields) == 3 and fields[1] == b'blob', fields
    size = int(fields[2]); offset = end + 1
    assert blobs[offset:offset+size] == (R / name.decode()).read_bytes(), name
    offset += size
    assert blobs[offset:offset+1] == b'\n'
    offset += 1
assert offset == len(blobs)
preflight = {'parent': parent, 'audit_sha256': sha(audit_path),
    'staged_paths': len(actual), 'staged_pathlist_sha256': sha(pathlist),
    'source_hashes_verified': len(sources), 'all_staged_blob_bytes_match': True,
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
emit('preflight.json', preflight)
for label, argv in [('commit', ['git', '-c', 'gc.auto=0', 'commit', '-m',
    'Preserve cardboard linear candidates and repair checkpoint arena validation']),
    ('push', ['git', '-c', 'gc.auto=0', 'push', 'origin', 'main'])]:
    p = subprocess.run(argv, cwd=R, capture_output=True)
    (O / (label + '.stdout')).write_bytes(p.stdout)
    (O / (label + '.stderr')).write_bytes(p.stderr)
    assert p.returncode == 0, (label, p.stderr.decode())
head = checked(['git', 'rev-parse', 'HEAD']).decode().strip()
remote = checked(['git', 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
assert head == remote
(O / 'publish.py').write_bytes(Path(__file__).read_bytes())
receipt = {'schema': 'numi.cardboard.publication030-commit-push.v1', **preflight,
    'commit': head, 'remote_main': remote, 'remote_verified': True,
    'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'qualification': 'Component solver/checkpoint milestone. All six qualification axes remain open.'}
emit('receipt.json', receipt)
print(json.dumps(receipt, indent=2))
