from pathlib import Path
import hashlib
import json
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
D = R / PREFIX
O = B / 'analysis/publication029-staged-source-check-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):
    run = subprocess.run(['git', *args], cwd=R, capture_output=True)
    assert run.returncode == 0, run.stderr.decode()
    return run.stdout
assert git('rev-parse', 'HEAD').decode().strip() == 'a5ccffa52ffe368b49c877094a5081b7ee246ebf'
assert git('diff', '--cached', '--name-only', '-z') == b''
frozen = B / 'snapshots/source-reference-jacobian-168/frozen-source.json'
sources = json.loads(frozen.read_text())['source_sha256']
base = json.loads((B / 'snapshots/source-right-grip-telemetry-157/frozen-source.json').read_text())['source_sha256']
assert len(sources) == 802 and len(base) == 787
for name, expected in sources.items():
    assert sha(R / name) == expected, name
code = {name for name, digest in sources.items() if base.get(name) != digest}
assert len(code) == 27
summary = json.loads((D / 'summary.json').read_text())
binding = json.loads((D / 'publication-binding-reference029.json').read_text())
assert sha(D / 'summary.json') == binding['summary_sha256']
assert len(summary['records']) == 1524 and binding['new_case_records'] == 44
expected = set(code) | {PREFIX + name for name in ['summary.json', 'summary-reference028.json', 'publication-binding-reference029.json']}
docs = ['matter/examples/cardboard/implementation-20261014.md',
        'matter/examples/cardboard/solver-progress-20261014.md',
        'matter/examples/cardboard/solver-milestone-source167.md']
expected.update(docs)
for record in summary['records'][1480:]:
    for row in record['files']:
        expected.add(PREFIX + row.get('repository_copy', row.get('repository_compressed_copy')))
pathlist = B / 'publication029-staged-paths.nul'
assert not pathlist.exists()
pathlist.write_bytes(b''.join(p.encode() + b'\0' for p in sorted(expected)))
git('add', '-f', '--pathspec-from-file=' + str(pathlist), '--pathspec-file-nul')
actual = {p.decode() for p in git('diff', '--cached', '--name-only', '-z').split(b'\0') if p}
assert actual == expected
assert git('diff', '--name-only', '-z') == b''
assert git('diff', '--cached', '--name-only', '--diff-filter=D', '-z') == b''
report = {
    'schema': 'numi.cardboard.publication029-staged-source.v1',
    'head': git('rev-parse', 'HEAD').decode().strip(),
    'selected_source_hashes_verified': len(sources), 'changed_or_added_code_paths': sorted(code),
    'frozen_source_sha256': sha(frozen), 'staged_paths': len(expected),
    'stage_pathlist_sha256': sha(pathlist), 'summary_sha256': sha(D / 'summary.json'),
    'publication_binding_sha256': sha(D / 'publication-binding-reference029.json'),
    'historical_records': 1480, 'new_closed_cases': 44,
    'active_native470_excluded': True, 'documentation_paths': docs,
    'unrelated_untracked_preserved': True, 'all_six_qualification_axes_open': True
}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(O / 'staged-paths.nul').write_bytes(pathlist.read_bytes())
(O / 'stage.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report))
