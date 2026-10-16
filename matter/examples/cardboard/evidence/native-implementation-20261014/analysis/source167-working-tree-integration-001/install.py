from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-diagnostic-and-resume-167-final'
O = B / 'analysis/source167-working-tree-integration-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):
    return subprocess.check_output(['git', *args], cwd=R)
assert git('rev-parse', 'HEAD').decode().strip() == 'a5ccffa52ffe368b49c877094a5081b7ee246ebf'
assert git('branch', '--show-current').decode().strip() == 'main'
assert git('diff', '--name-only', '-z') == b''
assert git('diff', '--cached', '--name-only', '-z') == b''
base = json.loads((B / 'snapshots/source-right-grip-telemetry-157/frozen-source.json').read_text())['source_sha256']
assert len(base) == 787
for name, digest in base.items():
    assert sha(R / name) == digest, name
report_path = B / 'analysis/source167-combined-integration-004/report.json'
assert sha(report_path) == '264556dda999ce32f3c4aff96142f141bd00f540a136c4dceedd3817c037038e'
report = json.loads(report_path.read_text())
changes = report['changed_or_added']
for name, digest in changes.items():
    assert sha(S / name) == digest, name
    if name not in base:
        assert not (R / name).exists(), name
for name in changes:
    destination = R / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S / name, destination)
frozen = json.loads((S / 'frozen-source.json').read_text())['source_sha256']
for name, digest in frozen.items():
    assert sha(R / name) == digest, name
result = {
    'schema': 'numi.cardboard.source167-working-tree-integration.v1',
    'parent': git('rev-parse', 'HEAD').decode().strip(),
    'integration_report_sha256': sha(report_path),
    'frozen_source_sha256': sha(S / 'frozen-source.json'),
    'baseline_files_verified': len(base),
    'final_files_verified': len(frozen),
    'changed_or_added': changes,
    'unrelated_untracked_preserved': True,
    'gpu_execution': False,
    'qualification': 'Source integration only; complete physical qualification remains open.'
}
(O / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
shutil.copyfile(__file__, O / 'install.py')
(O / 'manifest.json').write_text(json.dumps({p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(O.iterdir()) if p.is_file()}, indent=2) + '\n')
print(json.dumps({'changed_or_added': len(changes), 'final_files_verified': len(frozen), 'report_sha256': sha(O / 'report.json')}))
