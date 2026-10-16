"""Verify the exact source173 transfer and build on the separate macmini host."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

B = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = B / 'build-source173-001'
O = B / 'analysis/source173-host-build-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists() and not O.exists()
assert sha(S / 'frozen-source.json') == '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3'
frozen = json.loads((S / 'frozen-source.json').read_text())
support = json.loads((S / 'configure-support-source.json').read_text())
assert len(frozen['source_sha256']) == 804 and len(support['files']) == 1034
for name, expected in dict(support['files'], **frozen['source_sha256']).items():
    p = S / name
    assert p.is_file() and not p.is_symlink() and sha(p) == expected, name
stats = os.statvfs(B)
free = stats.f_bavail * stats.f_frsize
assert free > 16 * 1024 ** 3
environment = dict(os.environ)
environment['PATH'] = '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
command = [sys.executable, '-B', str(S / 'matter/tools/cardboard_source_bound_build.py'),
    '--source-snapshot', str(S), '--build-dir', str(D), '--jobs', '2']
for target in ('numi-matter-cardboard-probe', 'numi-matter-cardboard-warm-start-check',
               'numi-matter-cardboard-shipping-snapshot-check', 'numi-matter-cardboard-coupon-resume-check'):
    command += ['--target', target]
O.mkdir(parents=True)
record = {'schema': 'numi.cardboard.macmini-source173-bootstrap.v1',
    'host_tag': 'macmini', 'platform': platform.platform(),
    'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
    'physical_memory_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
    'source_snapshot': str(S), 'source_files_verified': 804, 'support_files_verified': 1034,
    'source_frozen_sha256': sha(S / 'frozen-source.json'),
    'support_manifest_sha256': sha(S / 'configure-support-source.json'),
    'available_bytes_before': free, 'argv': command,
    'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'gpu_dispatch_requested': False, 'qualification': 'Fresh host build only; no native trajectory or M4 memory qualification.'}
(O / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
with (O / 'builder.stdout.log').open('x') as out, (O / 'builder.stderr.log').open('x') as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err)
record.update(returncode=result.returncode,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_build_sha256=sha(D / 'source-build.json') if (D / 'source-build.json').is_file() else None)
(O / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(result.returncode)
