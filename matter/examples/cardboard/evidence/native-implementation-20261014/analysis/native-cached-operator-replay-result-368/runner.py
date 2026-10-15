from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

base = Path('/Users/home/cardboard-implementation-20261014')
package = base / 'analysis/native-cached-operator-replay-360-003'
output = base / 'analysis/native-cached-operator-replay-result-368'
python = '/Users/home/.pyenv/versions/3.10.12/bin/python3'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(package / 'package-manifest.json') == '756f064c072cc292f99bc1ebefbfa4868a9fc37ff75541c26659f4be696f27e9'
assert sha(package / 'cached_operator_replay.mm') == 'f6bb34dbf514c9bfbd27474d3f84575f9c38389088d6182970bf0d41f54f5cda'
output.mkdir(exist_ok=False)
preflight = [python, str(package / 'preflight.py')]
native = [str(package / 'build/cached_operator_replay'),
          str(base / 'solver/reset-augmentation-first-failure-capture-360/output/compiled.nmatterpack'),
          str(package / 'prepared/replay-input.bin'), str(output / 'result')]
inventory = {str(p.relative_to(package)): sha(p)
             for p in sorted(package.rglob('*')) if p.is_file()}
prediction = dict(schema='numi.cardboard.native-diagnostic-prediction.v1',
    package_manifest_sha256=sha(package / 'package-manifest.json'),
    preflight_argv=preflight, native_argv=native, package_files=inventory,
    question='Does matched native cached A(delta) explain the control2/Newton23 recurrence/true-residual discrepancy?',
    prediction='Replay exact source101 captured state and unscaled correction through production cached and fresh FEM operator/gather kernels. Require finite healthy complete outputs and matching captured fresh action before attributing any difference. No noise floor inferred and no physical transaction.',
    physical_transactions=0,
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(output / 'prediction.json').write_text(json.dumps(prediction, indent=2) + '\n')
start = time.monotonic()
results = {}
for name, argv in [('preflight', preflight), ('native', native), ('postflight', preflight)]:
    if name == 'native' and results['preflight']['exit_code'] != 0:
        results[name] = dict(skipped='preflight failed')
        continue
    with (output / f'{name}.stdout.log').open('x') as out, (output / f'{name}.stderr.log').open('x') as err:
        try:
            result = subprocess.run(argv, stdout=out, stderr=err, timeout=180)
            results[name] = dict(exit_code=result.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            results[name] = dict(exit_code=None, timed_out=True)
for relative, digest in inventory.items():
    assert sha(package / relative) == digest, relative
receipt = dict(schema='numi.cardboard.native-diagnostic-receipt.v1',
    commands=results, wall_seconds=time.monotonic() - start,
    prepared_package_unchanged=True, physical_transactions=0,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(output / 'runner.py').write_bytes(Path(__file__).read_bytes())
files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(output.rglob('*')) if p.is_file()]
(output / 'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.native-diagnostic-evidence.v1', files=files), indent=2) + '\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if all(v.get('exit_code') == 0 for v in results.values()) else 1)
