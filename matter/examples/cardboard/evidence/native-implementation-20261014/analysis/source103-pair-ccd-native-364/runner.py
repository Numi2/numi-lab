from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

base = Path('/Users/home/cardboard-implementation-20261014')
package = base / 'analysis/source103-pair-ccd-diagnostic-002'
output = base / 'analysis/source103-pair-ccd-native-364'
python = '/Users/home/.pyenv/versions/3.10.12/bin/python3'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(package / 'manifest.json') == '2aaf4a4e71c205530599e5ee2205f4269213a4d4ffca86b2e6ebccea9ab0884a'
manifest = json.loads((package / 'manifest.json').read_text())
def check_package():
    for path, expected in manifest['compiled_artifacts'].items():
        assert sha(package / path) == expected, path
check_package()
output.mkdir(exist_ok=False)
native = [str(package / 'compile/pair_ccd_native_check'),
          str(package / 'prepared/prepared.json'),
          str(package / 'prepared/cases.bin'),
          str(package / 'compile/diagnostic_pair.metallib'),
          str(output / 'native-result.json')]
verify = [python, str(package / 'verify_pair_package.py'),
          '--prepared', str(package / 'prepared/prepared.json'),
          '--native-result', str(output / 'native-result.json'),
          '--output', str(output / 'native-audit.json')]
preflight = [python, str(package / 'verify_pair_package.py'),
             '--prepared', str(package / 'prepared/prepared.json'),
             '--output', str(output / 'preflight-audit.json')]
prediction = dict(schema='numi.cardboard.native-diagnostic-prediction.v1',
    package_manifest_sha256=sha(package / 'manifest.json'),
    preflight_argv=preflight, native_argv=native, verifier_argv=verify,
    question='Which exact unchanged production CCD subcall becomes uncertified at source103 first fatal self-contact trial?',
    prediction='Require direct-production/trace-clone physical-output bit equality on all 30 calls before interpreting traces. Certification failures are retained and are not interpreted as physical penetration. No state, threshold or iteration budget changes.',
    physical_transactions=0,
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(output / 'prediction.json').write_text(json.dumps(prediction, indent=2) + '\n')
start = time.monotonic()
results = {}
for name, argv in [('preflight', preflight), ('native', native), ('verify', verify)]:
    if name != 'preflight' and results['preflight']['exit_code'] != 0:
        results[name] = dict(skipped='preflight failed')
        continue
    if name == 'verify' and not (output / 'native-result.json').exists():
        results[name] = dict(skipped='native result unavailable')
        continue
    with (output / f'{name}.stdout.log').open('x') as out, (output / f'{name}.stderr.log').open('x') as err:
        try:
            result = subprocess.run(argv, stdout=out, stderr=err, timeout=180)
            results[name] = dict(exit_code=result.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            results[name] = dict(exit_code=None, timed_out=True)
check_package()
receipt = dict(schema='numi.cardboard.native-diagnostic-receipt.v1',
    commands=results, wall_seconds=time.monotonic() - start,
    compiled_package_unchanged=True, physical_transactions=0,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(output / 'runner.py').write_bytes(Path(__file__).read_bytes())
files = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(output.iterdir()) if p.is_file()]
(output / 'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.native-diagnostic-evidence.v1', files=files), indent=2) + '\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if all(v.get('exit_code') == 0 for v in results.values()) else 1)
