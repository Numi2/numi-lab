from pathlib import Path
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
old_source = B / 'snapshots/source-cache-projection-reuse-runtime-136'
new_source = B / 'snapshots/source-coupon-cache-admission-139'
old_case = B / 'solver/cache-reuse-full-history-parity-437'
new_case = B / 'solver/cache-reuse-full-history-parity-441'
assert not new_case.exists()
build = R / 'build-cardboard-provenance-139/source-build.json'
assert json.loads(build.read_text())['success']
command = json.loads((B / 'native-437-command.json').read_text())
command = [part.replace(str(old_source), str(new_source))
           .replace('build-cardboard-provenance-136', 'build-cardboard-provenance-139')
           .replace(str(old_case), str(new_case)) for part in command]
prediction = command.index('--prediction') + 1
command[prediction] = (
    'Corrected opt-in admission now includes the compiler enabled IPC capability '
    'and the existing finite elastic starch elements, whose tangent stays on the '
    'original path. Expect complete attempted-history parity with native434: '
    'same geometry, histories, traces and terminal Newton48 rejection, apart from '
    'provenance/timing/notices. This is not completed coupon, speedup or numerical '
    'convergence qualification. Source136 initialization failure437 is preserved.')
command += []
(B / 'native-441-command.json').write_text(json.dumps(command, indent=2) + '\n')
binding = {
    'build_sha256': hashlib.sha256(build.read_bytes()).hexdigest(),
    'source': str(new_source),
    'preserved_failed_predecessor': str(old_case),
    'reference': str(B / 'solver/base-cauchy-disabled-parity-434'),
    'command': command,
}
(B / 'native-441-preflight.json').write_text(json.dumps(binding, indent=2) + '\n')
print('prepared441')
