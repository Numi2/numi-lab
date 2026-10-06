"""Native build reconciliation, retaining the unchanged six-second input scene."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time

E = Path('/Users/n/numi-human-resting-evidence-20261005')
template = E / 'bladder-taenia-native-114/invocation.json'
source = Path('/Users/n/numi-human-resting-final-source-028')
build = Path('/Users/n/numi-human-resting-final-build-028')
old_build = Path('/Users/n/numi-human-resting-cardiac-build-026')
out = E / 'native-source-reconciliation-123'
out.mkdir(exist_ok=False)
original = json.loads(template.read_text())
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
argv = original['argv'].copy()
argv[0] = str(build / 'bin/numi-human-native')
argv[4] = str(out)
argv[argv.index('--resting-movie') + 1] = str(out / 'native-viewer.mov')
env = {**os.environ, **original['environment'], 'DYLD_PRINT_LIBRARIES': '1'}
assets = {p: sha(p) for p in original['asset_sha256'] if not Path(p).is_relative_to(old_build)}
assert all(value == original['asset_sha256'][p] for p, value in assets.items())
for name in ('bin/numi-human-native', 'lib/libmetalrobo.dylib', 'shaders/MetalRobo.metallib',
             'shaders/MetalRoboHyperPolicy.metallib', 'shaders/NumiNeuron.metallib',
             'matter/shaders/HumanRespiration.metallib', 'matter/shaders/NumiMatter.metallib',
             'matter/shaders/NumiMatterPhysicalStateDigest.metallib'):
    assets[str(build / name)] = sha(build / name)
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
(out / 'source.patch').write_bytes(patch)
inv = {
    'scope': 'Six-second same-input native build reconciliation. Current main plus the exact frozen026 cardiac runtime changes; known geometry defects remain, not final acceptance.',
    'template_invocation': str(template), 'template_sha256': sha(template),
    'driver_sha256': sha(__file__), 'source_root': str(source), 'source_revision': revision,
    'source_diff_sha256': sha(out / 'source.patch'),
    'source_file_sha256': {p: sha(source / p) for p in original['source_file_sha256']},
    'binary_sha256': sha(argv[0]), 'asset_sha256': assets, 'argv': argv,
    'environment': {k: v for k, v in env.items() if k.startswith(('NUMI_', 'DYLD_'))},
    'device': original['device'], 'cwd': str(source),
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path = out / 'invocation.json'
path.write_text(json.dumps(inv, indent=2) + '\n')
start = time.monotonic()
with (out / 'native.log').open('w') as log:
    result = subprocess.run(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
elapsed = time.monotonic() - start
inv.update(return_code=result.returncode, wall_seconds=elapsed,
           real_time_factor=6 / elapsed if result.returncode == 0 else None,
           finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           asset_sha256_after={p: sha(p) for p in assets})
path.write_text(json.dumps(inv, indent=2) + '\n')
assert inv['asset_sha256_after'] == assets
print(json.dumps({'return_code': result.returncode, 'wall_seconds': elapsed,
                  'real_time_factor': inv['real_time_factor']}), flush=True)
raise SystemExit(result.returncode)
