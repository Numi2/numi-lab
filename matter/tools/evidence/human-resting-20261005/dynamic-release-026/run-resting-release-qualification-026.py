from pathlib import Path
import csv, datetime, hashlib, json, os, subprocess, sys, time

root = Path('/Users/n/numi-human-resting-evidence-20261005')
mode = sys.argv.pop(1)
assert mode == "candidate"
source = Path('/Users/n/numi-human-resting-passive-release-source-016')
build = Path('/Users/n/numi-human-resting-passive-release-build-016')
if mode == "control":
    source=Path("/Users/n/numi-human-resting-liver-parser-source-012")
    build=Path("/Users/n/numi-human-resting-liver-parser-build-012")
name = sys.argv[1]
count = int(sys.argv[2])
inputs = Path(sys.argv[3])
captures = sys.argv[4]
activation_cap = sys.argv[5]
config = Path(sys.argv[6]) if len(sys.argv)>6 else root / "costal-fit-native-config-014/resting-reference-respiration.json"
assert activation_cap == "default" or (0 < float(activation_cap) <= 1)
out = root / name
out.mkdir(exist_ok=False)

old = json.loads((root / 'cardiac-wall-native-cycle-001/invocation.json').read_text())
argv = old['argv'].copy()
if activation_cap != "default":
    argv += ["--muscle-activation", activation_cap, "--resting-release-initialization"]
argv[0] = str(build / 'bin/numi-human-native')
argv[4] = str(out)
argv[argv.index('--muscle-step-count') + 1] = str(count)
argv[argv.index('--resting-scene') + 2] = str(config)
argv[argv.index('--torso-anatomy-payload') + 1] = str(inputs / 'resting-thorax.nhanatomy')
argv[argv.index('--resting-anatomy-receipt') + 1] = str(inputs / 'resting-anatomy-receipt.json')
movie_flag = argv.index('--resting-movie')
argv[movie_flag + 1] = str(out / 'native-viewer.mov')

env = old['environment'].copy()
env['NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS'] = '5'
for key in ('NUMI_HUMAN_EXECUTION_STAGES','NUMI_HUMAN_GPU_TIMING','NUMI_HUMAN_GPU_TIMING_STAGE','NUMI_HUMAN_STAND_FINISH_COUNTER_ROOTS','NUMI_HUMAN_SUPPORT_GPU_TIMING','NUMI_MATTER_GPU_TIMING'):
    env.pop(key,None)
if captures == '-':
    env.pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS', None)
else:
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS'] = captures

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
paths = [Path(x) for x in argv if Path(x).is_file()]
paths += [build / x for x in [
    'lib/libmetalrobo.dylib',
    'shaders/MetalRobo.metallib',
    'shaders/MetalRoboHyperPolicy.metallib',
    'shaders/NumiNeuron.metallib',
    'matter/shaders/HumanRespiration.metallib',
    'matter/shaders/NumiMatter.metallib',
    'matter/shaders/NumiMatterPhysicalStateDigest.metallib',
    'CMakeCache.txt',
]]
patch_file = Path('/Users/n/numi-human-operator-profile-timing-001.patch')
paths.append(patch_file)
assets = {str(path): sha(path) for path in paths}
files = [
    'apps/NumiHumanRestingAnatomy.hpp',
    'apps/NumiHumanRestingVisual.hpp',
    'apps/NumiHumanRestingWindow.hpp',
    'apps/numilab_human_myosim_visual_probe.mm',
    'include/metalrobo/numi_human_resting_visual_gpu.h',
    'matter/src/human_respiration.metal',
    'matter/src/metal/vascular.metalinc',
    'matter/tools/cardiac_geometry_binding.py',
    'matter/tools/resting_intervention_study.py',
    'matter/tools/test_resting_intervention_study.py',
    'src/metal/MetalArticulatedOperator.mm',
    'matter/src/metal_encoder_timing.hpp',
    'matter/tools/human_resting_runtime.hpp',
    'apps/NumiHumanRestingSupportGeometry.hpp',
]
source_sha = {file: sha(source / file) for file in files}
(out / 'source.diff').write_bytes(subprocess.check_output([
    'git', '-C', str(source), 'diff', '--binary'
]))
invocation = {
    'argv': argv,
    'environment': env,
    'asset_sha256': assets,
    'source_revision': subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
    ).strip(),
    'source_file_sha256': source_sha,
    'source_diff_sha256': sha(out / 'source.diff'),
    'driver_sha256': sha(Path(__file__)),
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'body_posture_intervention': f"Postural recruitment cap={activation_cap}; release enabled only for a non-default cap. The cap is an explicit numerical initialization parameter, not a measured resting excitation. MyoSim passive muscle/tendon forces retained. Upper passive-joint rows disabled. No claim of equilibrium.",
    'qualification': f"Resting dynamic-release engineering qualification: {count} native steps, cap={activation_cap}, input={inputs}. Default-state regression and cap sensitivity are separate from long body-posture qualification. Known thoracic/cardiac anatomical defects remain; not final anatomical or physiological acceptance.",
    'contention_note': 'Independent CPU cardiac/anatomy preparation active on Mini during this run; GPU slot reserved for this run. Treat timing as contention-affected.',
}
(out / 'invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')

start = time.monotonic()
with (out / 'native.log').open('wb') as log:
    result = subprocess.run(argv, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
changed = [str(path) for path in paths if sha(path) != assets[str(path)]]
changed += [str(source / file) for file in files if sha(source / file) != source_sha[file]]
record = {
    'returncode': result.returncode,
    'wrapper_wall_seconds': time.monotonic() - start,
    'changed_inputs': changed,
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
if (out / 'resting-surface-audit.csv').is_file():
    rows = list(csv.DictReader((out / 'resting-surface-audit.csv').open()))
    if rows:
        record['last_surface_frame'] = rows[-1]
(out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)

