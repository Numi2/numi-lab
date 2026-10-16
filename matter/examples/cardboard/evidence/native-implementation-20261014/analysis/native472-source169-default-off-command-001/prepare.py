from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-warm-cgls-refinement-169'
D = R / 'build-cardboard-provenance-169'
P = B / 'analysis/native472-source169-default-off-command-001'
P.mkdir(exist_ok=False)
(P / 'inputs').mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = B / 'analysis/native470-accepted-checkpoint-prefix-command-002'
assert sha(old / 'package-manifest.json') == 'f7ebdb1bb9d3ca103fb0190370037f69260f787cdbb5a661ad28c1460e713b29'
previous = json.loads((old / 'native-470-command.json').read_text())
native = []
i = 0
while i < len(previous['native_argv']):
    token = previous['native_argv'][i]
    if token == '--coupon-checkpoint-after-control':
        i += 2
        continue
    native.append(token)
    i += 1
case = B / 'solver/source169-default-off-parity-472'
native[0] = str(D / 'numi-matter-cardboard-probe')
native[native.index('--metallib') + 1] = str(D / 'shaders/NumiMatter.metallib')
native[native.index('--output') + 1] = str(case / 'output')
env = dict(previous['environment_overrides'])
del env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR']
assert len(env) == 11 and 'NM_FEM_BASE_CGLS_WARM_START_REFINE' not in env
inputs = []
for name, original in [
    ('baseline469-prediction.json', B / 'solver/final-newton-fd-capture-469/prediction.json'),
    ('baseline468-receipt.json', B / 'solver/cgls-dt-half-cycle-468/receipt.json'),
    ('source169-cpu-manifest.json', B / 'analysis/source169-build-cpu-validation-001/manifest.json'),
    ('source169-cpu-verification.json', B / 'analysis/source169-build-cpu-validation-001/verification.json'),
    ('warm-kernel-receipt.json', B / 'analysis/warm-start-native-kernel-check-002/receipt.json'),
    ('liner-material.nmatter', S / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
    ('medium-material.nmatter', S / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
]:
    target = P / 'inputs' / name
    shutil.copy2(original, target)
    inputs.append({'alias': name, 'original': str(original), 'package': str(target), 'sha256': sha(original)})
wrapper = ['/Users/home/.pyenv/versions/3.10.12/bin/python3', '-B', str(S / 'matter/tools/cardboard_source_bound_run.py'),
           '--case', str(case), '--source-snapshot', str(S), '--runtime-dir', str(D),
           '--build-provenance', str(D / 'source-build.json'), '--timeout', '3600', '--diagnostic-run']
for row in inputs:
    wrapper += ['--input-file', row['package']]
for key, value in sorted(env.items()):
    wrapper += ['--env', key + '=' + value]
wrapper += ['--'] + native
build = json.loads((D / 'source-build.json').read_text())
spec = {'schema': 'numi.cardboard.native472-default-off-command.v1',
        'purpose': 'Source169 default-off full-attempt compatibility against468; no warm-enabled or full-protocol qualification.',
        'source_snapshot': str(S), 'runtime_dir': str(D), 'case': str(case),
        'frozen_source_sha256': sha(S / 'frozen-source.json'), 'source_file_count': 804,
        'source_build_sha256': sha(D / 'source-build.json'), 'artifact_sha256': build['artifacts_sha256'],
        'input_bindings': inputs, 'environment_overrides': env, 'native_argv': native, 'wrapper_argv': wrapper,
        'baseline470_command': str(old / 'native-470-command.json'),
        'baseline470_command_sha256': sha(old / 'native-470-command.json')}
(P / 'native-472-command.json').write_text(json.dumps(spec, indent=2) + '\n')
runner = (old / 'runner.py').read_text()
start = runner.index('def verify():\n')
end = runner.index('\ndef _process_rows():', start)
verify = '''def verify():
    spec = json.loads(SPEC_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest['schema'] == 'numi.cardboard.native472-package-manifest.v1'
    actual = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*')
              if p.is_file() and p != MANIFEST_PATH and '__pycache__' not in p.parts}
    assert actual == set(manifest['files_sha256'])
    for name, digest in manifest['files_sha256'].items():
        assert sha(PACKAGE / name) == digest, name
    source = Path(spec['source_snapshot'])
    build = Path(spec['runtime_dir'])
    assert sha(source / 'frozen-source.json') == spec['frozen_source_sha256']
    frozen = json.loads((source / 'frozen-source.json').read_text())
    assert len(frozen['source_sha256']) == 804
    for name, digest in frozen['source_sha256'].items():
        assert sha(source / name) == digest, name
    assert sha(build / 'source-build.json') == spec['source_build_sha256']
    built = json.loads((build / 'source-build.json').read_text())
    assert built['success'] is True and built['source_sha256'] == frozen['source_sha256']
    assert built['artifacts_sha256'] == spec['artifact_sha256']
    for name, digest in built['artifacts_sha256'].items():
        assert Path(name).resolve().is_relative_to(build.resolve()) and sha(name) == digest
    for row in spec['input_bindings']:
        assert sha(row['original']) == row['sha256'] == sha(row['package'])
    cpu = json.loads((PACKAGE / 'inputs/source169-cpu-verification.json').read_text())
    assert cpu['source_snapshot_sha256'] == spec['frozen_source_sha256']
    assert cpu['build_record_sha256'] == spec['source_build_sha256']
    assert cpu['cpu_validation']['total_test_cases_as_reported'] == 27
    kernel = json.loads((PACKAGE / 'inputs/warm-kernel-receipt.json').read_text())
    assert kernel['returncode'] == 0 and kernel['timed_out'] is False and kernel['physical_transactions'] == 0
    baseline = json.loads(Path(spec['baseline470_command']).read_text())
    assert sha(spec['baseline470_command']) == spec['baseline470_command_sha256']
    def physical(values):
        result = []; i = 1
        while i < len(values):
            if values[i] in ('--metallib', '--output', '--coupon-checkpoint-after-control'):
                i += 2
            else:
                result.append(values[i]); i += 1
        return result
    assert physical(spec['native_argv']) == physical(baseline['native_argv'])
    expected_env = dict(baseline['environment_overrides'])
    del expected_env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR']
    assert spec['environment_overrides'] == expected_env and len(expected_env) == 11
    assert 'NM_FEM_BASE_CGLS_WARM_START_REFINE' not in expected_env
    native = spec['native_argv']; wrapper = spec['wrapper_argv']; split = wrapper.index('--')
    assert native[0] == str(build / 'numi-matter-cardboard-probe')
    assert native[native.index('--metallib') + 1] == str(build / 'shaders/NumiMatter.metallib')
    assert native[native.index('--output') + 1] == str(Path(spec['case']) / 'output')
    assert '--coupon-checkpoint-after-control' not in native and '--coupon-resume-checkpoint' not in native
    assert wrapper[split + 1:] == native
    assert wrapper[1] == '-B' and wrapper[2] == str(source / 'matter/tools/cardboard_source_bound_run.py')
    assert wrapper[wrapper.index('--timeout') + 1] == '3600'
    wrapper_env = {wrapper[i+1].split('=', 1)[0]: wrapper[i+1].split('=', 1)[1]
                   for i, token in enumerate(wrapper[:split]) if token == '--env'}
    assert wrapper_env == expected_env
    case = Path(spec['case'])
    assert case.parent.resolve() == B / 'solver' and not case.exists() and not EXECUTIONS.exists()
    active = []; stopped = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,stat=,comm='], text=True).splitlines():
        f = line.split(None, 3)
        if len(f) != 4 or f[2].startswith('Z'): continue
        if 'numi-matter-' not in Path(f[3]).name and Path(f[3]).name not in ('warm_start_check','sparse_scan_check'): continue
        if f[:3] == ['77441', '76106', 'T'] and f[3] == '/Users/home/numi-cardboard-20261006/build-cardboard-world/matter/numi-matter-cardboard-paper-material-check':
            parent = subprocess.check_output(['ps', '-p', '76106', '-o', 'comm='], text=True).strip()
            assert Path(parent).name == 'lldb'; stopped.append(line)
        else: active.append(line)
    assert not active, active
    free = available(); assert free >= MINIMUM
    return spec, {'available_bytes': free, 'active_gpu_workers': active,
                  'preserved_stopped_debugger_targets': stopped, 'verified_source_files': 804,
                  'verified_build_artifacts': len(built['artifacts_sha256']),
                  'verified_inputs': len(spec['input_bindings']),
                  'source_build_sha256': spec['source_build_sha256'],
                  'frozen_source_sha256': spec['frozen_source_sha256'], 'cpu_test_cases': 27}

'''
runner = runner[:start] + verify + runner[end:]
runner = runner.replace('native470', 'native472').replace('native-470-command.json', 'native-472-command.json')
runner = runner.replace("'checkpoint_control_index': 19", "'checkpoint_control_index': None")
(P / 'runner.py').write_text(runner)
(P / 'prepare.py').write_bytes(Path(__file__).read_bytes())
(P / 'package-manifest.json').write_text(json.dumps({'schema':'numi.cardboard.native472-package-manifest.v1',
    'files_sha256': {str(p.relative_to(P)): sha(p) for p in sorted(P.rglob('*')) if p.is_file()}}, indent=2) + '\n')
print(json.dumps({'package': str(P), 'manifest_sha256': sha(P / 'package-manifest.json')}))
