from pathlib import Path
import hashlib, json

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
PKG = B / 'analysis/native473-474-warm-checkpoint-smoke-001'
S = B / 'snapshots/source-checkpoint-warm-breakdown-173'
D = R / 'build-cardboard-provenance-173'
CPU = B / 'analysis/source173-build-cpu-validation-001'
INTEGRATION = B / 'analysis/source173-checkpoint-warm-integration-001/report.json'
PYTHON = '/Users/home/.pyenv/versions/3.10.12/bin/python3'
RUNNER = S / 'matter/tools/cardboard_source_bound_run.py'
PROBE = D / 'numi-matter-cardboard-probe'
METALLIB = D / 'shaders/NumiMatter.metallib'
CASE473 = B / 'solver/warm-cgls-control0-prefix-473'
CASE474 = B / 'solver/warm-cgls-control0-resume-474'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

frozen = json.loads((S / 'frozen-source.json').read_text())
built = json.loads((D / 'source-build.json').read_text())
cpu_report = json.loads((CPU / 'report.json').read_text())

# Exact source173 warm-enabled controls: remove the source470 consumer capture,
# retain its other controls, and add the warm refinement gate from source173.
env = {
    'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR': 'all:all',
    'NM_FEM_BASE_CGLS_ITERATIONS': '64',
    'NM_FEM_BASE_CGLS_PRIMARY_AFTER': '48',
    'NM_FEM_BASE_CGLS_RIGHT_SCALE': 'block3',
    'NM_FEM_BASE_CGLS_WARM_START_REFINE': '1',
    'NM_FEM_CACHE_PROJECTION_REUSE': '1',
    'NM_FEM_GRAPH_PARALLEL_LU_SEARCH': '1',
    'NM_FEM_REGIONAL_CURRENT_RESTART_AUGMENTATION': '1',
    'NM_FEM_RESIDUAL_ARMIJO_ROOT': 'all',
    'NM_FEM_TRUE_RESIDUAL_CERTIFY': '1',
    'NM_FEM_TRUE_RESIDUAL_ROOT': 'all',
    'NUMI_CARDBOARD_STEP_TIMING': '1',
}

physical = [
    '--preset', 'literature2009',
    '--line-search-steps', '24',
    '--without-self-contact',
    '--reference-displacement-gradient',
    '--compensated-paper-arithmetic',
    '--residual-tolerance', '5e-7',
    '--newton-iterations', '64',
    '--fgmres-restart', '16',
    '--fgmres-iterations', '1024',
    '--fgmres-preconditioner', 'regional-galerkin-schwarz',
    '--steps', '24', '--hold-steps', '6', '--unload-steps', '12',
    '--relax-steps', '6', '--post-release-relax-steps', '6', '--release-steps', '6',
    '--nx', '8', '--ny', '2', '--thickness-slices', '1', '--bend-angle', '.375',
    '--dt', '.00005', '--material-iterations', '32',
    '--metallib', str(METALLIB),
]
inputs_473 = [
    str(S / 'frozen-source.json'),
    str(PKG / 'inputs/source173-cpu-report.json'),
    str(PKG / 'inputs/source173-cpu-manifest.json'),
    str(PKG / 'inputs/source173-integration-report.json'),
    str(S / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
    str(S / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
]

def native_for(case, mode):
    args = [str(PROBE), *physical, '--output', str(case / 'output'),
            '--liner-material', str(S / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
            '--medium-material', str(S / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
            '--cache-fem-algorithmic-tangent', '--compensated-source-paper-projection',
            '--accurate-source-paper-projection']
    if mode == 'prefix':
        args += ['--coupon-checkpoint-after-control', '0']
    else:
        args += ['--coupon-resume-checkpoint', str(CASE473 / 'output/checkpoints/coupon-control-000000.nmcb'),
                 '--coupon-resume-metadata', str(CASE473 / 'output/checkpoints/coupon-control-000000.resume')]
    return args

def wrapper(case, native, question, prediction, inputs):
    out = [PYTHON, '-B', str(RUNNER),
           '--source-snapshot', str(S), '--runtime-dir', str(D),
           '--build-provenance', str(D / 'source-build.json'),
           '--case', str(case), '--timeout', '14400', '--question', question,
           '--prediction', prediction]
    for k, v in sorted(env.items()):
        out += ['--env', f'{k}={v}']
    out += ['--diagnostic-run']
    for path in inputs:
        out += ['--input-file', path]
    out += ['--', *native]
    return out

question473 = ('Does the source173 warm-start CGLS experiment accept the first control and write a two-environment native checkpoint at control 0?')
prediction473 = ('Run the unchanged 60-control literature2009 strip schedule with the source173 warm-start refinement enabled from startup. Stop only after the accepted zero-based control 0 checkpoint. This is a diagnostic prefix; it does not establish complete-history parity, convergence of the full schedule, or physical qualification.')
question474 = ('Can source173 restore the accepted warm-enabled control-0 checkpoint and continue the original 60-control schedule from control 1 through its scheduled end?')
prediction474 = ('Restore only the accepted checkpoint and metadata produced by case 473, with identical source, device program, material inputs, protocol, and non-path runtime settings. Continue controls 1 through 59 under the original full schedule. This is a diagnostic resumed segment, not uninterrupted trajectory parity or physical qualification.')

spec = {
  'schema': 'numi.cardboard.native473-474-warm-checkpoint-smoke.v1',
  'qualification': 'diagnostic_only_checkpoint_and_resume_smoke; no trajectory or physical qualification',
  'source': {
    'snapshot': str(S), 'frozen_source_file': str(S / 'frozen-source.json'),
    'frozen_source_sha256': sha(S / 'frozen-source.json'),
    'source_file_count': len(frozen['source_sha256']),
    'build_dir': str(D),
    'source_build': str(D / 'source-build.json'),
    'source_build_sha256': sha(D / 'source-build.json'),
    'build_success': built['success'], 'build_targets': built['targets'],
    'artifact_sha256': built['artifacts_sha256'],
    'build_metadata_sha256': built['build_metadata_sha256'],
    'cpu_validation_report': str(CPU / 'report.json'),
    'cpu_validation_report_sha256': sha(CPU / 'report.json'),
    'cpu_validation_manifest': str(CPU / 'manifest.json'),
    'cpu_validation_manifest_sha256': sha(CPU / 'manifest.json'),
    'integration_report': str(INTEGRATION), 'integration_report_sha256': sha(INTEGRATION),
    'warm_kernel_check_receipt': str(B / 'analysis/source173-warm-kernel-check-001/receipt.json'),
    'warm_kernel_check_manifest': str(B / 'analysis/source173-warm-kernel-check-001/manifest.json'),
    'warm_kernel_check_report_dir': str(B / 'analysis/source173-warm-kernel-check-001'),
    'warm_kernel_check_required': {
      'receipt_schema': 'numi.cardboard.source173.warm-kernel-check.v1',
      'returncode': 0, 'timed_out': False, 'gpu_execution': True,
      'physical_transactions': 0, 'production_material_operator_or_full_cgls_recurrence_exercised': False,
    },
  },
  'environment_overrides': env,
  'environment_constraints': {
    'warm_start_exact_value': '1',
    'capture_path_keys_set': [],
    'operator_fd_keys_set': [],
    'regional_consumer_capture_selector_set': False,
    'same_overrides_for_both_cases': True,
  },
  'protocol': {
    'total_controls': 60,
    'phase_controls': {'loading': 24, 'hold': 6, 'unloading': 12,
                       'relaxation': 6, 'release': 6, 'post_release_relaxation': 6},
    'native_physical_arguments': physical,
    'materials': {
      'liner': {'path': str(S / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
                'sha256': sha(S / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter')},
      'medium': {'path': str(S / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
                 'sha256': sha(S / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter')},
    },
    'full_schedule_preserved_on_resume': True,
    'warm_flag_present_from_control0_and_resume': True,
  },
  'cases': {
    '473': {
      'role': 'warm_enabled_control0_prefix', 'case': str(CASE473),
      'checkpoint_after_control': 0, 'accepted_prefix_count': 1, 'next_control_index': 1,
      'expected_checkpoint': str(CASE473 / 'output/checkpoints/coupon-control-000000.nmcb'),
      'expected_metadata': str(CASE473 / 'output/checkpoints/coupon-control-000000.resume'),
      'expected_prefix': str(CASE473 / 'output/checkpoints/coupon-control-000000.prefix'),
      'native_argv': native_for(CASE473, 'prefix'),
      'wrapper_argv': wrapper(CASE473, native_for(CASE473, 'prefix'), question473, prediction473, inputs_473),
      'question': question473, 'prediction': prediction473,
    },
    '474': {
      'role': 'warm_enabled_restore_then_full_remaining_schedule', 'case': str(CASE474),
      'source_case': str(CASE473), 'source_checkpoint_control': 0,
      'resume_starts_at_control': 1, 'scheduled_controls_remaining': 59,
      'checkpoint_creation_during_resume': False,
      'native_argv': native_for(CASE474, 'resume'),
      'wrapper_argv_template_static_inputs_only': wrapper(CASE474, native_for(CASE474, 'resume'), question474, prediction474, inputs_473),
      'question': question474, 'prediction': prediction474,
      'dynamic_source_inputs': [
        str(CASE473 / 'prediction.json'), str(CASE473 / 'native-environment.json'),
        str(CASE473 / 'receipt.json'), str(CASE473 / 'diagnostic-captures.json'),
        str(CASE473 / 'output/manifest.json'), str(CASE473 / 'output/result.json'),
        str(CASE473 / 'output/observations.csv'),
        str(CASE473 / 'output/coupon-checkpoint-prefix.json'),
        str(CASE473 / 'output/coupon-checkpoint-saved.txt'),
        str(CASE473 / 'output/checkpoints/coupon-control-000000.nmcb'),
        str(CASE473 / 'output/checkpoints/coupon-control-000000.resume'),
        str(CASE473 / 'output/checkpoints/coupon-control-000000.prefix'),
      ],
    },
  },
  'launch_policy': {
    'prepare_script_never_launches': True,
    'minimum_free_bytes_before_manual_wrapper_launch': 2 * 1024**3,
    'abort_floor_bytes': 512 * 1024**2,
    'one_active_native_probe_worker': True,
    'wrapper_native_timeout_seconds': 14400,
    'runner_lock': 'source-bound runner owns checkout native-worker lock',
    'execution': 'not performed by package preparation',
  },
  'package_input_sha256': {
    name: sha(PKG / 'inputs' / name)
    for name in ('source173-cpu-report.json', 'source173-cpu-manifest.json',
                 'source173-integration-report.json', 'source173-frozen-source.json')
  },
}
(PKG / 'run-spec.json').write_text(json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + '\n')
