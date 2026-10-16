#!/usr/bin/env python3
"""Prepare (never execute) the source-bound dt/2 native 468 command and clock audit."""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, shutil, struct

ROOT = pathlib.Path('/Users/home/cardboard-implementation-20261014')
SNAP = ROOT / 'snapshots/source-cgls-all-controls-newton128-154'
BUILD = pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-154')
BASE_PACKAGE = ROOT / 'analysis/native467-source154-allcontrol-outer128-command-003'
BASE_COMMAND = BASE_PACKAGE / 'native-467-command.json'
CASE467 = ROOT / 'solver/cgls-complete-outer128-allcontrols-467'
PKG = ROOT / 'analysis/native468-source154-dt-half-cycle-command-002'
CASE468 = ROOT / 'solver/cgls-dt-half-cycle-468'


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def set_pair(argv: list[str], option: str, value: str) -> None:
    matches = [i for i, token in enumerate(argv[:-1]) if token == option]
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one {option}, found {len(matches)}')
    argv[matches[0] + 1] = value


def phase_point(counts: dict[str, int], bend: float, step: int) -> tuple[str, float, bool]:
    remaining = step
    order = ('loading', 'hold', 'unloading', 'relaxation', 'release', 'post_release_relaxation')
    for phase in order:
        n = counts[phase]
        if remaining < n:
            if phase == 'loading':
                angle = bend * float(remaining + 1) / n
            elif phase == 'hold':
                angle = bend
            elif phase == 'unloading':
                angle = bend * (1.0 - float(remaining + 1) / n)
            else:
                angle = 0.0
            return phase, angle, phase in ('release', 'post_release_relaxation')
        remaining -= n
    raise IndexError(step)


def target_position(rest: list[float], angle_deg: float,
                    length: float, width: float, height: float) -> tuple[float, float, float]:
    if angle_deg == 0.0:
        return (f32(rest[0]), f32(rest[1]), f32(rest[2]))
    theta = angle_deg * math.pi / 180.0
    px, py, pz = length, 0.5 * width, 0.5 * height
    rx, rz = rest[0] - px, rest[2] - pz
    return (f32(px + math.cos(theta) * rx + math.sin(theta) * rz),
            f32(rest[1]),
            f32(pz - math.sin(theta) * rx + math.cos(theta) * rz))


def pack_float(x: float) -> bytes:
    return struct.pack('<f', x)


if PKG.exists() and any(p.name != 'prepare.py' for p in PKG.iterdir()):
    raise SystemExit(f'refusing to overwrite generated package artifacts: {PKG}')
if CASE468.exists():
    raise SystemExit(f'468 case path already exists: {CASE468}')
PKG.mkdir(parents=True, exist_ok=True)
inputs_dir = PKG / 'inputs'
inputs_dir.mkdir(exist_ok=True)

argv = json.loads(BASE_COMMAND.read_text())
if argv[0] != '/Users/home/.pyenv/versions/3.10.12/bin/python3' or argv[1] != str(SNAP / 'matter/tools/cardboard_source_bound_run.py'):
    raise RuntimeError('unexpected source-bound runner ABI in command003')
if argv[argv.index('--source-snapshot') + 1] != str(SNAP):
    raise RuntimeError('source snapshot mismatch')
if argv[argv.index('--runtime-dir') + 1] != str(BUILD):
    raise RuntimeError('runtime build mismatch')

# Preserve all 54 existing source154/466/149 evidence inputs byte-for-byte,
# under unique basenames in this package. The runner checks basename uniqueness.
old_inputs = [pathlib.Path(argv[i + 1]) for i, token in enumerate(argv[:-1]) if token == '--input-file']
if len(old_inputs) != 54:
    raise RuntimeError(f'expected 54 command003 inputs, found {len(old_inputs)}')
extra_sources = [
    (CASE467 / 'prediction.json', '467 prediction'),
    (CASE467 / 'receipt.json', '467 incomplete native receipt'),
    (CASE467 / 'output/manifest.json', '467 partial output manifest'),
    (CASE467 / 'output/observations.csv', '467 accepted-step observations'),
    (CASE467 / 'output/mesh.json', '467 compiled authored mesh'),
    (ROOT / 'analysis/native467-capacity-interruption-audit-001/report.json', '467 capacity interruption audit'),
    (ROOT / 'analysis/native467-capacity-interruption-audit-001/manifest.json', '467 capacity audit manifest'),
    (ROOT / 'analysis/native467-capacity-interruption-audit-001/stderr-lifecycle-excerpt.txt', '467 lifecycle excerpt'),
    (ROOT / 'analysis/native467-launch-guard-001/receipt.json', '467 launch guard receipt'),
    (ROOT / 'analysis/native467-launch-guard-001/launch.json', '467 launch guard launch record'),
    (BASE_COMMAND, '467 source-bound command'),
    (BASE_PACKAGE / 'manifest.json', '467 command package manifest'),
    (BASE_PACKAGE / 'admission.json', '467 admission record'),
    (BUILD / 'source-build.json', 'source154 build provenance with 787 source hashes'),
    (SNAP / 'frozen-source.json', 'source154 frozen source hash inventory'),
    (SNAP / 'configure-support-source.json', 'source154 configure-support binding'),
]
for path, _ in extra_sources:
    if not path.is_file():
        raise FileNotFoundError(path)
all_sources = [(p, 'command003 evidence input') for p in old_inputs] + extra_sources
copy_records = []
new_input_paths = []
seen_names: set[str] = set()
for index, (source, role) in enumerate(all_sources):
    if not source.is_file():
        raise FileNotFoundError(source)
    basename = f'evidence-{index:03d}-{source.name}'
    if basename in seen_names:
        raise RuntimeError(f'duplicate copied external-input basename {basename}')
    seen_names.add(basename)
    dest = inputs_dir / basename
    shutil.copyfile(source, dest)
    digest = sha(dest)
    if digest != sha(source):
        raise RuntimeError(f'copy hash mismatch for {source}')
    copy_records.append({
        'index': index,
        'role': role,
        'source_path': str(source),
        'copied_path': str(dest),
        'bytes': dest.stat().st_size,
        'sha256': digest,
    })
    new_input_paths.append(str(dest))

# Bind full live source/build hashes before making the command. Build provenance
# has 787 source files, three binary artifacts and two build metadata files.
build_json_path = BUILD / 'source-build.json'
build = json.loads(build_json_path.read_text())
if build.get('source_snapshot') != str(SNAP) or build.get('build_dir') != str(BUILD):
    raise RuntimeError('source-build provenance points to another source/build')
source_bad = []
for rel, expected in build['source_sha256'].items():
    path = SNAP / rel
    if not path.is_file() or sha(path) != expected:
        source_bad.append(rel)
artifacts = {}
for path_string, expected in build['artifacts_sha256'].items():
    path = pathlib.Path(path_string)
    actual = sha(path) if path.is_file() else None
    artifacts[path_string] = {'expected_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected}
metadata = {}
for rel, expected in build['build_metadata_sha256'].items():
    path = BUILD / rel
    actual = sha(path) if path.is_file() else None
    metadata[rel] = {'expected_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected}
if source_bad or not all(x['matches'] for x in artifacts.values()) or not all(x['matches'] for x in metadata.values()):
    raise RuntimeError(f'live source/build mismatch: source={source_bad[:5]} artifacts={artifacts} metadata={metadata}')

# Replace only declared run changes and rebind copied external inputs.
rebound = []
i = 0
while i < len(argv):
    token = argv[i]
    if token == '--input-file':
        old_path = pathlib.Path(argv[i + 1])
        index = next(j for j, (src, _) in enumerate(all_sources) if src == old_path)
        rebound.extend(['--input-file', new_input_paths[index]])
        i += 2
    else:
        rebound.append(token)
        i += 1
argv = rebound
# Add source154/467 capacity lineage after the copied base evidence and before probe argv.
insert_at = argv.index('--')
for index in range(len(old_inputs), len(all_sources)):
    argv[insert_at:insert_at] = ['--input-file', new_input_paths[index]]
    insert_at += 2

set_pair(argv, '--case', str(CASE468))
set_pair(argv, '--question',
         'Under source154, does dt/2 complete the same physical-duration coupon path with the same force-accuracy bound using an explicitly lower 64-Newton budget?')
set_pair(argv, '--prediction',
         'Run a dt/2 diagnostic under source154. Double each of the six phase counts and halve dt so the loading, hold, unloading, relaxation, release and post-release durations remain 0.003 seconds total and the source prescribed angular path/rate is identical at common physical times. Keep geometry, mesh, materials, material iterations, line-search, FGMRES and all-control CGLS64/block3/primary48 settings from source154 command 467. Tighten the solver certificate from 1e-6 to 5e-7: in this pure mechanical case the residual norm is an impulse in N s with implicit unit normalization while its scale remains below 1; at dt=5e-5 this conditionally preserves the 0.01 N bound. Use a 64 rather than 128 Newton ceiling to bound runtime; therefore this is a dt/2 diagnostic with a different nonlinear iteration budget, not a pure time-refinement qualification. Preserve full histories and verify post-run residual scaling and force imbalance; no physical/calibration qualification is implied.')
set_pair(argv, '--output', str(CASE468 / 'output'))
set_pair(argv, '--newton-iterations', '64')
set_pair(argv, '--steps', '24')
set_pair(argv, '--hold-steps', '6')
set_pair(argv, '--unload-steps', '12')
set_pair(argv, '--relax-steps', '6')
set_pair(argv, '--release-steps', '6')
set_pair(argv, '--post-release-relax-steps', '6')
set_pair(argv, '--dt', '.00005')
set_pair(argv, '--residual-tolerance', '5e-7')

# Assert the selected ABI/controls exactly.
if argv[argv.index('--fgmres-restart') + 1] != '16' or argv[argv.index('--fgmres-iterations') + 1] != '1024':
    raise RuntimeError('FGMRES controls drifted')
if argv[argv.index('--material-iterations') + 1] != '32':
    raise RuntimeError('material local iteration budget drifted')
if argv.count('--input-file') != len(all_sources) or len({pathlib.Path(new_input_paths[j]).name for j in range(len(new_input_paths))}) != len(new_input_paths):
    raise RuntimeError('external inputs are not basename-unique')
for name, expected in {
    'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR': 'all:all',
    'NM_FEM_BASE_CGLS_ITERATIONS': '64',
    'NM_FEM_BASE_CGLS_RIGHT_SCALE': 'block3',
    'NM_FEM_BASE_CGLS_PRIMARY_AFTER': '48',
    'NM_FEM_TRUE_RESIDUAL_ROOT': 'all',
    'NM_FEM_RESIDUAL_ARMIJO_ROOT': 'all',
}.items():
    pairs = [argv[j + 1] for j, token in enumerate(argv[:-1]) if token == '--env' and argv[j + 1].startswith(name + '=')]
    if pairs != [name + '=' + expected]:
        raise RuntimeError(f'environment drift for {name}: {pairs}')

(PKG / 'native-468-command.json').write_text(json.dumps(argv, indent=2) + '\n')
(PKG / 'external-input-bindings.json').write_text(json.dumps({
    'schema': 'numi.cardboard.native468.external-input-bindings.v1',
    'command003_input_count': len(old_inputs),
    'supersedes_preparation': 'native468-source154-dt-half-cycle-command-001; its rate metric compared one coarse interval to only one half-step instead of the two-half-step interval average.',
    'additional_467_capacity_and_build_binding_count': len(extra_sources),
    'unique_basename_count': len(new_input_paths),
    'inputs': copy_records,
}, indent=2) + '\n')

# Source-equivalent phase clock/target replay; no native physics is run here.
# v2 corrects v1's rate comparison to use both refined half-steps over a coarse interval.
old = {'loading': 12, 'hold': 3, 'unloading': 6, 'relaxation': 3, 'release': 3, 'post_release_relaxation': 3}
new = {'loading': 24, 'hold': 6, 'unloading': 12, 'relaxation': 6, 'release': 6, 'post_release_relaxation': 6}
bend = 0.375
old_dt = f32(0.0001)
new_dt = f32(0.00005)
old_count = sum(old.values())
new_count = sum(new.values())
old_duration = old_count * old_dt
new_duration = new_count * new_dt
if old_duration != new_duration or new_dt * 2.0 != old_dt:
    raise RuntimeError('runtime float32 timesteps do not preserve total duration exactly')
mesh = json.loads((CASE467 / 'output/mesh.json').read_text())
right_indices = mesh['right_grip_nodes']
nodes = mesh['nodes_m']
length, width, height = 0.0316, 0.02, 0.00421
rows = []
max_angle_common = 0.0
max_time_common = 0.0
max_target_abs_common = 0.0
exact_target_components = 0
target_components = 0
max_rate_diff = 0.0
max_local_rate_diff = 0.0
previous_old_angle = 0.0
previous_new_angle = 0.0
for k in range(1, old_count + 1):
    old_phase, old_angle, old_free = phase_point(old, bend, k - 1)
    new_phase, new_angle, new_free = phase_point(new, bend, 2 * k - 1)
    t_old, t_new = k * old_dt, (2 * k) * new_dt
    max_time_common = max(max_time_common, abs(t_old - t_new))
    max_angle_common = max(max_angle_common, abs(old_angle - new_angle))
    old_rate = (old_angle - previous_old_angle) / old_dt
    previous_common_new_angle = phase_point(new, bend, 2 * k - 3)[1] if k > 1 else 0.0
    new_interval_rate = (new_angle - previous_common_new_angle) / (2.0 * new_dt)
    previous_fine_angle = phase_point(new, bend, 2 * k - 2)[1]
    new_last_half_rate = (new_angle - previous_fine_angle) / new_dt
    max_rate_diff = max(max_rate_diff, abs(old_rate - new_interval_rate))
    max_local_rate_diff = max(max_local_rate_diff, abs(old_rate - new_last_half_rate))
    if old_phase != new_phase or old_free != new_free:
        raise RuntimeError(f'phase/grip mode mismatch at common sample {k}: {old_phase}/{new_phase}, {old_free}/{new_free}')
    sample_error = 0.0
    sample_exact = 0
    for node_index in right_indices:
        rest = nodes[node_index]
        a = target_position(rest, old_angle, length, width, height)
        b = target_position(rest, new_angle, length, width, height)
        for av, bv in zip(a, b):
            target_components += 1
            sample_error = max(sample_error, abs(av - bv))
            if pack_float(av) == pack_float(bv):
                exact_target_components += 1
                sample_exact += 1
    max_target_abs_common = max(max_target_abs_common, sample_error)
    rows.append({
        'common_sample': k,
        'coarse_step_0_based': k - 1,
        'refined_step_0_based': 2 * k - 1,
        'coarse_time_s_runtime_f32_dt': t_old,
        'refined_time_s_runtime_f32_dt': t_new,
        'phase': old_phase,
        'coarse_target_angle_deg': old_angle,
        'refined_target_angle_deg': new_angle,
        'right_grip_free': old_free,
        'coarse_discrete_angular_rate_deg_s': old_rate,
        'refined_interval_average_angular_rate_deg_s': new_interval_rate,
        'refined_final_half_step_angular_rate_deg_s': new_last_half_rate,
        'right_grip_target_float_components_exact': sample_exact,
        'right_grip_target_float_components_total': len(right_indices) * 3,
        'max_target_coordinate_abs_difference_m': sample_error,
    })
    previous_old_angle, previous_new_angle = old_angle, new_angle
with (PKG / 'common-clock-target-comparison.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# Check every refined endpoint against the same continuous piecewise-linear
# angular path defined by source counts*dt, not merely common coarse endpoints.
old_durations = {p: n * old_dt for p, n in old.items()}
new_durations = {p: n * new_dt for p, n in new.items()}
def continuous_angle(t: float, durations: dict[str, float]) -> float:
    load = durations['loading']; hold = durations['hold']; unload = durations['unloading']
    if t <= load: return bend * t / load
    t -= load
    if t <= hold: return bend
    t -= hold
    if t <= unload: return bend * (1.0 - t / unload)
    return 0.0
max_fine_continuous_error = 0.0
for idx in range(1, new_count + 1):
    phase, angle, _ = phase_point(new, bend, idx - 1)
    t = idx * new_dt
    max_fine_continuous_error = max(max_fine_continuous_error, abs(angle - continuous_angle(t, new_durations)))

max_radius = max(math.hypot(nodes[i][0] - length, nodes[i][2] - height / 2.0) for i in right_indices)
omega_deg_s = bend / old_durations['loading']
dtheta_old = math.radians(bend / old['loading'])
dtheta_new = math.radians(bend / new['loading'])
coarse_chord_speed = 2.0 * max_radius * math.sin(dtheta_old / 2.0) / old_dt
fine_chord_speed = 2.0 * max_radius * math.sin(dtheta_new / 2.0) / new_dt
angular_summary = {
    'source_clock': {
        'total_steps': 'sum of phase counts',
        'sample_time': 'runtime effective float32 dt * (zero_based_step + 1)',
        'loading_angle': 'bend * (zero_based_step_within_loading + 1) / loading_steps',
        'unloading_angle': 'bend * (1 - (zero_based_step_within_unloading + 1) / unloading_steps)',
        'release_flags': 'fillTargets marks right grip free in both release and post_release_relaxation phases',
    },
    '467': {'phase_counts': old, 'effective_runtime_dt_s': old_dt, 'phase_durations_s': old_durations, 'total_duration_s': old_duration, 'bend_angle_deg': bend},
    '468': {'phase_counts': new, 'effective_runtime_dt_s': new_dt, 'phase_durations_s': new_durations, 'total_duration_s': new_duration, 'bend_angle_deg': bend},
    'common_sample_checks': {
        'count': old_count,
        'common_time_max_abs_difference_s': max_time_common,
        'target_angle_max_abs_difference_deg': max_angle_common,
        'target_float32_component_matches': exact_target_components,
        'target_float32_component_count': target_components,
        'target_coordinate_max_abs_difference_m': max_target_abs_common,
        'phase_and_right_grip_state_match': True,
        'coarse_interval_vs_refined_two_halfsteps_rate_max_abs_difference_deg_s': max_rate_diff,
        'coarse_step_vs_refined_last_halfstep_rate_max_abs_difference_deg_s': max_local_rate_diff,
    },
    'refined_path_checks': {
        'fine_endpoint_count': new_count,
        'max_angle_difference_from_same_continuous_piecewise_linear_path_deg': max_fine_continuous_error,
        'continuous_loading_and_unloading_rate_magnitude_deg_s': omega_deg_s,
        'max_right_grip_radius_from_rotation_pivot_m': max_radius,
        'coarse_step_chord_speed_at_max_radius_m_s': coarse_chord_speed,
        'fine_step_chord_speed_at_max_radius_m_s': fine_chord_speed,
        'fine_vs_coarse_chord_speed_relative_difference': (fine_chord_speed / coarse_chord_speed - 1.0),
        'interpretation': 'Underlying prescribed angular path and physical angular rate are unchanged. Per-step chord velocities differ at the expected O(dt^2) sampling level because the same curved rotational path is more finely sampled.',
    },
    'force_accuracy_contract': {
        '467_relative_residual_cli_threshold': 1e-6,
        '468_relative_residual_cli_threshold': 5e-7,
        'units_and_condition': 'For this pure mechanical coupon, source runtime forms certificate as generalized residual impulse norm divided by max(reference scale, 1). While the denominator remains 1, certificate is numerically an N s residual; divided by dt it gives the equivalent force bound.',
        '467_conditional_force_bound_N': 1e-6 / old_dt,
        '468_conditional_force_bound_N': 5e-7 / new_dt,
        'conditional_bounds_equal': (1e-6 / old_dt) == (5e-7 / new_dt),
        'postrun_gate': 'Verify the residual scale remains 1 for every accepted row and separately compare measured free_force_imbalance_l2_N/max_N; no force-accuracy conclusion can be made from this prepared command.',
    },
    'telemetry_caveat': {
        'source_file': str(SNAP / 'matter/tools/cardboard_probe.mm'),
        'line_ranges': {'target_flags': '4245-4255', 'CSV_right_grip_constrained': '6180-6187'},
        'detail': 'fillTargets sets right-grip target w=2 (free) for release and post_release_relaxation, while writeObservation emits right_grip_constrained=0 only when phase == release and emits 1 for post_release_relaxation. Treat this CSV field as incorrect for post-release; derive the grip mode from authored phase/target code and the retained requested phase counts.',
    },
}
if exact_target_components != target_components or max_angle_common != 0.0 or max_time_common != 0.0 or max_fine_continuous_error > 1e-15 or not angular_summary['force_accuracy_contract']['conditional_bounds_equal']:
    raise RuntimeError('clock/target/force-accuracy equivalence check failed')
(PKG / 'protocol-clock.json').write_text(json.dumps(angular_summary, indent=2, sort_keys=True) + '\n')

# Keep source line/hash provenance next to the transcription.
probe_source = SNAP / 'matter/tools/cardboard_probe.mm'
driver_binding = {
    'path': str(probe_source),
    'sha256': sha(probe_source),
    'source_build_map_sha256': build['source_sha256']['matter/tools/cardboard_probe.mm'],
    'code_anchors': {
        'phase counts and target schedule': '4067-4086',
        'right-grip rotated target': '4202-4258',
        'time column uses (step+1)*runtime timestep': '6150-6156',
        'right_grip_constrained diagnostic field': '6180-6187',
    },
    'method': 'Python double transcription of the displayed source formulas, float32 output cast, exact output mesh 467 and runtime float32 dt. This is schedule equivalence only, not a native physics/reaction comparison.',
}
(PKG / 'driver-source-binding.json').write_text(json.dumps(driver_binding, indent=2, sort_keys=True) + '\n')

question_text = argv[argv.index('--question') + 1]
prediction_text = argv[argv.index('--prediction') + 1]
(PKG / 'README.md').write_text(f'''# Native 468 dt/2 complete-cycle command preparation\n\nPrepared only; no native process, GPU launch, or build was run.\n\n## Decision\n\nThe source154 clock permits an exact schedule-preserving dt/2 command: 467 uses 12/3/6/3/3/3 steps at an effective runtime dt of {old_dt:.17g} s. 468 uses 24/6/12/6/6/6 at {new_dt:.17g} s. Both total {old_duration:.17g} s. For all {old_count} original sample timestamps, phase, target angle, right-grip mode, and all {target_components} right-grip target float components match exactly. The intermediate fine endpoints lie on the same piecewise-linear angle(t) curve.\n\nThe prescribed angular rate is {omega_deg_s:.9g} deg/s in both loading and unloading. The per-step chord-speed difference is {angular_summary['refined_path_checks']['fine_vs_coarse_chord_speed_relative_difference']:.9g}, a sampling effect on the same curved rotational path.\n\nThe command sets the certificate scalar from 1e-6 to 5e-7. Under the source runtime's pure-mechanical residual normalization of max(reference, 1), while the residual scale stays 1, this preserves a conditional 0.01 N force bound at both dt values. The new command must verify that scale row-by-row and inspect the CSV force-imbalance columns after execution; this prepared package makes no force-accuracy claim.\n\nThe nonlinear ceiling changes from 128 to 64, as requested for capacity/runtime bounding. This is therefore a dt/2 diagnostic with a different nonlinear budget, not a pure time-refinement qualification. Other physical and solver settings from 467 are retained.\n\n## Native 467 predecessor status\n\n467 was interrupted by its capacity guard after only control step 0 completed for both environments. It was not a nonlinear solver failure or completed cycle. The case receipt, partial observations/mesh/manifest, launch guard, and capacity-audit report are hash-bound as external command inputs. 466 full-attempt default64 parity, source154 build-topology erratum, the 787 source-file hashes, and binary/build-metadata hashes are also bound.\n\n## Limits\n\n- No native 468 run has occurred.\n- The C++ observation field `right_grip_constrained` is wrong during post-release relaxation: the target writer frees the right grip in both release phases, but the CSV column marks post-release as constrained. Use the phase/target source binding, not that CSV field, for post-release classification.\n- Exact common-time prescribed target equality does not prove equal discrete FEM trajectories or reactions; dt/2 changes the integration discretization.\n- This is not physical/calibration qualification.\n\n## Command\n\nSee `native-468-command.json`. Inputs are copied under unique basenames in `inputs/`; every copy is hash-bound in `external-input-bindings.json`.\n\nQuestion: {question_text}\n\nPrediction: {prediction_text}\n''')

# Record source/build binding in a compact verification file.
bindings = {
    'schema': 'numi.cardboard.native468.source-build-bindings.v1',
    'source_snapshot': str(SNAP),
    'source_snapshot_manifest_sha256': sha(SNAP / 'frozen-source.json'),
    'configure_support_sha256': sha(SNAP / 'configure-support-source.json'),
    'source_build_provenance_path': str(build_json_path),
    'source_build_provenance_sha256': sha(build_json_path),
    'source_file_hash_count': len(build['source_sha256']),
    'source_file_hash_mismatches': source_bad,
    'artifacts': artifacts,
    'build_metadata': metadata,
    'build_topology_erratum_manifest_sha256': sha(ROOT / 'analysis/source154-build-topology-erratum-001/manifest.json'),
    'build_topology_note': 'The source154 build used CMake source root matter and emitted the recorded source-relative target/artifacts; preserve the source154 topology erratum in the inherited evidence. This command reuses that exact build and does not rebuild.',
    'all_live_source_and_build_hashes_match': True,
}
(PKG / 'source-build-bindings.json').write_text(json.dumps(bindings, indent=2, sort_keys=True) + '\n')

# Final package inventory excluding the manifest itself.
files = {}
for path in sorted(PKG.rglob('*')):
    if path.is_file() and path.name != 'manifest.json':
        files[str(path.relative_to(PKG))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
manifest = {
    'schema': 'numi.cardboard.native468-dt-half-command-preparation.v1',
    'package': str(PKG),
    'case_path': str(CASE468),
    'native_execution': 'not launched',
    'build': 'not run',
    'gpu': 'not used',
    'base_command_path': str(BASE_COMMAND),
    'base_command_sha256': sha(BASE_COMMAND),
    'new_command_sha256': sha(PKG / 'native-468-command.json'),
    'source_build_provenance_sha256': sha(build_json_path),
    'source_snapshot_hash_count': len(build['source_sha256']),
    'all_source_artifact_build_metadata_hashes_match_live': True,
    'copied_external_input_count': len(all_sources),
    'external_input_basenames_unique': True,
    'protocol_clock_path_preserved': True,
    'force_accuracy_equivalence': 'conditional on residual scale == 1; verifier required after run',
    'outer_newton_budget_changed': {'467': 128, '468': 64},
    'files': files,
}
(PKG / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({
    'package': str(PKG),
    'command_sha256': manifest['new_command_sha256'],
    'inputs': len(all_sources),
    'source_hashes_verified': len(build['source_sha256']),
    'effective_dt_467': old_dt,
    'effective_dt_468': new_dt,
    'total_duration_s': old_duration,
    'common_time_samples': old_count,
    'target_components_bitwise_equal': exact_target_components,
    'conditional_force_bounds_N': [1e-6 / old_dt, 5e-7 / new_dt],
}, indent=2))
