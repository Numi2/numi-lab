#!/usr/bin/env python3
"""Prepare source157 command003 from sealed 468002; never launches native work."""
from __future__ import annotations
import csv, hashlib, json, math, pathlib, shutil, struct, subprocess

ROOT = pathlib.Path('/Users/home/cardboard-implementation-20261014')
SNAP154 = ROOT / 'snapshots/source-cgls-all-controls-newton128-154'
SNAP157 = ROOT / 'snapshots/source-right-grip-telemetry-157'
BUILD154 = pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-154')
BUILD157 = pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157')
BASE_PACKAGE = ROOT / 'analysis/native468-source154-dt-half-cycle-command-002'
BASE_COMMAND = BASE_PACKAGE / 'native-468-command.json'
CASE467 = ROOT / 'solver/cgls-complete-outer128-allcontrols-467'
PKG = ROOT / 'analysis/native468-source157-dt-half-cycle-command-003'
CASE468 = ROOT / 'solver/cgls-dt-half-cycle-468'
OVERLAY = ROOT / 'analysis/source157-right-grip-telemetry-fix-001'
BUILD_PACKAGE = ROOT / 'analysis/source157-telemetry-native-build-001'


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]


def phase_point(counts: dict[str, int], bend: float, step: int) -> tuple[str, float, bool]:
    remaining = step
    order = ('loading', 'hold', 'unloading', 'relaxation', 'release', 'post_release_relaxation')
    for phase in order:
        n = counts[phase]
        if remaining < n:
            if phase == 'loading': angle = bend * float(remaining + 1) / n
            elif phase == 'hold': angle = bend
            elif phase == 'unloading': angle = bend * (1.0 - float(remaining + 1) / n)
            else: angle = 0.0
            return phase, angle, phase in ('release', 'post_release_relaxation')
        remaining -= n
    raise IndexError(step)


def target_position(rest: list[float], angle_deg: float, length: float,
                    width: float, height: float) -> tuple[float, float, float]:
    if angle_deg == 0.0: return (f32(rest[0]), f32(rest[1]), f32(rest[2]))
    theta = angle_deg * math.pi / 180.0
    px, py, pz = length, 0.5 * width, 0.5 * height
    rx, rz = rest[0] - px, rest[2] - pz
    return (f32(px + math.cos(theta) * rx + math.sin(theta) * rz),
            f32(rest[1]), f32(pz - math.sin(theta) * rx + math.cos(theta) * rz))


def pack_float(x: float) -> bytes:
    return struct.pack('<f', x)


def extract_function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == '{': depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0: return source[brace + 1:i]
    raise RuntimeError(f'unclosed function: {signature}')


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


if not PKG.is_dir(): raise SystemExit(f'package directory must exist: {PKG}')
if any(p.name != 'prepare.py' and not (p.name == 'inputs' and p.is_dir() and not any(p.iterdir())) for p in PKG.iterdir()):
    raise SystemExit(f'refusing to overwrite generated package artifacts: {PKG}')
if CASE468.exists(): raise SystemExit(f'468 case path already exists: {CASE468}')
inputs_dir = PKG / 'inputs'; inputs_dir.mkdir(exist_ok=True)

# Rebind the exact already-declared 468002 command; do not reconstruct or
# silently alter its dt, six phase counts, Newton cap, certificate, or solver flags.
base_argv = json.loads(BASE_COMMAND.read_text())
if base_argv[0] != '/Users/home/.pyenv/versions/3.10.12/bin/python3':
    raise RuntimeError('unexpected Python ABI in sealed 468002 command')
if base_argv[1] != str(SNAP154 / 'matter/tools/cardboard_source_bound_run.py'):
    raise RuntimeError('unexpected source-bound runner in sealed 468002 command')
old_input_paths = [pathlib.Path(base_argv[i + 1])
                   for i, token in enumerate(base_argv[:-1]) if token == '--input-file']
if len(old_input_paths) != 70: raise RuntimeError(f'expected 70 sealed 468002 inputs, found {len(old_input_paths)}')
old_bindings = json.loads((BASE_PACKAGE / 'external-input-bindings.json').read_text())
old_record_by_path = {r['copied_path']: r for r in old_bindings['inputs']}
old_sources = []
for p in old_input_paths:
    r = old_record_by_path.get(str(p))
    if not r or not p.is_file() or sha(p) != r['sha256']:
        raise RuntimeError(f'sealed 468002 input does not match binding: {p}')
    old_sources.append((p, f'inherited sealed 468002: {r["role"]}'))

build_path = BUILD157 / 'source-build.json'
build = json.loads(build_path.read_text())
frozen = json.loads((SNAP157 / 'frozen-source.json').read_text())
support = json.loads((SNAP157 / 'configure-support-source.json').read_text())
if build.get('source_snapshot') != str(SNAP157) or build.get('build_dir') != str(BUILD157):
    raise RuntimeError('source157 build provenance points to another snapshot/build')
if len(build.get('source_sha256', {})) != 787 or len(frozen.get('source_sha256', {})) != 787:
    raise RuntimeError('source157 source-count mismatch')
if build['source_sha256'] != frozen['source_sha256']:
    raise RuntimeError('source157 build source map differs from frozen source map')
if len(support.get('files', {})) != 1034:
    raise RuntimeError('source157 configure-support count mismatch')
source_mismatches = [rel for rel, expected in build['source_sha256'].items()
                     if not (SNAP157 / rel).is_file() or sha(SNAP157 / rel) != expected]
support_mismatches = [rel for rel, expected in support['files'].items()
                      if not (SNAP157 / rel).is_file() or sha(SNAP157 / rel) != expected]
if source_mismatches or support_mismatches:
    raise RuntimeError(f'live source mismatch: {source_mismatches[:3]} {support_mismatches[:3]}')
build_package_manifest = json.loads((BUILD_PACKAGE / 'manifest.json').read_text())
if build_package_manifest.get('frozen_source_sha256') != sha(SNAP157 / 'frozen-source.json'):
    raise RuntimeError('source157 frozen-source digest mismatch')
if build_package_manifest.get('source_overlay_manifest_sha256') != sha(SNAP157 / 'source-overlay.json'):
    raise RuntimeError('source157 source-overlay digest mismatch')

# Bind exact source157 overlay/build, compile-only budgets, and the source154
# parent. All extra evidence files are small metadata/text outputs, not build trees.
extra_sources: list[tuple[pathlib.Path, str]] = [
    (BASE_COMMAND, 'sealed source154 command002, the exact parent command being versioned'),
    (BASE_PACKAGE / 'manifest.json', 'sealed source154 command002 manifest'),
    (BASE_PACKAGE / 'external-input-bindings.json', 'sealed source154 command002 inherited input bindings'),
    (BASE_PACKAGE / 'source-build-bindings.json', 'sealed source154 command002 source/build binding'),
    (BASE_PACKAGE / 'protocol-clock.json', 'source154 command002 prior clock comparison'),
    (SNAP157 / 'frozen-source.json', 'source157 exact 787-file frozen source map'),
    (SNAP157 / 'configure-support-source.json', 'source157 exact 1034-file configure support map'),
    (SNAP157 / 'source-overlay.json', 'source157 overlay manifest'),
    (SNAP157 / 'source157-recovery.json', 'source157 recovery/parent metadata'),
    (build_path, 'source157 full source/build provenance with 787 source hashes'),
    (BUILD_PACKAGE / 'manifest.json', 'source157 fresh build and artifact manifest'),
    (BUILD_PACKAGE / 'verification-report.json', 'source157 compile-only and topology verification'),
    (BUILD_PACKAGE / 'build-metadata/CMakeCache.txt', 'source157 build CMake metadata'),
    (BUILD_PACKAGE / 'build-metadata/compile_commands.json', 'source157 compile command metadata'),
    (BUILD_PACKAGE / 'compile-only-64-command.json', 'source157 compile-only 64 command'),
    (BUILD_PACKAGE / 'compile-only-64/manifest.json', 'source157 compile-only 64 manifest'),
    (BUILD_PACKAGE / 'compile-only-64/result.json', 'source157 compile-only 64 result'),
    (BUILD_PACKAGE / 'compile-only-128-command.json', 'source157 compile-only 128 command'),
    (BUILD_PACKAGE / 'compile-only-128/manifest.json', 'source157 compile-only 128 manifest'),
    (BUILD_PACKAGE / 'compile-only-128/result.json', 'source157 compile-only 128 result'),
    (BUILD_PACKAGE / 'resource-preflight.txt', 'source157 build resource preflight record'),
    (OVERLAY / 'manifest.json', 'source157 telemetry overlay manifest'),
    (OVERLAY / 'source157.patch', 'source157 telemetry patch'),
    (OVERLAY / 'syntax-check.txt', 'source157 Objective-C++ syntax check'),
    (OVERLAY / 'tests/report.json', 'source157 CPU phase/telemetry source-contract tests'),
]
for path, _ in extra_sources:
    if not path.is_file(): raise FileNotFoundError(path)

# Verify real built artifacts and metadata against the source-bound build record.
artifact_checks = {}
for pstr, expected in build['artifacts_sha256'].items():
    p = pathlib.Path(pstr); actual = sha(p) if p.is_file() else None
    artifact_checks[pstr] = {'expected_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected}
metadata_checks = {}
for rel, expected in build['build_metadata_sha256'].items():
    p = BUILD157 / rel; actual = sha(p) if p.is_file() else None
    metadata_checks[rel] = {'expected_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected}
if not all(v['matches'] for v in artifact_checks.values()) or not all(v['matches'] for v in metadata_checks.values()):
    raise RuntimeError('source157 built artifact or build metadata hash mismatch')
def artifact_hash_by_name(mapping, basename):
    values=[v for k,v in mapping.items() if pathlib.Path(k).name==basename]
    if len(values)!=1: raise RuntimeError(f'expected one artifact {basename}, found {len(values)}')
    return values[0]
if artifact_hash_by_name(build['artifacts_sha256'],'NumiMatter.metallib') != '89f9d756a0298fad44786b5333d67172636a8255e8c4f71b2696e4e5d6271114':
    raise RuntimeError('source157 physics metallib identity changed unexpectedly')
if artifact_hash_by_name(build['artifacts_sha256'],'NumiMatterPhysicalStateDigest.metallib') != '0401b29d3ac2165153f36bea19786be653a10a242eed140e1c9008595ec1f729':
    raise RuntimeError('source157 digest metallib identity changed unexpectedly')

# Source-bound unchanged function proof: target generation itself is identical.
probe154 = SNAP154 / 'matter/tools/cardboard_probe.mm'
probe157 = SNAP157 / 'matter/tools/cardboard_probe.mm'
body154 = extract_function(probe154.read_text(), 'void fillTargets(const Arguments& arguments,')
body157 = extract_function(probe157.read_text(), 'void fillTargets(const Arguments& arguments,')
body154_sha = hashlib.sha256(body154.encode()).hexdigest()
body157_sha = hashlib.sha256(body157.encode()).hexdigest()
expected_body = 'cfec49a4be92145586acbf0a3d99c59585475790148859165835e36bafafdbb4'
if body154_sha != expected_body or body157_sha != expected_body:
    raise RuntimeError(f'fillTargets function changed: 154={body154_sha},157={body157_sha}')
fill_binding_path = PKG / 'fillTargets-source-equivalence.json'
write_json(fill_binding_path, {
    'schema': 'numi.cardboard.source157.fill-targets-equivalence.v1',
    'source154_probe_sha256': sha(probe154), 'source157_probe_sha256': sha(probe157),
    'source154_fillTargets_body_sha256': body154_sha,
    'source157_fillTargets_body_sha256': body157_sha,
    'equal': True,
    'meaning': 'The target-generation function is unchanged; source157 corrects the CSV right-grip telemetry by reading generated target constraint modes. This is not physical/grip-behavior qualification.'
})
extra_sources.append((fill_binding_path, 'source154/source157 unchanged fillTargets function-body digest proof'))

# Shader identity binding: source157 was freshly built; both shader libraries
# are byte-identical to source154, while the host probe executable is not.
art154 = json.loads((BUILD154 / 'source-build.json').read_text())['artifacts_sha256']
shader_binding_path = PKG / 'shader-identity.json'
shader_items = {}
for name in ('NumiMatter.metallib', 'NumiMatterPhysicalStateDigest.metallib'):
    src157 = BUILD157 / 'shaders' / name
    src154 = BUILD154 / 'shaders' / name
    shader_items[name] = {
        'source157_path': str(src157), 'source157_sha256': sha(src157),
        'source154_path': str(src154), 'source154_sha256': sha(src154),
        'byte_identical': sha(src157) == sha(src154),
        'source154_build_record_sha256': artifact_hash_by_name(art154,name),
        'source157_build_record_sha256': artifact_hash_by_name(build['artifacts_sha256'],name),
    }
if not all(x['byte_identical'] for x in shader_items.values()):
    raise RuntimeError('source157 shader byte identity check failed')
write_json(shader_binding_path, {
    'schema': 'numi.cardboard.source157.shader-identity.v1',
    'build_topology': 'source root matter, target numi-matter-cardboard-probe; source157 built with 2 jobs versus source154 4 jobs. CMake source-relative topology and release target are the same; job-count difference is recorded, not hidden.',
    'shaders': shader_items,
    'host_probe': {'source154_sha256': artifact_hash_by_name(art154,'numi-matter-cardboard-probe'),
                   'source157_sha256': artifact_hash_by_name(build['artifacts_sha256'],'numi-matter-cardboard-probe'),
                   'byte_identical': artifact_hash_by_name(art154,'numi-matter-cardboard-probe') == artifact_hash_by_name(build['artifacts_sha256'],'numi-matter-cardboard-probe')}
})
extra_sources.append((shader_binding_path, 'source154/source157 exact shader artifact identity proof'))

# Regenerate the paired 467/source154 and 468/source157 clock transcription.
old = {'loading': 12, 'hold': 3, 'unloading': 6, 'relaxation': 3, 'release': 3, 'post_release_relaxation': 3}
new = {'loading': 24, 'hold': 6, 'unloading': 12, 'relaxation': 6, 'release': 6, 'post_release_relaxation': 6}
bend = 0.375; old_dt = f32(0.0001); new_dt = f32(0.00005)
old_count, new_count = sum(old.values()), sum(new.values())
old_duration, new_duration = old_count * old_dt, new_count * new_dt
if old_duration != new_duration or new_dt * 2.0 != old_dt:
    raise RuntimeError('runtime float32 timesteps do not preserve duration exactly')
mesh = json.loads((CASE467 / 'output/mesh.json').read_text())
right_indices, nodes = mesh['right_grip_nodes'], mesh['nodes_m']
length, width, height = 0.0316, 0.02, 0.00421
rows=[]; max_angle=max_time=max_target=max_interval_rate=max_last_rate=0.0
exact_components=component_count=0
previous_old=0.0
for k in range(1, old_count+1):
    p0,a0,free0=phase_point(old,bend,k-1); p1,a1,free1=phase_point(new,bend,2*k-1)
    t0,t1=k*old_dt,(2*k)*new_dt
    max_time=max(max_time,abs(t0-t1)); max_angle=max(max_angle,abs(a0-a1))
    coarse_rate=(a0-previous_old)/old_dt
    previous_new_common=phase_point(new,bend,2*k-3)[1] if k>1 else 0.0
    interval_rate=(a1-previous_new_common)/(2*new_dt)
    fine_prev=phase_point(new,bend,2*k-2)[1]
    last_rate=(a1-fine_prev)/new_dt
    max_interval_rate=max(max_interval_rate,abs(coarse_rate-interval_rate))
    max_last_rate=max(max_last_rate,abs(coarse_rate-last_rate))
    if p0!=p1 or free0!=free1: raise RuntimeError(f'clock phase mismatch at {k}')
    sample_max=0.0; sample_exact=0
    for ni in right_indices:
        xyz0=target_position(nodes[ni],a0,length,width,height)
        xyz1=target_position(nodes[ni],a1,length,width,height)
        for x,y in zip(xyz0,xyz1):
            component_count+=1; sample_max=max(sample_max,abs(x-y))
            if pack_float(x)==pack_float(y): exact_components+=1; sample_exact+=1
    max_target=max(max_target,sample_max)
    rows.append({'common_sample':k,'coarse_step_0_based':k-1,'refined_step_0_based':2*k-1,
                 'coarse_time_s_runtime_f32_dt':t0,'refined_time_s_runtime_f32_dt':t1,
                 'phase':p0,'coarse_target_angle_deg':a0,'refined_target_angle_deg':a1,
                 'right_grip_free':free0,'coarse_discrete_angular_rate_deg_s':coarse_rate,
                 'refined_interval_average_angular_rate_deg_s':interval_rate,
                 'refined_final_half_step_angular_rate_deg_s':last_rate,
                 'right_grip_target_float_components_exact':sample_exact,
                 'right_grip_target_float_components_total':len(right_indices)*3,
                 'max_target_coordinate_abs_difference_m':sample_max})
    previous_old=a0
with (PKG/'common-clock-target-comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def continuous_angle(t: float, d: dict[str,float]) -> float:
    load=d['loading']; hold=d['hold']; unload=d['unloading']
    if t<=load:return bend*t/load
    t-=load
    if t<=hold:return bend
    t-=hold
    if t<=unload:return bend*(1-t/unload)
    return 0.0
old_durs={p:n*old_dt for p,n in old.items()}; new_durs={p:n*new_dt for p,n in new.items()}
max_cont=0.0
for idx in range(1,new_count+1):
    _,angle,_=phase_point(new,bend,idx-1)
    max_cont=max(max_cont,abs(angle-continuous_angle(idx*new_dt,new_durs)))
max_radius=max(math.hypot(nodes[i][0]-length,nodes[i][2]-height/2) for i in right_indices)
omega=bend/old_durs['loading']
dtheta0=math.radians(bend/old['loading']); dtheta1=math.radians(bend/new['loading'])
chord0=2*max_radius*math.sin(dtheta0/2)/old_dt
chord1=2*max_radius*math.sin(dtheta1/2)/new_dt

# Explicit post-release telemetry comparison: the physical targets/modes are
# free in both sources, but 154 emitted a phase-derived false constrained=1.
telemetry_rows=[]
first_post=sum(new[p] for p in ('loading','hold','unloading','relaxation','release'))
for step in range(first_post, new_count):
    phase,angle,free=phase_point(new,bend,step)
    for env in (0,1):
        target_w=2.0 if free else 1.0
        telemetry_rows.append({
            'zero_based_step':step,'one_based_step':step+1,'environment':env,
            'phase':phase,'target_mode_w':int(target_w),
            'source154_logged_right_grip_constrained':int(phase!='release'),
            'source157_logged_right_grip_constrained':int(target_w==1.0),
            'expected_observation_difference_source154_minus_source157':int(phase!='release')-int(target_w==1.0),
            'target_angle_deg': angle if env==0 else 0.0,
            'interpretation':'expected CSV telemetry correction only; not a native result or a physical grip qualification'
        })
if any(r['source157_logged_right_grip_constrained'] != 0 for r in telemetry_rows) or len(telemetry_rows)!=12:
    raise RuntimeError('unexpected post-release target/telemetry schedule')
with (PKG/'post-release-telemetry-difference.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(telemetry_rows[0])); w.writeheader(); w.writerows(telemetry_rows)

clock = {
    'schema':'numi.cardboard.native468.source157-clock-protocol.v1',
    'source_pair':{'coarse_source':'source-cgls-all-controls-newton128-154','refined_source':'source-right-grip-telemetry-157',
                   'coarse_probe_sha256':sha(probe154),'refined_probe_sha256':sha(probe157),
                   'unchanged_fillTargets_body_sha256':body154_sha},
    'source_clock':{'total_steps':'sum of phase counts','sample_time':'runtime effective float32 dt * (zero_based_step + 1)',
                    'release_target_mode':'fillTargets sets right grip target w=2 (free) in release and post_release_relaxation'},
    '467':{'phase_counts':old,'effective_runtime_dt_s':old_dt,'phase_durations_s':old_durs,'total_duration_s':old_duration,'bend_angle_deg':bend},
    '468':{'phase_counts':new,'effective_runtime_dt_s':new_dt,'phase_durations_s':new_durs,'total_duration_s':new_duration,'bend_angle_deg':bend},
    'common_sample_checks':{'count':old_count,'common_time_max_abs_difference_s':max_time,
       'target_angle_max_abs_difference_deg':max_angle,'target_float32_component_matches':exact_components,
       'target_float32_component_count':component_count,'target_coordinate_max_abs_difference_m':max_target,
       'phase_and_target_mode_match':True,'coarse_interval_vs_refined_two_halfsteps_rate_max_abs_difference_deg_s':max_interval_rate,
       'coarse_step_vs_refined_last_halfstep_rate_max_abs_difference_deg_s':max_last_rate},
    'refined_path_checks':{'fine_endpoint_count':new_count,'max_angle_difference_from_same_continuous_piecewise_linear_path_deg':max_cont,
       'continuous_loading_and_unloading_rate_magnitude_deg_s':omega,'max_right_grip_radius_from_rotation_pivot_m':max_radius,
       'coarse_step_chord_speed_at_max_radius_m_s':chord0,'fine_step_chord_speed_at_max_radius_m_s':chord1,
       'fine_vs_coarse_chord_speed_relative_difference':chord1/chord0-1,
       'interpretation':'Prescribed angle path and physical angular rate are unchanged. Per-step chord velocities differ at the expected sampling level.'},
    'force_accuracy_contract':{'467_relative_residual_cli_threshold':1e-6,'468_relative_residual_cli_threshold':5e-7,
       'certificate_source_formula':'||generalizedResidual|| / max(referenceScale, 1)',
       'condition':'For this pure mechanical coupon, while referenceScale remains below 1, the denominator is 1 and the certificate is numerically an impulse residual in N s.',
       '467_conditional_force_bound_N':1e-6/old_dt,'468_conditional_force_bound_N':5e-7/new_dt,
       'conditional_bounds_equal':(1e-6/old_dt)==(5e-7/new_dt),
       'postrun_gate':'Verify referenceScale == 1 for each accepted sample and inspect measured free_force_imbalance_l2_N/max_N separately. The numeric 0.01 N equivalence is conditional on that scale; no force-accuracy conclusion is made here.'},
    'telemetry_correction':{'source154':{'csv_expression':'phase == release ? 0 : 1','post_release_value':1},
       'source157':{'csv_expression':'commandedRightGripConstrained(targets, mesh, environment)','post_release_value':0},
       'post_release_rows':len(telemetry_rows),'source154_false_constrained_rows':sum(r['source154_logged_right_grip_constrained'] for r in telemetry_rows),
       'source157_false_constrained_rows':sum(r['source157_logged_right_grip_constrained'] for r in telemetry_rows),
       'csv_path':'post-release-telemetry-difference.csv','scope':'expected observational column correction in six post-release steps across two environments; physics, targets, solver inputs and acceptance are unchanged.'}
}
if exact_components!=component_count or max_angle!=0 or max_time!=0 or max_cont>1e-15 or not clock['force_accuracy_contract']['conditional_bounds_equal']:
    raise RuntimeError('source154/source157 schedule equivalence check failed')
write_json(PKG/'protocol-clock.json',clock)

driver_binding={'schema':'numi.cardboard.native468.source157-driver-binding.v1',
  'source154_probe_path':str(probe154),'source154_probe_sha256':sha(probe154),
  'source157_probe_path':str(probe157),'source157_probe_sha256':sha(probe157),
  'source_map_sha256':build['source_sha256']['matter/tools/cardboard_probe.mm'],
  'anchors':{'phase_count_parsing':'4067-4086','target generation':'4232-4265',
             'source154 hardcoded CSV flag':'6180-6187','source157 target-mode CSV flag':'6179-6212',
             'source157 actual target-mode lookup':'12565-12570'},
  'method':'Source-bound Python transcription of the schedule formulas using the retained 467 compiled mesh and runtime float32 dt. This checks prescribed clocks/targets and logging logic only; it is not a native physics/reaction comparison.'}
write_json(PKG/'driver-source-binding.json',driver_binding)
extra_sources.extend([(PKG/'protocol-clock.json','regenerated source154/source157 schedule clock comparison'),
                      (PKG/'common-clock-target-comparison.csv','regenerated 30-point common clock/target comparison'),
                      (PKG/'post-release-telemetry-difference.csv','expected source154/source157 post-release telemetry rows'),
                      (PKG/'fillTargets-source-equivalence.json','unchanged source154/source157 target function proof'),
                      (PKG/'shader-identity.json','source154/source157 shader artifact identity proof')])

# Copy both inherited run inputs and the new source157 admission evidence under
# unique basenames. The wrapper requires unique basename, not just unique path.
all_sources = old_sources + extra_sources
copy_records=[]; new_paths=[]; seen=set()
for index,(src,role) in enumerate(all_sources):
    if not src.is_file(): raise FileNotFoundError(src)
    name=f'evidence-{index:03d}-{src.name}'
    if name in seen: raise RuntimeError(f'duplicate copied basename {name}')
    seen.add(name); dest=inputs_dir/name; shutil.copyfile(src,dest)
    srcsha,destsha=sha(src),sha(dest)
    if srcsha!=destsha: raise RuntimeError(f'copy mismatch {src}')
    new_paths.append(str(dest))
    copy_records.append({'index':index,'role':role,'source_path':str(src),'copied_path':str(dest),
                         'bytes':dest.stat().st_size,'sha256':destsha})

# Source build binding records the fresh 157 artifacts and the deliberate
# CMake topology difference (same CMake source-relative root; jobs differ).
checksums={}
for rel,expected in build['build_metadata_sha256'].items():
    actual=sha(BUILD157/rel); checksums[rel]={'expected_sha256':expected,'actual_sha256':actual,'matches':actual==expected}
if not all(x['matches'] for x in checksums.values()): raise RuntimeError('build metadata changed')
source154build=json.loads((BUILD154/'source-build.json').read_text())
source157_bindings={
 'schema':'numi.cardboard.native468.source-build-bindings.v1','source_snapshot':str(SNAP157),
 'source_snapshot_manifest_sha256':sha(SNAP157/'frozen-source.json'),
 'configure_support_sha256':sha(SNAP157/'configure-support-source.json'),
 'source_file_hash_count':len(build['source_sha256']),'source_file_hash_mismatches':source_mismatches,
 'configure_support_file_hash_count':len(support['files']),'configure_support_hash_mismatches':support_mismatches,
 'source_build_provenance_path':str(build_path),'source_build_provenance_sha256':sha(build_path),
 'source_overlay_manifest_sha256':sha(SNAP157/'source-overlay.json'),
 'source_recovery_sha256':sha(SNAP157/'source157-recovery.json'),
 'build_package_manifest_sha256':sha(BUILD_PACKAGE/'manifest.json'),
 'verification_report_sha256':sha(BUILD_PACKAGE/'verification-report.json'),
 'compile_only':{'64':json.loads((BUILD_PACKAGE/'compile-only-64/result.json').read_text()),
                 '128':json.loads((BUILD_PACKAGE/'compile-only-128/result.json').read_text())},
 'artifacts':artifact_checks,'build_metadata':checksums,
 'source154_parent_artifacts_sha256':source154build['artifacts_sha256'],
 'build_topology':{'source_relative_root':'matter','target':'numi-matter-cardboard-probe',
                   'source154_jobs':4,'source157_jobs':2,'same_release_target_and_cmake_source_relative_root':True,
                   'warning':'Job count differs; no claim of identical build command. Both source-bound builds are individually hashed.'},
 'metallibs_byte_identical_to_source154':all(x['byte_identical'] for x in shader_items.values()),
 'host_probe_byte_identical_to_source154':False,
 'all_live_hashes_match':True}
write_json(PKG/'source-build-bindings.json',source157_bindings)
# Bind the generated source/build binder itself as an external run input.
sourcebind=PKG/'source-build-bindings.json'
source_index=len(all_sources)
dest=inputs_dir/f'evidence-{source_index:03d}-{sourcebind.name}'
shutil.copyfile(sourcebind,dest)
new_paths.append(str(dest));copy_records.append({'index':source_index,'role':'source157 full source/build/artifact/topology binding','source_path':str(sourcebind),'copied_path':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest)})
all_sources.append((sourcebind,'source157 full source/build/artifact/topology binding'))

# Rewrite command only for source/build/artifact/material paths, copied external
# evidence paths, and explicit source identity text. Physics settings from 002
# remain exact, including dt/phase/Newton/certificate/CGLS controls.
argv=list(base_argv)
argv[1]=str(SNAP157/'matter/tools/cardboard_source_bound_run.py')
newargv=[]; source_idx=0; i=0
while i<len(argv):
    if argv[i]=='--input-file':
        if source_idx>=70: raise RuntimeError('too many inherited input paths')
        newargv.extend(['--input-file',new_paths[source_idx]]);source_idx+=1;i+=2
    else:newargv.append(argv[i]);i+=1
if source_idx!=70: raise RuntimeError(f'expected to rebind 70 inherited paths, rebound {source_idx}')
argv=newargv
insert_at=argv.index('--')
for p in new_paths[70:]:
    argv[insert_at:insert_at]=['--input-file',p];insert_at+=2

def set_pair(option,value):
    matches=[i for i,t in enumerate(argv[:-1]) if t==option]
    if len(matches)!=1: raise RuntimeError(f'expected exactly one {option}, found {len(matches)}')
    argv[matches[0]+1]=value
set_pair('--source-snapshot',str(SNAP157))
set_pair('--runtime-dir',str(BUILD157))
set_pair('--build-provenance',str(build_path))
set_pair('--question','Under source157, does dt/2 complete the same physical-duration coupon path with the same force-accuracy bound using an explicitly lower 64-Newton budget?')
set_pair('--prediction',
  'Run the source157 dt/2 diagnostic while preserving the source154 physical-duration schedule: double each of the six phase counts and halve dt so loading, hold, unloading, relaxation, release and post-release durations remain 0.003 seconds total, with identical prescribed path/rate at common times. Geometry, mesh, material assets, solver/acceptance settings and the 64-Newton diagnostic budget remain as declared in sealed command002. The certificate scalar remains 5e-7; in this pure mechanical case its residual is generalized impulse divided by max(referenceScale,1), so the numerical 0.01 N equivalence is conditional on referenceScale staying 1 for every accepted row. Inspect free-force imbalance separately. Source157 corrects only the right_grip_constrained CSV field after release by using generated target modes; fillTargets and the actual target path are unchanged. This telemetry correction does not establish physical grip behavior. The 64-vs-128 Newton ceiling means this is not a pure time-refinement qualification.')
set_pair('--metallib',str(BUILD157/'shaders/NumiMatter.metallib'))
set_pair('--liner-material',str(SNAP157/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'))
set_pair('--medium-material',str(SNAP157/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'))
if argv[argv.index('--case')+1]!=str(CASE468) or argv[argv.index('--output')+1]!=str(CASE468/'output'):
    raise RuntimeError('case/output path drift')
# Immutable protocol numeric values expected from sealed 002.
expected_values={'--newton-iterations':'64','--steps':'24','--hold-steps':'6','--unload-steps':'12',
 '--relax-steps':'6','--release-steps':'6','--post-release-relax-steps':'6','--dt':'.00005',
 '--residual-tolerance':'5e-7','--fgmres-restart':'16','--fgmres-iterations':'1024','--material-iterations':'32'}
for opt,val in expected_values.items():
    if argv[argv.index(opt)+1]!=val:raise RuntimeError(f'protocol numeric drift {opt}')
for name,val in {'NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR':'all:all','NM_FEM_BASE_CGLS_ITERATIONS':'64',
 'NM_FEM_BASE_CGLS_RIGHT_SCALE':'block3','NM_FEM_BASE_CGLS_PRIMARY_AFTER':'48',
 'NM_FEM_TRUE_RESIDUAL_ROOT':'all','NM_FEM_RESIDUAL_ARMIJO_ROOT':'all'}.items():
    actual=[argv[j+1] for j,t in enumerate(argv[:-1]) if t=='--env' and argv[j+1].startswith(name+'=')]
    if actual!=[name+'='+val]:raise RuntimeError(f'control drift {name}: {actual}')
if argv.count('--input-file')!=len(new_paths):raise RuntimeError('input path count mismatch')
if len({pathlib.Path(new_paths[j]).name for j in range(len(new_paths))})!=len(new_paths):raise RuntimeError('input basenames not unique')
command_path=PKG/'native-468-command.json'; command_path.write_text(json.dumps(argv,indent=2)+'\n')
write_json(PKG/'external-input-bindings.json',{'schema':'numi.cardboard.native468.source157.external-input-bindings.v1',
 'inherited_468002_input_count':70,'source157_new_admission_input_count':len(new_paths)-70,
 'unique_basename_count':len(new_paths),'inputs':copy_records,
 'note':'The binding file is package metadata, not an input to avoid self-reference. All listed copied inputs are bound into native-468-command.json.'})

# Snapshot the current capacity observation, but make it an explicit dispatch
# prerequisite rather than stale evidence of future free space.
free_bytes=int(subprocess.check_output(['df','-k','/Users/home'],text=True).splitlines()[-1].split()[3])*1024
capacity={'schema':'numi.cardboard.native468.source157.capacity-preflight.v1','observed_free_bytes_at_preparation':free_bytes,
 'native_launch_minimum_free_bytes':1610612736,'native_launch_allowed_by_this_observation':free_bytes>=1610612736,
 '512MiB_abort_floor_bytes':536870912,'native_execution':'not launched',
 'instruction':'Recheck live free space before native launch; this preparation-time measurement is not a launch authorization.'}
write_json(PKG/'capacity-preflight.json',capacity)

readme=f'''# Native 468 source157 dt/2 command preparation

Preparation only: no native process, GPU launch, or simulation was run. The command is a source157 version of sealed command002; command001/002 remain unchanged and unrun.

## Rebound source/build identity

- Snapshot: `{SNAP157}`; frozen-source digest `{sha(SNAP157/'frozen-source.json')}`; 787/787 source entries and 1034/1034 configure-support entries verified.
- Fresh build: `{BUILD157}`; source-build provenance `{sha(build_path)}`; source157 build package `{sha(BUILD_PACKAGE/'manifest.json')}`; compile-only 64 and 128 both report zero accepted steps and no runtime execution.
- Metallibs are byte-identical to source154: NumiMatter `{artifact_hash_by_name(build['artifacts_sha256'],'NumiMatter.metallib')}`; physical digest `{artifact_hash_by_name(build['artifacts_sha256'],'NumiMatterPhysicalStateDigest.metallib')}`. The host probe executable is source157-specific and bound separately.
- CMake source-relative root is `matter`; target is `numi-matter-cardboard-probe`. Source157 used 2 build jobs versus source154 4; this topology difference is recorded rather than called identical.
- `fillTargets` is byte-identical at function-body SHA `{body157_sha}`. The telemetry-only change reads the commanded target mode for `right_grip_constrained`.

## Preserved protocol versus 467

The command retains the already-declared phase counts 24/6/12/6/6/6, dt `.00005`, outer Newton budget 64, certificate scalar `5e-7`, material iterations 32, FGMRES restart/limit 16/1024 and CGLS all-control settings from command002. Compared with 467 (12/3/6/3/3/3, dt `0.0001`, Newton 128, certificate `1e-6`), phase durations and total `0.003` second duration match. The 64-vs-128 Newton ceiling is different, so this is a dt/2 diagnostic, not pure temporal qualification.

The regenerated `protocol-clock.json` and `common-clock-target-comparison.csv` bind coarse source154 to refined source157. At all 30 common samples, phase, time, angle, target mode, and all {component_count} right-grip target float components match; max coarse interval-average rate discrepancy is {max_interval_rate:.17g} deg/s. This is schedule/target transcription only, not native trajectory equivalence.

## Telemetry correction

The old source154 log expression marks post-release relaxation constrained (`phase == release ? 0 : 1`) despite the unchanged target generator assigning free mode `w=2`. Source157 logs the generated constraint mode and emits 0 for post-release. `post-release-telemetry-difference.csv` lists the six post-release steps across both environments. This changes reporting only; it does not change targets, solver state, or physical contact/retention behavior. No post-release physical qualification is implied.

## Conditional force bound

The source certificate is `||generalizedResidual|| / max(referenceScale, 1)`. For the pure mechanical coupon, while `referenceScale` remains below 1, the denominator is 1 and the certificate is numerically an impulse residual in N s. The selected 5e-7 threshold at dt 5e-5 corresponds conditionally to 0.01 N. After a run, verify scale==1 on each accepted row and separately evaluate `free_force_imbalance_l2_N` and `free_force_imbalance_max_N`. This command preparation makes no force-accuracy claim.

## Capacity and execution

At preparation, observed free space was {free_bytes} bytes; the requested 1.5 GiB native-launch minimum is {1610612736}. The observed value is below that threshold. Root must recheck storage and publication audit before launch. The command is not dispatched here.

The canonical argument vector is `native-468-command.json`; 70 prior inputs and {len(new_paths)-70} source157 admission inputs are copied with unique basenames and content hashes in `external-input-bindings.json`. No raw build tree or metallib is copied into this package.
'''
(PKG/'README.md').write_text(readme)

# Build a command-specific admission record and final inventory.
files={}
for p in sorted(PKG.rglob('*')):
    if p.is_file() and p.name not in ('manifest.json','admission.json'):
        files[str(p.relative_to(PKG))]={'bytes':p.stat().st_size,'sha256':sha(p)}
command_sha=sha(command_path)
admission={'schema':'numi.cardboard.native468.source157.admission.v1','command_path':str(command_path),
 'command_sha256':command_sha,'base_command002_sha256':sha(BASE_COMMAND),'case_path':str(CASE468),
 'source157_snapshot_hash':sha(SNAP157/'frozen-source.json'),'source157_overlay_hash':sha(SNAP157/'source-overlay.json'),
 'source157_build_provenance_hash':sha(build_path),'source157_build_package_manifest_hash':sha(BUILD_PACKAGE/'manifest.json'),
 'source157_source_count':787,'configure_support_count':1034,'source_hash_mismatches':source_mismatches,
 'support_hash_mismatches':support_mismatches,'artifacts':artifact_checks,'build_metadata':checksums,
 'shader_byte_identity_to_source154':all(x['byte_identical'] for x in shader_items.values()),
 'fillTargets_body_sha256_equal_to_source154':body154_sha==body157_sha==expected_body,
 'compile_only_protocols':{k:{'status':json.loads((BUILD_PACKAGE/f'compile-only-{k}/result.json').read_text())['status'],
    'runtime_execution':False,
    'physical_validation':json.loads((BUILD_PACKAGE/f'compile-only-{k}/result.json').read_text())['physical_validation'],
    'accepted_steps':json.loads((BUILD_PACKAGE/f'compile-only-{k}/result.json').read_text())['accepted_steps']}
    for k in ('64','128')},
 'preserved_protocol':expected_values,'new_admission_input_count':len(new_paths)-70,
 'input_basename_unique':True,'capacity_preflight':capacity,
 'execution_status':'prepared_not_dispatched','physical_qualification':'not established'}
write_json(PKG/'admission.json',admission)
files['admission.json']={'bytes':(PKG/'admission.json').stat().st_size,'sha256':sha(PKG/'admission.json')}
manifest={'schema':'numi.cardboard.native468-source157-dt-half-command-preparation.v1',
 'package':str(PKG),'base_package':str(BASE_PACKAGE),'base_command_sha256':sha(BASE_COMMAND),
 'case_path':str(CASE468),'native_execution':'not launched','gpu':'not used','build':'source157 build was previously performed and independently bound; no build in this task',
 'command_sha256':command_sha,'source_count':787,'configure_support_count':1034,'input_count':len(new_paths),
 'unique_external_input_basenames':True,'protocol_clock_regenerated_source154_to_source157':True,
 'fillTargets_unchanged':body154_sha==body157_sha==expected_body,
 'post_release_csv_telemetry_correction_only':True,'conditional_force_bound_requires_reference_scale_1':True,
 'native_launch_capacity_preflight':capacity,'files':files}
write_json(PKG/'manifest.json',manifest)
print(json.dumps({'package':str(PKG),'command_sha256':command_sha,'input_count':len(new_paths),
 'source157_evidence_count':len(new_paths)-70,'source_count':len(build['source_sha256']),
 'support_count':len(support['files']),'target_float32_components_equal':exact_components,
 'target_float32_component_count':component_count,'clock_total_s':new_duration,
 'free_bytes':free_bytes,'native_launch_threshold_bytes':1610612736,
 'native_launch_ready_by_preparation_observation':capacity['native_launch_allowed_by_this_observation']},indent=2))
