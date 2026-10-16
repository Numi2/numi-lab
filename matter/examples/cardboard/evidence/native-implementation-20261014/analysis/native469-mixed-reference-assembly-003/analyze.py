"""Replay frame/element/nodal actions with explicitly mixed FP64/MP roots.

This supplements, and never overwrites, the incomplete all-FP64 replay.
It does not qualify a native finite difference or smooth branch/noise floor.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
B = Path('/Users/home/cardboard-implementation-20261014')
R = B / 'analysis/bilinear-capture-layer-audit-source166-003/candidate/matter/tools'
M = B / 'analysis/native469-mp-fallback-005'
C = B / 'solver/final-newton-fd-capture-469'
O = B / 'analysis/native469-mixed-reference-assembly-003'
O.mkdir(exist_ok=False)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

manifest = M / 'manifest.json'
assert sha(manifest) == 'f176666da43b50a441c038cbd88b5c9c804df32fa8ea9d46483a086ab5616c19'
mp = json.loads(manifest.read_text())
for name, digest in mp['external_inputs_sha256'].items():
    assert sha(Path(name)) == digest, name
for name, digest in mp['included_files_sha256'].items():
    assert sha(M / name) == digest, name
sys.path.insert(0, str(B / 'snapshots/source-right-grip-telemetry-157/matter/tools'))
sys.path.insert(0, str(R))
layers = load('mixed_assembly_layers', R / 'cardboard_bilinear_capture_layers.py')
geometry = layers.geometry
capture = C / 'diagnostics/final-newton'
bindings = geometry.bind_inputs(capture, C / 'prediction.json', C / 'receipt.json', C / 'stderr.log', 0)
header, nodes, tets, integrity = geometry.capture_records(capture)
layers.validate_identity(header, nodes, tets, bindings)
geometry.validate_and_assemble(header, nodes, tets, integrity, 0)
nodes = sorted((n for n in nodes if n['environment'] == 0), key=lambda n: n['node_id'])
tets = sorted((t for t in tets if t['environment'] == 0), key=lambda t: t['tet_id'])
f32 = layers.f32
mixed_path = M / 'run-002/mixed_reference_arrays.npz'
assert sha(mixed_path) == '107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad'
mixed = np.load(mixed_path, allow_pickle=False)
rotation = np.stack([geometry.rotation_from_xyzw(f32(t['material_frame_quaternion_xyzw']))
                     if t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3] == 1 else np.eye(3)
                     for t in tets])
inverse = np.stack([f32(t['inverse_rest_rows_per_m']) for t in tets])
volume = f32([t['rest_volume_m3'] for t in tets])
ids = np.asarray([t['nodes'] for t in tets], dtype=int)
direction = np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])
dt = float(f32(header['dt_seconds']))
mass = f32([n['center']['mass_kg'] for n in nodes])
dv = np.stack([f32(n['correction_delta_v_m_s']) for n in nodes])
free = ~np.isin(f32([n['center']['constraint_tag'] for n in nodes]), [1, 3])
assert len(tets) == 1920 and len(nodes) == 882
assert dt > 0 and np.isfinite(mass).all() and np.all(mass >= 0)
Ct = mixed['center_tangent']
assert Ct.shape == (1920, 9, 9) and np.isfinite(Ct).all()
original_path = B / 'analysis/bilinear-capture-layer-audit-source166-003/validation/regression469-001/arrays.npz'
original = np.load(original_path, allow_pickle=False)
original_C = original['reference_C']
original_finite = np.isfinite(original_C).all(axis=(1, 2))
assert original_finite.sum() == 1916
assert np.array_equal(Ct[original_finite], original_C[original_finite]), 'Successful source166003 center rows were changed by fallback'
local = np.einsum('tij,tj->ti', Ct, direction.reshape(1920, 9)).reshape(1920, 3, 3)
world = local @ np.swapaxes(rotation, -1, -2)
element = layers.forces(world, inverse, volume)
nodal = layers.gather(element, ids, len(nodes))
mass_action = mass[:, None] * dv
stiffness_action = -dt * dt * nodal
total = mass_action + stiffness_action
native = np.stack([f32(n['production_A_delta_v_impulse_Ns'])[:3] for n in nodes])
cached = np.stack([f32(t['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9, 9) for t in tets])
def compare(actual, expected, gate=None):
    return layers.comparison(actual, expected, gate)[0]
report = {
    'schema': 'numi.cardboard.native469-mixed-reference-assembly.v1',
    'classification': 'Supplemental mixed FP64/MP reference at captured center; not all-FP64 projection or native finite-difference qualification.',
    'source_frozen_sha256': sha(B / 'snapshots/source-right-grip-telemetry-157/frozen-source.json'),
    'source166003_successful_center_rows_bitwise_preserved': int(original_finite.sum()),
    'superseded_invalid_input_report': 'native469-mixed-reference-assembly-002',
    'mp_manifest_sha256': sha(manifest), 'mp_arrays_sha256': sha(mixed_path),
    'binding': bindings, 'free_nodes': int(free.sum()), 'total_nodes': len(nodes),
    'tets': len(tets), 'dt_s': dt,
    'center_tangent_columns': compare(np.swapaxes(cached, -1, -2).reshape(-1, 9), np.swapaxes(Ct, -1, -2).reshape(-1, 9), 1e-3),
    'center_local_action': compare(np.stack([f32(t['center']['tangent_local_Pa_per_s']) for t in tets]), local, 1e-3),
    'center_frame_action': compare(np.stack([f32(t['center']['tangent_world_Pa_per_s']) for t in tets]), world, 1e-3),
    'center_element_action': compare(np.stack([f32(t['center']['operator_force_node_world_N_per_s']) for t in tets]), element, 1e-3),
    'center_assembled_action': compare(native[free][None], total[free][None], .01),
    'separated_actions': {'mass_l2_Ns': float(np.linalg.norm(mass_action[free])),
                          'stiffness_l2_Ns': float(np.linalg.norm(stiffness_action[free])),
                          'total_l2_Ns': float(np.linalg.norm(total[free])),
                          'sign_convention': 'K=-df/dx; A=M+dt^2*K; delta_x=dt*epsilon*delta_v'},
    'state_stress': [],
    'qualification': {'all_fp64_projection': False, 'native_projection': False,
                      'native_branch_and_noise_floor': False, 'full_solver': False,
                      'numerical_convergence': False, 'physical_validation': False},
    'limitations': ['Four center and thirteen side reference projections use separately identified MP roots; original FP64 failures remain failures.',
                   'Relative threshold comparisons are descriptive where a measured native absolute noise floor is unavailable.',
                   'The earlier perturbed liner mismatch was a wrong-grade tool error in004. Native branch/noise-floor and complete projection gates remain open.'],
}
for i, label in enumerate(['center', 's1', 's2', 's3', 's4']):
    states = [t['center'] if i == 0 else t['sides'][i - 1] for t in tets]
    reference = mixed[label + '_stress']
    assert reference.shape == (1920, 3, 3) and np.isfinite(reference).all()
    native_stress = np.stack([f32(s['P_local_Pa']) for s in states])
    error = np.linalg.norm((native_stress - reference).reshape(1920, 9), axis=1)
    scale = np.linalg.norm(reference.reshape(1920, 9), axis=1)
    resolved = scale > 0
    relative = np.divide(error, scale, out=np.full(1920, np.nan), where=resolved)
    worst = int(np.nanargmax(relative))
    report['state_stress'].append({'state': label, 'comparison': compare(native_stress, reference, 1e-3),
                                   'worst_tet': worst, 'worst_absolute_l2_Pa': float(error[worst]),
                                   'worst_relative_l2': float(relative[worst]),
                                   'tets_above_1e_minus3_relative': [int(j) for j in np.flatnonzero(relative > 1e-3)]})
np.savez_compressed(O / 'arrays.npz', local=local, world=world, element=element,
                    mass_action=mass_action, stiffness_action=stiffness_action,
                    assembled_action=total, native_assembled_action=native, free_mask=free)
report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
report, nonfinite = layers.finite_report(report)
assert nonfinite == [], nonfinite
(O / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(O / 'analyze.py').write_bytes(Path(__file__).read_bytes())
files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(O.iterdir()) if p.is_file()}
(O / 'manifest.json').write_text(json.dumps({'schema': 'numi.cardboard.mixed-assembly-files.v1', 'files': files,
    'input_files': {str(p): sha(p) for p in [manifest, mixed_path, original_path, capture, R / 'cardboard_bilinear_capture_layers.py',
                                           Path(layers.ref.__file__), Path(geometry.__file__)]}}, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['center_tangent_columns', 'center_element_action', 'center_assembled_action', 'state_stress']}))
