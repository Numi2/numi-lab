from __future__ import annotations
import argparse, hashlib, importlib.util, json, struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot import owner module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def materialize(*, base_path: Path, base_sha256: str,
                displacement_path: Path, displacement_sha256: str,
                ancestry_path: Path, ancestry_sha256: str,
                report_path: Path, report_sha256: str,
                rest_pose_path: Path, rest_pose_sha256: str,
                rebind_owner_path: Path, rebind_owner_sha256: str,
                emission_owner_path: Path, emission_owner_sha256: str,
                fraction: float, output_path: Path) -> dict:
    pinned = {
        'base': (base_path, base_sha256),
        'displacement': (displacement_path, displacement_sha256),
        'ancestry': (ancestry_path, ancestry_sha256),
        'line_search_report': (report_path, report_sha256),
        'rest_pose': (rest_pose_path, rest_pose_sha256),
        'rebind_owner': (rebind_owner_path, rebind_owner_sha256),
        'emission_owner': (emission_owner_path, emission_owner_sha256),
    }
    for label, (path, expected) in pinned.items():
        if not path.is_file():
            raise ValueError(f'missing {label}: {path}')
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'{label} hash mismatch: {actual} != {expected}')
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('fraction must be in [0,1]')
    report = json.loads(report_path.read_text())
    expected_output = report['fractions'][str(fraction)]['candidate_sha256']
    rb = load_module('pinned_scalar_fraction_rebind', rebind_owner_path)
    geom = load_module('pinned_scalar_fraction_emission', emission_owner_path)
    base = rb.decode(base_path.read_bytes())
    displaced = rb.decode(displacement_path.read_bytes())
    n = int(base['nv'])
    faces = np.asarray(displaced['inds'], dtype=np.int64).reshape(-1, 3)
    with np.load(ancestry_path, allow_pickle=False) as z:
        patch_faces = np.asarray(z['patch_source_face_rows'], dtype=np.int64)
        clone_sources = np.asarray(z['cloned_source_vertex_ids'], dtype=np.int64)
        delta_mm = np.asarray(z['displacement_mm'], dtype=np.float64)
        boundary_edges = np.asarray(z['fixed_outer_boundary_edges'], dtype=np.int64)
        parent_offsets = np.asarray(z['face_source_parent_offsets'], dtype=np.uint32)
        parent_rows = np.asarray(z['face_source_parent_rows'], dtype=np.uint32)
    if len(clone_sources) != len(delta_mm) or int(displaced['nv']) != n + len(clone_sources):
        raise ValueError('clone/displacement row ancestry mismatch')
    base_face_count = len(np.asarray(base['inds']).reshape(-1, 3))
    if len(patch_faces) == 0 or patch_faces.min() < 0 or patch_faces.max() >= base_face_count:
        raise ValueError('patch source face rows are invalid')
    kept_faces = np.setdiff1d(np.arange(base_face_count, dtype=np.int64), patch_faces)
    pose = json.loads(rest_pose_path.read_text())
    pose_rest = {'source_poses': [{'core_body': int(row['body_index']),
        'position': row['rest']['position_world_m'],
        'quaternion_xyzw': row['rest']['orientation_world_xyzw']}
        for row in pose.get('all_bodies', pose.get('bodies', []))]}
    A, b = rb.reference_transforms(base, pose_rest, [0, 0, 0])
    base_world = rb.positions(base, A, b)
    candidate_world = np.vstack((base_world, base_world[clone_sources] + fraction * delta_mm / 1000.0))
    offsets = np.arange(len(candidate_world) + 1, dtype=np.int64)
    ancestry_ids = np.r_[np.arange(n, dtype=np.int64), clone_sources]
    ancestry_weights = np.ones(len(candidate_world), dtype=np.float64)
    blob, _ = rb.append_registered_surface_patch(base, pose_rest, pose_rest, [0, 0, 0],
        candidate_world, faces, offsets, ancestry_ids, ancestry_weights,
        maximum_error_m=2e-6, protected_source_face_ids=kept_faces)
    candidate = rb.decode(blob)
    pose['vertex_count'] = int(candidate['nv'])
    normal_rows = np.asarray(sorted(set(map(int, boundary_edges.ravel())) |
        set(range(n, n + len(clone_sources)))), dtype=np.int64)
    normals = rb.registered_rest_vertex_normals_float32(candidate, pose, normal_rows)
    emitted = bytearray(blob)
    for row, normal in zip(normal_rows, normals):
        struct.pack_into('<3f', emitted, int(candidate['vertoff']) + int(row) * 56 + 12,
                         *map(float, normal))
    blob = bytes(emitted)
    actual_output = hashlib.sha256(blob).hexdigest()
    if actual_output != expected_output:
        raise ValueError(f'candidate differs from audited payload: {actual_output} != {expected_output}')
    # Recheck the repaired patch's canonical Float32 emission before publishing bytes.
    reference = rb.positions(candidate, *rb.reference_transforms(candidate, pose_rest, [0, 0, 0]))
    positions_f32 = rb.registered_positions_float32(candidate, pose, frame='rest')
    geom.validate_float32_surface_emission(reference, positions_f32, faces[patch_faces])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
    return {'fraction': fraction, 'candidate_path': str(output_path),
        'candidate_sha256': actual_output, 'expected_audited_sha256': expected_output,
        'exact_audited_payload_match': True, 'float32_rest_emission': 'pass',
        'base_vertex_prefix_count': n, 'clone_count': len(clone_sources),
        'replacement_face_count': len(patch_faces), 'fixed_boundary_edge_count': len(boundary_edges),
        'face_parent_csr_sha256': hashlib.sha256(parent_offsets.astype('<u4').tobytes() + parent_rows.astype('<u4').tobytes()).hexdigest(),
        'native_or_mass_admission': False}

def main() -> None:
    p = argparse.ArgumentParser(description='Re-emit a pinned scalar fraction of a connected NHSKIN patch and verify the audited payload hash.')
    for arg in ('base', 'displacement', 'ancestry', 'report', 'rest-pose', 'rebind-owner', 'emission-owner'):
        p.add_argument('--' + arg, type=Path, required=True)
        p.add_argument('--' + arg + '-sha256', required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = materialize(
        base_path=a.base, base_sha256=a.base_sha256,
        displacement_path=a.displacement, displacement_sha256=a.displacement_sha256,
        ancestry_path=a.ancestry, ancestry_sha256=a.ancestry_sha256,
        report_path=a.report, report_sha256=a.report_sha256,
        rest_pose_path=a.rest_pose, rest_pose_sha256=a.rest_pose_sha256,
        rebind_owner_path=a.rebind_owner, rebind_owner_sha256=a.rebind_owner_sha256,
        emission_owner_path=a.emission_owner, emission_owner_sha256=a.emission_owner_sha256,
        fraction=a.fraction, output_path=a.output)
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
