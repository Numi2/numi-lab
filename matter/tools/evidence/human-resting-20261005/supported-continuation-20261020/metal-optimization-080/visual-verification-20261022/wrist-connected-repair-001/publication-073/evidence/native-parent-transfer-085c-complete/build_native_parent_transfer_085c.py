#!/usr/bin/env python3
"""Full candidate-to-native reference overlap and mass ledger using exact 085b clip."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, resource, sys, time
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
RUN = ROOT / 'native-parent-transfer-085c' / 'run-001'
CAND = ROOT / 'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
GATES = ROOT / 'connectivity-001/run-001/root-canonical-cell-gates.json'
HUMAN = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
EXPORT = HUMAN / 'connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight'
NODES = EXPORT / 'compiled-skin-fem-nodes.csv'
TETS = EXPORT / 'compiled-skin-fem-tets.csv'
CONTRIB = EXPORT / 'premerge-layer-contributions.csv'
OWNER073 = HUMAN / 'connected-skin-parent-mass-transfer-067/native-parent-transfer-073/build_native_parent_transfer_073.py'
STACKED = ROOT / 'native-parent-transfer-085b/plane_filter_085b.py'
HALFSPACE = ROOT / 'native-parent-transfer-085b/plane_filter_085.py'
EXPECTED = {
    CAND: '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    GATES: 'ced7f2ec3c45bfa1ece35b230f1bded46c9cadc39297ac8b8854a549cad61dd3',
    NODES: '26c607f9835241ef5d274169e4ac2aeae9c0f6ab7ad94a06b307ca25844862c9',
    TETS: 'a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461'
          ,
    CONTRIB: '75bd67cc6f7aeab9dccbb73da29220fcd35896b217042e8ce9c8181d89ca7c6f',
    OWNER073: '167c176edef02cbb2360d68e3c011809722030bd00009e20afa80d2b8ad59586',
    STACKED: 'ae3cbe1e9f79a580b3a0f22b5126a2d91f11293874f9c867f46a83e81d6f3f61',
    HALFSPACE: '0ba4109b943f7e1b4312ca339f8e5f2f931b81413daca4074fa2be42f3b9c305',
}
AABB_TOL = 2e-10
INSIDE_TOL = 2e-10
UNIQUE_TOL = 2e-10
CHILD_TIMEOUT_S = 600.0
OUTER_TIMEOUT_S = 720.0
MATERIAL_MAP = {1: 15, 2: 16}
DENSITY_BOUNDS = {15: (900., 1100.), 16: (850., 950.)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def volume6(points):
    return abs(float(np.linalg.det(points[1:] - points[0]))) / 6.0


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def main():
    outer0 = time.monotonic()
    if RUN.exists():
        raise RuntimeError(f'refusing to overwrite existing run directory: {RUN}')
    for p, expected in EXPECTED.items():
        actual = sha(p)
        if '?' in expected or actual != expected:
            raise RuntimeError(f'input pin mismatch {p}: expected={expected} actual={actual}')
    if time.monotonic() - outer0 > OUTER_TIMEOUT_S:
        raise TimeoutError('outer 12-minute bound elapsed during pin preflight')

    # Import only the frozen unculled 085b stacked owner. No plane/separator filter is used.
    sys.path.insert(0, str(STACKED.parent))
    from plane_filter_085b import tetra_intersection_volume_stacked

    RUN.mkdir(parents=True)
    progress = RUN / 'clip-progress.jsonl'
    manifest = {
        'status': 'PREFLIGHT_PASS_CLIPPING_NOT_STARTED',
        'candidate': str(CAND), 'candidate_sha256': sha(CAND),
        'candidate_gate_sha256': sha(GATES),
        'owner_073_sha256': sha(OWNER073), 'stacked_085b_sha256': sha(STACKED),
        'halfspace_085_sha256': sha(HALFSPACE),
        'input_sha256': {p.name: sha(p) for p in (NODES, TETS, CONTRIB)},
        'clip': {'aabb_tolerance_m': AABB_TOL, 'inside_tolerance_m': INSIDE_TOL,
                 'unique_tolerance_m': UNIQUE_TOL,
                 'method': 'all AABB-overlapping pairs, unculled tetra_intersection_volume_stacked from pinned 085b'},
        'limits': {'clip_seconds': CHILD_TIMEOUT_S, 'outer_seconds': OUTER_TIMEOUT_S},
    }
    write_json(RUN / 'preflight.json', manifest)

    cand = np.load(CAND, allow_pickle=False)
    required = {'reference_points_m', 'current037_points_m_f32', 'tetrahedra_zero_based',
                'material_ids', 'cell_uids', 'source_global_node_ids'}
    missing = required - set(cand.files)
    if missing:
        raise RuntimeError(f'candidate missing required arrays {sorted(missing)}')
    points = np.asarray(cand['reference_points_m'], dtype=np.float64)
    current = np.asarray(cand['current037_points_m_f32'])
    new_t = np.asarray(cand['tetrahedra_zero_based'], dtype=np.int64)
    new_mat = np.asarray(cand['material_ids'], dtype=np.int64)
    cell_uids = np.asarray(cand['cell_uids'], dtype=np.int64)
    source_global = np.asarray(cand['source_global_node_ids'], dtype=np.int64)
    if points.shape != (7360, 3) or points.dtype != np.float64 or not np.all(np.isfinite(points)):
        raise RuntimeError(f'candidate reference points invalid: shape={points.shape} dtype={points.dtype}')
    if current.shape != points.shape or current.dtype != np.float32 or not np.all(np.isfinite(current)):
        raise RuntimeError('candidate emitted current F32 array invalid')
    if new_t.shape != (32796, 4) or np.min(new_t) < 0 or np.max(new_t) >= len(points):
        raise RuntimeError('candidate tetrahedra malformed/out of range')
    if new_mat.shape != (len(new_t),) or cell_uids.shape != (len(new_t),):
        raise RuntimeError('candidate cell metadata row counts mismatch')
    if len(np.unique(cell_uids)) != len(cell_uids) or not set(new_mat.tolist()) <= set(MATERIAL_MAP):
        raise RuntimeError('candidate cell ids/material values invalid')

    node_rows = csv_rows(NODES)
    tet_rows = [r for r in csv_rows(TETS) if int(r['identity_active_bit']) == 1]
    contrib = csv_rows(CONTRIB)
    if (len(node_rows), len(tet_rows), len(contrib)) != (7233, 28632, 9644):
        raise RuntimeError(f'frozen native dimensions changed: nodes={len(node_rows)} tets={len(tet_rows)} source={len(contrib)}')
    native_global = np.asarray([int(r['global_node']) for r in node_rows], dtype=np.int64)
    native_local = np.asarray([int(r['object_local_node']) for r in node_rows], dtype=np.int64)
    native_rest = np.asarray([[float(r[f'rest_{a}']) for a in 'xyz'] for r in node_rows], dtype=np.float64)
    if not np.array_equal(native_local, np.arange(len(node_rows))):
        raise RuntimeError('native object-local node rows are not dense ordered rows')
    if not np.array_equal(source_global, native_global):
        raise RuntimeError('candidate source global-node prefix identity differs from native node export')
    if not np.array_equal(points[:len(native_rest)], native_rest):
        raise RuntimeError('candidate reference prefix is not bitwise the native F32-promoted-F64 domain')

    global_to_local = {int(g): int(i) for i, g in enumerate(native_global)}
    old_global = np.asarray([[int(r[f'global_node{j}']) for j in range(4)] for r in tet_rows], dtype=np.int64)
    old_t = np.asarray([[int(r[f'source_node{j}']) for j in range(4)] for r in tet_rows], dtype=np.int64)
    old_t_from_global = np.vectorize(global_to_local.__getitem__, otypes=[np.int64])(old_global)
    if not np.array_equal(old_t, old_t_from_global):
        raise RuntimeError('compiled source_node rows do not match global-node to object-local identity')
    if old_t.min() < 0 or old_t.max() >= len(native_rest):
        raise RuntimeError('native tet object-local node row outside node domain')
    old_mat = np.asarray([int(r['material_index']) for r in tet_rows], dtype=np.int64)
    old_source_row = np.asarray([int(r['source_tet_row']) for r in tet_rows], dtype=np.int64)
    old_compiled_volume = np.asarray([float(r['inverse_rest_w_m3']) for r in tet_rows], dtype=np.float64)
    old_density = np.asarray([float(r['compiled_bulk_x']) for r in tet_rows], dtype=np.float64)
    old_corner_mass = np.asarray([float(r['bulk_density_corner_mass_kg']) for r in tet_rows], dtype=np.float64)
    if not np.array_equal(old_source_row, np.arange(len(tet_rows))):
        raise RuntimeError('native source_tet_row is not dense row identity')
    old_cell_mass = old_compiled_volume * old_density
    if not np.array_equal(old_corner_mass, old_cell_mass / 4.0):
        raise RuntimeError('compiled per-corner mass identity does not exactly equal parent mass/4')
    old_xyz = native_rest[old_t]
    old_geom_volume = np.asarray([volume6(x) for x in old_xyz], dtype=np.float64)
    if np.any(old_geom_volume <= 0) or np.any(~np.isfinite(old_geom_volume)):
        raise RuntimeError('native reference tet geometry contains nonpositive/nonfinite volume')

    child_xyz = points[new_t]
    child_geom_volume = np.asarray([volume6(x) for x in child_xyz], dtype=np.float64)
    if np.any(child_geom_volume <= 0) or np.any(~np.isfinite(child_geom_volume)):
        raise RuntimeError('candidate reference tet geometry contains nonpositive/nonfinite volume')
    old_min, old_max = old_xyz.min(axis=1), old_xyz.max(axis=1)
    max_x_order = np.argsort(old_max[:, 0], kind='mergesort')
    sorted_old_max_x = old_max[max_x_order, 0]

    overlap_rows = [None] * len(new_t)
    aabb_pairs = 0
    positive_pairs = 0
    material_cross_pairs = 0
    tclip = time.monotonic()
    with progress.open('w') as pf:
        for ci, vnew in enumerate(child_xyz):
            if (ci & 255) == 0:
                elapsed = time.monotonic() - tclip
                if elapsed > CHILD_TIMEOUT_S:
                    payload = {'status': 'FAILED_CLIP_TIMEOUT', 'processed_children': ci,
                               'candidate_children': len(new_t), 'aabb_pairs': aabb_pairs,
                               'positive_overlap_pairs': positive_pairs,
                               'clip_elapsed_seconds': elapsed,
                               'remaining_estimate_seconds': (elapsed * (len(new_t)-ci) / max(ci, 1))}
                    write_json(RUN / 'timeout.json', payload)
                    raise TimeoutError('085c exact clipping exceeded 600 second budget')
            lo = vnew.min(axis=0)
            hi = vnew.max(axis=0)
            first = int(np.searchsorted(sorted_old_max_x, lo[0] - AABB_TOL, side='left'))
            ids = max_x_order[first:]
            if len(ids):
                keep = (old_min[ids, 0] <= hi[0] + AABB_TOL)
                keep &= np.all(old_max[ids, 1:] >= lo[None, 1:] - AABB_TOL, axis=1)
                keep &= np.all(old_min[ids, 1:] <= hi[None, 1:] + AABB_TOL, axis=1)
                ids = ids[keep]
            hits = []
            for oi in ids:
                aabb_pairs += 1
                ov = tetra_intersection_volume_stacked(vnew, old_xyz[int(oi)], INSIDE_TOL, UNIQUE_TOL)
                threshold = max(min(child_geom_volume[ci], old_geom_volume[int(oi)]) * 1e-13, 1e-25)
                if ov > threshold:
                    hits.append((int(oi), float(ov)))
                    positive_pairs += 1
                    if MATERIAL_MAP[int(new_mat[ci])] != int(old_mat[int(oi)]):
                        material_cross_pairs += 1
            overlap_rows[ci] = sorted(hits, key=lambda pair: pair[0])
            if (ci + 1) % 1000 == 0 or ci + 1 == len(new_t):
                elapsed = time.monotonic() - tclip
                payload = {'processed_children': ci+1, 'candidate_children': len(new_t),
                           'aabb_pairs': aabb_pairs, 'positive_overlap_pairs': positive_pairs,
                           'material_cross_positive_pairs': material_cross_pairs,
                           'clip_elapsed_seconds': elapsed,
                           'estimated_remaining_seconds': elapsed * (len(new_t)-ci-1)/(ci+1)}
                pf.write(json.dumps(payload, sort_keys=True) + '\n')
                pf.flush()
                print(json.dumps(payload, sort_keys=True), flush=True)
            if time.monotonic() - outer0 > OUTER_TIMEOUT_S:
                raise TimeoutError('085c exceeded 720 second outer budget during exact clipping')
    clip_elapsed = time.monotonic() - tclip

    if any(row is None for row in overlap_rows):
        raise RuntimeError('clip did not produce all child rows')
    child_coverage = np.zeros(len(new_t), dtype=np.float64)
    parent_covered = np.zeros(len(tet_rows), dtype=np.float64)
    cross_material = []
    for ci, row in enumerate(overlap_rows):
        if not row:
            raise RuntimeError(f'candidate child has no native parent overlap: child={ci}, uid={int(cell_uids[ci])}')
        child_coverage[ci] = sum(v for _, v in row)
        cm = MATERIAL_MAP[int(new_mat[ci])]
        for oi, ov in row:
            if old_mat[oi] != cm:
                cross_material.append((ci, oi, ov, cm, int(old_mat[oi])))
            parent_covered[oi] += ov
    child_cov_rel = (child_coverage - child_geom_volume) / child_geom_volume
    parent_cov_rel = (parent_covered - old_geom_volume) / old_geom_volume
    if material_cross_pairs or cross_material:
        raise RuntimeError(f'positive overlap crosses materials: count={len(cross_material)}')
    if np.max(np.abs(child_cov_rel)) > 1e-7:
        ix = int(np.argmax(np.abs(child_cov_rel)))
        raise RuntimeError(f'child reference-volume coverage failed at {ix}: rel={child_cov_rel[ix]}')
    if np.max(np.abs(parent_cov_rel)) > 1e-7:
        ix = int(np.argmax(np.abs(parent_cov_rel)))
        raise RuntimeError(f'native parent reference-volume coverage failed at {ix}: rel={parent_cov_rel[ix]}')

    offsets = [0]
    parent_rows, overlap_volumes = [], []
    child_parent_fraction, parent_mass_fraction = [], []
    for ci, row in enumerate(overlap_rows):
        for oi, ov in row:
            parent_rows.append(oi)
            overlap_volumes.append(ov)
            child_parent_fraction.append(ov / child_geom_volume[ci])
            parent_mass_fraction.append(ov / old_geom_volume[oi])
        offsets.append(len(parent_rows))
    parent_rows = np.asarray(parent_rows, dtype=np.int32)
    overlap_volumes = np.asarray(overlap_volumes, dtype=np.float64)
    child_parent_fraction = np.asarray(child_parent_fraction, dtype=np.float64)
    parent_mass_fraction = np.asarray(parent_mass_fraction, dtype=np.float64)
    child_mass = np.zeros(len(new_t), dtype=np.float64)
    for ci in range(len(new_t)):
        a, z = offsets[ci], offsets[ci+1]
        child_mass[ci] = np.dot(old_cell_mass[parent_rows[a:z]], parent_mass_fraction[a:z])
    child_density = child_mass / child_geom_volume
    native_mat_for_child = np.asarray([MATERIAL_MAP[int(x)] for x in new_mat], dtype=np.int16)
    density_violations = []
    density_summary = {}
    for m, (dlo, dhi) in DENSITY_BOUNDS.items():
        ix = np.flatnonzero(native_mat_for_child == m)
        bad = ix[(child_density[ix] < dlo) | (child_density[ix] > dhi)]
        density_violations.extend((int(i), int(m), float(child_density[i])) for i in bad)
        density_summary[str(m)] = {'min_kg_m3': float(child_density[ix].min()),
                                   'max_kg_m3': float(child_density[ix].max()),
                                   'child_rows': int(len(ix)), 'bounds_kg_m3': [dlo, dhi]}
    parent_mass_received = np.zeros(len(tet_rows), dtype=np.float64)
    np.add.at(parent_mass_received, parent_rows, old_cell_mass[parent_rows] * parent_mass_fraction)
    parent_mass_rel = (parent_mass_received - old_cell_mass) / old_cell_mass
    material_mass = {}
    for m in (15, 16):
        oi = old_mat == m
        ci = native_mat_for_child == m
        m0, m1 = float(old_cell_mass[oi].sum()), float(child_mass[ci].sum())
        material_mass[str(m)] = {'native_parent_mass_kg': m0, 'child_mass_kg': m1,
                                 'relative_delta': (m1 - m0) / m0}
    if density_violations:
        raise RuntimeError(f'child effective-density bounds failed: count={len(density_violations)} first={density_violations[:5]}')
    if np.max(np.abs(parent_mass_rel)) > 1e-7:
        ix = int(np.argmax(np.abs(parent_mass_rel)))
        raise RuntimeError(f'native parent mass closure failed at {ix}: rel={parent_mass_rel[ix]}')

    # Preserve exact cooked-parent corner provenance from the native source-only contribution export.
    source_by_key = defaultdict(list)
    for ri, row in enumerate(contrib):
        key = (int(row['combined_source_global_node_index']), int(row['source_material_index']))
        source_by_key[key].append(ri)
    if any(len(v) != 1 for v in source_by_key.values()):
        raise RuntimeError('native source contribution key is not unique; donor corner provenance is ambiguous')
    parent_corner_source_row = np.empty((len(tet_rows), 4), dtype=np.int32)
    parent_corner_body = np.empty((len(tet_rows), 4), dtype=np.int32)
    parent_corner_local = np.empty((len(tet_rows), 4, 3), dtype=np.float64)
    for oi, row in enumerate(tet_rows):
        mat = int(row['material_index'])
        for j in range(4):
            key = (int(row[f'global_node{j}']), mat)
            if key not in source_by_key:
                raise RuntimeError(f'missing source donor row for parent corner {oi}/{j}/{key}')
            ri = source_by_key[key][0]
            source = contrib[ri]
            parent_corner_source_row[oi, j] = ri
            parent_corner_body[oi, j] = int(source['node_frame_owner_body'])
            parent_corner_local[oi, j] = [float(source[f'donor_local_{a}_m']) for a in 'xyz']

    map_path = RUN / 'native-parent-transfer-085c.npz'
    np.savez_compressed(map_path,
        candidate_cell_uids=cell_uids,
        candidate_child_material_ids=new_mat.astype(np.int8),
        candidate_child_native_material_ids=native_mat_for_child,
        candidate_child_reference_volume_m3=child_geom_volume,
        child_native_parent_indptr=np.asarray(offsets, dtype=np.int64),
        child_native_parent_rows=parent_rows,
        overlap_native_reference_volume_m3=overlap_volumes,
        child_parent_fraction_overlap_over_child_volume=child_parent_fraction,
        parent_mass_fraction_overlap_over_native_parent_geometry_volume=parent_mass_fraction,
        child_transferred_mass_kg=child_mass,
        child_effective_reference_density_kg_m3=child_density,
        native_parent_source_tet_rows=old_source_row.astype(np.int32),
        native_parent_material_ids=old_mat.astype(np.int16),
        native_parent_compiled_reference_volume_m3=old_compiled_volume,
        native_parent_native_f32_geometry_volume_m3=old_geom_volume,
        native_parent_compiled_density_bulk_x=old_density,
        native_parent_compiled_mass_kg=old_cell_mass,
        native_parent_corner_source_contribution_row_ids=parent_corner_source_row,
        native_parent_corner_frame_owner_body_ids=parent_corner_body,
        native_parent_corner_donor_local_points_m=parent_corner_local,
        native_parent_corner_lumped_cell_mass_kg=np.repeat((old_cell_mass/4.)[:,None],4,axis=1),
    )
    final = {
        'status': 'PASS_REFERENCE_DOMAIN_OVERLAP_AND_MASS_LEDGER',
        'admission': {'physical_mass_transfer': False, 'current_geometry': False,
                      'native_emission': False, 'native_physics': False, 'bone_clearance': False},
        'method': 'Every candidate child was tested against every native parent whose AABB overlaps under the pinned 073 tolerance; all overlap volumes use the unculled stacked 085b exact clip. No identity shortcut, approximate rejection, masked renormalization, or global mass rescale.',
        'candidate': {'path': str(CAND), 'sha256': sha(CAND), 'nodes': int(len(points)),
                      'children': int(len(new_t)), 'cell_uid_sha256': hashlib.sha256(cell_uids.tobytes()).hexdigest()},
        'input_sha256': {str(p): sha(p) for p in (CAND, GATES, NODES, TETS, CONTRIB, OWNER073, HALFSPACE, STACKED)},
        'native_domain': {'node_rows': len(node_rows), 'active_parent_rows': len(tet_rows),
                          'source_tet_row_domain': 'dense 0..28631',
                          'candidate_source_global_node_prefix_exact': True,
                          'candidate_reference_prefix_bitwise_native_rest': True,
                          'native_reference_points': 'compiled rest_x/y/z F32 decimal values promoted to float64',
                          'candidate_reference_points': 'candidate reference_points_m float64, exact original 7233-row prefix verified'},
        'coverage': {'aabb_tolerance_m': AABB_TOL, 'inside_tolerance_m': INSIDE_TOL,
                     'unique_tolerance_m': UNIQUE_TOL, 'aabb_pairs_tested': int(aabb_pairs),
                     'positive_overlap_pairs': int(len(parent_rows)),
                     'candidate_child_max_abs_relative_volume_error': float(np.max(np.abs(child_cov_rel))),
                     'native_parent_max_abs_relative_volume_error': float(np.max(np.abs(parent_cov_rel))),
                     'positive_cross_material_pairs': int(len(cross_material)),
                     'all_candidate_children_covered': bool(np.max(np.abs(child_cov_rel)) <= 1e-7),
                     'all_native_parents_covered': bool(np.max(np.abs(parent_cov_rel)) <= 1e-7),
                     'cross_material_overlap_count': len(cross_material)},
        'mass': {'native_cell_mass_formula': 'inverse_rest_w_m3 * compiled_bulk_x',
                 'native_total_kg': float(old_cell_mass.sum()), 'candidate_total_kg': float(child_mass.sum()),
                 'relative_total_delta': float((child_mass.sum()-old_cell_mass.sum())/old_cell_mass.sum()),
                 'per_material': material_mass,
                 'max_parent_mass_relative_closure_error': float(np.max(np.abs(parent_mass_rel))),
                 'candidate_density_bounds_kg_m3': DENSITY_BOUNDS,
                 'candidate_effective_density_by_material': density_summary,
                 'density_violation_count': len(density_violations),
                 'native_compiled_corner_mass_exactly_parent_mass_div4': True},
        'timing': {'clip_seconds': clip_elapsed, 'outer_seconds': time.monotonic()-outer0,
                   'maximum_resident_set_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        'limitations': ['Reference-domain geometric overlap and mass transfer only. This does not certify current-emitted cells, interface/contact geometry, native app emission, or physical simulation.',
                        'Compiled cooked parent cell masses are authoritative; raw source contribution masses are retained only as donor/frame provenance and are reconciled by the separate 074-style moment audit.']}
    write_json(RUN / 'transfer-report.json', final)
    write_json(RUN / 'run-manifest.json', {'candidate_sha256': sha(CAND), 'map_sha256': sha(map_path),
              'report_sha256': sha(RUN/'transfer-report.json'), 'builder_sha256': sha(Path(__file__)),
              'status': final['status']})
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Preserve a clear failure receipt. Never write a candidate map on failed gates.
        run = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/native-parent-transfer-085c/run-001')
        if run.exists():
            write_json(run / 'failure.json', {'status': 'FAILED', 'exception_type': type(exc).__name__,
                                               'message': str(exc), 'time_unix': time.time()})
        raise
