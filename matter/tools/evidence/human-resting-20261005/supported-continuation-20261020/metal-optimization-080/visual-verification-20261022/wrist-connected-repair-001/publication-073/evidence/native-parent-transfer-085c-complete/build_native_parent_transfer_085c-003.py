#!/usr/bin/env python3
"""Full candidate-to-native reference overlap and mass ledger using exact 085b clip."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, resource, sys, time, itertools
from fractions import Fraction
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
RUN = ROOT / 'native-parent-transfer-085c' / 'run-003'
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


def exact_binary64_intersection_rank(a_points, b_points):
    """Return exact affine rank of two binary64 tetrahedra's closed intersection.

    Input floats are interpreted exactly via Fraction.from_float. A full-dimensional
    convex intersection has rank 3; face/edge/point contacts have ranks 2/1/0;
    disjoint tetrahedra return rank -1. This is used only after the unchanged
    tolerance clip reports a positive opposite-material pair.
    """
    def to_fraction_tet(points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.shape != (4, 3) or not np.all(np.isfinite(arr)):
            raise ValueError('exact intersection requires finite (4,3) tetrahedra')
        return [tuple(Fraction.from_float(float(x)) for x in row) for row in arr]

    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    def dot(a, b):
        return sum((a[i]*b[i] for i in range(3)), Fraction(0))

    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

    def solve3(m, rhs):
        det = det3(m)
        if det == 0:
            return None
        out = []
        for col in range(3):
            replaced = [list(row) for row in m]
            for row in range(3):
                replaced[row][col] = rhs[row]
            out.append(det3(replaced) / det)
        return tuple(out)

    def tetra_planes(points):
        result = []
        for opposite in range(4):
            face_ids = [i for i in range(4) if i != opposite]
            p0, p1, p2 = (points[i] for i in face_ids)
            n = cross(sub(p1, p0), sub(p2, p0))
            h = dot(n, p0)
            side = dot(n, points[opposite]) - h
            if side == 0:
                raise ValueError('exact intersection input contains a degenerate tetrahedron')
            if side > 0:
                n = tuple(-x for x in n)
                h = -h
            result.append((n, h))  # closed interior halfspace n.x <= h
        return result

    a = to_fraction_tet(a_points)
    b = to_fraction_tet(b_points)
    planes = tetra_planes(a) + tetra_planes(b)
    vertices = set()
    for plane_ids in itertools.combinations(range(8), 3):
        matrix = [list(planes[i][0]) for i in plane_ids]
        rhs = [planes[i][1] for i in plane_ids]
        point = solve3(matrix, rhs)
        if point is None:
            continue
        if all(dot(n, point) <= h for n, h in planes):
            vertices.add(point)
    if not vertices:
        return {'affine_rank': -1, 'vertex_count': 0,
                'classification': 'exactly_disjoint'}

    points = list(vertices)
    origin = points[0]
    vectors = [sub(point, origin) for point in points[1:]]
    first = next((v for v in vectors if any(x != 0 for x in v)), None)
    if first is None:
        rank = 0
    else:
        normal = None
        for vector in vectors:
            candidate = cross(first, vector)
            if any(x != 0 for x in candidate):
                normal = candidate
                break
        if normal is None:
            rank = 1
        elif any(dot(normal, vector) != 0 for vector in vectors):
            rank = 3
        else:
            rank = 2
    classes = {
        0: 'exact_point_contact',
        1: 'exact_edge_contact',
        2: 'exact_face_contact',
        3: 'exact_positive_volume_intersection',
    }
    return {'affine_rank': rank, 'vertex_count': len(vertices),
            'classification': classes[rank]}


def classify_exact_cross_material_pair(a_points, b_points):
    result = exact_binary64_intersection_rank(a_points, b_points)
    rank = result['affine_rank']
    if rank == 3:
        action = 'retain_and_fail_closed_genuine_cross_material_overlap'
    elif rank == -1:
        action = 'omit_tolerance_only_positive_exactly_disjoint_pair'
    else:
        action = 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'
    return {**result, 'transfer_action': action}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def flatten_overlap_rows(rows, count=None):
    use = rows if count is None else rows[:count]
    offsets = [0]
    parents, vols = [], []
    for row in use:
        if row is not None:
            for oi, ov in row:
                parents.append(int(oi)); vols.append(float(ov))
        offsets.append(len(parents))
    return np.asarray(offsets, np.int64), np.asarray(parents, np.int32), np.asarray(vols, np.float64)


def save_partial_overlap(rows, processed, candidate_sha):
    ptr, parent, volume = flatten_overlap_rows(rows, processed)
    target = RUN / 'partial-overlap-checkpoint-085c.npz'
    tmp = RUN / 'partial-overlap-checkpoint-085c.tmp.npz'
    np.savez_compressed(tmp, processed_child_rows=np.asarray(processed, np.int32),
        child_native_parent_indptr=ptr, child_native_parent_rows=parent,
        overlap_native_reference_volume_m3=volume,
        candidate_sha256=np.asarray(candidate_sha),
        clip_tolerance_m=np.asarray([AABB_TOL, INSIDE_TOL, UNIQUE_TOL], np.float64))
    tmp.replace(target)


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
                save_partial_overlap(overlap_rows, ci + 1, sha(CAND))
                print(json.dumps(payload, sort_keys=True), flush=True)
            if time.monotonic() - outer0 > OUTER_TIMEOUT_S:
                raise TimeoutError('085c exceeded 720 second outer budget during exact clipping')
    clip_elapsed = time.monotonic() - tclip

    if any(row is None for row in overlap_rows):
        raise RuntimeError('clip did not produce all child rows')
    tolerance_positive_cross = []
    zero_measure_cross = []
    boundary_contacts = []
    exactly_disjoint_cross = []
    genuine_cross = []
    # The tolerance clip is intentionally unchanged. Only its positive
    # opposite-material results receive an exact binary64 affine-rank check.
    # Rank < 3 has zero geometric measure and is retained as a typed diagnostic,
    # then omitted from transfer CSR. Rank 3 remains a hard failure.
    for ci, row in enumerate(overlap_rows):
        if row is None:
            raise RuntimeError(f'clip left child row unprocessed: child={ci}')
        keep = []
        cm = MATERIAL_MAP[int(new_mat[ci])]
        for oi, ov in row:
            if old_mat[oi] == cm:
                keep.append((oi, ov))
                continue
            exact = classify_exact_cross_material_pair(child_xyz[ci], old_xyz[oi])
            record = {'candidate_child_row': int(ci), 'candidate_cell_uid': int(cell_uids[ci]),
                      'candidate_material_id': int(new_mat[ci]),
                      'candidate_native_material_id': int(cm),
                      'native_parent_row': int(oi),
                      'native_source_tet_row': int(old_source_row[oi]),
                      'native_material_id': int(old_mat[oi]),
                      'tolerance_clip_overlap_volume_m3': float(ov),
                      'candidate_child_geometry_volume_m3': float(child_geom_volume[ci]),
                      'native_parent_geometry_volume_m3': float(old_geom_volume[oi]),
                      'fraction_of_candidate_child': float(ov / child_geom_volume[ci]),
                      'fraction_of_native_parent': float(ov / old_geom_volume[oi]),
                      'exact_binary64_intersection': exact}
            tolerance_positive_cross.append(record)
            record['transfer_action'] = exact['transfer_action']
            if exact['affine_rank'] < 3:
                zero_measure_cross.append(record)
                if exact['affine_rank'] == -1:
                    exactly_disjoint_cross.append(record)
                else:
                    boundary_contacts.append(record)
            else:
                genuine_cross.append(record)
                keep.append((oi, ov))
        overlap_rows[ci] = keep

    write_json(RUN / 'cross-material-exact-rank-classification-085c.json', {
        'status': 'CLASSIFIED_AFTER_UNCHANGED_TOLERANCE_CLIP',
        'inside_tolerance_m': INSIDE_TOL, 'unique_tolerance_m': UNIQUE_TOL,
        'positive_cutoff': 'max(min(Vchild,Vnative)*1e-13,1e-25)',
        'exact_coordinate_interpretation': 'Fraction.from_float over the candidate/native reference point values actually passed to the clip.',
        'tolerance_positive_pair_count': len(tolerance_positive_cross),
        'zero_measure_pair_count_omitted_from_transfer': len(zero_measure_cross),
        'boundary_contact_pair_count': len(boundary_contacts),
        'exactly_disjoint_tolerance_positive_pair_count': len(exactly_disjoint_cross),
        'genuine_full_dimensional_cross_material_count': len(genuine_cross),
        'zero_measure_pairs': zero_measure_cross,
        'boundary_contacts': boundary_contacts,
        'exactly_disjoint_tolerance_positive_pairs': exactly_disjoint_cross,
        'genuine_cross_material_pairs': genuine_cross,
    })
    raw_offsets, raw_parent_rows, raw_overlap_volumes = flatten_overlap_rows(overlap_rows)
    child_coverage = np.zeros(len(new_t), dtype=np.float64)
    parent_covered = np.zeros(len(tet_rows), dtype=np.float64)
    cross_material = []
    uncovered_child_rows = []
    for ci, row in enumerate(overlap_rows):
        if not row:
            uncovered_child_rows.append(int(ci))
        child_coverage[ci] = sum(v for _, v in row) if row else 0.0
        cm = MATERIAL_MAP[int(new_mat[ci])]
        for oi, ov in row:
            if old_mat[oi] != cm:
                cross_material.append((ci, oi, ov, cm, int(old_mat[oi])))
            parent_covered[oi] += ov
    child_cov_rel = (child_coverage - child_geom_volume) / child_geom_volume
    parent_cov_rel = (parent_covered - old_geom_volume) / old_geom_volume
    # Preserve complete geometric CSR and coverage arrays before any material/coverage gate.
    np.savez_compressed(RUN / 'geometric-overlap-085c.npz',
        candidate_cell_uids=cell_uids,
        candidate_child_material_ids=new_mat.astype(np.int8),
        candidate_child_reference_volume_m3=child_geom_volume,
        child_native_parent_indptr=raw_offsets,
        child_native_parent_rows=raw_parent_rows,
        overlap_native_reference_volume_m3=raw_overlap_volumes,
        child_overlap_coverage_m3=child_coverage,
        child_overlap_relative_error=child_cov_rel,
        native_parent_material_ids=old_mat.astype(np.int16),
        native_parent_geometry_volume_m3=old_geom_volume,
        native_parent_overlap_coverage_m3=parent_covered,
        native_parent_overlap_relative_error=parent_cov_rel,
        candidate_sha256=np.asarray(sha(CAND)),
        native_tet_csv_sha256=np.asarray(sha(TETS)),
        tolerance_positive_cross_material_pair_count=np.asarray(len(tolerance_positive_cross), np.int32),
        zero_measure_cross_material_pair_count_omitted=np.asarray(len(zero_measure_cross), np.int32),
        genuine_full_dimensional_cross_material_pair_count=np.asarray(len(genuine_cross), np.int32),
        uncovered_candidate_child_rows=np.asarray(uncovered_child_rows, dtype=np.int32))
    # All pair rows and coverage arrays are persisted before any admission gate.
    if uncovered_child_rows:
        raise RuntimeError(f'candidate children have no native overlap after exact zero-measure omission: count={len(uncovered_child_rows)} first={uncovered_child_rows[:10]}')
    if genuine_cross or cross_material:
        write_json(RUN / 'cross-material-witnesses.json', {
            'count': len(genuine_cross), 'pairs': genuine_cross,
            'note': 'Only exact affine-rank-3 cross-material intersections are retained as positive-volume failures.'})
        raise RuntimeError(f'genuine full-dimensional overlap crosses materials: count={len(genuine_cross)}; exact CSR and boundary diagnostics preserved')
    if np.max(np.abs(child_cov_rel)) > 1e-7:
        ix = int(np.argmax(np.abs(child_cov_rel)))
        raise RuntimeError(f'child reference-volume coverage failed at {ix}: rel={child_cov_rel[ix]}')
    if np.max(np.abs(parent_cov_rel)) > 1e-7:
        ix = int(np.argmax(np.abs(parent_cov_rel)))
        raise RuntimeError(f'native parent reference-volume coverage failed at {ix}: rel={parent_cov_rel[ix]}')

    offsets = raw_offsets.tolist()
    parent_rows = raw_parent_rows
    overlap_volumes = raw_overlap_volumes
    child_parent_fraction = []
    parent_mass_fraction = []
    for ci, row in enumerate(overlap_rows):
        for oi, ov in row:
            child_parent_fraction.append(ov / child_geom_volume[ci])
            parent_mass_fraction.append(ov / old_geom_volume[oi])
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
        'method': 'Every candidate child was tested against every native parent whose AABB overlaps under the pinned 073 tolerance; all overlap volumes use the unchanged unculled stacked 085b clip. Tolerance-positive cross-material pairs receive exact binary64 affine-rank classification: rank<3 contacts are saved as typed boundary diagnostics and omitted as zero-measure transfer pairs; rank 3 fails closed. No identity shortcut, approximate rejection, masked renormalization, or global mass rescale.',
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
                     'cross_material_overlap_count': len(cross_material),
                     'tolerance_positive_cross_material_pairs_before_exact_rank': len(tolerance_positive_cross),
                     'zero_measure_cross_material_pairs_omitted': len(zero_measure_cross),
                     'exact_boundary_contact_pairs': len(boundary_contacts),
                     'exactly_disjoint_tolerance_positive_pairs': len(exactly_disjoint_cross),
                     'genuine_full_dimensional_cross_material_pairs': len(genuine_cross)},
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
        run = RUN
        if run.exists():
            write_json(run / 'failure.json', {'status': 'FAILED', 'exception_type': type(exc).__name__,
                                               'message': str(exc), 'time_unix': time.time()})
        raise
