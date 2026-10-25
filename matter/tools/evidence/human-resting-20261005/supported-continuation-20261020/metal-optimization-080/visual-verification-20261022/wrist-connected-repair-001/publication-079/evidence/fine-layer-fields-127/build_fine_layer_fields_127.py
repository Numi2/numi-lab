from __future__ import annotations
import csv, gzip, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

E = Path('/Users/n/Work/numi-human-connected-surface-069')
BASE = E / 'whole-reference-tetgen-084-bounded'
COMPACT = BASE / 'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
SOURCE = BASE / 'whole-skin-fat-reference-084-candidate.npz'
WPATH = BASE / 'connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'
SIDECAR = BASE / 'connectivity-001/run-001/transfer-map-075-084-002/transfer-sidecar.npz'
XFER_DIR = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight')
COARSE_REF = XFER_DIR / 'accepted-coarse-reference.csv'
COARSE_FACES = XFER_DIR / 'accepted-coarse-faces.csv'
TRANSFER_ROWS = XFER_DIR / 'accepted-transfer-rows.csv'
INITIAL_NODES = XFER_DIR / 'initial-skin-fem-nodes.csv'
MARKER_MAP = Path('/Users/n/Work/numi-human-connected-surface-069/publication-075/evidence/prepared-inputs/facet-marker-map.csv.gz')
TRANSFER_REPORT = BASE / 'connectivity-001/run-001/transfer-map-075-084-002/transfer-report.json'
OUT = E / 'fine-layer-fields-127'
ASSET = OUT / 'baseline-coarse-to-detail-layer-fields.npz'
REPORT = OUT / 'report.json'

EXPECTED = {
    'candidate_compacted_npz': '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    'source_084_candidate_npz': 'd3d0abb590f5858de4a27e0362e98bc9b3088831183ea045096cefc4ebba26e1',
    'Wnew_npz': '55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519',
    'transfer_sidecar_npz': '050875e760a1b3e3ddbbf18da5aff145b1405eeb4d3aa5d6dc31a09b15908f5d',
    'accepted_coarse_reference_csv': '9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2',
    'accepted_coarse_faces_csv': '7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2',
    'accepted_transfer_rows_csv': '71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
    'facet_marker_map_csv_uncompressed_sha256': '8e7ac9430c037d29e74d4c2f7f9f838995f6541c0f5fdf29a7b7095d9a6d8ff7',
    'transfer_report_json': '5452f7c24b9201734e0823a40a02d7cabcee6f945fd4755ea7acc5ee09fe8413',
}
COLUMN_COUNT = 2411
SOURCE_NODE_COUNT = COLUMN_COUNT * 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_gzip_uncompressed(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def csv_dicts(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def csv_rows_gz(path: Path):
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def unit_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.zeros_like(p)
    tri = p[f]
    area2n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, f[:, k], area2n)
    lengths = np.linalg.norm(n, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError('source O winding normal undefined at a coarse vertex')
    return n / lengths[:, None]


def column_layer_fields(points_m: np.ndarray, normals: np.ndarray):
    p = np.asarray(points_m, dtype=np.float64)
    if p.shape != (SOURCE_NODE_COUNT, 3):
        raise ValueError(f'expected source O/I/D rows {(SOURCE_NODE_COUNT, 3)}, got {p.shape}')
    o, i, d = p[:COLUMN_COUNT], p[COLUMN_COUNT:2*COLUMN_COUNT], p[2*COLUMN_COUNT:]
    skin = i - o
    fat = d - i
    whole = d - o
    hskin = np.linalg.norm(skin, axis=1)
    hfat = np.linalg.norm(fat, axis=1)
    if np.any(~np.isfinite(hskin)) or np.any(~np.isfinite(hfat)) or np.any(hskin <= 0) or np.any(hfat <= 0):
        raise ValueError('nonpositive/nonfinite Euclidean layer depth in source columns')
    n = np.asarray(normals, dtype=np.float64)
    return {
        'O': o, 'I': i, 'D': d,
        'skin_offset_I_minus_O': skin,
        'fat_offset_D_minus_I': fat,
        'total_offset_D_minus_O': whole,
        'hskin_euclidean': hskin,
        'hfat_euclidean': hfat,
        'skin_signed_projection': np.einsum('ij,ij->i', skin, n),
        'fat_signed_projection': np.einsum('ij,ij->i', fat, n),
    }


def interpolate_parent_field(parent_tet: np.ndarray, bary: np.ndarray, source_field: np.ndarray):
    tet = np.asarray(parent_tet, dtype=np.int64)
    b = np.asarray(bary, dtype=np.float64)
    f = np.asarray(source_field, dtype=np.float64)
    if tet.shape != (4,) or b.shape != (4,) or np.any(tet < 0) or np.any(tet >= len(f)):
        raise ValueError('bad source parent tetrahedron/barycentric shape or index')
    if not np.isfinite(b).all() or abs(float(b.sum()) - 1.0) > 1e-10:
        raise ValueError('source parent barycentric weights fail sum check')
    return b @ f[tet]


def summarize_mm(x_m: np.ndarray):
    x = np.asarray(x_m, dtype=np.float64) * 1000.0
    return {
        'count': int(x.size),
        'min': float(np.min(x)),
        'p01': float(np.quantile(x, .01)),
        'median': float(np.median(x)),
        'p99': float(np.quantile(x, .99)),
        'max': float(np.max(x)),
        'positive_count': int(np.count_nonzero(x > 0)),
    }


def main():
    t0 = time.time()
    paths = {
        'candidate_compacted_npz': COMPACT,
        'source_084_candidate_npz': SOURCE,
        'Wnew_npz': WPATH,
        'transfer_sidecar_npz': SIDECAR,
        'accepted_coarse_reference_csv': COARSE_REF,
        'accepted_coarse_faces_csv': COARSE_FACES,
        'accepted_transfer_rows_csv': TRANSFER_ROWS,
        'facet_marker_map_csv_gz': MARKER_MAP,
        'transfer_report_json': TRANSFER_REPORT,
        'initial_skin_fem_nodes_csv': INITIAL_NODES,
    }
    actual = {k: sha256(v) for k, v in paths.items()}
    actual['facet_marker_map_csv_uncompressed_sha256'] = sha256_gzip_uncompressed(MARKER_MAP)
    for k, expected in EXPECTED.items():
        if actual[k] != expected:
            raise RuntimeError(f'{k} SHA mismatch {actual[k]} != {expected}')

    compact = np.load(COMPACT, allow_pickle=False)
    source = np.load(SOURCE, allow_pickle=False)
    wz = np.load(WPATH, allow_pickle=False)
    side = np.load(SIDECAR, allow_pickle=False)
    transfer_report = json.loads(TRANSFER_REPORT.read_text())
    wshape = tuple(int(x) for x in wz['shape'])
    W = csr_matrix((wz['data'].astype(np.float64), wz['indices'], wz['indptr']), shape=wshape)
    W.sort_indices()

    R = np.asarray(compact['reference_points_m'], dtype=np.float64)
    source_rows = np.asarray(compact['source_input_row'], dtype=np.int64)
    source_R = np.asarray(source['source_reference_points_m'], dtype=np.float64)
    source_ids = np.asarray(source['source_global_node_ids'], dtype=np.int64)
    src_tets = np.asarray(source['source_tetrahedra_zero_based'], dtype=np.int64)
    if R.shape != (7360, 3) or source_rows.shape != (7360,) or source_R.shape != (SOURCE_NODE_COUNT, 3):
        raise RuntimeError('candidate/source geometry schema changed')
    if source_ids.shape != (SOURCE_NODE_COUNT,) or src_tets.shape != (28632, 4):
        raise RuntimeError('source node/tetrahedron schema changed')
    valid_src = np.flatnonzero(source_rows >= 0)
    if len(valid_src) != SOURCE_NODE_COUNT or not np.array_equal(np.sort(source_rows[valid_src]), np.arange(SOURCE_NODE_COUNT)):
        raise RuntimeError('compact candidate no longer maps all 7233 source rows exactly once')
    out_for_src = np.full(SOURCE_NODE_COUNT, -1, dtype=np.int64)
    out_for_src[source_rows[valid_src]] = valid_src
    if not np.array_equal(out_for_src, np.arange(SOURCE_NODE_COUNT)):
        raise RuntimeError('expected source input rows to remain the compact output prefix')
    if not np.array_equal(R[out_for_src], source_R):
        raise RuntimeError('compact reference source prefix is not bitwise-equal to pinned source reference')

    coarse_rows = csv_dicts(COARSE_REF)
    coarse_rows.sort(key=lambda r: int(r['coarse_row']))
    if len(coarse_rows) != COLUMN_COUNT or [int(r['coarse_row']) for r in coarse_rows] != list(range(COLUMN_COUNT)):
        raise RuntimeError('accepted coarse reference O rows are not exact 0..2410')
    accepted_O = np.asarray([[float(r[f'world_{axis}_m']) for axis in 'xyz'] for r in coarse_rows], dtype=np.float64)
    accepted_O_source_max_delta = float(np.max(np.abs(accepted_O - source_R[:COLUMN_COUNT])))
    if accepted_O_source_max_delta > 1e-7:
        raise RuntimeError(f'accepted 075 coarse O reference exceeds documented CSV precision tolerance: {accepted_O_source_max_delta} m')
    coarse_faces_rows = csv_dicts(COARSE_FACES)
    coarse_faces_rows.sort(key=lambda r: int(r['coarse_face_row']))
    coarse_faces = np.asarray([[int(r[f'vertex{i}']) for i in range(3)] for r in coarse_faces_rows], dtype=np.int64)
    candidate_O_faces = np.asarray(compact['source_O_faces'], dtype=np.int64)
    if not np.array_equal(coarse_faces, candidate_O_faces):
        raise RuntimeError('coarse O face identity/order differs from compact candidate')

    initial_rows = csv_dicts(INITIAL_NODES)
    initial_rows.sort(key=lambda r: int(r['global_node']))
    if len(initial_rows) != SOURCE_NODE_COUNT:
        raise RuntimeError('initial FEM node table is not 7233 rows')
    for r, row in enumerate(initial_rows):
        layer, col = divmod(r, COLUMN_COUNT)
        if int(row['layer']) != layer or int(row['row']) != col:
            raise RuntimeError(f'initial FEM node layer/column ordering mismatch at input row {r}')
        xyz = np.asarray([float(row[f'{axis}_m']) for axis in 'xyz'], dtype=np.float64)
        if not np.array_equal(xyz, source_R[r]):
            raise RuntimeError(f'initial FEM row {r} differs from 084 source reference position')

    # Source O face-winding normal is used only for signed diagnostics; it is not used as thickness.
    coarse_normal = unit_normals(source_R[:COLUMN_COUNT], coarse_faces)
    fields = column_layer_fields(source_R, coarse_normal)
    # The source FE extension repeats each O/I/D column scalar/vector attribute on its three
    # corresponding source vertices. This makes parent-tet interpolation continuous; W uses O only.
    source_fields = {
        'hskin': np.tile(fields['hskin_euclidean'], 3),
        'hfat': np.tile(fields['hfat_euclidean'], 3),
        'skin_offset': np.tile(fields['skin_offset_I_minus_O'], (3, 1)),
        'fat_offset': np.tile(fields['fat_offset_D_minus_I'], (3, 1)),
        'total_offset': np.tile(fields['total_offset_D_minus_O'], (3, 1)),
        'skin_signed_projection': np.tile(fields['skin_signed_projection'], 3),
        'fat_signed_projection': np.tile(fields['fat_signed_projection'], 3),
    }

    marker_rows = csv_rows_gz(MARKER_MAP)
    o_markers = {int(r['marker']) for r in marker_rows if r['surface'] == 'O'}
    surf_faces = np.asarray(compact['surface_faces_zero_based'], dtype=np.int64)
    surf_markers = np.asarray(compact['surface_face_markers'], dtype=np.int64)
    if surf_faces.shape != (14761, 3) or surf_markers.shape != (14761,):
        raise RuntimeError('compact marked surface schema changed')
    o_face_mask = np.isin(surf_markers, np.fromiter(o_markers, dtype=np.int64))
    o_nodes = np.unique(surf_faces[o_face_mask])
    if len(o_nodes) != 2464:
        raise RuntimeError(f'expected 2464 candidate O-surface output nodes, got {len(o_nodes)}')
    o_node_mask = np.zeros(len(R), dtype=bool)
    o_node_mask[o_nodes] = True

    # Candidate O field at original rows is the baseline per-column field. Generated O nodes
    # use their frozen source-tetra ancestry, which must lie on the O layer (up to roundoff).
    candidate_fields = {k: np.full((len(R),) + sf.shape[1:], np.nan, dtype=np.float64) for k, sf in source_fields.items()}
    for outrow in o_nodes:
        srcrow = int(source_rows[outrow])
        if srcrow >= 0:
            if srcrow >= COLUMN_COUNT:
                raise RuntimeError(f'O-surface output row {outrow} maps to non-O source row {srcrow}')
            for key, sf in source_fields.items():
                candidate_fields[key][outrow] = sf[srcrow]

    ancestry_out = np.asarray(compact['newnode_ancestry_output_row'], dtype=np.int64)
    ancestry_tet = np.asarray(compact['newnode_ancestry_source_tet_row'], dtype=np.int64)
    ancestry_bary = np.asarray(compact['newnode_ancestry_barycentric'], dtype=np.float64)
    if ancestry_out.shape != ancestry_tet.shape or ancestry_bary.shape != (len(ancestry_out), 4):
        raise RuntimeError('generated-node ancestry schema mismatch')
    ancestry_by_row: dict[int, list[int]] = {}
    for j, outrow in enumerate(ancestry_out):
        ancestry_by_row.setdefault(int(outrow), []).append(j)
    max_parent_position_residual = 0.0
    max_nonO_bary_l1 = 0.0
    max_parent_field_disagreement = 0.0
    generated_o_rows = o_nodes[source_rows[o_nodes] < 0]
    if len(generated_o_rows) != 53:
        raise RuntimeError(f'expected 53 generated O rows, got {len(generated_o_rows)}')
    for outrow in generated_o_rows:
        entries = ancestry_by_row.get(int(outrow), [])
        if not entries:
            raise RuntimeError(f'generated O row {outrow} has no source ancestry')
        preds_by_key = {k: [] for k in source_fields}
        for j in entries:
            tetrow = int(ancestry_tet[j])
            bary = ancestry_bary[j]
            if tetrow < 0 or tetrow >= len(src_tets):
                raise RuntimeError(f'generated O row {outrow} parent tet out of range')
            tet = src_tets[tetrow]
            if np.any(tet < 0) or np.any(tet >= SOURCE_NODE_COUNT):
                raise RuntimeError('source parent tet has non-source input row')
            if abs(float(bary.sum()) - 1.0) > 1e-10:
                raise RuntimeError(f'generated O row {outrow} parent barycentric sum error')
            non_o = float(np.abs(bary[tet >= COLUMN_COUNT]).sum())
            max_nonO_bary_l1 = max(max_nonO_bary_l1, non_o)
            if non_o > 1e-10:
                raise RuntimeError(f'generated O row {outrow} ancestry is not on source O face (non-O weight {non_o})')
            parent_xyz = bary @ source_R[tet]
            pos_res = float(np.linalg.norm(parent_xyz - R[outrow]))
            max_parent_position_residual = max(max_parent_position_residual, pos_res)
            if pos_res > 1e-8:
                raise RuntimeError(f'generated O row {outrow} parent coordinate residual {pos_res} m')
            for key, sf in source_fields.items():
                preds_by_key[key].append(interpolate_parent_field(tet, bary, sf))
        for key, preds in preds_by_key.items():
            ref = np.asarray(preds[0], dtype=np.float64)
            for value in preds[1:]:
                diff = float(np.max(np.abs(np.asarray(value) - ref)))
                max_parent_field_disagreement = max(max_parent_field_disagreement, diff)
                if diff > 1e-10:
                    raise RuntimeError(f'generated O row {outrow} ancestry parents disagree for {key}: {diff}')
            candidate_fields[key][outrow] = ref

    wzdata = np.asarray(wz['data'], dtype=np.float32)
    windices = np.asarray(wz['indices'], dtype=np.int32)
    windptr = np.asarray(wz['indptr'], dtype=np.int64)
    if W.shape != (61950, len(R)) or len(transfer_report['075']['active_rows'] if isinstance(transfer_report['075'].get('active_rows'), list) else []) not in (0, 54551):
        raise RuntimeError('W or transfer report schema changed')
    transfer_rows = csv_dicts(TRANSFER_ROWS)
    transfer_rows.sort(key=lambda r: int(r['detail_row']))
    n_detail = len(transfer_rows)
    if n_detail != W.shape[0] or [int(r['detail_row']) for r in transfer_rows] != list(range(n_detail)):
        raise RuntimeError('detail transfer row identity/order changed')
    active = np.asarray([int(r['active']) == 1 for r in transfer_rows], dtype=bool)
    row_nnz = np.diff(windptr)
    row_sums = np.asarray(W.sum(axis=1)).reshape(-1)
    if int(active.sum()) != 54551 or int((~active).sum()) != 7399:
        raise RuntimeError('active/inactive detail row counts differ from pinned report')
    if np.any(active != (row_nnz > 0)):
        raise RuntimeError('accepted transfer active mask differs from W zero-row mask')
    if np.any(row_nnz[~active] != 0) or np.any(row_sums[~active] != 0):
        raise RuntimeError('inactive W rows are not exact zero rows')
    if np.any(row_nnz[active] <= 0) or np.max(np.abs(row_sums[active] - 1.0)) > 1e-7:
        raise RuntimeError('active W row sum is outside pinned Float32 tolerance')
    if len(wzdata) != int(row_nnz.sum()) or np.any(wzdata < 0) or np.any(~np.isfinite(wzdata)):
        raise RuntimeError('W has invalid support/weights')
    support = np.unique(windices)
    if np.any(~o_node_mask[support]):
        bad = support[~o_node_mask[support]][:10].tolist()
        raise RuntimeError(f'W support is not restricted to candidate O surface: {bad}')
    if np.any(~np.isfinite(candidate_fields['hskin'][support])) or np.any(~np.isfinite(candidate_fields['hfat'][support])):
        raise RuntimeError('W support includes an undefined source layer field')

    fine_fields = {}
    for key, nodefield in candidate_fields.items():
        finite_node = np.where(o_node_mask[:, None], nodefield, 0.0) if nodefield.ndim == 2 else np.where(o_node_mask, nodefield, 0.0)
        mapped = np.asarray(W @ finite_node, dtype=np.float64)
        if nodefield.ndim == 1:
            mapped = mapped.reshape(-1)
        mapped[~active] = np.nan
        fine_fields[key] = mapped
    if not np.all(np.isfinite(fine_fields['hskin'][active])) or not np.all(np.isfinite(fine_fields['hfat'][active])):
        raise RuntimeError('active detail fields contain nonfinite values')
    if np.any(fine_fields['hskin'][active] <= 0) or np.any(fine_fields['hfat'][active] <= 0):
        raise RuntimeError('active detail Euclidean thickness field is nonpositive')

    fine_id = np.asarray([int(r['detail_source_vertex_id']) for r in transfer_rows], dtype=np.int64)
    fine_parent_face = np.full(n_detail, -1, dtype=np.int64)
    fine_parent_stable = np.full(n_detail, -1, dtype=np.int64)
    fine_parent_vertices = np.full((n_detail, 3), -1, dtype=np.int64)
    fine_parent_bary = np.full((n_detail, 3), np.nan, dtype=np.float64)
    for j, r in enumerate(transfer_rows):
        if not active[j]:
            continue
        fine_parent_face[j] = int(r['selected_face_row'])
        fine_parent_stable[j] = int(r['selected_face_stable_id'])
        fine_parent_vertices[j] = [int(r[f'coarse_vertex{k}']) for k in range(3)]
        fine_parent_bary[j] = [float(r[f'weight{k}']) for k in range(3)]
        if np.any(fine_parent_vertices[j] < 0) or np.any(fine_parent_vertices[j] >= COLUMN_COUNT):
            raise RuntimeError(f'active fine row {j} parent coarse face vertex out of range')
        if abs(float(fine_parent_bary[j].sum()) - 1.0) > 1e-6 or np.any(fine_parent_bary[j] < -1e-7):
            raise RuntimeError(f'active fine row {j} source face barycentric invalid')

    coarse_direct = np.asarray([int(r['source_direct_id']) if r['source_direct_id'].strip() else -1 for r in coarse_rows], dtype=np.int64)
    coarse_virtual_vertices = np.asarray([[int(r[f'source_face_v{k}']) if r[f'source_face_v{k}'].strip() else -1 for k in range(3)] for r in coarse_rows], dtype=np.int64)
    coarse_virtual_bary = np.asarray([[float(r[f'source_bary{k}']) if r[f'source_bary{k}'].strip() else np.nan for k in range(3)] for r in coarse_rows], dtype=np.float64)
    coarse_donor = np.asarray([int(r['donor_body']) for r in coarse_rows], dtype=np.int32)
    if np.any((coarse_direct < 0) & (~np.isfinite(coarse_virtual_bary).all(axis=1))):
        raise RuntimeError('virtual coarse anchor missing explicit source-face barycentric provenance')
    fine_parent_source_direct_ids = np.full((n_detail, 3), -1, dtype=np.int64)
    fine_parent_source_direct_ids[active] = coarse_direct[fine_parent_vertices[active]]

    arrays = {
        'coarse_column_ids': np.arange(COLUMN_COUNT, dtype=np.int32),
        'coarse_source_global_node_ids_O': source_ids[:COLUMN_COUNT],
        'coarse_source_global_node_ids_I': source_ids[COLUMN_COUNT:2*COLUMN_COUNT],
        'coarse_source_global_node_ids_D': source_ids[2*COLUMN_COUNT:],
        'coarse_O_reference_m': fields['O'],
        'coarse_I_reference_m': fields['I'],
        'coarse_D_reference_m': fields['D'],
        'coarse_skin_offset_I_minus_O_m': fields['skin_offset_I_minus_O'],
        'coarse_fat_offset_D_minus_I_m': fields['fat_offset_D_minus_I'],
        'coarse_total_offset_D_minus_O_m': fields['total_offset_D_minus_O'],
        'coarse_hskin_euclidean_m': fields['hskin_euclidean'],
        'coarse_hfat_euclidean_m': fields['hfat_euclidean'],
        'coarse_source_O_winding_normal_unit': coarse_normal,
        'coarse_skin_signed_projection_on_O_winding_m': fields['skin_signed_projection'],
        'coarse_fat_signed_projection_on_O_winding_m': fields['fat_signed_projection'],
        'coarse_source_direct_id': coarse_direct,
        'coarse_source_virtual_face_vertex_ids': coarse_virtual_vertices,
        'coarse_source_virtual_face_barycentric': coarse_virtual_bary,
        'coarse_donor_body': coarse_donor,
        'candidate_O_surface_output_node_rows': o_nodes.astype(np.int32),
        'candidate_O_surface_node_field_defined_mask': o_node_mask,
        'candidate_O_surface_hskin_field_m': candidate_fields['hskin'],
        'candidate_O_surface_hfat_field_m': candidate_fields['hfat'],
        'candidate_O_surface_skin_offset_I_minus_O_m': candidate_fields['skin_offset'],
        'candidate_O_surface_fat_offset_D_minus_I_m': candidate_fields['fat_offset'],
        'candidate_O_surface_total_offset_D_minus_O_m': candidate_fields['total_offset'],
        'candidate_O_surface_skin_signed_projection_on_O_winding_m': candidate_fields['skin_signed_projection'],
        'candidate_O_surface_fat_signed_projection_on_O_winding_m': candidate_fields['fat_signed_projection'],
        'fine_detail_row': np.arange(n_detail, dtype=np.int32),
        'fine_detail_source_vertex_id': fine_id,
        'fine_active_mask': active,
        'fine_W_row_nnz': row_nnz.astype(np.int32),
        'fine_W_row_sum': row_sums,
        'fine_selected_source_face_row': fine_parent_face,
        'fine_selected_source_face_stable_id': fine_parent_stable,
        'fine_selected_source_face_coarse_vertex_rows': fine_parent_vertices,
        'fine_selected_source_face_barycentric': fine_parent_bary,
        'fine_parent_coarse_vertex_source_direct_ids': fine_parent_source_direct_ids,
        'fine_hskin_euclidean_m': fine_fields['hskin'],
        'fine_hfat_euclidean_m': fine_fields['hfat'],
        'fine_skin_offset_I_minus_O_m': fine_fields['skin_offset'],
        'fine_fat_offset_D_minus_I_m': fine_fields['fat_offset'],
        'fine_total_offset_D_minus_O_m': fine_fields['total_offset'],
        'fine_skin_signed_projection_on_O_winding_m': fine_fields['skin_signed_projection'],
        'fine_fat_signed_projection_on_O_winding_m': fine_fields['fat_signed_projection'],
        'generated_O_output_rows': generated_o_rows.astype(np.int32),
        'generated_O_source_parent_tet_rows': np.asarray([ancestry_tet[ancestry_by_row[int(r)][0]] for r in generated_o_rows], dtype=np.int32),
        'generated_O_source_parent_barycentric': np.asarray([ancestry_bary[ancestry_by_row[int(r)][0]] for r in generated_o_rows], dtype=np.float64),
    }
    np.savez_compressed(ASSET, **arrays)
    asset_sha = sha256(ASSET)

    def negative_stats(x):
        y = np.asarray(x, dtype=np.float64)
        return {'negative_count': int(np.count_nonzero(y < 0)), 'min_m': float(y.min()), 'max_m': float(y.max())}
    def stats(x):
        return summarize_mm(np.asarray(x, dtype=np.float64))

    report = {
        'schema': 'baseline_084_column_layer_fields_interpolated_to_075_detail_v1',
        'status': 'FIELD_INTERPOLATION_DIAGNOSTIC_ONLY',
        'classification': 'Euclidean O-I and I-D lengths are inferred geometric fields from the pinned baseline reference mesh; they are not measured anatomical thicknesses and do not qualify geometry, materials, or native physics.',
        'field_semantics': {
            'coarse_source': '084 baseline reference_points_m, with original 7233 source rows ordered as O[0:2411], I[2411:4822], D[4822:7233].',
            'thickness': 'hskin = norm(I-O); hfat = norm(D-I), both Euclidean and positive by construction.',
            'offsets': 'I-O, D-I, and D-O retained as vector fields in metres.',
            'generated_O_rows': 'For the 53 generated O surface nodes, field values are P1 interpolations from frozen source tetrahedron ancestry. Parent barycentrics place each supported point on its O parent face within the checked residual; no nearest-anchor substitution.',
            'W_use': 'The exact pinned Wnew CSR is used only to interpolate these scalar/vector fields from candidate O surface nodes to active detailed rows. It is not used as a geometric anchor or position transfer here.',
            'inactive_rows': 'The 7,399 inactive detail rows have no W support; their fine fields are NaN in the asset, not zero thickness.',
            'signed_projection': 'Diagnostic dot product of each offset with the area-weighted source O face-winding normal. Sign is relative to that winding only; negative values are retained and are not clamped or used as thickness.',
            'source_provenance': 'Each active detail row carries its detail source vertex ID and exact selected coarse source face row/stable ID, coarse vertex rows, and face barycentrics. Coarse vertices carry direct source IDs or explicit source-face vertex IDs and barycentrics for virtual anchors.',
        },
        'counts': {
            'source_columns': COLUMN_COUNT,
            'source_layer_nodes': SOURCE_NODE_COUNT,
            'candidate_output_nodes': int(len(R)),
            'candidate_O_surface_nodes': int(len(o_nodes)),
            'generated_O_surface_nodes': int(len(generated_o_rows)),
            'W_support_nodes': int(len(support)),
            'W_support_original_O_nodes': int(np.count_nonzero(support < COLUMN_COUNT)),
            'W_support_generated_O_nodes': int(np.count_nonzero(support >= SOURCE_NODE_COUNT)),
            'detail_rows': int(n_detail),
            'active_detail_rows': int(active.sum()),
            'inactive_detail_rows': int((~active).sum()),
            'coarse_direct_source_anchors': int(np.count_nonzero(coarse_direct >= 0)),
            'coarse_virtual_source_anchors': int(np.count_nonzero(coarse_direct < 0)),
        },
        'checks': {
            'candidate_source_prefix_bitwise_matches_source_reference': True,
            'accepted_075_O_reference_max_abs_delta_from_source_O_m': accepted_O_source_max_delta,
            'accepted_075_O_reference_within_1e-7m_csv_precision_bound': accepted_O_source_max_delta <= 1e-7,
            'source_initial_node_order_and_positions_match_layer_blocks': True,
            'candidate_O_faces_bitwise_match_accepted_075_faces': True,
            'all_W_support_columns_are_candidate_O_surface_nodes': True,
            'all_W_support_columns_have_defined_fields': True,
            'W_active_mask_matches_transfer_table': True,
            'active_W_rows_have_positive_nonnegative_support': True,
            'inactive_W_rows_are_zero': True,
            'generated_O_parent_position_max_residual_m': max_parent_position_residual,
            'generated_O_parent_nonO_barycentric_l1_max': max_nonO_bary_l1,
            'generated_O_parent_field_disagreement_max': max_parent_field_disagreement,
            'W_active_row_sum_error_max': float(np.max(np.abs(row_sums[active] - 1.0))),
            'W_min_weight': float(wzdata.min()),
            'W_max_weight': float(wzdata.max()),
        },
        'ranges_mm': {
            'coarse_hskin_euclidean': stats(fields['hskin_euclidean']),
            'coarse_hfat_euclidean': stats(fields['hfat_euclidean']),
            'fine_active_hskin_euclidean': stats(fine_fields['hskin'][active]),
            'fine_active_hfat_euclidean': stats(fine_fields['hfat'][active]),
            'coarse_skin_signed_projection_on_O_winding': negative_stats(fields['skin_signed_projection']),
            'coarse_fat_signed_projection_on_O_winding': negative_stats(fields['fat_signed_projection']),
            'fine_skin_signed_projection_on_O_winding': negative_stats(fine_fields['skin_signed_projection'][active]),
            'fine_fat_signed_projection_on_O_winding': negative_stats(fine_fields['fat_signed_projection'][active]),
        },
        'input_sha256': actual,
        'output': {'asset_path': str(ASSET), 'asset_sha256': asset_sha, 'asset_bytes': ASSET.stat().st_size},
        'runtime': {'host': platform.node(), 'python': sys.version, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'seconds': time.time() - t0, 'builder_sha256': sha256(Path(__file__))},
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'asset': str(ASSET), 'asset_sha256': asset_sha, 'report': str(REPORT), 'counts': report['counts'], 'ranges_mm': report['ranges_mm']}, indent=2))

if __name__ == '__main__':
    main()
