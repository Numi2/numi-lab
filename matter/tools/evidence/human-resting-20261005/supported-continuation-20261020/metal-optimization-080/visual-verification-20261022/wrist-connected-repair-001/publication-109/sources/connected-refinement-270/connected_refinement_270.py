#!/usr/bin/env python3
"""Affine conforming refinement wrapper with explicit ancestry composition.

This is a construction-only wrapper around the pinned P084/P037 owners. It
preserves the current point prefix and historical lineage, composes the new
immediate map with the current source map, and prolongates a matched 037 area
reference through the same immediate map. It performs no geometry movement or
physical admission.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

BASE = Path('/Users/n/Work/numi-human-connected-surface-069')
REFINER_PATH = BASE / 'full-connected-clearance-150/run-037-local-refinement-006-prep/refine_conforming_006.py'
REFINER_SHA256 = 'd3a6e0c31e34f2165d853ffaf411ad67faaa1d307a3af1a6c79ba41a129f4f31'
P084_PATH = BASE / 'publication-084/canonical/matter/tools/human_skin_fat_boundary_refinement.py'
P084_SHA256 = 'eab8deda28164964051a6fa6ff346b8d94bc8bede04e896411dfdd02d46594f6'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned(path: Path, expected_sha: str, module_name: str):
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f'pinned source hash mismatch for {path}: {actual}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load pinned module {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _csr_from_arrays(arrays: dict[str, np.ndarray], prefix: str, rows: int,
                     cols: int | None = None) -> csr_matrix:
    ptr = np.asarray(arrays[prefix + 'indptr'])
    indices = np.asarray(arrays[prefix + 'node_rows'])
    values = np.asarray(arrays[prefix + 'weights'])
    if ptr.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu':
        raise ValueError(f'{prefix} CSR pointers and indices must be integers')
    if values.dtype.kind != 'f':
        raise ValueError(f'{prefix} CSR weights must be floating point')
    if ptr.shape != (rows + 1,) or ptr[0] != 0 or ptr[-1] != len(indices) or len(values) != len(indices):
        raise ValueError(f'{prefix} CSR shape/pointer mismatch')
    if np.any(ptr[1:] < ptr[:-1]) or np.any(indices < 0):
        raise ValueError(f'{prefix} CSR pointers or indices are invalid')
    if not np.isfinite(values).all() or np.any(values < -1e-14):
        raise ValueError(f'{prefix} CSR weights must be finite and nonnegative')
    inferred_cols = int(indices.max(initial=-1)) + 1
    ncols = inferred_cols if cols is None else int(cols)
    if ncols < inferred_cols or ncols < 0 or np.any(indices >= ncols):
        raise ValueError(f'{prefix} CSR index exceeds declared column domain')
    row_sums = np.zeros(rows, dtype=np.float64)
    for row in range(rows):
        row_sums[row] = values[int(ptr[row]):int(ptr[row + 1])].sum(dtype=np.float64)
    if np.any(ptr[1:] == ptr[:-1]) or not np.allclose(row_sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError(f'{prefix} CSR rows must be nonempty affine weights')
    return csr_matrix((values.astype(np.float64, copy=False),
                       indices.astype(np.int64, copy=False),
                       ptr.astype(np.int64, copy=False)), shape=(rows, ncols))


def _csr_arrays(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return (matrix.indptr.astype(np.int64, copy=True),
            matrix.indices.astype(np.int64, copy=True),
            matrix.data.astype(np.float64, copy=True))


def _snapshot_prior_arrays(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in source.items() if key.startswith('prior_')}


def _same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes(order='C') == bb.tobytes(order='C')


def build_refined_arrays(source: dict[str, np.ndarray], crossing_report: dict[str, Any],
                         area_reference_points_world_m: np.ndarray,
                         area_reference_faces: np.ndarray,
                         *, candidate_sha256: str, crossing_report_sha256: str,
                         area_reference_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Refine current surface and preserve lineage with Q @ P composition."""
    refiner = _load_pinned(REFINER_PATH, REFINER_SHA256, '_refine_conforming_037_270')
    X = np.asarray(source.get('points_world_m'))
    F = np.asarray(source.get('source_O_faces'))
    if X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError('candidate points_world_m must be finite Float32 Nx3')
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in 'iu' or len(F) == 0:
        raise ValueError('candidate source_O_faces must be nonempty integer Mx3')
    F64 = F.astype(np.int64, copy=False)
    if F64.min(initial=0) < 0 or F64.max(initial=-1) >= len(X):
        raise ValueError('candidate face references node outside point domain')
    if crossing_report.get('candidate_sha256') != candidate_sha256:
        raise ValueError('crossing report candidate hash does not match candidate input')
    if crossing_report.get('complete') is not True or crossing_report.get('geometry_or_native_admission') is not False:
        raise ValueError('crossing report must be complete and diagnostic-only')
    crossed, crossing_pair_count = refiner.crossing_face_rows(crossing_report, len(F))

    area_points = np.asarray(area_reference_points_world_m)
    area_faces = np.asarray(area_reference_faces)
    if area_points.shape != X.shape or area_points.dtype.kind != 'f' or not np.isfinite(area_points).all():
        raise ValueError('matched 037 area reference must be finite Nx3 in candidate node domain')
    if area_faces.shape != F.shape or area_faces.dtype.kind not in 'iu' or not np.array_equal(area_faces, F):
        raise ValueError('matched 037 area-reference faces must exactly match candidate face rows')

    # The current top-level CSR maps this mesh to its source-parent domain.
    # The historical prior_input_006__ map identifies that column count and is
    # intentionally retained byte-for-byte below.
    hist = _snapshot_prior_arrays(source)
    if 'prior_input_parent__source_prolongation_indptr' in source:
        raise ValueError('reserved prior_input_parent__ namespace already exists')
    if 'prior_input_006__refinement_node_kind' not in source:
        raise ValueError('missing typed historical input-domain row count')
    n_source = len(np.asarray(source['prior_input_006__refinement_node_kind']))
    P = _csr_from_arrays(source, 'source_prolongation_', len(X), n_source)

    markers = np.asarray(source.get('parent_source_face_marker'))
    if markers.shape != (len(F),) or markers.dtype.kind not in 'iu':
        raise ValueError('candidate parent_source_face_marker must be an integer face vector')
    result = refiner.p084_affine_refinement(X, F64, crossed, markers)
    Q = _csr_from_arrays(result, 'source_prolongation_', len(result['points_world_m']), len(X))
    composed = (Q @ P).tocsr()
    composed.sum_duplicates()
    composed.eliminate_zeros()
    composed.sort_indices()
    qptr, qidx, qval = _csr_arrays(Q)
    cptr, cidx, cval = _csr_arrays(composed)

    # The P037 extension code names the current parent `prior_input_006__`.
    # Feed it only current-domain arrays, then relabel that newly captured
    # parent snapshot without touching older historical stages.
    protected_anchor_keys = ('coarse_boundary_anchor_rows', 'coarse_boundary_anchor_source_nodes',
                             'typed_coarse_anchor_output_rows_018', 'typed_coarse_anchor_source_ids_018')
    protected_anchors = {key: np.asarray(source[key]).copy() for key in protected_anchor_keys if key in source}
    current = {key: np.asarray(value) for key, value in source.items()
               if not key.startswith('prior_') and key not in protected_anchors}
    extended = refiner.extend_candidate_arrays(current, result, candidate_sha256, crossing_report_sha256)
    # These arrays encode coarse-anchor IDs, not arbitrary node- or face-aligned
    # data. Restore them explicitly so dimension coincidences cannot remap them.
    extended.update(protected_anchors)
    for key in list(extended):
        if key.startswith('prior_input_006__'):
            new_key = 'prior_input_parent__' + key[len('prior_input_006__'):]
            if new_key in extended:
                raise ValueError(f'parent namespace collision for {new_key}')
            extended[new_key] = extended.pop(key)
    for key, value in hist.items():
        if key in extended:
            raise ValueError(f'historical lineage key collision for {key}')
        extended[key] = value

    # Keep both maps: Q is the immediate current-parent map; QP is the full
    # source-column mapping. The pre-existing P is retained in prior_input_parent__.
    extended['immediate_refinement_prolongation_indptr'] = qptr
    extended['immediate_refinement_prolongation_node_rows'] = qidx
    extended['immediate_refinement_prolongation_weights'] = qval
    extended['immediate_refinement_prolongation_semantics'] = np.asarray([
        f'rows map refined nodes to the immediate {len(X)}-node candidate parent using identity, edge midpoint, or affine face-centroid weights'
    ])
    extended['source_prolongation_indptr'] = cptr
    extended['source_prolongation_node_rows'] = cidx
    extended['source_prolongation_weights'] = cval
    extended['source_prolongation_semantics'] = np.asarray([
        f'rows map refined nodes through Q times the preserved immediate-parent map P to the {n_source}-node source domain'
    ])

    # Apply the exact same Q to the matched accepted-037 reference geometry.
    area_refined = np.asarray(Q @ area_points.astype(np.float64), dtype=np.float64)
    if area_refined.shape != (len(extended['points_world_m']), 3) or not np.isfinite(area_refined).all():
        raise AssertionError('refined 037 area reference has invalid shape or values')
    extended['area_reference_037_points_world_m_f64'] = area_refined
    extended['area_reference_037_source_O_faces'] = np.asarray(extended['source_O_faces']).copy()
    old_ref_cross = np.cross(area_points[ F64[:, 1] ].astype(np.float64) - area_points[ F64[:, 0] ].astype(np.float64),
                             area_points[ F64[:, 2] ].astype(np.float64) - area_points[ F64[:, 0] ].astype(np.float64))
    new_faces = np.asarray(extended['source_O_faces'], dtype=np.int64)
    new_ref_cross = np.cross(area_refined[new_faces[:, 1]] - area_refined[new_faces[:, 0]],
                             area_refined[new_faces[:, 2]] - area_refined[new_faces[:, 0]])
    old_ref_area = 0.5 * np.linalg.norm(old_ref_cross, axis=1)
    new_ref_area = 0.5 * np.linalg.norm(new_ref_cross, axis=1)
    if np.any(old_ref_area <= 0.0) or np.any(new_ref_area <= 0.0):
        raise ValueError('matched 037 area reference has a degenerate parent or child face')
    if np.any(np.einsum('ij,ij->i', new_ref_cross, old_ref_cross[np.asarray(result['parent_face_rows'], dtype=np.int64)]) <= 0.0):
        raise ValueError('matched 037 area-reference child winding disagrees with its parent')
    ref_area_sum = np.bincount(np.asarray(result['parent_face_rows'], dtype=np.int64), weights=new_ref_area, minlength=len(F))
    ref_area_rel = np.abs(ref_area_sum - old_ref_area) / np.maximum(old_ref_area, 1e-300)
    extended['area_reference_037_semantics'] = np.asarray([
        'matched accepted-037 area-reference points prolonged through the same immediate Q map and child-face topology; area constraints only'
    ])
    extended['area_reference_037_input_sha256'] = np.asarray([area_reference_sha256])
    extended['refinement_crossing_report_sha256'] = np.asarray([crossing_report_sha256])

    # Prefix and historical maps are hard invariants, not tolerance checks.
    if not np.array_equal(extended['points_world_m'][:len(X)].view(np.uint8), X.view(np.uint8)):
        raise AssertionError('old point-coordinate prefix changed')
    for key in ('coarse_boundary_anchor_rows', 'coarse_boundary_anchor_source_nodes',
                'typed_coarse_anchor_output_rows_018', 'typed_coarse_anchor_source_ids_018'):
        if key in source and not _same_bytes(extended[key], source[key]):
            raise AssertionError(f'anchor identity changed: {key}')
    for key, value in hist.items():
        if not _same_bytes(extended[key], value):
            raise AssertionError(f'historical lineage changed: {key}')
    for key in ('node_kind', 'node_source_row', 'node_source_virtual_row',
                'node_parent_coarse_face_row', 'node_parent_coarse_edge_a', 'node_parent_coarse_edge_b'):
        if key in source and not np.array_equal(extended[key][:len(X)].view(np.uint8), np.asarray(source[key]).view(np.uint8)):
            raise AssertionError(f'typed node identity prefix changed: {key}')

    # Validate current parent snapshot, immediate Q, and composed QP.
    parent_captured = _csr_from_arrays(extended, 'prior_input_parent__source_prolongation_', len(X), n_source)
    if parent_captured.shape != P.shape or (parent_captured != P).nnz:
        raise AssertionError('preserved immediate-parent CSR differs from input P')
    if Q.shape != (len(extended['points_world_m']), len(X)) or composed.shape != (len(extended['points_world_m']), n_source):
        raise AssertionError('prolongation composition dimensions are inconsistent')
    if not np.allclose(np.asarray(Q.sum(axis=1)).ravel(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError('immediate Q rows are not affine')
    if not np.allclose(np.asarray(composed.sum(axis=1)).ravel(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError('composed QP rows are not affine')

    report = {
        'status': 'REFINEMENT_CONSTRUCTION_READY',
        'candidate_sha256': candidate_sha256,
        'crossing_report_sha256': crossing_report_sha256,
        'area_reference_037_sha256': area_reference_sha256,
        'p037_refiner_sha256': REFINER_SHA256,
        'p084_owner_sha256': P084_SHA256,
        'old_node_count': int(len(X)),
        'new_node_count': int(len(extended['points_world_m'])),
        'old_face_count': int(len(F)),
        'new_face_count': int(len(extended['source_O_faces'])),
        'crossing_bank_triangle_pair_count': int(crossing_pair_count),
        'unique_crossed_parent_face_count': int(len(crossed)),
        'selected_edge_count': int(len(result['selected_edges'])),
        'affected_parent_face_count': int(len(result['affected_parent_face_rows'])),
        'boundary_components_before': int(result['boundary_components_before']),
        'boundary_components_after': int(result['boundary_components_after']),
        'boundary_edges_before': int(result['boundary_edges_before']),
        'boundary_edges_after': int(result['boundary_edges_after']),
        'max_parent_area_relative_drift_current_geometry': float(np.max(result['relative_parent_area_error'])),
        'immediate_Q_shape': [int(Q.shape[0]), int(Q.shape[1])],
        'parent_P_shape': [int(P.shape[0]), int(P.shape[1])],
        'composed_QP_shape': [int(composed.shape[0]), int(composed.shape[1])],
        'immediate_Q_nnz': int(Q.nnz),
        'composed_QP_nnz': int(composed.nnz),
        'historical_maps_preserved': sorted(hist),
        'old_point_prefix_bitwise_unchanged': True,
        'area_reference_uses_same_Q': True,
        'area_reference_faces_match_candidate_exactly': bool(np.array_equal(extended['area_reference_037_source_O_faces'], extended['source_O_faces'])),
        'max_parent_area_relative_drift_037_reference': float(np.max(ref_area_rel)),
        'existing_node_coordinates_moved': False,
        'new_nodes_created_by_affine_subdivision': True,
        'physical_steps': 0,
        'geometry_admitted': False,
        'scope_limit': 'conforming affine subdivision and ancestry only; no bone clearance, mass transfer, material, native runtime, or physical admission',
    }
    return extended, report


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--crossing-report', required=True, type=Path)
    parser.add_argument('--crossing-report-sha256', required=True)
    parser.add_argument('--area-reference-npz', required=True, type=Path)
    parser.add_argument('--area-reference-sha256', required=True)
    parser.add_argument('--area-reference-key', default='points_world_m')
    parser.add_argument('--output-npz', required=True, type=Path)
    parser.add_argument('--output-report', required=True, type=Path)
    args = parser.parse_args()
    for path in (args.output_npz, args.output_report):
        if path.exists():
            raise SystemExit(f'refuse to overwrite {path}')
    hashes = ((args.candidate, args.candidate_sha256),
              (args.crossing_report, args.crossing_report_sha256),
              (args.area_reference_npz, args.area_reference_sha256))
    for path, expected in hashes:
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f'input SHA mismatch for {path}: {actual}')
    source = _load_npz(args.candidate)
    crossing = json.loads(args.crossing_report.read_text())
    area = _load_npz(args.area_reference_npz)
    if args.area_reference_key not in area or 'source_O_faces' not in area:
        raise SystemExit('area-reference NPZ must contain requested points key and source_O_faces')
    output, report = build_refined_arrays(
        source, crossing, area[args.area_reference_key], area['source_O_faces'],
        candidate_sha256=args.candidate_sha256,
        crossing_report_sha256=args.crossing_report_sha256,
        area_reference_sha256=args.area_reference_sha256,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_npz.open('xb') as stream:
        np.savez_compressed(stream, **output)
    report['candidate_path'] = str(args.candidate)
    report['crossing_report_path'] = str(args.crossing_report)
    report['area_reference_037_path'] = str(args.area_reference_npz)
    report['output_npz'] = str(args.output_npz)
    report['output_npz_sha256'] = sha256_file(args.output_npz)
    report['output_report'] = str(args.output_report)
    args.output_report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
