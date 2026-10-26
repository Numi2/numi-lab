#!/usr/bin/env python3
"""Inert adapter around the pinned 270/P037/P084 conforming subdivision.

The current candidate already has ``prior_input_parent__``.  This adapter
preserves that and every other prior_* array byte-for-byte, rekeys only the
new immediate-parent snapshot to ``prior_input_parent_546__``, composes Q@P,
and remaps complete multi-parent face-ancestry CSR rows. It creates affine
nodes only; it does not move geometry or claim clearance/admission.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any
import numpy as np

BASE = Path('/Users/numi/Work/numi-human-connected-surface-069')
OWNER270 = BASE / 'publication-109/sources/connected-refinement-270/connected_refinement_270.py'
OWNER270_SHA256 = '70e391867c4aa5680dacdcba1244ec15962a5dc74dc9d67c31467cdaffb608db'
PARENT_PREFIX = 'prior_input_parent__'
HISTORICAL_PARENT_PREFIX = 'prior_input_parent_546__'
ANCHOR_KEYS = ('coarse_boundary_anchor_rows', 'coarse_boundary_anchor_source_nodes',
               'typed_coarse_anchor_output_rows_018', 'typed_coarse_anchor_source_ids_018')
FACE_ANCESTRY_CSR = (
    ('face_source_ancestry_indptr', 'face_source_ancestry_face_rows'),
    ('face_fine_subface_ancestry_indptr', 'face_fine_subface_ancestry_rows'),
    ('face_fine_detail_ancestry_indptr', 'face_fine_detail_ancestry_rows'),
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def _load_owner270():
    actual = sha256_file(OWNER270)
    if actual != OWNER270_SHA256:
        raise ValueError(f'pinned 270 owner hash mismatch: {actual}')
    name = '_connected_refinement_270_adapter546'
    spec = importlib.util.spec_from_file_location(name, OWNER270)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load pinned 270 owner: {OWNER270}')
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    spec.loader.exec_module(owner)
    return owner

def _same_bytes(a, b) -> bool:
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes(order='C') == bb.tobytes(order='C')

def _face_csr_rows(ptr, values, nface: int, name: str):
    p, v = np.asarray(ptr), np.asarray(values)
    if p.dtype.kind not in 'iu' or v.dtype.kind not in 'iu':
        raise ValueError(f'{name} ancestry CSR must use integer pointers and rows')
    if p.shape != (nface + 1,) or int(p[0]) != 0 or int(p[-1]) != len(v):
        raise ValueError(f'{name} ancestry CSR has a stale face domain or pointer end')
    if np.any(p[1:] < p[:-1]):
        raise ValueError(f'{name} ancestry CSR pointers must be monotone')
    return [v[int(p[i]):int(p[i + 1])].copy() for i in range(nface)]

def _validate_immediate_parent_contract(arrays, nface: int, parent_count: int | None, *, label: str):
    ptr_key, rows_key = 'refinement_face_parent_ancestry_indptr', 'refinement_face_parent_ancestry_face_rows'
    if ptr_key not in arrays or rows_key not in arrays:
        raise ValueError(f'{label} is missing immediate-parent ancestry CSR')
    rows_by_face = _face_csr_rows(arrays[ptr_key], arrays[rows_key], nface, label)
    scalar = np.asarray(arrays.get('refinement_face_parent_face_row'))
    bary = np.asarray(arrays.get('refinement_face_parent_barycentric'), dtype=np.float64)
    if scalar.shape != (nface,) or scalar.dtype.kind not in 'iu' or bary.shape != (nface, 3, 3):
        raise ValueError(f'{label} scalar/barycentric immediate-parent rows have invalid shapes')
    for face, union in enumerate(rows_by_face):
        if len(union) == 0:
            raise ValueError(f'{label} immediate-parent ancestry has empty face row {face}')
        if parent_count is not None and (min(map(int, union)) < 0 or max(map(int, union)) >= parent_count):
            raise ValueError(f'{label} immediate-parent row {face} references outside its parent face domain')
        if len(union) == 1:
            if int(scalar[face]) != int(union[0]) or not np.isfinite(bary[face]).all():
                raise ValueError(f'{label} single-parent face {face} disagrees across CSR/scalar/barycentric metadata')
            if np.any(bary[face] < -1e-8) or np.any(bary[face] > 1 + 1e-8) or not np.allclose(bary[face].sum(axis=1), 1., atol=1e-8, rtol=0):
                raise ValueError(f'{label} single-parent face {face} has invalid barycentric rows')
        elif int(scalar[face]) != -1 or not np.isnan(bary[face]).all():
            raise ValueError(f'{label} multi-parent face {face} must use scalar -1 and NaN barycentrics')
    return rows_by_face


def _face_edge_set(faces, rows):
    f = np.asarray(faces, dtype=np.int64)
    out = set()
    for row in rows:
        a, b, c = map(int, f[int(row)])
        out.update((tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))))
    return out

def _select_crossing_faces_with_frozen_faces(faces, validated_crossed_rows, frozen_face_rows):
    """Drop only directly conflicting crossed-face operations.

    The selected-edge guard checks that remaining operations cannot split a
    frozen face. It intentionally does not transitively exclude neighbors of
    an omitted row; P084 remains responsible for its unchanged conforming fan.
    """
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("frozen-edge selection requires integer Fx3 faces")
    nface = len(f)
    full = np.asarray(validated_crossed_rows, dtype=np.int64)
    frozen = sorted(set(map(int, frozen_face_rows)))
    if full.ndim != 1 or (full.size and (int(full.min()) < 0 or int(full.max()) >= nface)):
        raise ValueError("validated crossing rows are outside the current face domain")
    if len(set(map(int, full))) != len(full):
        raise ValueError("validated crossing rows are not unique")
    if any(row < 0 or row >= nface for row in frozen):
        raise ValueError("frozen face row is outside current face domain")
    crossed = set(map(int, full))
    conflict = crossed.intersection(frozen)
    if conflict:
        raise ValueError(f"frozen faces are themselves crossed; refusing to omit them: {sorted(conflict)}")
    frozen_edges = _face_edge_set(f, frozen)
    excluded = sorted(row for row in crossed
                      if _face_edge_set(f, (row,)).intersection(frozen_edges))
    selected = np.asarray(sorted(crossed - set(excluded)), dtype=np.int64)
    if not len(selected):
        raise ValueError("frozen-edge selection removed every crossed face operation")
    selected_edges = _face_edge_set(f, selected)
    if selected_edges.intersection(frozen_edges):
        raise AssertionError("selected edges still touch a frozen face")
    return selected, {
        "original_validated_crossed_face_rows": sorted(crossed),
        "selected_crossed_face_rows": selected.tolist(),
        "excluded_crossed_face_rows": excluded,
        "frozen_face_rows": frozen,
        "protected_unchanged_face_rows": frozen,
        "frozen_edge_rows": [list(e) for e in sorted(frozen_edges)],
        "selected_edge_set_disjoint_from_frozen_faces": True,
    }

def _remap_face_union_csr(source, output, parent_face_rows, n_old_faces: int):
    parent = np.asarray(parent_face_rows)
    if parent.ndim != 1 or parent.dtype.kind not in 'iu' or (parent.size and
            (int(parent.min()) < 0 or int(parent.max()) >= n_old_faces)):
        raise ValueError('refined child faces do not map to valid immediate parent rows')
    stats = {}
    for ptr_key, rows_key in FACE_ANCESTRY_CSR:
        has_ptr, has_rows = ptr_key in source, rows_key in source
        if has_ptr != has_rows:
            raise ValueError(f'incomplete multi-parent ancestry pair: {ptr_key}, {rows_key}')
        if not has_ptr:
            continue
        old_rows = _face_csr_rows(source[ptr_key], source[rows_key], n_old_faces, ptr_key)
        child_rows = [old_rows[int(p)] for p in parent]
        ptr = np.zeros(len(child_rows) + 1, dtype=np.asarray(source[ptr_key]).dtype)
        for i, row in enumerate(child_rows):
            ptr[i + 1] = int(ptr[i]) + len(row)
        values = (np.concatenate(child_rows).astype(np.asarray(source[rows_key]).dtype, copy=False)
                  if child_rows else np.asarray([], dtype=np.asarray(source[rows_key]).dtype))
        output[ptr_key], output[rows_key] = ptr, values
        new_rows = _face_csr_rows(ptr, values, len(parent), ptr_key)
        for child, p in enumerate(parent):
            if not np.array_equal(new_rows[child], old_rows[int(p)]):
                raise AssertionError(f'{ptr_key} child {child} lost part of parent union')
        stats[ptr_key] = {
            'parent_rows_with_multi_parent_union': int(sum(len(row) > 1 for row in old_rows)),
            'parent_rows_with_empty_ancestry': int(sum(len(row) == 0 for row in old_rows)),
        }
    return stats

def build_refined_arrays546(source: dict[str, np.ndarray], crossing_report: dict[str, Any],
                            area_reference_points_world_m: np.ndarray,
                            area_reference_faces: np.ndarray, *, candidate_sha256: str,
                            crossing_report_sha256: str, area_reference_sha256: str,
                            frozen_face_rows=(), selection_exclusion_reason: str | None = None):
    """Build one affine subdivision generation while preserving all lineage."""
    original = {k: np.asarray(v).copy() for k, v in source.items()}
    if any(k.startswith(HISTORICAL_PARENT_PREFIX) for k in original):
        raise ValueError(f'{HISTORICAL_PARENT_PREFIX} is already occupied')
    if PARENT_PREFIX + 'source_prolongation_indptr' not in original:
        raise ValueError('expected existing prior_input_parent__ lineage namespace is missing')
    if 'prior_input_006__refinement_node_kind' not in original:
        raise ValueError('270 requires the existing prior_input_006__ node-domain witness')
    if ('refinement_face_parent_ancestry_indptr' in original or
            'refinement_face_parent_ancestry_face_rows' in original):
        if 'prior_input_parent__refinement_face_parent_face_row' not in original:
            raise ValueError('input immediate-parent CSR lacks its prior parent face-domain snapshot')
        _validate_immediate_parent_contract(
            original, len(original['source_O_faces']),
            len(original['prior_input_parent__refinement_face_parent_face_row']), label='input')
    history = {k: v.copy() for k, v in original.items() if k.startswith('prior_')}
    # 270's reserved parent namespace is removed from the staged call. Retain
    # the real 006 witness because 270 uses its length for source-column count.
    staged = {k: v.copy() for k, v in original.items() if not k.startswith('prior_')}
    for k, v in original.items():
        if k.startswith('prior_input_006__'):
            staged[k] = v.copy()
    owner = _load_owner270()
    original_snapshot = owner._snapshot_prior_arrays
    original_load_pinned = owner._load_pinned
    selection_receipt = {}
    frozen_rows = tuple(map(int, frozen_face_rows))
    if frozen_rows and not str(selection_exclusion_reason or '').strip():
        raise ValueError('frozen-face selection requires an explicit exclusion reason')
    try:
        # Preserve the pinned refiner and full report; filter only the validated
        # face-row selection returned by the loaded P037 owner's own parser.
        def _load_with_frozen_edge_selection(path, expected_sha, module_name):
            module = original_load_pinned(path, expected_sha, module_name)
            if Path(path).resolve() == Path(owner.REFINER_PATH).resolve() and frozen_rows:
                original_crossing_rows = module.crossing_face_rows
                def _validated_then_filtered(report_value, face_count):
                    full_rows, pair_count = original_crossing_rows(report_value, face_count)
                    selected_rows, policy = _select_crossing_faces_with_frozen_faces(
                        staged['source_O_faces'], full_rows, frozen_rows)
                    if selection_receipt:
                        raise RuntimeError('crossing selection hook was called more than once')
                    selection_receipt.update(policy)
                    selection_receipt['original_crossing_pair_count'] = int(pair_count)
                    selection_receipt['selection_exclusion_reason'] = str(selection_exclusion_reason)
                    return selected_rows, pair_count
                module.crossing_face_rows = _validated_then_filtered
            return module
        owner._load_pinned = _load_with_frozen_edge_selection
        # Ignore staged historical keys for this call; restore the real history
        # explicitly after 270 emits a fresh immediate-parent snapshot.
        owner._snapshot_prior_arrays = lambda _arrays: {}
        output, report = owner.build_refined_arrays(
            staged, crossing_report, area_reference_points_world_m, area_reference_faces,
            candidate_sha256=candidate_sha256,
            crossing_report_sha256=crossing_report_sha256,
            area_reference_sha256=area_reference_sha256)
        if selection_receipt:
            affected_rows = set(map(int, np.asarray(output["refinement_face_parent_face_row"])))
            excluded_rows = set(selection_receipt["excluded_crossed_face_rows"])
            selection_receipt["excluded_crossed_face_rows_affected_by_conforming_neighbor_edges"] = sorted(excluded_rows & affected_rows)
            selection_receipt["excluded_crossed_face_rows_remain_unresolved"] = sorted(excluded_rows)
            selection_receipt["excluded_crossing_faces_selected_for_clearance_refinement"] = False
    finally:
        owner._snapshot_prior_arrays = original_snapshot
        owner._load_pinned = original_load_pinned
    if frozen_rows and not selection_receipt:
        raise AssertionError('frozen-edge crossing-selection receipt was not produced')
    generated = [k for k in output if k.startswith(PARENT_PREFIX)]
    if not generated:
        raise AssertionError('270 did not emit its immediate-parent snapshot')
    # Preserve the older, already-existing parent namespace under a distinct
    # historical name. The fresh 270 snapshot remains at its standard prefix,
    # so downstream ancestry validators see the actual immediate input domain.
    renamed_history = {}
    for key, value in history.items():
        new_key = (HISTORICAL_PARENT_PREFIX + key[len(PARENT_PREFIX):]
                   if key.startswith(PARENT_PREFIX) else key)
        if new_key in output or new_key in renamed_history:
            raise ValueError(f'historical lineage key collision: {new_key}')
        renamed_history[new_key] = value.copy()
    for key, value in renamed_history.items():
        output[key] = value
    old_faces = np.asarray(original['source_O_faces'])
    parents = np.asarray(output['refinement_face_parent_face_row'])
    ancestry_stats = _remap_face_union_csr(original, output, parents, len(old_faces))

    # The input CSR is a union of the input faces' own immediate parents. Keep
    # it as history; the new top-level CSR describes each affine child -> its
    # actual immediate input face, matching 270's scalar and barycentric maps.
    old_parent_ptr = 'refinement_face_parent_ancestry_indptr'
    old_parent_rows = 'refinement_face_parent_ancestry_face_rows'
    if (old_parent_ptr in original) != (old_parent_rows in original):
        raise ValueError('incomplete input immediate-parent ancestry CSR')
    if old_parent_ptr in original:
        _face_csr_rows(original[old_parent_ptr], original[old_parent_rows], len(old_faces), old_parent_ptr)
        archive = {
            HISTORICAL_PARENT_PREFIX + 'input_refinement_face_parent_ancestry_indptr': original[old_parent_ptr].copy(),
            HISTORICAL_PARENT_PREFIX + 'input_refinement_face_parent_ancestry_face_rows': original[old_parent_rows].copy(),
        }
        semantics_key = 'refinement_face_parent_ancestry_semantics'
        if semantics_key in original:
            archive[HISTORICAL_PARENT_PREFIX + 'input_' + semantics_key] = original[semantics_key].copy()
        for key, value in archive.items():
            if key in output:
                raise ValueError(f'input immediate-parent CSR archive collision: {key}')
            output[key] = value
    if parents.shape != (len(output['source_O_faces']),) or parents.dtype.kind not in 'iu':
        raise ValueError('270 immediate parent rows do not match child face domain')
    output[old_parent_ptr] = np.arange(len(parents) + 1, dtype=np.int64)
    output[old_parent_rows] = parents.astype(np.int64, copy=True)
    output['refinement_face_parent_ancestry_semantics'] = np.asarray([
        'CSR singleton immediate input-face row for each affine child; older input unions are archived under prior_input_parent_546__input_refinement_face_parent_ancestry_*'
    ])
    scalar = np.asarray(output['refinement_face_parent_face_row'])
    bary = np.asarray(output['refinement_face_parent_barycentric'], dtype=np.float64)
    child_faces = np.asarray(output['source_O_faces'], dtype=np.int64)
    old_points = np.asarray(original['points_world_m'], dtype=np.float64)
    child_points = np.asarray(output['points_world_m'], dtype=np.float64)
    if scalar.shape != parents.shape or bary.shape != (len(parents), 3, 3):
        raise ValueError('immediate scalar/barycentric parent metadata has the wrong child domain')
    if not np.array_equal(scalar.astype(np.int64), parents.astype(np.int64)):
        raise ValueError('immediate-parent CSR and scalar rows identify different face domains')
    if not np.isfinite(bary).all() or not np.allclose(bary.sum(axis=2), 1.0, atol=1e-10, rtol=0):
        raise ValueError('immediate-parent barycentric rows are invalid')
    max_parent_reconstruction_error = 0.0
    for child, parent in enumerate(parents):
        tri = old_faces[int(parent)]
        reconstructed = bary[child] @ old_points[np.asarray(tri, dtype=np.int64)]
        actual = child_points[child_faces[child]]
        max_parent_reconstruction_error = max(max_parent_reconstruction_error,
                                                float(np.max(np.abs(reconstructed - actual))))
    f32_scale = max(1.0, float(np.max(np.abs(old_points), initial=0.0)))
    f32_tol = max(8.0 * abs(float(np.spacing(np.float32(f32_scale)))), 1e-12)
    if max_parent_reconstruction_error > f32_tol:
        raise ValueError(f'immediate-parent barycentrics do not reconstruct child vertices: {max_parent_reconstruction_error} > {f32_tol}')
    protected_rows = selection_receipt.get('protected_unchanged_face_rows', [])
    unchanged_protected = []
    for input_face in protected_rows:
        children = np.flatnonzero(parents == int(input_face))
        if len(children) != 1:
            raise ValueError(f'protected face {input_face} was subdivided by a remaining selected edge')
        child = int(children[0])
        if not np.array_equal(child_faces[child], old_faces[int(input_face)]):
            raise ValueError(f'protected face {input_face} changed connectivity')
        if not np.array_equal(bary[child], np.eye(3, dtype=np.float64)):
            raise ValueError(f'protected face {input_face} changed its parent barycentric identity')
        unchanged_protected.append(int(input_face))
    ptr = np.asarray(output[old_parent_ptr], dtype=np.int64)
    rows = np.asarray(output[old_parent_rows], dtype=np.int64)
    if not np.array_equal(np.diff(ptr), np.ones(len(parents), dtype=np.int64)) or not np.array_equal(rows, parents.astype(np.int64)):
        raise AssertionError('current immediate-parent CSR is not singleton child-to-input-face ancestry')
    for key, value in history.items():
        preserved_key = (HISTORICAL_PARENT_PREFIX + key[len(PARENT_PREFIX):]
                         if key.startswith(PARENT_PREFIX) else key)
        if not _same_bytes(output[preserved_key], value):
            raise AssertionError(f'historical prior lineage changed: {key} -> {preserved_key}')
    for key in ANCHOR_KEYS:
        if key in original and (key not in output or not _same_bytes(output[key], original[key])):
            raise AssertionError(f'typed anchor changed: {key}')
    for suffix in ('indptr', 'node_rows', 'weights', 'semantics'):
        parent_key = PARENT_PREFIX + 'source_prolongation_' + suffix
        if parent_key not in output or not _same_bytes(output[parent_key], original['source_prolongation_' + suffix]):
            raise AssertionError(f'immediate parent source map changed or missing: {suffix}')
    report = dict(report)
    report.update({
        'adapter': 'connected_refinement_adapter546',
        'adapter_sha256': sha256_file(Path(__file__)),
        'pinned_270_sha256': OWNER270_SHA256,
        'existing_prior_input_parent_namespace_preserved_under': HISTORICAL_PARENT_PREFIX,
        'new_immediate_parent_namespace': PARENT_PREFIX,
        'historical_prior_keys_preserved_count': len(history),
        'historical_prior_keys_preserved': sorted(history),
        'historical_namespace_migrations': {PARENT_PREFIX: HISTORICAL_PARENT_PREFIX},
        'old_input_immediate_parent_csr_archived': bool(old_parent_ptr in original),
        'old_input_immediate_parent_csr_archive_prefix': HISTORICAL_PARENT_PREFIX + 'input_',
        'new_immediate_parent_csr_singleton_child_to_input_face': True,
        'max_child_barycentric_parent_reconstruction_error_m': max_parent_reconstruction_error,
        'child_barycentric_reconstruction_tolerance_m': f32_tol,
        'face_ancestry_parent_row_stats': ancestry_stats,
        'construction_face_selection': selection_receipt or {
            'original_validated_crossed_face_rows': None,
            'selected_crossed_face_rows': None,
            'excluded_crossed_face_rows': [],
            'frozen_face_rows': [],
        },
        'protected_faces_proven_unchanged': unchanged_protected,
        'new_faces_inherit_full_parent_union': True,
        'geometry_moved': False,
        'containment_or_clearance_claim': False,
        'geometry_admitted': False,
        'physical_steps': 0,
    })
    return output, report
