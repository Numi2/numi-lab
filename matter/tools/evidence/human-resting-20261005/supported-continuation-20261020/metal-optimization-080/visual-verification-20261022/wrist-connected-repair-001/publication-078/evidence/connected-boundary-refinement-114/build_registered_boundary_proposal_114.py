from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/Users/numi/Work/numi-human-connected-surface-069')
OUTROOT = ROOT / 'connected-boundary-refinement-114'
TOOL = OUTROOT / 'matter/tools'
BATCH = ROOT / 'digit-source-diagnosis-110/batch-145-curved-source-midpoints-004.json'
COMPACT = ROOT / 'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
REFERENCE = ROOT / 'connected-capacity-reference-115/candidate.npz'
MARKER_MAP = ROOT / 'whole-reference-tetgen-084-bounded/facet-marker-map.csv'
OUT = OUTROOT / 'run-001'
EXPECTED = {
    'batch': 'e4c6c76d497e176e40ad802cb3159a4a143279cc131ae632fe515498aa38fafa',
    'compact': '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    'reference115': 'abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9',
    'marker_map': '8e7ac9430c037d29e74d4c2f7f9f838995f6541c0f5fdf29a7b7095d9a6d8ff7',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode('ascii'))
    h.update(json.dumps(list(a.shape), separators=(',', ':')).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f'refusing existing output directory: {OUT}')
    actual = {'batch': sha(BATCH), 'compact': sha(COMPACT),
              'reference115': sha(REFERENCE), 'marker_map': sha(MARKER_MAP)}
    if actual != EXPECTED:
        raise RuntimeError(f'frozen input hash mismatch: {actual}')

    z = np.load(COMPACT, allow_pickle=False)
    r115 = np.load(REFERENCE, allow_pickle=False)
    batch = json.loads(BATCH.read_text())
    if len(batch['edges']) != 145 or batch['counts']['proposal_eligible'] != 123:
        raise RuntimeError('source midpoint batch inventory changed')
    if not np.array_equal(r115['source_reference_points_m'], z['reference_points_m']):
        raise RuntimeError('115 reference source no longer exactly matches 084 reference')
    if not np.array_equal(r115['source_points_world_f32'], z['current037_points_m_f32']):
        raise RuntimeError('115 current source no longer exactly matches 084 current037')
    points_by_frame = {
        'reference': np.asarray(r115['proposed_reference_points_m']),
        'current037': np.asarray(r115['points_world_f32']),
    }
    if points_by_frame['reference'].dtype != np.float64 or points_by_frame['current037'].dtype != np.float32:
        raise RuntimeError('115 coordinate precision contract changed')
    if points_by_frame['reference'].shape != (7360, 3) or points_by_frame['current037'].shape != (7360, 3):
        raise RuntimeError('115 point array domain changed')

    source_global = np.asarray(z['source_global_node_ids'], dtype=np.int64)
    source_input = np.asarray(z['source_input_row'], dtype=np.int64)
    if source_global.shape != (7233,) or not np.array_equal(source_input[:7233], np.arange(7233)):
        raise RuntimeError('084 original input-row identity is not the pinned 0..7232 prefix')
    global_to_row = {int(g): i for i, g in enumerate(source_global)}
    if len(global_to_row) != len(source_global):
        raise RuntimeError('084 source global IDs are not unique')

    # The existing marker map is the owner of source parent marker identity. Validate
    # every layer source face by its exact source-face row and vertex set.
    marker_rows = list(csv.DictReader(MARKER_MAP.open(newline='')))
    layer_code = {'O': 1, 'I': 2, 'D': 3}
    faces_by_layer = {layer: np.asarray(z[f'source_{layer}_faces'], dtype=np.int64) for layer in layer_code}
    parent_markers: dict[str, np.ndarray] = {}
    marker_by_surface_row: dict[tuple[str, int], int] = {}
    cap_rows: list[tuple[int, str, int, tuple[int, int, int]]] = []
    for rec in marker_rows:
        surface = rec['surface']
        marker = int(rec['marker'])
        parent_row = int(rec['source_face_row'])
        verts = tuple(int(v) for v in rec['vertex_ids_zero_based'].split())
        if len(verts) != 3 or min(verts) < 0 or max(verts) >= points_by_frame['reference'].shape[0]:
            raise RuntimeError(f'malformed marker-map face {rec}')
        if surface in layer_code:
            key = (surface, parent_row)
            if key in marker_by_surface_row or not 0 <= parent_row < len(faces_by_layer[surface]):
                raise RuntimeError(f'duplicate/out-of-range marker parent {key}')
            face = faces_by_layer[surface][parent_row]
            if sorted(verts) != sorted(map(int, face)):
                raise RuntimeError(f'{surface} marker {marker} does not identify its source face row {parent_row}')
            if marker != layer_code[surface] * 100000 + parent_row + 1:
                raise RuntimeError(f'{surface} marker numbering changed at source row {parent_row}')
            marker_by_surface_row[key] = marker
        elif surface.startswith('SIDE-'):
            cap_rows.append((marker, surface, parent_row, verts))
        else:
            raise RuntimeError(f'unrecognized marker-map surface {surface!r}')
    for layer in layer_code:
        if len([k for k in marker_by_surface_row if k[0] == layer]) != len(faces_by_layer[layer]):
            raise RuntimeError(f'{layer} marker map does not cover every source parent face')
        parent_markers[layer] = np.asarray([
            marker_by_surface_row[(layer, row)] for row in range(len(faces_by_layer[layer]))
        ], dtype=np.int64)
        if len(np.unique(parent_markers[layer])) != len(parent_markers[layer]):
            raise RuntimeError(f'{layer} source parent markers are not unique')
    cap_rows.sort(key=lambda x: x[0])
    cap_faces = np.asarray([r[3] for r in cap_rows], dtype=np.int64).reshape(-1, 3)
    cap_markers = np.asarray([r[0] for r in cap_rows], dtype=np.int64)
    cap_parent_rows = np.asarray([r[2] for r in cap_rows], dtype=np.int64)
    cap_surface_names = [r[1] for r in cap_rows]
    if len(cap_rows) != 180 or len(np.unique(cap_markers)) != len(cap_markers):
        raise RuntimeError('side-cap marker table changed')

    # Explicitly reconstruct column->layer row mappings from the batch's row-aligned
    # O/I/D endpoints and verify the old offsets against the frozen 084 source.
    column_layer_rows: dict[int, dict[str, int]] = {}
    selections = []
    eligible_records = []
    held_records = []
    old_current = np.asarray(z['current037_points_m_f32'])
    old_reference = np.asarray(z['reference_points_m'])
    for rec in batch['edges']:
        if not rec['proposal_eligible']:
            held_records.append(rec)
            continue
        columns = tuple(map(int, rec['edge_global_node_ids']))
        o_rows = tuple(map(int, rec['midpoint_layer_positions']['O']['endpoint_rows']))
        if columns != tuple(map(int, rec['edge_endpoint_source_vertex_ids'])):
            raise RuntimeError('batch coarse edge IDs and endpoint IDs differ')
        if o_rows != tuple(map(int, rec['edge_output_rows'])):
            raise RuntimeError('batch O endpoint rows disagree with output rows')
        if tuple(map(int, source_global[list(o_rows)])) != columns:
            raise RuntimeError('batch O output rows do not match exact 084 source global IDs')
        layer_edges = {}
        for layer in ('O', 'I', 'D'):
            rows = tuple(map(int, rec['midpoint_layer_positions'][layer]['endpoint_rows']))
            if len(rows) != 2 or any(row not in set(faces_by_layer[layer].ravel().tolist()) for row in rows):
                raise RuntimeError(f'{layer} endpoints are not on the exact source surface layer')
            layer_edges[layer] = rows
            for cid, row in zip(columns, rows):
                mapping = column_layer_rows.setdefault(cid, {})
                if layer in mapping and mapping[layer] != row:
                    raise RuntimeError(f'column {cid} has inconsistent {layer} source row')
                mapping[layer] = row
        # Independent row-order check: batch source offsets reproduce the original
        # 084 endpoints before the 115 corrected D reference/current field is applied.
        for layer in ('I', 'D'):
            rows = layer_edges[layer]
            got_ref = old_reference[list(rows)].astype(np.float64) - old_reference[list(o_rows)].astype(np.float64)
            got_cur = old_current[list(rows)].astype(np.float64) - old_current[list(o_rows)].astype(np.float64)
            expected_ref = np.asarray(rec['midpoint_layer_positions'][layer]['offset_from_O_reference_endpoints_m'], dtype=np.float64)
            expected_cur = np.asarray(rec['midpoint_layer_positions'][layer]['offset_from_O_current037_endpoints_m'], dtype=np.float64)
            if not np.array_equal(got_ref, expected_ref):
                if np.max(np.abs(got_ref - expected_ref)) > 2e-12:
                    raise RuntimeError(f'{layer} reference endpoint offset crosswalk failed')
            if not np.array_equal(got_cur, expected_cur):
                if np.max(np.abs(got_cur - expected_cur)) > 2e-12:
                    raise RuntimeError(f'{layer} current endpoint offset crosswalk failed')
        for frame_name, midpoint_key in (('reference', 'proposed_reference_position_m'),
                                         ('current037', 'proposed_current037_position_m_f64')):
            sample = rec['midpoint_layer_positions']['O'][midpoint_key]
            source = {
                'source_kind': 'detailed_skin_surface_path',
                'source_face_rows': rec['path_selected_source_face_rows'],
                'source_path_vertex_ids': rec['source_path_vertex_ids'],
                'edge_endpoint_source_vertex_ids': rec['endpoint_detail_source_vertex_ids'],
                'sample_point_world_m': sample,
                'ambiguity_count': rec['ambiguity_count'],
                'minimum_normal_cosine': rec['minimum_normal_cosine'],
                'path_selection_rule': batch['method']['path_choice'],
                'normal_orientation_status': (
                    'reversed' if rec['normal_orientation'].get('consistent_path_parity') == -1
                    else 'aligned' if rec['normal_orientation'].get('consistent_path_parity') == 1
                    else 'unresolved'
                ),
                'path_detail_rows': rec['path_detail_rows'],
                'normal_orientation': rec['normal_orientation'],
                'source_face_rows_reported': rec['source_face_rows'],
                'parent_source_O_face_rows': rec['parent_source_O_face_rows'],
                'anchor_method': rec['anchor_method'],
                'endpoint_anchor_options': rec['endpoint_anchor_options'],
                'ambiguity_flags': rec['ambiguity_flags'],
            }
            rec.setdefault('_adapted_outer', {})[frame_name] = source
        selections.append({
            'column_ids': list(columns),
            'column_endpoint_detail_vertex_ids': rec['endpoint_detail_source_vertex_ids'],
            'layer_edges': {layer: list(edge) for layer, edge in layer_edges.items()},
            'sample_fraction': float(rec['path_arclength_fraction']),
            'outer_source_by_frame': rec['_adapted_outer'],
            'batch_edge_global_node_ids': columns,
            'source_path_detail_rows': rec['path_detail_rows'],
            'source_face_rows': rec['path_selected_source_face_rows'],
            'source_parent_O_face_rows': rec['parent_source_O_face_rows'],
            'normal_orientation': rec['normal_orientation'],
            'ambiguity_flags': rec['ambiguity_flags'],
        })
        eligible_records.append(rec)
    for cid, layer_rows in column_layer_rows.items():
        if set(layer_rows) != {'O', 'I', 'D'}:
            raise RuntimeError(f'column {cid} missing explicit O/I/D rows')
    if len(selections) != 123 or len(held_records) != 22:
        raise RuntimeError(f'eligible/held edge counts changed: {len(selections)}/{len(held_records)}')

    sys.path.insert(0, str(TOOL))
    from human_skin_fat_boundary_refinement import subdivide_registered_column_edges
    field_sha = array_sha(points_by_frame['reference'])
    mapping_sha = hashlib.sha256(json.dumps(
        {str(k): column_layer_rows[k] for k in sorted(column_layer_rows)},
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    result = subdivide_registered_column_edges(
        points_by_frame, faces_by_layer, parent_markers, column_layer_rows, selections,
        reference_field_sha256=field_sha,
        source_column_mapping_sha256=mapping_sha,
    )
    # Full source PLC surface: refined O/I/D parent facets plus unchanged side caps.
    face_arrays = [result.faces_by_layer[layer] for layer in ('O', 'I', 'D')]
    marker_arrays = [result.child_parent_face_markers_by_layer[layer] for layer in ('O', 'I', 'D')]
    parent_arrays = [result.child_parent_face_rows_by_layer[layer] for layer in ('O', 'I', 'D')]
    surface_id_arrays = [np.full(len(face_arrays[i]), i, dtype=np.int8) for i in range(3)]
    face_arrays.append(cap_faces.astype(np.int32)); marker_arrays.append(cap_markers.astype(np.int64))
    parent_arrays.append(cap_parent_rows); surface_id_arrays.append(np.full(len(cap_faces), 3, dtype=np.int8))
    all_faces = np.concatenate(face_arrays, axis=0).astype(np.int32)
    all_markers = np.concatenate(marker_arrays).astype(np.int32)
    all_parent_rows = np.concatenate(parent_arrays).astype(np.int64)
    all_surface_ids = np.concatenate(surface_id_arrays)
    if np.any(all_faces < 0) or np.any(all_faces >= len(result.points_by_frame['reference'])):
        raise RuntimeError('output PLC face references an invalid point row')
    if len(all_faces) != len(all_markers) or len(all_faces) != len(all_parent_rows):
        raise RuntimeError('output face provenance arrays are not aligned')

    # Keep variable-length source path and face provenance in compact CSR arrays.
    path_lists = []
    path_indptr = [0]
    source_face_lists = []
    source_face_indptr = [0]
    for edge in result.selected_outer_edges:
        key = tuple(map(int, edge))
        rec = next(x for x in eligible_records if tuple(sorted(map(int, x['edge_global_node_ids']))) == key)
        path = list(map(int, rec['path_detail_rows']))
        src_faces = list(map(int, rec['path_selected_source_face_rows']))
        path_lists.extend(path); path_indptr.append(len(path_lists))
        source_face_lists.extend(src_faces); source_face_indptr.append(len(source_face_lists))
    edge_to_rec = {tuple(sorted(map(int, x['edge_global_node_ids']))): x for x in eligible_records}
    edge_columns = np.asarray([edge_to_rec[tuple(map(int, edge))]['edge_global_node_ids'] for edge in result.selected_outer_edges], dtype=np.int64)
    edge_detail_endpoints = np.asarray([edge_to_rec[tuple(map(int, edge))]['endpoint_detail_source_vertex_ids'] for edge in result.selected_outer_edges], dtype=np.int64)
    edge_parent_faces = np.asarray([edge_to_rec[tuple(map(int, edge))]['parent_source_O_face_rows'] for edge in result.selected_outer_edges], dtype=np.int64)
    edge_batch_O_rows = np.asarray([edge_to_rec[tuple(map(int, edge))]['edge_output_rows'] for edge in result.selected_outer_edges], dtype=np.int64)
    edge_layer_rows = np.asarray([[result.source_layer_edges_by_edge[i][layer] for layer in ('O', 'I', 'D')]
                                  for i in range(len(result.selected_outer_edges))], dtype=np.int64)
    edge_midpoint_rows = np.asarray([[result.midpoint_rows_by_layer[layer][i] for layer in ('O', 'I', 'D')]
                                     for i in range(len(result.selected_outer_edges))], dtype=np.int64)
    edge_fractions = result.sample_fractions_from_path_start.astype(np.float64)
    edge_weights = result.source_column_weights_by_edge.astype(np.float64)
    edge_ambiguity = np.asarray([edge_to_rec[tuple(map(int, edge))]['ambiguity_count'] for edge in result.selected_outer_edges], dtype=np.int32)
    edge_normal_parity = np.asarray([edge_to_rec[tuple(map(int, edge))]['normal_orientation']['consistent_path_parity'] for edge in result.selected_outer_edges], dtype=np.int8)
    edge_min_cos = np.asarray([edge_to_rec[tuple(map(int, edge))]['minimum_normal_cosine'] for edge in result.selected_outer_edges], dtype=np.float64)

    # Compare the regenerated offset samples with the old 110 suggestions. The I/D
    # differences are expected where 115 changed D; no old offsets are copied.
    max_batch_midpoint_delta = {layer: {'reference_m': 0.0, 'current037_m': 0.0} for layer in ('O', 'I', 'D')}
    for i, edge in enumerate(result.selected_outer_edges):
        rec = edge_to_rec[tuple(map(int, edge))]
        for li, layer in enumerate(('O', 'I', 'D')):
            row = int(edge_midpoint_rows[i, li])
            for frame, batch_key, out_key in (
                ('reference', 'proposed_reference_position_m', 'reference_m'),
                ('current037', 'proposed_current037_position_m_f64', 'current037_m'),
            ):
                old = np.asarray(rec['midpoint_layer_positions'][layer][batch_key], dtype=np.float64)
                new = result.points_by_frame[frame][row].astype(np.float64)
                delta = float(np.linalg.norm(new - old))
                max_batch_midpoint_delta[layer][out_key] = max(max_batch_midpoint_delta[layer][out_key], delta)

    # Build provenance rows in exactly the selected O edge order.
    edge_provenance = []
    for i, edge in enumerate(result.selected_outer_edges):
        rec = edge_to_rec[tuple(map(int, edge))]
        edge_provenance.append({
            'edge_global_column_ids': list(map(int, rec['edge_global_node_ids'])),
            'layer_endpoint_rows': {layer: list(map(int, result.source_layer_edges_by_edge[i][layer])) for layer in ('O', 'I', 'D')},
            'detail_endpoint_vertex_ids': rec['endpoint_detail_source_vertex_ids'],
            'detail_path_vertex_ids': rec['source_path_vertex_ids'],
            'detail_path_rows': rec['path_detail_rows'],
            'selected_detail_source_face_rows': rec['path_selected_source_face_rows'],
            'coarse_source_O_parent_face_rows': rec['parent_source_O_face_rows'],
            'path_arclength_fraction': rec['path_arclength_fraction'],
            'source_path_tie_count': rec['ambiguity_count'],
            'path_selection_rule': batch['method']['path_choice'],
            'normal_orientation': rec['normal_orientation'],
            'eligible': True,
        })

    OUT.mkdir(parents=True)
    candidate = OUT / 'surface-proposal.npz'
    np.savez_compressed(
        candidate,
        reference_points_m=result.points_by_frame['reference'],
        current037_points_m_f32=result.points_by_frame['current037'],
        surface_faces_zero_based=all_faces,
        surface_face_markers=all_markers,
        surface_face_parent_rows=all_parent_rows,
        surface_face_surface_ids=all_surface_ids,
        source_O_faces=result.faces_by_layer['O'].astype(np.int32),
        source_I_faces=result.faces_by_layer['I'].astype(np.int32),
        source_D_faces=result.faces_by_layer['D'].astype(np.int32),
        source_O_child_parent_rows=result.child_parent_face_rows_by_layer['O'],
        source_I_child_parent_rows=result.child_parent_face_rows_by_layer['I'],
        source_D_child_parent_rows=result.child_parent_face_rows_by_layer['D'],
        source_O_child_parent_markers=result.child_parent_face_markers_by_layer['O'],
        source_I_child_parent_markers=result.child_parent_face_markers_by_layer['I'],
        source_D_child_parent_markers=result.child_parent_face_markers_by_layer['D'],
        selected_outer_edges=result.selected_outer_edges.astype(np.int64),
        selected_edge_global_column_ids=edge_columns,
        selected_edge_detail_endpoint_vertex_ids=edge_detail_endpoints,
        selected_edge_layer_endpoint_rows=edge_layer_rows,
        selected_edge_midpoint_rows=edge_midpoint_rows,
        selected_edge_path_fraction=edge_fractions,
        selected_edge_column_weights=edge_weights,
        selected_edge_source_path_tie_count=edge_ambiguity,
        selected_edge_normal_parity=edge_normal_parity,
        selected_edge_minimum_normal_cosine=edge_min_cos,
        source_path_indptr=np.asarray(path_indptr, dtype=np.int64),
        source_path_detail_vertex_ids=np.asarray(path_lists, dtype=np.int64),
        source_face_indptr=np.asarray(source_face_indptr, dtype=np.int64),
        selected_detail_source_face_rows=np.asarray(source_face_lists, dtype=np.int64),
        cap_faces_zero_based=cap_faces.astype(np.int32),
        cap_face_markers=cap_markers.astype(np.int32),
        cap_face_parent_rows=cap_parent_rows.astype(np.int64),
    )
    report = {
        'status': 'surface_only_conforming_subdivision_proposal',
        'interpretation': 'Construction proposal only. No TetGen, tetrahedralization, collision, capacity, conservative mass transfer, native run, or physical admission was performed.',
        'input_sha256': {**actual, 'reference_field_array': field_sha, 'source_column_mapping': mapping_sha},
        'source_field': {
            'reference': 'connected-capacity-reference-115/candidate.npz::proposed_reference_points_m',
            'current037': 'connected-capacity-reference-115/candidate.npz::points_world_f32',
            'source_reference_exactly_matches_084': True,
            'source_current037_exactly_matches_084': True,
            'D_rows_modified_by_115': int(len(r115['modified_D_rows'])),
            'max_abs_084_to_115_reference_delta_m': float(np.max(np.abs(r115['proposed_reference_points_m'] - r115['source_reference_points_m']))),
        },
        'midpoint_source_batch': {
            'outer_surface_edges': 145,
            'eligible_used': len(selections),
            'orientation_unresolved_held': len(held_records),
            'excluded_internal_chords': batch['excluded_internal_chords'],
            'shortest_path_ties_used_with_recorded_deterministic_rule': int(sum(int(r['ambiguity_count']) > 0 for r in eligible_records)),
            'eligible_normal_parity_counts': {
                str(int(p)): int(sum(int(r['normal_orientation']['consistent_path_parity']) == int(p) for r in eligible_records))
                for p in sorted({int(r['normal_orientation']['consistent_path_parity']) for r in eligible_records})
            },
            'held_out_edge_ids_and_status': [
                {'edge_global_node_ids': r['edge_global_node_ids'], 'status': r['status'],
                 'ambiguity_flags': r['ambiguity_flags'], 'minimum_normal_cosine': r['minimum_normal_cosine']}
                for r in held_records
            ],
            'max_regenerated_midpoint_delta_vs_old_batch_suggestion_m': max_batch_midpoint_delta,
        },
        'topology': {
            'input_node_rows': int(points_by_frame['reference'].shape[0]),
            'appended_rows': int(len(result.points_by_frame['reference']) - points_by_frame['reference'].shape[0]),
            'appended_rows_per_layer': {'O': len(selections), 'I': len(selections), 'D': len(selections)},
            'input_source_faces_per_layer': {layer: int(len(faces_by_layer[layer])) for layer in ('O', 'I', 'D')},
            'output_source_faces_per_layer': {layer: int(len(result.faces_by_layer[layer])) for layer in ('O', 'I', 'D')},
            'unchanged_side_cap_faces': int(len(cap_faces)),
            'combined_plc_surface_faces': int(len(all_faces)),
            'original_reference_prefix_bitwise_preserved': bool(np.array_equal(result.points_by_frame['reference'][:7360], r115['proposed_reference_points_m'])),
            'original_current037_prefix_bitwise_preserved': bool(np.array_equal(result.points_by_frame['current037'][:7360], r115['points_world_f32'])),
            'child_parent_rows_and_markers_retained': True,
            'tetrahedra_included': False,
        },
        'selected_edges': edge_provenance,
        'candidate_sha256': sha(candidate),
        'candidate_size_bytes': candidate.stat().st_size,
    }
    report_path = OUT / 'proposal-report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        'builder_sha256': sha(Path(__file__)),
        'refiner_sha256': sha(TOOL / 'human_skin_fat_boundary_refinement.py'),
        'test_sha256': sha(TOOL / 'test_human_skin_fat_boundary_refinement.py'),
        'candidate_sha256': sha(candidate),
        'report_sha256': sha(report_path),
        'input_sha256': actual,
        'no_geometry_or_native_run': True,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'candidate': str(candidate), 'candidate_sha256': manifest['candidate_sha256'],
        'report': str(report_path), 'report_sha256': manifest['report_sha256'],
        'manifest': str(OUT / 'manifest.json'), 'manifest_sha256': sha(OUT / 'manifest.json'),
        'selected': len(selections), 'held_orientation': len(held_records),
        'nodes': len(result.points_by_frame['reference']), 'faces': len(all_faces),
        'midpoint_delta': max_batch_midpoint_delta,
    }, indent=2))

if __name__ == '__main__':
    main()
