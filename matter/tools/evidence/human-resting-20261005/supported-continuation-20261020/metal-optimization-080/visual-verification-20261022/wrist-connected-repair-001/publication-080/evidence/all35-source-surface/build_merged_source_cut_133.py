#!/usr/bin/env python3
import csv, gzip, hashlib, importlib.util, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

BASE = Path('/Users/numi/Work/numi-human-connected-surface-069')
ROOT = BASE / 'connected-source-patch-cut-123'
OLD_DIR = ROOT / 'run-019-route-harmonized-old-input'
NEW_DIR = ROOT / 'run-016-best-reroute-expanded-cut'
COARSE = BASE / 'publication-072/evidence/whole-reference-TetGen-084/whole-skin-fat-reference-084-candidate.npz'
DETAIL = BASE / 'publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz'
ACCEPTED = BASE / 'publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-reference.csv.gz'
OUT = ROOT / 'run-026-global-seed-closure-source-cut'
EXPECTED = {
    'coarse': 'd3d0abb590f5858de4a27e0362e98bc9b3088831183ea045096cefc4ebba26e1',
    'detail': '21f1869b418a2eca1e5a0522059c8f9cc6f8afe1a270ccc7c7d99c9c19fdd2a4',
    'old_cut': '4a065fea7f347e5c2c7c9a1c82f2a598af509a4cce8fde1cf4ec32068abc2b61',
    'new_cut': '21e2b65d85ba9bac176aaa670b916bc13189975a295c9ab328d1e835bb2e0b78',
    'old_report': '292d68c19d326a4c39570e2d2cb8c109c6e42ea8dffb91fd83f0d9d0d11d1646',
    'new_report': 'd9f22dc8fda1ae6a3db5950c313b096871db2c9814fa09ad27987a2f9d7f8684',
    'accepted': '00b24dd1f09d3f890cf085ceeb1665715a2de7e116275e84165b5c14f43585d4',
}
OLD_COMPONENTS = [0, 3, 4, 5, 6, *range(8, 31), 32, 33, 34]
NEW_COMPONENTS = [0, 1, 2]
ALT_PATHS = {
    (310, 351): [1673, 1468, 1493, 1492, 1513, 1514, 1709, 1904, 1912, 2233, 2588, 3194],
    (2208, 2241): [49898, 49896, 50385, 51123, 51124, 51335, 51333],
    (2241, 2246): [51333, 51335, 51125, 51337, 51338, 51341, 51342, 51602, 51598, 51604, 51910, 51912, 51609],
}
FAN_ANCHORS = [351, 2241, 232, 240, 150, 229, 142, 141]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def ek(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def load_graft_module():
    src = ROOT / 'graft_hybrid_O_surface_129e.py'
    spec = importlib.util.spec_from_file_location('graft129e', src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    require(not OUT.exists(), f'output exists: {OUT}')
    input_paths = {
        'coarse': COARSE,
        'detail': DETAIL,
        'old_cut': OLD_DIR / 'source-patch-cut.npz',
        'new_cut': NEW_DIR / 'source-patch-cut.npz',
        'old_report': OLD_DIR / 'source-patch-cut-report.json',
        'new_report': NEW_DIR / 'source-patch-cut-report.json',
        'accepted': ACCEPTED,
    }
    pins = {k: sha(v) for k, v in input_paths.items()}
    for k, v in EXPECTED.items():
        require(pins[k] == v, f'{k} hash mismatch: {pins[k]}')

    graft = load_graft_module()
    old = np.load(input_paths['old_cut'], allow_pickle=False)
    new = np.load(input_paths['new_cut'], allow_pickle=False)
    old_report = json.load(open(input_paths['old_report']))
    new_report = json.load(open(input_paths['new_report']))
    coarse = np.load(COARSE, allow_pickle=False)
    detail = np.load(DETAIL, allow_pickle=False)
    coarse_faces = np.asarray(coarse['source_O_faces'], dtype=np.int32)
    coarse_points = np.asarray(coarse['reference_points_m'], dtype=np.float64)
    fine_faces = np.asarray(old['fine_subfaces'], dtype=np.int32)
    fine_parent = np.asarray(old['fine_subface_parent_rows'], dtype=np.int32)
    require(np.array_equal(fine_parent, np.asarray(old['fine_subface_parent_rows'], dtype=np.int32)), 'bad source parent map')

    old_patch = graft.selected_coarse_rows(old, OLD_COMPONENTS)
    new_patch = graft.selected_coarse_rows(new, NEW_COMPONENTS)
    patch = set(old_patch) | set(new_patch)

    # Full incident-face fans convert the two shared route anchors into patch interiors.
    all_edge_faces = defaultdict(list)
    incident_faces = defaultdict(set)
    for fi, tri in enumerate(coarse_faces):
        for v in tri:
            incident_faces[int(v)].add(fi)
        for a, b in zip(tri, np.roll(tri, -1)):
            all_edge_faces[ek(a, b)].append(fi)
    fan_counts = {}
    for v in FAN_ANCHORS:
        fan_counts[str(v)] = len(incident_faces[v])
        patch.update(incident_faces[v])

    # Connected coarse patch groups and oriented external edges. Expand only complete
    # incident fans at branch/touch vertices until every merged patch boundary is a
    # disjoint union of cycles; this removes shared-anchor branches from the boundary.
    fan_expansions = []
    def make_groups_and_boundaries(face_set):
        patch_edge_faces = defaultdict(list)
        for fi in face_set:
            for a, b in zip(coarse_faces[fi], np.roll(coarse_faces[fi], -1)):
                patch_edge_faces[ek(a, b)].append(int(fi))
        unseen = set(face_set)
        result_groups = []
        while unseen:
            seed = unseen.pop()
            comp = {seed}
            stack = [seed]
            while stack:
                fi = stack.pop()
                for a, b in zip(coarse_faces[fi], np.roll(coarse_faces[fi], -1)):
                    for fj in patch_edge_faces[ek(a, b)]:
                        if fj in unseen:
                            unseen.remove(fj)
                            comp.add(fj)
                            stack.append(fj)
            result_groups.append(comp)
        result_edges = []
        result_direction = {}
        branch_vertices = set()
        for comp in result_groups:
            boundary = []
            for edge, incident in all_edge_faces.items():
                inside = [fi for fi in incident if fi in comp]
                if len(inside) == 1:
                    fi = inside[0]
                    tri = coarse_faces[fi]
                    direction = next((int(a), int(b)) for a, b in zip(tri, np.roll(tri, -1)) if ek(a, b) == edge)
                    boundary.append((edge, direction))
                    require(edge not in result_direction, f'coarse patch groups share boundary edge {edge}')
                    result_direction[edge] = direction
            outgoing = defaultdict(int)
            incoming = defaultdict(int)
            for _, (a, b) in boundary:
                outgoing[a] += 1
                incoming[b] += 1
            for v in set(outgoing) | set(incoming):
                if outgoing[v] != 1 or incoming[v] != 1:
                    branch_vertices.add(v)
            result_edges.append(boundary)
        return result_groups, result_edges, result_direction, branch_vertices

    # Prescribed fans first, then local full-fan closure at related touch vertices.
    for iteration in range(len(coarse_points)):
        groups, group_edges, boundary_direction, branch_vertices = make_groups_and_boundaries(patch)
        if not branch_vertices:
            break
        added = set()
        for v in sorted(branch_vertices):
            added.update(incident_faces[v] - patch)
        require(added, f'boundary remains branched after full fans at {sorted(branch_vertices)}')
        patch.update(added)
        fan_expansions.append({'iteration': iteration, 'branch_vertices': sorted(map(int, branch_vertices)),
                               'added_coarse_faces': sorted(map(int, added))})
    else:
        raise RuntimeError('full-fan boundary closure did not converge')
    groups, group_edges, boundary_direction, branch_vertices = make_groups_and_boundaries(patch)
    require(not branch_vertices, f'boundary fan closure left branches {sorted(branch_vertices)}')
    # Collect original registered paths; only exact reverse duplicates may coincide.
    routes = {}
    route_origin = {}
    def add_routes(cut, component_ids, label):
        rc = np.asarray(cut['route_segment_component'], dtype=np.int64)
        ce = np.asarray(cut['route_segment_coarse_endpoints'], dtype=np.int64)
        ip = np.asarray(cut['route_path_indptr'], dtype=np.int64)
        pn = np.asarray(cut['route_path_nodes'], dtype=np.int32)
        for i, ci in enumerate(rc):
            if int(ci) not in component_ids:
                continue
            key = ek(*ce[i])
            path = pn[ip[i]:ip[i + 1]].copy()
            prior = routes.get(key)
            require(prior is None or np.array_equal(path, prior) or np.array_equal(path[::-1], prior),
                    f'incompatible route definitions for shared coarse edge {key}')
            if prior is None:
                routes[key] = path
                route_origin[key] = label
    add_routes(old, set(OLD_COMPONENTS), 'old-harmonized')
    add_routes(new, set(NEW_COMPONENTS), 'expanded-016')
    for key, path in ALT_PATHS.items():
        key = ek(*key)
        if key in boundary_direction:
            routes[key] = np.asarray(path, dtype=np.int32)
            route_origin[key] = 'bounded-reroute-prior'

    # Accepted coarse references provide direct source IDs; the four exact virtual
    # anchors are carried by the old frozen cut.
    direct = np.full(len(coarse_points), -1, dtype=np.int64)
    with gzip.open(ACCEPTED, 'rt', newline='') as f:
        for row in csv.DictReader(f):
            i = int(row['coarse_row'])
            d = int(row['source_direct_id'])
            if 0 <= d < 4294967295:
                direct[i] = d
    virtual_anchor = {int(cr): int(node) for cr, node in zip(old['virtual_anchor_coarse_rows'], old['virtual_anchor_nodes'])}
    def anchor(row):
        row = int(row)
        if direct[row] >= 0:
            return int(direct[row])
        require(row in virtual_anchor, f'no exact accepted source anchor for coarse row {row}')
        return virtual_anchor[row]

    # Build the detailed source graph once and route only the four newly exposed fan edges.
    detail_points = np.asarray(detail['detail_baseline_world_m'], dtype=np.float64)
    coords = np.vstack([detail_points, np.asarray(old['virtual_anchor_points_world_m'], dtype=np.float64)])
    fine_edges = set()
    for tri in fine_faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            fine_edges.add(ek(a, b))
    edge_array = np.asarray(sorted(fine_edges), dtype=np.int32)
    lengths = np.linalg.norm(coords[edge_array[:, 0]] - coords[edge_array[:, 1]], axis=1)
    ncoords = len(coords)
    graph = coo_matrix(
        (np.r_[lengths, lengths],
         (np.r_[edge_array[:, 0], edge_array[:, 1]], np.r_[edge_array[:, 1], edge_array[:, 0]])),
        shape=(ncoords, ncoords)).tocsr()

    missing = sorted(set(boundary_direction) - set(routes))
    new_route_keys = []
    for key in missing:
        # Use the oriented coarse face edge direction for the path.
        direction = boundary_direction[key]
        start, end = anchor(direction[0]), anchor(direction[1])
        dist, pred = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
        require(np.isfinite(dist[end]), f'no source graph path for newly exposed edge {key}')
        path = [end]
        x = end
        while x != start:
            x = int(pred[x])
            require(x >= 0, f'broken predecessor chain for new edge {key}')
            path.append(x)
        path = np.asarray(path[::-1], dtype=np.int32)
        routes[key] = path
        route_origin[key] = 'shortest-source-graph-after-full-fan'
        new_route_keys.append(key)

    # Remove routes that became internal after face-fan union, then orient every path.
    oriented = {}
    path_lengths = {}
    for key, direction in boundary_direction.items():
        path = routes[key]
        start, end = anchor(direction[0]), anchor(direction[1])
        if int(path[0]) == end and int(path[-1]) == start:
            path = path[::-1]
        require(int(path[0]) == start and int(path[-1]) == end,
                f'path endpoint mismatch for coarse edge {key}: expected {(start, end)}, got {(int(path[0]), int(path[-1]))}')
        for a, b in zip(path[:-1], path[1:]):
            require(ek(a, b) in fine_edges, f'route uses non-source edge {a}-{b} for coarse edge {key}')
        oriented[key] = path
        path_lengths[str(key)] = {'nodes': int(len(path)), 'source_length_m': float(np.linalg.norm(np.diff(coords[path], axis=0), axis=1).sum()),
                                  'origin': route_origin[key]}

    # There may be shared coarse endpoints, but no route interiors or source edges may be multiply owned.
    interior_owners = defaultdict(list)
    source_edge_owners = defaultdict(list)
    for ce, path in oriented.items():
        for v in path[1:-1]:
            interior_owners[int(v)].append(ce)
        for a, b in zip(path[:-1], path[1:]):
            source_edge_owners[ek(a, b)].append(ce)
    shared_interiors = {str(k): v for k, v in interior_owners.items() if len(v) > 1}
    shared_source_edges = {str(k): v for k, v in source_edge_owners.items() if len(v) > 1}
    require(not shared_interiors, f'shared interior source path nodes remain: {list(shared_interiors.items())[:5]}')
    require(not shared_source_edges, f'shared source path edges remain: {list(shared_source_edges.items())[:5]}')

    # Directed boundary edges must form disjoint simple cycles for every coarse group.
    loop_records = []
    segment_records = []
    for gi, (comp, boundary) in enumerate(zip(groups, group_edges)):
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for edge, (a, b) in boundary:
            outgoing[a].append((b, edge))
            incoming[b].append((a, edge))
        vertices = set(outgoing) | set(incoming)
        bad = {v: (len(outgoing[v]), len(incoming[v])) for v in vertices if len(outgoing[v]) != 1 or len(incoming[v]) != 1}
        require(not bad, f'coarse group {gi} boundary is not a set of simple cycles: {bad}')
        left = {edge for edge, _ in boundary}
        loop_id = 0
        while left:
            first = min(left)
            start_edge = next(pair for pair in boundary if pair[0] == first)
            edge = first
            start_vertex = start_edge[1][0]
            cycle_vertices = []
            cycle_edges = []
            current = start_vertex
            while True:
                candidates = [(e, d) for e, d in boundary if e in left and d[0] == current]
                require(len(candidates) == 1, f'group {gi} cycle traversal branch at coarse row {current}')
                e, (a, b) = candidates[0]
                cycle_vertices.append(a)
                cycle_edges.append(e)
                left.remove(e)
                current = b
                if current == start_vertex:
                    break
                require(len(cycle_edges) <= len(boundary), f'group {gi} cycle failed to close')
            anchors = [anchor(v) for v in cycle_vertices]
            loop_records.append({
                'component': gi, 'loop': loop_id, 'coarse_faces': sorted(map(int, comp)),
                'coarse_loop': list(map(int, cycle_vertices)), 'anchor_nodes': list(map(int, anchors)),
                'boundary_edge_count': len(cycle_edges), 'status': 'closed_disjoint_source_edge_cycle',
                'fine_group_boundary_expected': True,
            })
            for e, a, b in zip(cycle_edges, cycle_vertices, cycle_vertices[1:] + cycle_vertices[:1]):
                require(boundary_direction[e] == (a, b), 'coarse cycle direction mismatch')
                segment_records.append({'component': gi, 'loop': loop_id, 'coarse_start': int(a), 'coarse_end': int(b),
                                        'source_path_origin': route_origin[e]})
            loop_id += 1

    # Store routes in the same established cut NPZ format, but clearly mark this as a
    # derived union cut rather than an upstream/native source cut.
    route_edges = []
    segment_component, segment_loop, segment_endpoints = [], [], []
    route_indptr, route_nodes = [0], []
    routed_fine_edges = set()
    for rec in segment_records:
        a, b = int(rec['coarse_start']), int(rec['coarse_end'])
        path = oriented[ek(a, b)]
        require(int(path[0]) == anchor(a) and int(path[-1]) == anchor(b), 'route cycle orientation lost')
        segment_component.append(int(rec['component']))
        segment_loop.append(int(rec['loop']))
        segment_endpoints.append([a, b])
        route_nodes.extend(map(int, path))
        route_indptr.append(len(route_nodes))
        for x, y in zip(path[:-1], path[1:]):
            routed_fine_edges.add(ek(x, y))
        route_edges.append(ek(a, b))

    # Verify the exact fine-side source component for every merged coarse group.
    edge_faces = defaultdict(list)
    for fi, tri in enumerate(fine_faces):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_faces[ek(a, b)].append(fi)
    owner = np.full(len(fine_faces), -1, dtype=np.int32)
    fine_components = []
    for start in range(len(fine_faces)):
        if owner[start] >= 0:
            continue
        ci = len(fine_components)
        owner[start] = ci
        stack, members = [start], []
        while stack:
            fi = stack.pop()
            members.append(fi)
            for a, b in zip(fine_faces[fi], np.roll(fine_faces[fi], -1)):
                ee = ek(a, b)
                if ee in routed_fine_edges:
                    continue
                for fj in edge_faces[ee]:
                    if owner[fj] < 0:
                        owner[fj] = ci
                        stack.append(fj)
        fine_components.append(members)
    component_boundaries = [set() for _ in fine_components]
    for ee, adjacent in edge_faces.items():
        ids = {int(owner[fi]) for fi in adjacent}
        if len(ids) > 1 or len(adjacent) == 1:
            for ci in ids:
                component_boundaries[ci].add(ee)
    fine_group_rows = []
    for gi, boundary in enumerate(group_edges):
        expected_edges = set()
        for edge, _ in boundary:
            path = oriented[edge]
            expected_edges.update(ek(a, b) for a, b in zip(path[:-1], path[1:]))
        matches = [ci for ci, bdy in enumerate(component_boundaries) if bdy == expected_edges]
        require(len(matches) == 1, f'merged coarse group {gi} has {len(matches)} matching fine-side components')
        fine_group_rows.append(sorted(map(int, fine_components[matches[0]])))

    # Resolve original 35 seed-group coverage explicitly; no global completion claim.
    original_seed_faces = set(map(int, old['seed_source_O_face_rows']))
    seed_covered = original_seed_faces & patch
    synthetic_cut = {
        'seed_source_O_face_rows': np.asarray(sorted(original_seed_faces), dtype=np.int32),
        'patch_component_indptr': np.asarray([0] + list(np.cumsum([len(sorted(g)) for g in groups])), dtype=np.int64),
        'patch_component_face_rows': np.asarray([x for group in groups for x in sorted(group)], dtype=np.int32),
        'virtual_anchor_nodes': np.asarray(old['virtual_anchor_nodes'], dtype=np.int32),
        'virtual_anchor_coarse_rows': np.asarray(old['virtual_anchor_coarse_rows'], dtype=np.int32),
        'virtual_anchor_points_world_m': np.asarray(old['virtual_anchor_points_world_m'], dtype=np.float64),
        'virtual_anchor_source_face_detail_rows': np.asarray(old['virtual_anchor_source_face_detail_rows'], dtype=np.int32),
        'virtual_anchor_barycentric': np.asarray(old['virtual_anchor_barycentric'], dtype=np.float64),
        'route_segment_component': np.asarray(segment_component, dtype=np.int32),
        'route_segment_loop': np.asarray(segment_loop, dtype=np.int32),
        'route_segment_coarse_endpoints': np.asarray(segment_endpoints, dtype=np.int32),
        'route_path_indptr': np.asarray(route_indptr, dtype=np.int64),
        'route_path_nodes': np.asarray(route_nodes, dtype=np.int32),
        'routed_cut_edges': np.asarray(sorted(routed_fine_edges), dtype=np.int32),
        'fine_subfaces': fine_faces,
        'fine_subface_parent_rows': fine_parent,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    cut_path = OUT / 'merged-source-patch-cut.npz'
    np.savez_compressed(cut_path, **synthetic_cut)
    loop_report = {
        'status': 'derived_merged_source_cut_topology_checked',
        'interpretation': 'Derived union of retained old patch components and expanded run-016 components, plus complete incident coarse face fans at shared rows 351 and 2241. New outer boundary routes use shortest paths on the frozen registered fine source graph. This is a surface construction input, not physical/native admission.',
        'inputs': {k: {'path': str(v), 'sha256': pins[k]} for k, v in input_paths.items()},
        'component_policy': {'old_retained_components': OLD_COMPONENTS, 'old_excluded_components': [1, 2, 7, 31],
                             'expanded_components': NEW_COMPONENTS, 'fan_anchor_rows': FAN_ANCHORS, 'fan_face_counts': fan_counts, 'related_touch_fan_expansions': fan_expansions},
        'coverage': {'original_seed_face_count': len(original_seed_faces), 'covered_original_seed_faces': len(seed_covered),
                     'uncovered_original_seed_faces': sorted(original_seed_faces - patch),
                     'original_seed_group_count': 35, 'coarse_patch_face_count_before_fans': len(old_patch | new_patch),
                     'coarse_patch_face_count_after_fans': len(patch)},
        'routes': {'external_coarse_edges': len(boundary_direction), 'shortest_paths_added_for_fans': len(new_route_keys),
                   'new_shortest_paths': {str(k): int(len(oriented[k])) for k in new_route_keys},
                   'source_route_nodes_shared_interior': 0, 'source_route_edges_shared': 0,
                   'route_lengths': path_lengths},
        'fine_side': {'fine_face_component_count_after_all_routes': len(fine_components),
                      'merged_coarse_group_count': len(groups), 'exact_boundary_matched_groups': len(fine_group_rows),
                      'selected_fine_face_count': sum(len(x) for x in fine_group_rows),
                      'fine_face_rows_by_coarse_group': fine_group_rows},
        'outputs': {'merged_cut_npz': str(cut_path), 'merged_cut_sha256': sha(cut_path)},
        'routed_segments': segment_records,
        'patch_loops': loop_records,
    }
    report_path = OUT / 'merged-source-patch-cut-report.json'
    report_path.write_text(json.dumps(loop_report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'directory': str(OUT), 'cut_sha256': loop_report['outputs']['merged_cut_sha256'],
                      'report': str(report_path), 'seed_coverage': loop_report['coverage'],
                      'groups': loop_report['fine_side']['merged_coarse_group_count'],
                      'fine_faces': loop_report['fine_side']['selected_fine_face_count'],
                      'routes_added': loop_report['routes']['new_shortest_paths_added_for_fans'] if 'new_shortest_paths_added_for_fans' in loop_report['routes'] else len(new_route_keys)},
                     indent=2))

if __name__ == '__main__':
    main()
