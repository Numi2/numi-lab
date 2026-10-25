"""Fresh nearest-boundary first-union exit proposals for the exact-gated Run032 best trial."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
D = Path('/Users/n/Work/numi-human-connected-surface-069')
ROOT = D / 'full-connected-clearance-150'
CAND = ROOT / 'run-034-progress-merit-connected-scp-001/attempt-001/candidate.npz'
QUERY = ROOT / 'run-034-progress-merit-connected-scp-001/attempt-001/scp-02-alpha-0p25/audit-130/query'
BASE = ROOT / 'run-028-current-target-connected-fit-001/candidate.npz'
DETAIL = D / 'publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz'
ORIENT = D / 'digit-source-diagnosis-110/source-path-orientation-audit-112.json'
AUDIT = D / 'publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py'
EVIDENCE = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANK = EVIDENCE / 'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
PRED = EVIDENCE / 'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
BANK_FIX = EVIDENCE / 'route-audit-counter-fix-030/canonical/matter'
OUT = ROOT / 'run-035-current-point-exit-targets-001'
RPT = OUT / 'nearest-union-exits-run034-current.json'
NPZ = OUT / 'nearest-union-exits-run034-current.npz'
CLEARANCE_M = 5e-05
OLDN = 10145

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'not JSON serializable: {type(o).__name__}')

def loadmod(p, n):
    spec = importlib.util.spec_from_file_location(n, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[n] = mod
    spec.loader.exec_module(mod)
    return mod

def csvrows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def ray_box(q, d, lo, hi):
    t0 = -math.inf
    t1 = math.inf
    for k in range(3):
        if d[k] == 0:
            if q[k] < lo[k] or q[k] > hi[k]:
                return None
            continue
        a = (lo[k] - q[k]) / d[k]
        b = (hi[k] - q[k]) / d[k]
        if a > b:
            a, b = (b, a)
        t0 = max(t0, a)
        t1 = min(t1, b)
        if t0 > t1:
            return None
    if t1 < 0:
        return None
    return (max(0.0, t0), t1)

def first_empty_union_interval(initial_inside, events, clearance_m):
    grouped = {}
    for ev in events:
        grouped.setdefault(float(ev['distance_m']), []).append(ev)
    distances = sorted(grouped)
    active = set(initial_inside)
    for i, dexit in enumerate(distances):
        toggles = {}
        for ev in grouped[dexit]:
            toggles[ev['bank']] = toggles.get(ev['bank'], 0) + 1
        for key, count in toggles.items():
            if count % 2:
                if key in active:
                    active.remove(key)
                else:
                    active.add(key)
        if not active:
            next_event = distances[i + 1] if i + 1 < len(distances) else None
            gap = math.inf if next_event is None else next_event - dexit
            if gap <= 0:
                continue
            margin = clearance_m if math.isinf(gap) else min(clearance_m, 0.5 * gap)
            if margin > 0:
                return {'exit_distance_m': dexit, 'target_distance_m': dexit + margin, 'next_event_distance_m': next_event, 'margin_m': margin, 'free_gap_m': gap}
    return None

def check_union_interval_fixture():
    result = first_empty_union_interval({'A'}, [{'distance_m': 0.002, 'bank': 'A'}, {'distance_m': 0.5, 'bank': 'B'}, {'distance_m': 0.501, 'bank': 'B'}], 5e-05)
    assert result is not None and abs(result['target_distance_m'] - 0.00205) < 1e-15 and (result['next_event_distance_m'] == 0.5)

def source_normal_fields(z, old, detail, parity, rows):
    P = np.asarray(old['points_world_m'], np.float64)
    F = np.asarray(old['source_O_faces'], np.int64)
    oldA = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    oldL = np.linalg.norm(oldA, axis=1)
    DP = np.asarray(detail['detail_baseline_world_m'], np.float64)
    DF = np.asarray(detail['source_surface_faces_detail_rows'], np.int64)
    detA = np.cross(DP[DF[:, 1]] - DP[DF[:, 0]], DP[DF[:, 2]] - DP[DF[:, 0]]) * parity
    detL = np.linalg.norm(detA, axis=1)
    out = {}
    prov = {}
    for node in rows:
        a, b = map(int, z['refinement_node_parent_face_indptr'][node:node + 2])
        parents = np.asarray(z['refinement_node_parent_face_rows'][a:b], np.int64)
        bary = np.asarray(z['refinement_node_parent_face_barycentric'][a:b], np.float64)
        if not len(parents) or bary.shape != (len(parents), 3):
            raise RuntimeError(f'node {node} has incomplete parent-face ancestry')
        acc = np.zeros(3)
        sumw = 0.0
        items = []
        for f in parents:
            kind = int(old['face_kind'][f])
            if kind == 2:
                df = int(old['parent_fine_detail_face_row'][f])
                if not 0 <= df < len(detA) or detL[df] <= 0 or oldL[f] <= 0:
                    raise RuntimeError(f'invalid detail ancestry face {f}->{df}')
                n = detA[df] / detL[df]
                align = float(np.dot(n, oldA[f] / oldL[f]))
                weight = float(oldL[f])
                items.append({'old_parent_face': int(f), 'source_detail_face': df, 'kind': 'detailed', 'normal_alignment': align, 'barycentric': bary[np.flatnonzero(parents == f)[0]].tolist()})
            elif kind == 1:
                if oldL[f] <= 0:
                    raise RuntimeError(f'degenerate coarse-only parent {f}')
                n = oldA[f] / oldL[f]
                weight = float(oldL[f])
                items.append({'old_parent_face': int(f), 'kind': 'coarse_parent_oriented_fallback', 'normal_alignment': None, 'barycentric': bary[np.flatnonzero(parents == f)[0]].tolist()})
            else:
                raise RuntimeError(f'unsupported parent face kind {kind}')
            acc += weight * n
            sumw += weight
        mag = float(np.linalg.norm(acc))
        if sumw <= 0 or mag <= 1e-12 * sumw:
            raise RuntimeError(f'source normals cancel at generated row {node}: {(mag / sumw if sumw else 0)}')
        n = acc / mag
        X = np.asarray(z['points_world_m'], np.float32).astype(np.float64)
        FF = np.asarray(z['source_O_faces'], np.int64)
        fi = np.flatnonzero(np.any(FF == node, axis=1))
        emitted = np.cross(X[FF[fi, 1]] - X[FF[fi, 0]], X[FF[fi, 2]] - X[FF[fi, 0]]).sum(axis=0)
        en = float(np.linalg.norm(emitted))
        align = float(np.dot(n, emitted / en)) if en > 0 else -1.0
        out[int(node)] = n
        prov[int(node)] = {'parent_sources': items, 'resultant_over_sum_norms': mag / sumw, 'incident_emitted_alignment': align}
    return (out, prov)

def main():
    if RPT.exists() or NPZ.exists():
        raise FileExistsError('exit-target output already exists')
    expected = EXPECTED_QUERY_HASHES
    for name, path in [('candidate', CAND), ('query_points', QUERY / 'points.csv'), ('query_pairs', QUERY / 'pairs.csv'), ('query_nodes', QUERY / 'nodes.csv')]:
        if sha(path) != expected[name]:
            raise RuntimeError(f'pinned {name} hash mismatch')
    started_at = time.monotonic()
    check_union_interval_fixture()
    z = np.load(CAND, allow_pickle=False)
    old = np.load(BASE, allow_pickle=False)
    detail = np.load(DETAIL, allow_pickle=False)
    orient = json.loads(ORIENT.read_text())
    parity = int(orient['component_fine_to_coarse_winding_anchors']['0']['chosen_outward_parity'])
    if parity not in (-1, 1):
        raise RuntimeError('bad fine/coarse winding parity')
    nodes = csvrows(QUERY / 'nodes.csv')
    pairrows = csvrows(QUERY / 'pairs.csv')
    node_by_row = {int(r['query_row']): r for r in nodes}
    pairs_by_row = {}
    for r in pairrows:
        pairs_by_row.setdefault(int(r['query_row']), []).append(r)
    targetrows = sorted({int(r['query_row']) for r in pairrows if r['classification'] == 'inside'})
    if not targetrows:
        raise RuntimeError('no inside rows in current exact query')
    normals = {}
    normalprov = {}
    normal_errors = {}
    for node in targetrows:
        if node < OLDN:
            continue
        try:
            nn, pp = source_normal_fields(z, old, detail, parity, [node])
            normals.update(nn)
            normalprov.update(pp)
        except Exception as exc:
            normal_errors[node] = f'{type(exc).__name__}: {exc}'
    reader = loadmod(AUDIT, 'bank_reader_018')
    triangles, ids, gids, bodies, M, offset, positions, bankkeys, bankpins, exact = reader.read_bank_inputs(BANK, BANK_FIX)
    if len(bankkeys) != 185 or len(triangles) != len(ids):
        raise RuntimeError('incomplete native bank inventory')
    banks = {}
    for b0, k0 in bankkeys:
        b, k = (int(b0), int(k0))
        mask = (ids[:, 0] == b) & (ids[:, 1] == k)
        T = np.asarray(triangles[mask], np.float64)
        TI = np.asarray(ids[mask, 2], np.int64)
        if not len(T):
            raise RuntimeError(f'empty bank {b}/{k}')
        banks[b, k] = {'T': T, 'id': TI, 'lo': T.min(axis=(0, 1)), 'hi': T.max(axis=(0, 1)), 'tl': T.min(axis=1), 'th': T.max(axis=1), 'M': M[b], 'pos': positions[b], 'off': offset[b]}
    source_normals = {int(k): np.asarray(v, float).copy() for k, v in normals.items()}
    direction_info = {}
    for node in targetrows:
        inside_rows = [r for r in pairs_by_row.get(node, []) if r['classification'] == 'inside']
        if not inside_rows:
            raise RuntimeError(f'inside target has no inside-bank witness at row {node}')
        choices = []
        for ir in inside_rows:
            key = (int(ir['bank_body']), int(ir['bank_stable_id']))
            bank = banks[key]
            qloc = np.array([float(ir[f'query_local_{a}']) for a in 'xyz'])
            cloc = np.array([float(ir[f'closest_{a}']) for a in 'xyz'])
            delta_local = cloc - qloc
            delta_world = np.linalg.solve(bank['M'], delta_local)
            distance = float(np.linalg.norm(delta_world))
            if not np.isfinite(distance) or distance <= 0:
                raise RuntimeError(f'invalid closest-boundary direction at {node} bank {key}')
            choices.append((distance, key, ir, delta_world))
        distance, key, ir, delta_world = min(choices, key=lambda x: (x[0], x[1]))
        nearest_world = delta_world / distance
        source_normal = source_normals.get(node)
        cosine = float(np.dot(source_normal, nearest_world)) if source_normal is not None else None
        normals[node] = nearest_world
        direction_info[node] = {'closest_bank_body': key[0], 'closest_bank_stable_id': key[1], 'nearest_distance_world_m': distance, 'query_reported_nearest_distance_m': float(ir['nearest_distance_m']), 'inside_bank_count': len(inside_rows), 'inside_banks': [[int(r['bank_body']), int(r['bank_stable_id'])] for r in inside_rows], 'source_normal_dot_nearest_direction': cosine, 'source_surface_normal_world_unit': None if source_normal is None else source_normal.tolist(), 'source_normal_status': 'unavailable_for_inherited_prefix' if source_normal is None and node < OLDN else 'ancestry_normal_unresolved' if source_normal is None else 'available_diagnostic_only', 'source_normal_error': normal_errors.get(node), 'closest_boundary_direction_world_unit': nearest_world.tolist(), 'closest_point_world_delta_m': delta_world.tolist()}
    X = np.asarray(z['points_world_m'], np.float32)
    qpoints = csvrows(QUERY / 'points.csv')
    if len(qpoints) != len(X):
        raise RuntimeError('point export row count mismatch')
    for i, (r, p) in enumerate(zip(qpoints, X)):
        if int(r['query_row']) != i:
            raise RuntimeError('point export row identity mismatch')
        for j, key in enumerate(('x_bits_u32', 'y_bits_u32', 'z_bits_u32')):
            if int(r[key]) != int(p.view(np.uint32)[j]):
                raise RuntimeError(f'candidate/query bit mismatch row {i}')
    results = []
    for node in targetrows:
        nr = node_by_row[node]
        if int(nr['total_bone_banks']) != 185 or int(nr['ambiguous_banks']) or int(nr['invalid_banks']):
            raise RuntimeError(f'incomplete point query row {node}')
        cls = {(int(r['bank_body']), int(r['bank_stable_id'])): r['classification'] for r in pairs_by_row.get(node, [])}
        if len(cls) != 185 - int(nr['aabb_proven_outside_banks']):
            raise RuntimeError(f'query class accounting mismatch row {node}')
        start = X[node].astype(np.float64)
        n = normals[node]
        bybank = {}
        events = []
        unresolved = []
        for key in bankkeys:
            key = (int(key[0]), int(key[1]))
            rec = banks[key]
            st = cls.get(key, 'outside_aabb')
            if st not in ('inside', 'outside', 'outside_aabb'):
                raise RuntimeError(f'bad start class {st}')
            q = (start - rec['pos']) @ rec['M'].T - rec['off']
            d = rec['M'] @ n
            iv = ray_box(q, d, rec['lo'], rec['hi'])
            if iv is None:
                if st == 'inside':
                    raise RuntimeError(f'inside row {node} lies outside AABB {key}')
                bybank[key] = {'start': st, 'hits': [], 'end_inside': False, 'aabb': None}
                continue
            t0, t1 = iv
            te = t1 + CLEARANCE_M
            p0 = q + t0 * d
            p1 = q + te * d
            seglo = np.minimum(p0, p1)
            seghi = np.maximum(p0, p1)
            cand = np.flatnonzero(np.all(rec['th'] >= seglo, axis=1) & np.all(rec['tl'] <= seghi, axis=1))
            hits = []
            for ti in cand:
                h = exact['classify_segment_triangle'](p0, p1, rec['T'][int(ti)])
                c = h['classification']
                if c == 'proper_interior_hit':
                    tv = t0 + float(h['t']) * (te - t0)
                    hits.append({'triangle_stable_id': int(rec['id'][int(ti)]), 'distance_m': tv, 'barycentric': [float(v) for v in h['barycentric']]})
                    events.append({'distance_m': tv, 'bank': key, 'triangle_stable_id': int(rec['id'][int(ti)])})
                elif c not in ('disjoint', 'parallel_disjoint'):
                    unresolved.append({'body': key[0], 'bone': key[1], 'triangle_stable_id': int(rec['id'][int(ti)]), 'classification': c, 'exact_result': h})
            hits.sort(key=lambda h: (h['distance_m'], h['triangle_stable_id']))
            endinside = (st == 'inside') ^ (len(hits) % 2 == 1)
            bybank[key] = {'start': st, 'hits': hits, 'end_inside': bool(endinside), 'aabb': [t0, t1]}
        if unresolved:
            results.append({'query_row': node, 'status': 'UNRESOLVED_EXACT_CONTACT', 'unresolved_count': len(unresolved), 'witnesses': unresolved[:16]})
            continue
        remain = [f'{b}/{k}' for (b, k), v in bybank.items() if v['end_inside']]
        if remain:
            results.append({'query_row': node, 'status': 'RAY_PARITY_FAILURE', 'remaining_inside_banks': remain})
            continue
        initial_inside = {key for key, v in bybank.items() if v['start'] == 'inside'}
        if not initial_inside:
            results.append({'query_row': node, 'status': 'STALE_QUERY_NO_INITIAL_INSIDE_BANK'})
            continue
        first_exit = first_empty_union_interval(initial_inside, events, CLEARANCE_M)
        if first_exit is None:
            results.append({'query_row': node, 'status': 'NO_POSITIVE_FIRST_UNION_EXIT_INTERVAL', 'initial_inside_bank_count': len(initial_inside), 'event_count': len(events)})
            continue
        exit_dist = first_exit['exit_distance_m']
        next_event = first_exit['next_event_distance_m']
        margin = first_exit['margin_m']
        dist = first_exit['target_distance_m']
        target = (start + dist * n).astype(np.float32)
        emitted_dist = float(np.dot(target.astype(np.float64) - start, n))
        if not (emitted_dist > exit_dist and (next_event is None or emitted_dist < next_event)):
            results.append({'query_row': node, 'status': 'FLOAT32_TARGET_LEFT_FIRST_FREE_INTERVAL', 'exit_distance_m': exit_dist, 'next_event_distance_m': next_event, 'emitted_distance_m': emitted_dist})
            continue
        active_emitted = set(initial_inside)
        for ev in events:
            if float(ev['distance_m']) < emitted_dist:
                key = ev['bank']
                if key in active_emitted:
                    active_emitted.remove(key)
                else:
                    active_emitted.add(key)
        if active_emitted:
            results.append({'query_row': node, 'status': 'EMITTED_TARGET_NOT_OUTSIDE_UNION', 'remaining_inside_banks': [list(k) for k in sorted(active_emitted)], 'emitted_distance_m': emitted_dist})
            continue
        results.append({'query_row': node, 'status': 'PASS_FIRST_UNION_EXIT', 'start_world_f32_m': start.tolist(), 'proposal_direction_world_unit': n.tolist(), 'source_surface_normal_world_unit': None if node not in source_normals else source_normals[node].tolist(), 'direction_evidence': direction_info[node], 'target_world_f32_m': target.tolist(), 'ray_distance_m': emitted_dist, 'union_exit_distance_m': exit_dist, 'next_union_event_distance_m': next_event, 'free_interval_margin_m': emitted_dist - exit_dist, 'clearance_m': CLEARANCE_M, 'hit_count': len(events), 'touched_banks': len({e['bank'] for e in events}), 'initial_inside_bank_count': len(initial_inside), 'end_outside_all185_by_event_parity': True, 'normal_provenance': normalprov.get(node), 'bank_hit_counts': {f'{b}/{k}': len(v['hits']) for (b, k), v in bybank.items() if v['hits']}})
    passing = [r for r in results if r['status'] == 'PASS_FIRST_UNION_EXIT']
    report = {'schema': 'numi.human.point-first-union-exits.run034-current.v1', 'classification': 'construction_target_proposal_not_clearance_admission', 'inputs': {'candidate_sha256': sha(CAND), 'query_points_sha256': sha(QUERY / 'points.csv'), 'query_nodes_sha256': sha(QUERY / 'nodes.csv'), 'query_pairs_sha256': sha(QUERY / 'pairs.csv'), 'base_candidate_sha256': sha(BASE), 'detail_source_sha256': sha(DETAIL), 'orientation_audit_sha256': sha(ORIENT), 'bank_triangle_export_sha256': sha(BANK / 'exact-emitted.triangles.csv'), 'bank_body_export_sha256': sha(BANK / 'exact-emitted.bodies.csv'), 'bank_reader_source_sha256': sha(AUDIT), 'predicate_sha256': sha(PRED), 'bank_count': len(bankkeys), 'triangle_count': len(triangles), 'parity': parity}, 'query_rows': len(targetrows), 'inside_query_nodes': len(targetrows), 'inside_node_bank_pairs': sum((1 for r in pairrows if r['classification'] == 'inside')), 'inherited_prefix_inside_nodes': sum((r < OLDN for r in targetrows)), 'new_refinement_inside_nodes': sum((r >= OLDN for r in targetrows)), 'source_normal_unavailable_rows': sorted(set(targetrows) - set(source_normals)), 'source_normal_errors': normal_errors, 'all_bank_ray_results': len(results), 'passing_targets': len(passing), 'status': 'PASS_FIRST_UNION_EXIT_TARGETS_ONLY' if len(passing) == len(targetrows) else 'INCOMPLETE_OR_REJECTED_TARGETS', 'normal_method': 'proposal direction is the exact native inside-query closest-boundary vector transformed to world by solving the raw body affine; source normals are ancillary diagnostics when direct parent-face ancestry is available; direction is proposal-only and anatomical orientation is unqualified', 'targets': results, 'construction_cap_m': 0.012, 'physical_steps': 0, 'first_union_fixture': 'PASS: inside A exits at 2mm; distant B begins at 500mm; chosen target is 2.05mm', 'wall_seconds': time.monotonic() - started_at}
    with NPZ.open('xb') as f:
        np.savez_compressed(f, query_rows=np.asarray(targetrows, np.int64), target_world_f32_m=np.asarray([next((r['target_world_f32_m'] for r in results if r['query_row'] == n), [np.nan] * 3) for n in targetrows], np.float32), proposal_direction_world_unit=np.asarray([normals[n] for n in targetrows], np.float64), source_surface_normal_world_unit=np.asarray([source_normals.get(n, [np.nan] * 3) for n in targetrows], np.float64), source_normal_available_mask=np.asarray([n in source_normals for n in targetrows], np.bool_), pass_mask=np.asarray([next((r['status'] == 'PASS_FIRST_UNION_EXIT' for r in results if r['query_row'] == n), False) for n in targetrows], np.bool_))
    RPT.write_text(json.dumps(report, indent=2, default=json_default) + '\n')
    print(json.dumps({'report': str(RPT), 'report_sha256': sha(RPT), 'npz': str(NPZ), 'npz_sha256': sha(NPZ), 'targets': len(targetrows), 'passing': len(passing), 'new_refinement_rows': sum((n >= OLDN for n in targetrows)), 'failed': [r for r in results if r['status'] != 'PASS_FIRST_UNION_EXIT'][:5], 'max_distance_m': max((r.get('ray_distance_m', 0) for r in passing), default=None), 'over_6mm': sum((r.get('ray_distance_m', 0) > 0.006 for r in passing)), 'seconds': report['wall_seconds']}, indent=2, default=json_default))
CAND = Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz')
QUERY = Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/outer-node-audit-130-run002/query')
OUT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/refined-point-178-run001/round-001/targets')
RPT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/refined-point-178-run001/round-001/targets/nearest-union-exits.json')
NPZ = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/refined-point-178-run001/round-001/targets/nearest-union-exits.npz')
EXPECTED_QUERY_HASHES = {'candidate': '0853407a1ce7fb9aed2d91a51ea712f8a85dadfcfb4b048e29979453dabab4dc', 'query_points': '54e671a3cd0b5896eed68904f6c5f1737ebe655175ce1e869209ec658ede88ae', 'query_pairs': 'be9990d0c9f08c9b10b6e21249f005a9c4d128f6a91901925f70bbe0c1af251e', 'query_nodes': '1ca2896fe04b46c72b29a9b34471f911ee0d416ccbfedb10428b9cbc2b150f17'}
if __name__ == '__main__':
    main()
