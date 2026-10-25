"""Run032: bounded global connected SCP with exact-gated partial ADMM proposals.

This is construction geometry only. Each accepted step requires a complete
exact self audit and 185-bank point gate with no new typed inside pairs and
strict reduction. Final bone-surface crossings are measured separately.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from area_scp_support_023 import project_ball_halfspace, InfeasibleProjection, ball_intersection_support, progressive_plane_lower, project_two_balls_halfspace
from self_separation_026 import build_pair_rows, project_linear_rows, relax_pair_rows_for_projection
D = Path('/Users/n/Work/numi-human-connected-surface-069')
R = D / 'full-connected-clearance-150'
START = R / 'run-032-partial-iterate-exact-gated-scp-001/attempt-001/scp-02-alpha-0p25/candidate.npz'
TARGET_START = R / 'run-032-partial-iterate-exact-gated-scp-001/attempt-001/scp-02-alpha-0p25/candidate.npz'
INSIDE_REPORT = R / 'run-032-partial-iterate-exact-gated-scp-001/attempt-001/scp-02-alpha-0p25/audit-130/summary.json'
TARGET_JSON = R / 'run-033-current-point-exit-targets-001/nearest-union-exits-run032-best.json'
TARGET_NPZ = R / 'run-033-current-point-exit-targets-001/nearest-union-exits-run032-best.npz'
BASE = R / 'run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
OLD_TRIAL = START
PRED = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py')
CROSSING_REPORT = R / 'run-032-partial-iterate-exact-gated-scp-001/attempt-001/scp-02-alpha-0p25/audit-159/report.json'
CROSSING_COUNT = R / 'run-032-partial-iterate-exact-gated-scp-001/attempt-001/crossing-count-correction-001.json'
AUD130 = D / 'audit_connected_outer_geometry_130.py'
AUD159 = D / 'audit_skin_bone_surfaces_159.py'
PYTHON = Path('/Users/n/numi-human-prep-venv-20261005/bin/python')
OUT = R / 'run-034-progress-merit-connected-scp-001/attempt-001'
PIN = {'start': '60ef3652e34a43ee9b5631767529f3b061ba8a5d8604666de1a17b429b73559b', 'target_start': '60ef3652e34a43ee9b5631767529f3b061ba8a5d8604666de1a17b429b73559b', 'inside_report': '4c22b843756cc350ca926b8780fef6f93c466759b2bef8730a544f377bc0fa20', 'target_json': '3ff96a691b993c66717ed85a805d114ccb87b94c0af584998da8b92717bf1612', 'target_npz': '65ef4acaf0aad28976d16b74ab2b7b36b98a3e151fe92d260ad93eaf1457b18b', 'base': '70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31', 'old_trial': '60ef3652e34a43ee9b5631767529f3b061ba8a5d8604666de1a17b429b73559b', 'predicate': 'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d', 'self_helper': 'ec647128549c31d38d02f1722de1ee584c1ff442b9275be92a2d3a25b0f52ee4', 'support_helper': 'aaa9dbe32fbfeb71818a2afd673ef4eb7dfc0c209cee6d0ee1dbff1a6ce71abd', 'crossing_report': '8a6403e2e6af105ce5d5f4365136451ebe1d528278db6124b69cd3b9a3bb6b71', 'crossing_count': '7ab372e1d572747f78b90655ab9ca53eec8df5a82976a3f8f9cb735abef5fe6f', 'audit130': '52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9', 'audit159': 'de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1', 'progress_helper': '6081aa3de64a4500e3bfb453e33aa9cc183a1f93765cceda091fc62ea524bfed', 'node_projector': '9175186e4a916dbfd8b2ce4d10f4bd7bb09938cecf442b291bc093823a1336e7', 'parent_fit': 'bc7f7a9a79553fde80066bb76915f17ed5aeec0a663b4066bdf9c66a46fc9aac'}
TARGET_MAX_MM = 12.0
PROGRESS_HELPER = D / 'publication-089/canonical/matter/tools/human_skin_fat_bone_audit.py'
NODE_PROJECTOR = D / 'publication-090/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
SOURCE_CAP_MM = 12.0
PAIR_NUMERICAL_RELAXATION_MM = 1.5e-05
STEP_MM = 2.0
AREA_FLOOR_RATIO = 0.1
MAX_SCP = 4
MAX_ADMM = 1200
MAX_SOLVE_S = 35.0
RHO = 3.0
LAMBDA = 1.5
TARGET_WEIGHT = 8.0
ETA = 0.02
AREA_SWEEPS = 80
NODE_SWEEPS = 120

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(name, path):
    deps = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src')
    sys.path.insert(0, str(deps))
    sys.path.insert(0, str(Path(path).parents[1]))
    import numilab_human
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def project_ball(v, c, r):
    v = np.asarray(v, float)
    c = np.asarray(c, float)
    w = v - c
    norm = np.linalg.norm(w, axis=1)
    return c + w * np.minimum(1.0, float(r) / np.maximum(norm, 1e-300))[:, None]

def project_node_sets(v, source_center, source_cap, plane_rows, plane_normals, plane_lower, anchors, trust_center, trust_radius):
    """Exact per-node active-set projection onto source/trust balls and exit plane."""
    v = np.asarray(v, float)
    source_center = np.asarray(source_center, float)
    trust_center = np.asarray(trust_center, float)
    out = project_ball(v, trust_center, trust_radius)
    plane_by_node = {int(i): (np.asarray(n, float), float(h)) for i, n, h in zip(plane_rows, plane_normals, plane_lower)}
    anchor_set = set(map(int, anchors))
    overlap_count = 0
    full_enumeration_count = 0
    for row in range(len(v)):
        if row in anchor_set:
            out[row] = 0.0
            continue
        pair = plane_by_node.get(row)
        source_contains_trust = np.linalg.norm(trust_center[row] - source_center[row]) + trust_radius <= source_cap + 1e-10
        if pair is not None:
            n, h = pair
            if source_contains_trust:
                out[row] = project_ball_halfspace(v[row:row + 1], trust_center[row:row + 1], trust_radius, n[None, :], np.array([h]), np.array([0], np.int64))[0]
            else:
                overlap_count += 1
                full_enumeration_count += 1
                out[row] = project_two_balls_halfspace(v[row], source_center[row], source_cap, trust_center[row], trust_radius, n, h)
        elif np.linalg.norm(out[row] - source_center[row]) > source_cap + 1e-10:
            overlap_count += 1
            full_enumeration_count += 1
            out[row] = project_two_balls_halfspace(v[row], source_center[row], source_cap, trust_center[row], trust_radius)
    for row, (n, h) in plane_by_node.items():
        if row in anchor_set and h > 1e-10:
            raise InfeasibleProjection(row, h, source_cap)
    ball_violation = np.maximum(0.0, np.linalg.norm(out - source_center, axis=1) - source_cap)
    trust_violation = np.maximum(0.0, np.linalg.norm(out - trust_center, axis=1) - trust_radius)
    plane_violation = np.array([max(0.0, h - float(np.dot(n, out[row]))) for row, (n, h) in plane_by_node.items()])
    err = max(float(ball_violation.max(initial=0.0)), float(trust_violation.max(initial=0.0)), float(plane_violation.max(initial=0.0)))
    return (out, {'method': 'exact_active_set_candidates', 'intersection_residual_mm': err, 'overlap_nodes': overlap_count, 'full_active_set_enumerations': full_enumeration_count})

def color_face_rows(tri):
    """Greedy deterministic coloring so rows in each batch share no vertices."""
    groups = []
    used = []
    for ri, nodes in enumerate(np.asarray(tri, np.int64)):
        node_set = set(map(int, nodes))
        placed = False
        for ci, occupied in enumerate(used):
            if node_set.isdisjoint(occupied):
                groups[ci].append(ri)
                occupied.update(node_set)
                placed = True
                break
        if not placed:
            groups.append([ri])
            used.append(set(node_set))
    return [np.asarray(g, np.int64) for g in groups]

def area_rows_at(current_mm, ref_cross, face_ids, faces):
    tri = np.asarray(faces, np.int64)[face_ids]
    ref_area = np.linalg.norm(ref_cross[face_ids], axis=1)
    if np.any(ref_area <= 1e-16):
        raise RuntimeError('degenerate reference face in area constraints')
    normals = ref_cross[face_ids] / ref_area[:, None]
    a, b, c = (current_mm[tri[:, 0]], current_mm[tri[:, 1]], current_mm[tri[:, 2]])
    e1 = b - a
    e2 = c - a
    gb = np.cross(e2, normals)
    gc = np.cross(normals, e1)
    ga = -gb - gc
    glocal = np.stack((ga, gb, gc), axis=1)
    q = np.einsum('ij,ij->i', normals, np.cross(e1, e2))
    return (tri, glocal, AREA_FLOOR_RATIO * ref_area - q, ref_area, normals, q)

def project_area_rows(v, tri, g, rhs, groups, max_sweeps=AREA_SWEEPS, tol=0.0001):
    """Dykstra projection onto sparse per-face linearized oriented-area halfspaces."""
    d = np.asarray(v, float).copy()
    corr = np.zeros(len(tri), float)
    g2 = np.einsum('mij,mij->m', g, g)
    bad = np.flatnonzero(g2 <= 1e-24)
    if len(bad) and np.any(rhs[bad] > tol):
        return (d, {'converged': False, 'reason': 'zero_gradient_infeasible_area_rows', 'bad_rows': bad[:20].tolist(), 'max_violation_mm2': float(rhs[bad].max())})
    g2 = np.maximum(g2, 1e-24)
    maxv = float('inf')
    for sweep in range(max_sweeps):
        for ids in groups:
            nodes = tri[ids]
            y = d[nodes] + corr[ids, None, None] * g[ids]
            val = np.einsum('mij,mij->m', y, g[ids])
            step = np.maximum(0.0, (rhs[ids] - val) / g2[ids])
            z = y + step[:, None, None] * g[ids]
            d[nodes] = z
            corr[ids] = -step
        val = np.einsum('mij,mij->m', d[tri], g)
        maxv = float(np.maximum(0.0, rhs - val).max(initial=0.0))
        if maxv <= tol:
            break
    return (d, {'converged': bool(maxv <= tol), 'sweeps': sweep + 1, 'max_violation_mm2': maxv})

def area_ratios_mm(points_m, faces, ref_cross):
    p = np.asarray(points_m, np.float32).astype(np.float64) * 1000.0
    tri = np.asarray(faces, np.int64)
    cc = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    ref_area = np.linalg.norm(ref_cross, axis=1)
    ratios = np.einsum('ij,ij->i', cc, ref_cross) / np.maximum(ref_area * ref_area, 1e-300)
    return (ratios, cc)

def add_one_ring(active, faces, violation_rows):
    bad = np.asarray(violation_rows, np.int64)
    if len(bad) == 0:
        return active
    verts = np.unique(np.asarray(faces, np.int64)[bad].ravel())
    touch = np.flatnonzero(np.isin(faces, verts).any(axis=1))
    return np.unique(np.r_[active, bad, touch]).astype(np.int64)

def run_point_gate(candidate_path, output_path):
    import subprocess
    candidate_path = Path(candidate_path)
    output_path = Path(output_path)
    cmd = [str(PYTHON), str(AUD130), str(candidate_path), str(output_path)]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / 'audit-130.log').write_text(cp.stdout + cp.stderr)
    if cp.returncode or not (output_path / 'summary.json').is_file():
        return (None, {'returncode': cp.returncode, 'command': cmd})
    j = json.loads((output_path / 'summary.json').read_text())
    typed = {(int(r['query_row']), int(r['bank_body']), int(r['bank_stable_id'])) for r in j.get('inside_witnesses', [])}
    return (j, {'returncode': 0, 'command': cmd, 'summary_sha256': sha(output_path / 'summary.json'), 'inside_typed_pair_count': len(typed), 'inside_typed_pairs': sorted([list(x) for x in typed])})

def main():
    started = time.monotonic()
    for p, k in [(START, 'start'), (TARGET_START, 'target_start'), (INSIDE_REPORT, 'inside_report'), (TARGET_JSON, 'target_json'), (TARGET_NPZ, 'target_npz'), (BASE, 'base'), (OLD_TRIAL, 'old_trial'), (PRED, 'predicate'), (Path(__file__).with_name('self_separation_026.py'), 'self_helper'), (Path(__file__).with_name('area_scp_support_023.py'), 'support_helper'), (CROSSING_REPORT, 'crossing_report'), (CROSSING_COUNT, 'crossing_count'), (AUD130, 'audit130'), (AUD159, 'audit159'), (PROGRESS_HELPER, 'progress_helper'), (NODE_PROJECTOR, 'node_projector')]:
        if sha(p) != PIN[k]:
            raise RuntimeError(f'input SHA mismatch {k}: {sha(p)}')
    parent_fit = PARENT_FIT
    if sha(parent_fit) != PIN['parent_fit']:
        raise RuntimeError('parent fit source pin mismatch')
    if any(((OUT / n).exists() for n in ('candidate.npz', 'fit-report.json'))):
        raise FileExistsError('Run034 output already frozen')
    OUT.mkdir(exist_ok=True)
    z = np.load(START, allow_pickle=False)
    target_start = np.load(TARGET_START, allow_pickle=False)
    base = np.load(BASE, allow_pickle=False)
    tj = json.loads(TARGET_JSON.read_text())
    tn = np.load(TARGET_NPZ, allow_pickle=False)
    inside_report = json.loads(INSIDE_REPORT.read_text())
    crossing_report = json.loads(CROSSING_REPORT.read_text())
    count_text = CROSSING_COUNT.read_text()
    crossing_count = json.loads(count_text[:-2] if count_text.endswith(chr(92) + chr(110)) else count_text)
    X = np.asarray(z['points_world_m'], np.float32).astype(np.float64)
    F = np.asarray(z['source_O_faces'], np.int32)
    if X.shape != (13518, 3) or F.shape != (26991, 3):
        raise RuntimeError('unexpected refined start geometry shape')
    if tj.get('inputs', {}).get('candidate_sha256') != PIN['target_start'] or tj.get('status') != 'PASS_FIRST_UNION_EXIT_TARGETS_ONLY' or int(tj.get('passing_targets', 0)) != TARGET_NODE_COUNT:
        raise RuntimeError('target source report mismatch')
    if inside_report.get('candidate_sha256') != PIN['start'] or inside_report.get('complete_surface_audit') is not True or int(inside_report.get('exact_PLC_surface_pairs', -1)) != 0:
        raise RuntimeError('current inside-point report mismatch')
    if crossing_report.get('candidate_sha256') != PIN['start'] or crossing_report.get('complete') is not True or int(crossing_report.get('bank_count', -1)) != 185 or (int(crossing_report.get('expected_bank_count', -1)) != 185):
        raise RuntimeError('current face crossing report incomplete')
    if int(crossing_count.get('total_crossing_pairs', -1)) != SURFACE_COUNT or int(crossing_count.get('bank_count', -1)) != 185:
        raise RuntimeError('strict face crossing count mismatch')
    all_cross_pairs = []
    for bank in crossing_report['banks']:
        result = bank.get('result')
        if result is None:
            if int(bank.get('skin_aabb_active', -1)) != 0 or int(bank.get('bank_aabb_active', -1)) != 0:
                raise RuntimeError('missing bank result without proven AABB exclusion')
            continue
        pairs = result.get('triangle_pairs')
        if type(result.get('count')) is not int or result['count'] != len(pairs):
            raise RuntimeError('159 per-bank crossing count mismatch')
        all_cross_pairs.extend(pairs)
    if len(all_cross_pairs) != SURFACE_COUNT:
        raise RuntimeError('159 total crossing witness mismatch')
    crossed_faces = np.unique(np.asarray(all_cross_pairs, np.int64)[:, 0])
    if not len(crossed_faces) or int(crossed_faces.max()) >= len(F):
        raise RuntimeError('159 skin face rows out of range')
    payload = {k: z[k] for k in z.files}
    target_start_points = np.asarray(target_start['points_world_m'], np.float32)
    rows = np.asarray(tn['query_rows'], np.int64)
    target_xyz = np.asarray(tn['target_world_f32_m'], np.float32)
    pm = np.asarray(tn['pass_mask'], bool)
    if len(rows) != TARGET_NODE_COUNT or target_xyz.shape != (len(rows), 3) or (not np.all(pm)) or (len(np.unique(rows)) != len(rows)):
        raise RuntimeError('incomplete target NPZ')
    target_map = {int(r['query_row']): r for r in tj['targets']}
    if len(target_map) != len(rows):
        raise RuntimeError('target JSON row mismatch')
    nold = len(base['points_before_joint_self_pair_repair_m'])
    ind = np.asarray(z['source_prolongation_indptr'], np.int64)
    cols = np.asarray(z['source_prolongation_node_rows'], np.int64)
    w = np.asarray(z['source_prolongation_weights'], np.float64)
    pind = np.asarray(z['prior_run028__source_prolongation_indptr'], np.int64)
    pcols = np.asarray(z['prior_run028__source_prolongation_node_rows'], np.int64)
    pw = np.asarray(z['prior_run028__source_prolongation_weights'], np.float64)
    nmid = len(pind) - 1
    if len(ind) != len(X) + 1 or ind[0] != 0 or ind[-1] != len(cols) or (len(cols) != len(w)) or np.any(np.diff(ind) < 0) or np.any(cols < 0) or np.any(cols >= nmid):
        raise RuntimeError('bad run029 source prolongation CSR')
    if len(pind) != nmid + 1 or pind[0] != 0 or pind[-1] != len(pcols) or (len(pcols) != len(pw)) or np.any(np.diff(pind) < 0) or np.any(pcols < 0) or np.any(pcols >= nold):
        raise RuntimeError('bad preserved Run028 source prolongation CSR')
    sums1 = np.asarray([w[ind[i]:ind[i + 1]].sum() for i in range(len(X))])
    sums0 = np.asarray([pw[pind[i]:pind[i + 1]].sum() for i in range(nmid)])
    if np.any(np.diff(ind) == 0) or np.any(np.diff(pind) == 0) or np.max(np.abs(sums1 - 1.0)) > 2e-10 or (np.max(np.abs(sums0 - 1.0)) > 2e-10):
        raise RuntimeError('source prolongation rows not affine')
    Pcur = sparse.csr_matrix((w, cols, ind), shape=(len(X), nmid))
    Pold = sparse.csr_matrix((pw, pcols, pind), shape=(nmid, nold))
    P = (Pcur @ Pold).tocsr()
    sums = np.asarray(P.sum(axis=1)).ravel()
    if np.max(np.abs(sums - 1.0)) > 4e-10:
        raise RuntimeError('composed source ancestry rows not affine')
    R0 = np.asarray(P @ np.asarray(base['points_before_joint_self_pair_repair_m'], np.float64), np.float64)
    anchors = np.asarray(z['typed_coarse_anchor_output_rows_018'], np.int64)
    anchor_ids = np.asarray(z['typed_coarse_anchor_source_ids_018'], np.int64)
    if len(anchors) != 237 or len(np.unique(anchors)) != 237 or len(anchor_ids) != 237:
        raise RuntimeError('typed anchor set incomplete')
    if not np.array_equal(np.ascontiguousarray(X[anchors], dtype=np.float32).tobytes(), np.ascontiguousarray(R0[anchors], dtype=np.float32).tobytes()):
        raise RuntimeError('true anchors not at registered reference')
    Xmm = X * 1000.0
    center = (R0 - X) * 1000.0
    start_disp = np.linalg.norm(center, axis=1)
    if float(start_disp.max()) > SOURCE_CAP_MM + 2e-05:
        raise RuntimeError('start exceeds declared source cap')
    rows = np.asarray(tn['query_rows'], np.int64)
    target_xyz = np.asarray(tn['target_world_f32_m'], np.float32)
    pm = np.asarray(tn['pass_mask'], bool)
    if len(rows) != TARGET_NODE_COUNT or target_xyz.shape != (len(rows), 3) or (not np.all(pm)) or (len(np.unique(rows)) != len(rows)):
        raise RuntimeError('incomplete target NPZ')
    target_map = {int(r['query_row']): r for r in tj['targets']}
    if len(target_map) != len(rows):
        raise RuntimeError('target JSON row mismatch')
    inside_records = inside_report.get('inside_witnesses', [])
    inside_pairs = [int(q['query_row']) for q in inside_records]
    inside_rows = set(inside_pairs)
    inside_pair_count = int(inside_report.get('inside_node_bank_pairs', -1))
    if len(inside_rows) != int(inside_report.get('inside_nodes', -1)) or inside_pair_count != int(tj.get('inside_node_bank_pairs', -2)):
        raise RuntimeError('inside query receipt node/pair counts mismatch')
    unresolved = []
    missing_targets = sorted(inside_rows - set(target_map))
    for row in missing_targets:
        unresolved.append({'query_row': row, 'reason': 'current_inside_node_has_no_frozen_first_union_target'})
    prop_rows = []
    prop_dirs = []
    prop_h = []
    prop_t = []
    already_outside_target_rows = []
    behind_rows = []
    for j, rr0 in enumerate(rows):
        row = int(rr0)
        rr = target_map[row]
        if rr.get('status') != 'PASS_FIRST_UNION_EXIT':
            raise RuntimeError(f'proposal {row} lacks first-union status')
        if not np.array_equal(target_xyz[j].view(np.uint32), np.asarray(rr['target_world_f32_m'], np.float32).view(np.uint32)):
            raise RuntimeError(f'proposal array/report mismatch at {row}')
        st = np.asarray(rr['start_world_f32_m'], np.float32)
        if row >= len(target_start_points) or not np.array_equal(st.view(np.uint32), target_start_points[row].view(np.uint32)):
            raise RuntimeError(f'proposal original-start mismatch at {row}')
        d = np.asarray(rr['proposal_direction_world_unit'], float)
        dn = float(np.linalg.norm(d))
        if dn <= 0:
            raise RuntimeError(f'zero proposal direction at {row}')
        d = d / dn
        original_dist = float(np.dot(d, target_xyz[j].astype(float) * 1000.0 - st.astype(float) * 1000.0))
        if original_dist < -1e-07 or abs(original_dist - float(rr['ray_distance_m']) * 1000.0) > 0.02:
            raise RuntimeError(f'original proposal distance mismatch at {row}')
        if row not in inside_rows:
            already_outside_target_rows.append(row)
            continue
        current = X[row].astype(float)
        vec = (target_xyz[j].astype(float) - current) * 1000.0
        dist = float(np.dot(d, vec))
        if dist <= 0:
            behind_rows.append(row)
            unresolved.append({'query_row': row, 'reason': 'current_inside_but_frozen_exit_plane_is_not_ahead', 'projected_remaining_mm': dist})
            continue
        source_support = float(np.dot(d, center[row]) + SOURCE_CAP_MM)
        if dist > TARGET_MAX_MM + 1e-07:
            unresolved.append({'query_row': row, 'reason': 'remaining_first_union_target_exceeds_12mm_construction_target_cap', 'distance_mm': dist})
        if dist > source_support + 1e-07:
            unresolved.append({'query_row': row, 'reason': 'remaining_target_outside_source_ball_projection', 'distance_mm': dist, 'max_source_ball_projection_mm': source_support})
        prop_rows.append(row)
        prop_dirs.append(d)
        prop_h.append(dist)
        prop_t.append(vec)
    prop_rows = np.asarray(prop_rows, np.int64)
    prop_dirs = np.asarray(prop_dirs, float)
    prop_h = np.asarray(prop_h, float)
    prop_t = np.asarray(prop_t, float)
    hard_mask = (prop_h <= TARGET_MAX_MM + 1e-07) & (prop_h <= np.einsum('ij,ij->i', prop_dirs, center[prop_rows]) + SOURCE_CAP_MM + 1e-07) & ~np.isin(prop_rows, anchors)
    hard_rows = prop_rows[hard_mask]
    hard_dirs = prop_dirs[hard_mask]
    hard_targets = prop_h[hard_mask]
    if len(hard_rows) == 0:
        raise RuntimeError('no feasible ≤12mm hard exit proposals')
    if len(np.unique(hard_rows)) != len(hard_rows):
        raise RuntimeError('duplicate hard target row')
    edges = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]])).astype(np.int64)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    positive = lengths[lengths > 0]
    if len(positive) != len(lengths):
        raise RuntimeError('zero-length O edge')
    med = float(np.median(positive))
    ew = np.clip(med / positive, 0.05, 20.0)
    er = np.r_[edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]]
    ec = np.r_[edges[:, 0], edges[:, 1], edges[:, 1], edges[:, 0]]
    ev = np.r_[ew, ew, -ew, -ew]
    L = sparse.coo_matrix((ev, (er, ec)), shape=(len(X), len(X))).tocsr()
    tw = np.zeros(len(X))
    tt = np.zeros((len(X), 3))
    for i, row in enumerate(prop_rows):
        tw[row] = TARGET_WEIGHT
        tt[row] = prop_t[i]
        nn = float(np.linalg.norm(tt[row]))
        if nn > TARGET_MAX_MM:
            tt[row] *= TARGET_MAX_MM / nn
    if sha(BARY_HELPER) != BARY_HELPER_SHA:
        raise RuntimeError('barycentric helper source mismatch')
    bary_owner = module('numilab_human._barycentric_constraints_170', BARY_HELPER)
    if sha(BARY_TARGETS) != BARY_TARGET_SHA:
        raise RuntimeError('barycentric target hash mismatch')
    bz = np.load(BARY_TARGETS, allow_pickle=False)
    B, btri, bg, btarget, bdelta = bary_owner.build_barycentric_exit_system(Xmm, F, bz['face_rows'], bz['weights'], bz['normals_world'], bz['targets_world_m'] * 1000.0, anchor_rows=anchors)
    bgroups = color_face_rows(btri)
    bcounts = np.bincount(bz['face_rows'], minlength=len(F))
    bw = 1.0 / bcounts[bz['face_rows']]
    A = L * LAMBDA + sparse.diags(tw + ETA) + BARY_WEIGHT * (B.T @ sparse.diags(bw) @ B)
    ref_cross = np.cross(Xmm[F[:, 1]] - Xmm[F[:, 0]], Xmm[F[:, 2]] - Xmm[F[:, 0]])
    ref_area = np.linalg.norm(ref_cross, axis=1)
    if np.any(ref_area <= 1e-10):
        raise RuntimeError('degenerate emitted start face')
    flipped = np.asarray(crossed_faces, np.int64)
    active = add_one_ring(np.empty(0, np.int64), F, flipped)
    rho = RHO
    Q = (A + sparse.eye(len(X), format='csr') * (3 * rho)).tocsc()
    factor_t = time.monotonic()
    lu = splu(Q)
    factor_s = time.monotonic() - factor_t
    pred = module('numilab_human._area_scp_021', PRED)
    progress_owner = module('numilab_human._clearance_progress_089', PROGRESS_HELPER)
    projector_owner = module('numilab_human._surface_fit_constraints_090', NODE_PROJECTOR)
    current_point_report = inside_report
    current_surface_count = BEST_SURFACE_COUNT
    current_surface_report = crossing_report

    def exact_self(points):
        rec, _ = pred.float32_surface_records(np.asarray(points, np.float32), F, source_seam_groups=())
        return pred._audit_self_pairs_touching_faces(rec, set(range(len(F))))
    start_self = exact_self(X)
    if not start_self.get('complete_surface_audit') or len(start_self.get('triangle_pairs', [])):
        raise RuntimeError('SCP start fails complete exact self audit')
    self_pair_ids = []
    self_pair_margin_fraction = {}
    inside_set = {(int(r['query_row']), int(r['bank_body']), int(r['bank_stable_id'])) for r in inside_records}
    if len(inside_set) != inside_pair_count:
        raise RuntimeError('initial typed inside-pair witness count mismatch')
    current_inside_set = set(inside_set)
    current_inside_pair_count = inside_pair_count
    accepted_point_audits = []
    rejected_point_audits = []
    self_pair_seed_records = None
    dcur = np.zeros_like(Xmm)
    stages = []
    accepted_any = False
    stop_reason = 'max_scp_iterations'
    for scp in range(MAX_SCP):
        t0 = time.monotonic()
        cur_m = (Xmm + dcur) / 1000.0
        cur_mm = cur_m.astype(np.float32).astype(np.float64) * 1000.0
        self_rows = []
        pair_geometry = []
        for pair in self_pair_ids:
            pair_rows, meta = build_pair_rows(cur_mm, F, [pair], margin_fraction=self_pair_margin_fraction[tuple(pair)])
            meta = relax_pair_rows_for_projection(pair_rows, meta, min(PAIR_NUMERICAL_RELAXATION_MM, 0.25 * min((float(r['preserved_margin_mm']) for r in meta))))
            for row in pair_rows:
                row['lower'] += sum((float(np.dot(c, dcur[i])) for i, c in zip(row['nodes'], row['coefficients'])))
                self_rows.append(row)
            pair_geometry.extend(meta)
        pair_base_residual = max([max(0.0, float(r['lower']) - sum((float(np.dot(c, dcur[i])) for i, c in zip(r['nodes'], r['coefficients'])))) for r in self_rows], default=0.0)
        if pair_base_residual > 0.0002:
            raise RuntimeError(f'current accepted iterate violates self-pair barrier by {pair_base_residual}mm')
        if self_pair_seed_records is None:
            self_pair_seed_records = pair_geometry
        self_safety_margin_mm = min((float(m['effective_preserved_margin_mm']) for m in pair_geometry), default=0.0001)
        if self_safety_margin_mm <= 0:
            raise RuntimeError('self-pair numerical allowance consumed the barrier margin')
        curproj = np.einsum('ij,ij->i', hard_dirs, dcur[hard_rows])
        maxproj = np.einsum('ij,ij->i', hard_dirs, center[hard_rows]) + SOURCE_CAP_MM
        support = np.asarray([ball_intersection_support(center[row], SOURCE_CAP_MM, dcur[row], STEP_MM, n) for row, n in zip(hard_rows, hard_dirs)], dtype=np.float64)
        lower = progressive_plane_lower(curproj, hard_targets, support, 0.95)
        tri, glocal, base_rhs, sel_area, normals, qcur = area_rows_at(cur_m * 1000.0, ref_cross, active, F)
        dloc = dcur[tri]
        arhs = base_rhs + np.einsum('mij,mij->m', glocal, dloc)
        groups = color_face_rows(tri)
        blower = bary_owner.progressive_barycentric_lower(bg, btri, dcur, btarget, STEP_MM)
        rhs = tw[:, None] * tt + BARY_WEIGHT * (B.T @ (bw[:, None] * bdelta))
        x = dcur.copy()
        zn = dcur.copy()
        za = dcur.copy()
        zs = dcur.copy()
        un = np.zeros_like(Xmm)
        ua = np.zeros_like(Xmm)
        us = np.zeros_like(Xmm)
        zb = dcur.copy()
        ub = np.zeros_like(Xmm)
        log = []
        converged = False
        node_projection = {}
        area_projection = {}
        solve_t = time.monotonic()
        for it in range(MAX_ADMM):
            x = lu.solve((rhs + rho * (zn - un + (za - ua) + (zs - us))).astype(np.float64))
            oldzn = zn
            oldza = za
            oldzs = zs
            oldzb = zb
            zn, node_projection = projector_owner.project_node_sets_batched(x + un, center, SOURCE_CAP_MM, hard_rows, hard_dirs, lower, anchors, dcur, STEP_MM)
            za, area_projection = project_area_rows(x + ua, tri, glocal, arhs, groups)
            zb = x.copy()
            bary_projection = {'converged': True, 'method': 'soft_objective_only_no_inconsistent_hard_exit_planes', 'max_violation_mm': None}
            zs, self_projection = project_linear_rows(x + us, self_rows, max_sweeps=120, tolerance_mm=1e-06)
            self_projection_safe = bool(float(self_projection.get('max_violation_mm', float('inf'))) < self_safety_margin_mm)
            un += x - zn
            ua += x - za
            us += x - zs
            ub += x - zb
            pr = max(float(np.max(np.linalg.norm(x - zn, axis=1))), float(np.max(np.linalg.norm(x - za, axis=1))), float(np.max(np.linalg.norm(x - zs, axis=1))))
            du = rho * max(float(np.max(np.linalg.norm(zn - oldzn, axis=1))), float(np.max(np.linalg.norm(za - oldza, axis=1))), float(np.max(np.linalg.norm(zs - oldzs, axis=1))))
            if it % 20 == 0 or it == MAX_ADMM - 1:
                log.append({'iteration': it + 1, 'primal_inf_mm': pr, 'dual_inf_mm': du, 'area_projection': area_projection, 'node_projection': node_projection, 'self_projection': self_projection, 'barycentric_projection': bary_projection, 'self_projection_safety_margin_mm': self_safety_margin_mm, 'self_projection_residual_below_margin': self_projection_safe})
            if pr < 0.0002 and du < 0.0002 and area_projection.get('converged', False) and (node_projection.get('intersection_residual_mm', float('inf')) < 0.0002) and bary_projection.get('converged', False) and (self_projection.get('converged', False) or self_projection_safe):
                converged = True
                break
            if time.monotonic() - solve_t > MAX_SOLVE_S:
                break
        solve_s = time.monotonic() - solve_t
        if area_projection.get('reason') or node_projection.get('reason') or self_projection.get('reason'):
            stop_reason = 'projection_infeasible'
            stages.append({'scp_iteration': scp + 1, 'status': stop_reason, 'area': area_projection, 'node': node_projection})
            break
        if not (np.isfinite(zn).all() and np.isfinite(za).all() and np.isfinite(zs).all() and np.isfinite(zb).all()):
            stop_reason = 'nonfinite_admm_partial_iterate'
            stages.append({'scp_iteration': scp + 1, 'status': stop_reason, 'admm': log[-1] if log else {}, 'solve_seconds': solve_s})
            break
        partial_iterate_used = not converged
        raw = (zn + za + zs) / 3.0
        chosen = None
        trial_records = []
        newly_bad = []
        current_proj = np.einsum('ij,ij->i', hard_dirs, dcur[hard_rows])
        for alpha in [0.25, 0.125, 0.0625]:
            td = dcur + alpha * (raw - dcur)
            yp = ((Xmm + td) / 1000.0).astype(np.float32).astype(np.float64)
            if len(anchors):
                yp[anchors] = X[anchors]
            trial_dir = OUT / f"scp-{scp + 1:02d}-alpha-{str(alpha).replace('.', 'p')}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            trial_candidate = trial_dir / 'candidate.npz'
            trial_payload = dict(payload)
            trial_payload['points_world_m'] = yp.astype(np.float32)
            with trial_candidate.open('xb') as fp:
                np.savez_compressed(fp, **trial_payload)
            ratios, _ = area_ratios_mm(yp, F, ref_cross)
            flipped_now = np.flatnonzero(ratios <= 0.0)
            active_low = np.flatnonzero(ratios[active] < AREA_FLOOR_RATIO - 5e-06)
            capdist = np.linalg.norm(yp - R0, axis=1) * 1000.0
            capok = bool(np.all(capdist <= SOURCE_CAP_MM + 0.002))
            anchorok = bool(np.ascontiguousarray(yp[anchors], dtype=np.float32).tobytes() == np.ascontiguousarray(X[anchors], dtype=np.float32).tobytes())
            stepmax = float(np.linalg.norm(yp - (Xmm + dcur) / 1000.0, axis=1).max() * 1000.0)
            stepok = stepmax <= STEP_MM + 0.003
            pnow = np.einsum('ij,ij->i', hard_dirs, yp[hard_rows] * 1000.0 - Xmm[hard_rows])
            progressok = bool(np.all(pnow >= current_proj - 0.002))
            local_area_ok = len(flipped_now) == 0 and len(active_low) == 0
            trial_delta = yp * 1000.0 - Xmm
            self_linear_violation = max((max(0.0, float(r['lower']) - sum((float(np.dot(c, trial_delta[i])) for i, c in zip(r['nodes'], r['coefficients'])))) for r in self_rows), default=0.0)
            self_linear_ok = bool(self_linear_violation < self_safety_margin_mm)
            self_report = {}
            self_pairs = np.empty((0, 2), dtype=np.int64)
            point_report = None
            point_meta = {}
            inside_trial_set = set()
            no_new_inside = False
            strict_reduction = False
            point_ok = False
            merit = None
            if capok and anchorok and stepok and local_area_ok:
                self_report = exact_self(yp)
                self_pairs = np.asarray(self_report.get('triangle_pairs', []), np.int64).reshape(-1, 2)
            exactok = bool(self_report.get('complete_surface_audit') and len(self_pairs) == 0)
            if exactok and capok and anchorok and stepok and local_area_ok:
                point_report, point_meta = run_point_gate(trial_candidate, trial_dir / 'audit-130')
                if point_report is not None and point_report.get('candidate_sha256') == sha(trial_candidate) and (point_report.get('complete_surface_audit') is True) and (int(point_report.get('exact_PLC_surface_pairs', -1)) == 0):
                    inside_trial_set = {(int(r['query_row']), int(r['bank_body']), int(r['bank_stable_id'])) for r in point_report.get('inside_witnesses', [])}
                    if len(inside_trial_set) == int(point_report.get('inside_node_bank_pairs', -1)):
                        try:
                            merit = progress_owner.assess_point_clearance_construction_progress(current_point_report, point_report)
                            no_new_inside = not bool(merit.get('introduced_typed_pairs'))
                            strict_reduction = bool(merit.get('strict_pair_count_reduction'))
                            point_ok = bool(merit['maximum_depth_nonincreasing'] and merit['squared_depth_nonincreasing'])
                            if point_ok:
                                import subprocess
                                ap = trial_dir / 'audit-159'
                                ac = subprocess.run([str(PYTHON), str(AUD159), str(trial_candidate), str(ap), '--max-seconds', '600'], capture_output=True, text=True)
                                (trial_dir / 'audit-159.log').write_text(ac.stdout + ac.stderr)
                                if ac.returncode:
                                    raise RuntimeError('full surface audit failed')
                                ar = json.loads((ap / 'report.json').read_text())
                                if ar['candidate_sha256'] != sha(trial_candidate):
                                    raise RuntimeError('surface audit source mismatch')
                                sc = progress_owner.summarize_surface_crossing_report(ar, expected_bank_keys={(b['body'], b['stable_bone_id']) for b in crossing_report['banks']})
                                before_surface = current_surface_count
                                joint_merit = progress_owner.assess_joint_clearance_construction_progress(current_point_report, point_report, current_surface_report, ar)
                                point_ok = joint_merit['construction_progress']
                                merit['joint_merit'] = joint_merit
                                merit.update({'surface_pairs_before': before_surface, 'surface_pairs_after': sc['total_crossing_pairs'], 'surface_report': str(ap / 'report.json'), 'joint_construction_progress': point_ok})
                                if point_ok:
                                    current_surface_count = sc['total_crossing_pairs']
                                    current_surface_report = ar
                        except (TypeError, ValueError) as exc:
                            point_meta['progress_merit_error'] = f'{type(exc).__name__}: {exc}'
                if point_report is not None:
                    point_meta.update({'status': point_report.get('status'), 'inside_node_bank_pairs': point_report.get('inside_node_bank_pairs'), 'inside_nodes': point_report.get('inside_nodes'), 'complete_surface_audit': point_report.get('complete_surface_audit'), 'exact_PLC_surface_pairs': point_report.get('exact_PLC_surface_pairs'), 'no_new_inside_typed_pairs': no_new_inside, 'strict_inside_pair_reduction': strict_reduction, 'progress_merit': merit, 'inside_pairs_first12': sorted([list(x) for x in inside_trial_set])[:12]})
            good = capok and anchorok and stepok and local_area_ok and exactok and point_ok
            trial_records.append({'alpha': alpha, 'max_cumulative_source_displacement_mm': float(capdist.max()), 'source_cap_12mm_ok': capok, 'anchors_bitwise_unchanged': anchorok, 'max_increment_mm': stepmax, 'step_2mm_ok': stepok, 'exit_progress_nonregressing': progressok, 'hard_exit_rows': len(hard_rows), 'hard_exit_max_remaining_mm': float(np.maximum(0, hard_targets - pnow).max(initial=0)), 'allface_flip_count': int(len(flipped_now)), 'first_flipped_face_rows': flipped_now[:20].tolist(), 'active_face_below_10pct_count': int(len(active_low)), 'self_linear_barrier_ok': self_linear_ok, 'self_linear_max_violation_mm': float(self_linear_violation), 'self_projection_safety_margin_mm': self_safety_margin_mm, 'exact_self_complete': bool(self_report.get('complete_surface_audit', False)), 'exact_self_pair_count': int(len(self_pairs)), 'self_pairs_first8': self_pairs[:8].tolist(), 'self_pair_witnesses': [{'face_rows': [int(ii), int(jj)], 'face_vertex_rows': [F[int(ii)].astype(int).tolist(), F[int(jj)].astype(int).tolist()], 'trial_triangle_points_world_f32_m': [yp[F[int(ii)]].astype(float).tolist(), yp[F[int(jj)]].astype(float).tolist()], 'safe_start_triangle_points_world_f32_m': [cur_m[F[int(ii)]].astype(float).tolist(), cur_m[F[int(jj)]].astype(float).tolist()]} for ii, jj in self_pairs], 'exact_point_gate': point_meta, 'trial_candidate_sha256': sha(trial_dir / 'candidate.npz') if (trial_dir / 'candidate.npz').is_file() else None, 'progress_merit': merit, 'accepted': bool(good)})
            if len(flipped_now) or len(active_low):
                newly_bad.extend(np.asarray(active[np.asarray(active_low, np.int64)], np.int64).tolist())
                newly_bad.extend(flipped_now.tolist())
                newly_bad.extend(np.flatnonzero(ratios < AREA_FLOOR_RATIO).tolist())
            if point_ok:
                accepted_point_audits.append({'scp_iteration': scp + 1, 'alpha': alpha, 'summary_sha256': point_meta.get('summary_sha256'), 'inside_node_bank_pairs': len(inside_trial_set), 'inside_nodes': int(point_report.get('inside_nodes', -1)), 'progress_merit': merit})
            elif point_report is not None:
                rejected_point_audits.append({'scp_iteration': scp + 1, 'alpha': alpha, 'summary_sha256': point_meta.get('summary_sha256'), 'inside_node_bank_pairs': int(point_report.get('inside_node_bank_pairs', -1)), 'progress_merit': merit})
            if good:
                chosen = (alpha, yp, self_report, self_pairs, ratios, inside_trial_set, int(point_report['inside_node_bank_pairs']), point_report, merit)
                break
        stage = {'scp_iteration': scp + 1, 'status': 'ACCEPTED' if chosen else 'NO_ACCEPTABLE_BACKTRACK', 'admm_converged': bool(converged), 'partial_iterate_exact_gated': bool(partial_iterate_used), 'active_area_face_count': len(active), 'active_face_color_count': len(groups), 'barycentric_rows': len(btri), 'barycentric_projection': bary_projection, 'area_linearized_max_violation_mm2': float(area_projection.get('max_violation_mm2', float('inf'))), 'admm_iterations': log[-1]['iteration'] if log else 0, 'admm_residual_log': log, 'node_projection': node_projection, 'area_projection': area_projection, 'factor_seconds': factor_s, 'solve_seconds': solve_s, 'elapsed_seconds': time.monotonic() - t0, 'exit_constraint_progress_max_mm': float((lower - curproj).max(initial=0)), 'trials': trial_records}
        stages.append(stage)
        if chosen:
            alpha, yp, sr, sp, ratios, accepted_inside_set, accepted_inside_count, accepted_report, accepted_merit = chosen
            previous_inside_set = set(current_inside_set)
            dcur = yp * 1000.0 - Xmm
            accepted_any = True
            current_inside_set = set(accepted_inside_set)
            current_inside_pair_count = int(accepted_inside_count)
            current_point_report = accepted_report
            newly_inside_query_rows = {int(k[0]) for k in current_inside_set - previous_inside_set}
            uncovered_rows = sorted(newly_inside_query_rows - set(map(int, hard_rows)))
            stage['uncovered_new_query_rows_without_frozen_target'] = uncovered_rows
            stage['accepted_progress_merit'] = accepted_merit
            if uncovered_rows:
                stop_reason = 'accepted_progress_requires_fresh_targets_for_new_query_rows'
                break
            if np.max(np.maximum(0, hard_targets - np.einsum('ij,ij->i', hard_dirs, dcur[hard_rows]))) <= 0.05:
                stop_reason = 'all_feasible_exit_targets_reached'
                break
            continue
        witnessed = set()
        for tr in trial_records:
            for witness in tr.get('self_pair_witnesses', []):
                key = tuple(sorted(map(int, witness['face_rows'])))
                witnessed.add(key)
        pair_changed = False
        for pair in sorted(witnessed):
            if pair not in self_pair_margin_fraction:
                self_pair_ids.append(pair)
                self_pair_ids.sort()
                self_pair_margin_fraction[pair] = 0.25
                pair_changed = True
            else:
                old_fraction = self_pair_margin_fraction[pair]
                new_fraction = min(0.75, old_fraction + 0.15)
                if new_fraction > old_fraction:
                    self_pair_margin_fraction[pair] = new_fraction
                    pair_changed = True
        if pair_changed:
            stage['rebuilt_threat_pair_ids'] = [list(x) for x in sorted(witnessed)]
            stage['pair_margin_fraction_after_reject'] = {f'{a}:{b}': self_pair_margin_fraction[a, b] for a, b in sorted(witnessed)}
            continue
        add = np.unique(np.asarray(newly_bad, np.int64))
        if len(add):
            active = add_one_ring(active, F, add)
            stage['added_area_rows_for_next_scp'] = int(len(active))
            continue
        stop_reason = 'no_safe_area_self_pair_backtrack'
        break
    final_m = (Xmm + dcur) / 1000.0
    final_f32 = final_m.astype(np.float32)
    final_ratios, _ = area_ratios_mm(final_f32, F, ref_cross)
    final_flip = np.flatnonzero(final_ratios <= 0)
    payload['points_world_m'] = final_f32
    CAND = OUT / 'candidate.npz'
    with CAND.open('xb') as fp:
        np.savez_compressed(fp, **payload)
    if len(final_flip):
        np.save(OUT / 'face-area-nonpositive-rows.npy', final_flip)
    report = {'schema': 'numi.human.refined-crossing-global-connected-scp.v2', 'status': 'CANDIDATE_WRITTEN_UNADMITTED' if accepted_any else 'NO_ACCEPTED_SCP_STEP_START_PRESERVED', 'classification': 'construction_geometry_only_no_anatomical_physical_or_native_admission', 'inputs': {'start_sha256': sha(START), 'base_702_sha256': sha(BASE), 'target_source_start_sha256': sha(TARGET_START), 'current_point_report_sha256': sha(INSIDE_REPORT), 'old_rejected_fit_sha256': None, 'nearest_union_report_sha256': sha(TARGET_JSON), 'crossing_report_sha256': sha(CROSSING_REPORT), 'crossing_count_receipt_sha256': sha(CROSSING_COUNT), 'refinement_candidate_sha256': sha(START), 'refinement_source_sha256': sha(R / 'run-029-affine-crossed-face-refinement/refine_conforming_029.py'), 'nearest_union_npz_sha256': sha(TARGET_NPZ), 'exact_predicate_sha256': sha(PRED), 'support_helper_sha256': sha(Path(__file__).with_name('area_scp_support_023.py')), 'self_pair_helper_sha256': sha(Path(__file__).with_name('self_separation_026.py')), 'parent_run032_fit_report_sha256': sha(PARENT_FIT), 'progress_merit_helper_sha256': sha(PROGRESS_HELPER), 'batched_node_projector_sha256': sha(NODE_PROJECTOR), 'barycentric_target_sha256': sha(BARY_TARGETS), 'barycentric_helper_sha256': sha(BARY_HELPER)}, 'constraints': {'source_cap_mm': SOURCE_CAP_MM, 'proposal_target_max_mm': TARGET_MAX_MM, 'local_trust_step_mm': STEP_MM, 'progressive_lens_support_fraction': 0.95, 'area_floor_ratio': AREA_FLOOR_RATIO, 'area_constrained_seed_faces': flipped.astype(int).tolist(), 'area_constrained_seed_semantics': 'all skin faces with at least one exact Run029 159 crossing pair', 'area_constrained_seed_face_count': int(len(flipped)), 'active_area_faces_final': int(len(active)), 'hard_exit_feasible_rows': int(len(hard_rows)), 'current_inside_nodes': len(inside_rows), 'current_inside_query_pairs': inside_pair_count, 'inside_unmatched_target_rows': missing_targets, 'target_rows_already_outside_currently': already_outside_target_rows, 'unresolved_proposals': unresolved, 'fixed_anchor_rows': len(anchors), 'self_pair_seed_ids': [], 'self_pair_projection_relaxation_mm': PAIR_NUMERICAL_RELAXATION_MM, 'self_pair_seed_effective_margins_mm': {f"{r['faces'][0]}:{r['faces'][1]}": r['effective_preserved_margin_mm'] for r in self_pair_seed_records or []}, 'self_pair_rows_initial': self_pair_seed_records, 'self_pair_ids_final': [list(x) for x in self_pair_ids], 'initial_inside_node_bank_pairs': inside_pair_count, 'final_accepted_inside_node_bank_pairs': current_inside_pair_count, 'accepted_point_gate_summaries': accepted_point_audits, 'rejected_point_gate_summaries': rejected_point_audits, 'self_pair_margin_fractions_final': {f'{a}:{b}': v for (a, b), v in self_pair_margin_fraction.items()}, 'all_faces_checked_without_rounding_mask': True}, 'solver': {'method': 'Global SCP using frozen Run033 first-union targets for the Run032 best partial start, graph-Laplacian soft targets, exact source/trust/exit projections, oriented-area guards, dynamic sparse self-pair barriers, and the verified P090 batched projector. Every accepted iterate passes a fresh exact self audit, complete exact 185-bank point audit, and P089 construction-progress merit (strict typed-pair reduction with nonincreasing maximum and squared penetration depth). Newly introduced shallow pairs are retained.', 'target_weight': TARGET_WEIGHT, 'laplacian_weight': LAMBDA, 'source_regularization': ETA, 'admm_rho': RHO, 'stop_reason': stop_reason, 'accepted_steps': sum((s['status'] == 'ACCEPTED' for s in stages)), 'stages': stages}, 'final_geometry': {'candidate_path': str(CAND), 'candidate_sha256': sha(CAND), 'points': len(final_f32), 'faces': len(F), 'changed_nodes': int(np.count_nonzero(np.any(final_f32 != X.astype(np.float32), axis=1))), 'max_source_displacement_mm': float(np.linalg.norm(final_f32.astype(float) - R0, axis=1).max() * 1000), 'source_cap_pass': bool(np.linalg.norm(final_f32.astype(float) - R0, axis=1).max() * 1000 <= SOURCE_CAP_MM + 0.002), 'anchors_bitwise_unchanged': bool(np.ascontiguousarray(final_f32[anchors]).tobytes() == np.ascontiguousarray(X.astype(np.float32)[anchors]).tobytes()), 'minimum_allface_oriented_area_ratio': float(final_ratios.min()), 'allface_nonpositive_count': int(len(final_flip)), 'allface_first_nonpositive_rows': final_flip[:30].astype(int).tolist(), 'self_audit_ran_on_each_accepted_iterate': True, 'every_backtracking_trial_candidate_saved_before_geometry_gates': True, 'linear_pair_projection_can_use_bounded_residual': True, 'wall_seconds': time.monotonic() - started}, 'limitations': 'Construction geometry only. Finite ADMM partial iterates are proposal-only; convergence is not an acceptance criterion. Every backtracking trial is saved before gating. Frozen Run033 first-union targets cover all 1,239 Run032-start inside query rows. Each accepted step passes exact self and fresh complete 185-bank point audits plus P089 strict count/depth merit; introduced typed pairs are allowed and recorded. If a newly inside query row lacks a frozen target, the run stops with a retarget-required status. The matched Run032 surface crossing-pair count is 4,156 and seeds oriented-area guards; later face counts are not directly comparable after geometry changes. The result has no anatomical, material, physical, or native admission.'}
    (OUT / 'fit-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'candidate': str(CAND), 'sha256': sha(CAND), 'report': str(OUT / 'fit-report.json'), 'report_sha256': sha(OUT / 'fit-report.json'), 'stop_reason': stop_reason, 'accepted_steps': report['solver']['accepted_steps'], 'allface_nonpositive': len(final_flip), 'max_source_displacement_mm': report['final_geometry']['max_source_displacement_mm']}, indent=2))
START = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/candidate.npz')
TARGET_START = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/candidate.npz')
INSIDE_REPORT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/scp-02-alpha-0p125/audit-130/summary.json')
TARGET_JSON = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-010-capped-current-normal-soft/point-targets/nearest-union-exits.json')
TARGET_NPZ = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-010-capped-current-normal-soft/point-targets/nearest-union-exits.npz')
OLD_TRIAL = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/candidate.npz')
CROSSING_REPORT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/scp-02-alpha-0p125/audit-159/report.json')
CROSSING_COUNT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-010-capped-current-normal-soft/surface-count.json')
OUT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-010-capped-current-normal-soft/attempt-001')
PARENT_FIT = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-006-rebased-soft/attempt-001/fit-report.json')
PIN = {'start': '931529f21497521436aa15d52ebebf71a1e243c5a32d9b301d3f0a99e3035cc7', 'target_start': '931529f21497521436aa15d52ebebf71a1e243c5a32d9b301d3f0a99e3035cc7', 'inside_report': '3257c9b17400e196b126df61ab862212f52fde49feaac7b892b05eff554f74e0', 'target_json': '2f6083e4346dc7d5c57ccaf24e217579fcfd05fbbaf44187a119c1523ce81755', 'target_npz': 'c5b90c8e359209f6ebc86904f270a3a943c9d171f3f2161dbababa14259fb2b6', 'base': '70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31', 'old_trial': '931529f21497521436aa15d52ebebf71a1e243c5a32d9b301d3f0a99e3035cc7', 'predicate': 'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d', 'self_helper': 'c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3', 'support_helper': 'aaa9dbe32fbfeb71818a2afd673ef4eb7dfc0c209cee6d0ee1dbff1a6ce71abd', 'crossing_report': '17036a56d8f4b7c38b8e91f3350f354954508fa157411285b1406c1b2c467f3b', 'crossing_count': 'a53babdd03a5ffb4ffee45ceda556ff7f6bb162048bb9ac5b83ddb2a75220fab', 'audit130': '52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9', 'audit159': 'de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1', 'progress_helper': '37c040a442f1fdf312745c1cc1fb6c543383f1cd0782db46c81ca1eadf0d8324', 'node_projector': '9175186e4a916dbfd8b2ce4d10f4bd7bb09938cecf442b291bc093823a1336e7', 'parent_fit': '2dbd7b56bf0535b7a54b7279ad7492de95efd47a1b9caa24b4009aacd35469c3'}
TARGET_NODE_COUNT = 13
SURFACE_COUNT = 3626
BARY_HELPER = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/human_skin_fat_barycentric_constraints.py')
BARY_TARGETS = Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-fit-010-capped-current-normal-soft/barycentric-targets.npz')
BARY_TARGET_SHA = '5f7d54a23cfa1dc6519cf4fcb327a1cc56d983ba6de130ab42f5e4f5167e4609'
BARY_HELPER_SHA = '965a166e86952d79c019302c98e7684f2f8ef3a9d63b328430ac9c8c8edbaff2'
PROGRESS_HELPER = Path('/Users/numi/Work/numi-human-connected-surface-069/D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py')
BARY_WEIGHT = 8.0
BEST_SURFACE_COUNT = 3626
MAX_SCP = 4
STEP_MM = 1.0
MAX_SOLVE_S = 20.0
if __name__ == '__main__':
    main()
