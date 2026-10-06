from pathlib import Path
import hashlib, heapq, json, sys, time
import numpy as np

E = Path('/Users/n/numi-human-resting-evidence-20261005')
sys.path.insert(0, '/Users/n/numi-human-resting-conforming-source-009/matter/tools')
import accepted_mrvpack_surface_audit as audit

run = E / 'joint-costal-conditioned-native-049'
step = int(sys.argv[1])
out = run / f'passive-neighbor-audit-step-{step}.json'
assert not out.exists()
pack = run / 'accepted-geometry' / f'step-{step}.mrvpack'
receipt = run / 'accepted-geometry' / f'step-{step}.receipt.json'
anatomy = E / 'costal-current003-full-chain-003/resting-anatomy-receipt.json'
r = json.loads(anatomy.read_text())
ids = r['functional_bindings']['passive_viscera_geometry_binding']['stable_ids']
predicate = Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py')
pred = audit.predicate_module(predicate)
mm, stream, offset, surfaces = audit.read_pack(pack)
provenance = audit.validate_accepted_receipt(pack, receipt, step, mm, offset, surfaces)
keys = [(51010, sid) for sid in ids]
assert all(k in surfaces for k in keys)
points = {sid: audit.surface_points(mm, offset, surfaces[(51010, sid)]['faces']) for sid in ids}
scale = max(p[2] for p in points.values())
cache, invalid = {}, {}
boxes = {}
for sid in ids:
    xyz = np.array(list(points[sid][1].values()))
    boxes[sid] = (xyz.min(axis=0), xyz.max(axis=0))
    try:
        cache[sid] = audit.make_records(points[sid], scale, surfaces[(51010, sid)]['faces'], pred)
    except ValueError as err:
        invalid[sid] = str(err)

def classify(a, b, intersections):
    common = set(a[0]) & set(b[0])
    if len(common) in (1, 2) and all(pred._allowed_shared_point(p, common) for p in intersections):
        return 'exact_shared_edge_or_vertex'
    if len(common) == 3:
        seq, seq2 = a[0], b[0]
        opposite = any(all(seq[i] == seq2[(j-i) % 3] for i in range(3)) for j in range(3))
        return 'exact_opposite_face' if opposite else 'exact_same_winding_face'
    return 'nonshared_intersection'

def first_nonshared(a, b):
    # Same exact sweep and predicates as the existing accepted-pack audit.
    ordered = sorted(b, key=lambda r: r[1][0])
    active, expiry, cursor = {}, [], 0
    counts = {'exact_shared_edge_or_vertex': 0, 'exact_opposite_face': 0,
              'exact_same_winding_face': 0}
    n = 0
    for row in sorted(a, key=lambda r: r[1][0]):
        while expiry and expiry[0][0] < row[1][0]:
            _, i = heapq.heappop(expiry)
            active.pop(i, None)
        while cursor < len(ordered) and ordered[cursor][1][0] <= row[2][0]:
            other = ordered[cursor]
            if other[2][0] >= row[1][0]:
                active[other[3]] = other
                heapq.heappush(expiry, (other[2][0], other[3]))
            cursor += 1
        for other in active.values():
            if any(row[2][d] < other[1][d] or other[2][d] < row[1][d] for d in range(3)):
                continue
            n += 1
            intersections = pred.triangle_intersection_points(row[0], other[0])
            if not intersections:
                continue
            kind = classify(row, other, intersections)
            if kind != 'nonshared_intersection':
                counts[kind] += 1
                continue
            return {'audit_complete': False, 'count_is_lower_bound': True,
                    'aabb_candidate_pairs': n, 'geometric_contacts': counts,
                    'first_nonshared': {'triangle_pair': [row[3], other[3]],
                        'shared_exact_points': len(set(row[0]) & set(other[0])),
                        'intersection_points_m': [[float(c)/scale for c in p] for p in intersections]}}
    return {'audit_complete': True, 'count_is_lower_bound': False,
            'aabb_candidate_pairs': n, 'geometric_contacts': counts, 'first_nonshared': None}

report = {'schema': 'accepted-mrvpack-pair-audit.v2', 'step': step,
          'accepted_receipt': provenance, 'pack_sha256': audit.sha256_file(pack),
          'driver_sha256': audit.sha256_file(Path(__file__)),
          'existing_audit_owner_sha256': audit.sha256_file(Path(audit.__file__)),
          'predicate_sha256': audit.sha256_file(predicate),
          'anatomy_receipt_sha256': audit.sha256_file(anatomy),
          'stable_ids': ids, 'invalid_surfaces': invalid, 'pairs': [],
          'limitation': 'First nonshared exact intersection for all pairs of the registered passive-viscera set in one accepted frame. Shared geometric contacts are classified, not asserted anatomically intended. Closed-surface containment, self-intersections, physical penetration depth and full-cycle clearance require separate checks.'}
start = time.monotonic()
for i, a in enumerate(ids):
    for b in ids[i+1:]:
        row = {'first': [51010, a], 'second': [51010, b],
               'names': [r['provenance']['source_id_map'][str(s)]['name'] for s in (a, b)]}
        if a in invalid or b in invalid:
            row['invalid_source'] = True
        elif np.any(boxes[a][1] < boxes[b][0]) or np.any(boxes[b][1] < boxes[a][0]):
            row.update({'aabb_disjoint': True, 'audit_complete': True, 'first_nonshared': None})
        else:
            row.update(first_nonshared(cache[a], cache[b]))
        report['pairs'].append(row)
        out.write_text(json.dumps(report, indent=2) + '\n')
        if row.get('first_nonshared'):
            print(json.dumps(row), flush=True)
report['all_requested_pairs_visited'] = len(report['pairs']) == len(ids) * (len(ids)-1) // 2
report['elapsed_seconds'] = time.monotonic()-start
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'path': str(out), 'sha256': audit.sha256_file(out),
    'pair_count': len(report['pairs']), 'nonshared_pair_count': sum(bool(p.get('first_nonshared')) for p in report['pairs']),
    'invalid_surfaces': invalid, 'elapsed_s': report['elapsed_seconds']}), flush=True)
mm.close()
stream.close()
