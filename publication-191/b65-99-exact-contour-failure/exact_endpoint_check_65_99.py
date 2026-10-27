from __future__ import annotations
import hashlib, importlib.util, json, math, struct, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np

S = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
C = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
A = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json"
O5 = S / "per-bank-workers-20261027/b115-118/attempt-005"
OUT = S / "per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity"
REPORT_OLD = O5 / "conforming-patch-003/report.json"
CONTOUR_OLD = O5 / "local-contour-geometry.json"
AUDIT_TOOL = ROOT / "audit_skin_bone_surfaces_159.py"
HELPER = Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT = E / "collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
SEGMENT_HELPER = E / "root_exact_segment_witness.py"
BANKDIR = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def frac(x):
    return Fraction.from_float(float(x))

def exact_tri(tri):
    return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3, 3))

def barycentric(p, tri, exact):
    n = exact._cross(exact._sub(tri[1], tri[0]), exact._sub(tri[2], tri[0]))
    axes = exact._axes(n)
    a = tuple(tri[0][k] for k in axes)
    b = tuple(tri[1][k] for k in axes)
    c = tuple(tri[2][k] for k in axes)
    q = tuple(p[k] for k in axes)
    den = exact._orient2(a, b, c)
    if den == 0:
        raise ValueError("exact degenerate triangle")
    return (exact._orient2(b, c, q) / den,
            exact._orient2(c, a, q) / den,
            exact._orient2(a, b, q) / den)

def feature_for(bary, vertex_ids, point, vertex_keys, side, body, bone):
    zeros = [i for i, w in enumerate(bary) if w == 0]
    if len(zeros) >= 2:
        i = next(i for i, w in enumerate(bary) if w == 1)
        if side == "skin":
            return ("skin-vertex", int(vertex_ids[i]))
        return ("bone-vertex", int(body), int(bone), vertex_keys[i])
    if len(zeros) == 1:
        ii = [i for i in range(3) if i != zeros[0]]
        i, j = ii
        if side == "skin":
            vi, vj = int(vertex_ids[i]), int(vertex_ids[j])
            lo, hi = sorted((vi, vj))
            t = bary[j] if vj == hi else bary[i]
            return ("skin-edge", lo, hi, t)
        ki, kj = vertex_keys[i], vertex_keys[j]
        lo, hi = sorted((ki, kj))
        t = bary[j] if kj == hi else bary[i]
        return ("bone-emitted-edge", int(body), int(bone), lo, hi, t)
    return None

def point_on_triangle(p, tri, exact):
    normal = exact._cross(exact._sub(tri[1], tri[0]), exact._sub(tri[2], tri[0]))
    if not any(normal) or exact._dot(normal, exact._sub(p, tri[0])) != 0:
        return False
    axes = exact._axes(normal)
    p2 = tuple(p[k] for k in axes)
    signs = [exact._orient2(tuple(a[k] for k in axes), tuple(b[k] for k in axes), p2)
             for a, b in zip(tri, tri[1:] + tri[:1])]
    return min(signs) >= 0 or max(signs) <= 0

def rounded_f32(q):
    return struct.unpack("<f", struct.pack("<f", float(q)))[0]

def fraction_string(q):
    return f"{q.numerator}/{q.denominator}"

def feature_string(feature):
    if feature is None:
        return None
    return json.dumps(feature, default=lambda v: fraction_string(v) if isinstance(v, Fraction) else str(v), separators=(",", ":"))

started = time.monotonic()
if sha(C) != "7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af":
    raise RuntimeError("candidate pin changed")
if sha(A) != "c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e":
    raise RuntimeError("audit report pin changed")
if sha(REPORT_OLD) != "7f72e5c45cac788ee3c39813d3d36da01687f65272ff4ab36fc5daefc42320ad":
    raise RuntimeError("prior patch report pin changed")
if sha(CONTOUR_OLD) != "41db72c7c5cd40c6a3058a2edd3bddeb25c9f68e2742c829472f9bb901601a90":
    raise RuntimeError("prior contour report pin changed")

audit_tool = loadmod("endpoint_identity_audit_owner", AUDIT_TOOL)
helper = loadmod("endpoint_identity_bank_helper", HELPER)
exact = audit_tool.exact_module()
with np.load(C, allow_pickle=False) as z:
    P32 = np.asarray(z["points_world_m"], dtype=np.float32).copy()
    F = np.asarray(z["source_O_faces"], dtype=np.int64).copy()
world64 = P32.astype(np.float64)
T, ids, gids, bodies, M, off, pos, bankkeys, bankpins, unused = helper.read_bank_inputs(BANKDIR, E)
bank_mask = (ids[:, 0] == 65) & (ids[:, 1] == 99)
bone_ids = ids[bank_mask, 3].astype(np.int64)
bone_raw = T[bank_mask].astype(np.float64)
bone_rows = np.flatnonzero(bank_mask)
bone_index = {int(stable): i for i, stable in enumerate(bone_ids)}
if len(bone_index) != len(bone_ids):
    raise RuntimeError("bank triangle stable IDs are not unique")
skin_local = (world64 - pos[65]) @ M[65].T - off[65]
bank_tri_exact = [exact_tri(t) for t in bone_raw]
skin_node_lattice = None
bank_vertex_keys = [[exact.float32_point_lattice_key(v) for v in tri] for tri in bone_raw]
audit_doc = json.loads(A.read_text())
bank_doc = next(x for x in audit_doc["banks"] if (x["body"], x["stable_bone_id"]) == (65, 99))
pairs = [tuple(map(int, pair)) for pair in bank_doc["result"]["triangle_pairs"]]
if len(pairs) != 148:
    raise RuntimeError(f"expected pinned 148 pair scope, found {len(pairs)}")

# Segment/triangle helper review: exact signs are useful, but its public hit
# parameters are floats; exact triangle_intersection_points is used below.
endpoint_rows = []
pair_status = []
segment_rows = []
point_contact_count = 0
area_contact_pairs = []
duplicate_exact_points = 0
for sf, bone_stable in pairs:
    if not 0 <= sf < len(F) or bone_stable not in bone_index:
        raise RuntimeError(f"pair references absent face or bank triangle {(sf, bone_stable)}")
    bi = bone_index[bone_stable]
    tri_s = exact_tri(skin_local[F[sf]])
    tri_b = bank_tri_exact[bi]
    raw_points = exact.triangle_intersection_points(tri_s, tri_b)
    uniq = sorted(set(raw_points))
    duplicate_exact_points += len(raw_points) - len(uniq)
    if not uniq:
        pair_status.append({"skin_face": sf, "bone_triangle": bone_stable, "dimension": "empty-after-exact-reconstruction"})
        continue
    if len(uniq) == 1:
        dimension = "point"
        selected = uniq
        point_contact_count += 1
    else:
        p0 = uniq[0]
        direction = next((exact._sub(p, p0) for p in uniq[1:] if p != p0), None)
        collinear = all(not any(exact._cross(direction, exact._sub(p, p0))) for p in uniq[1:])
        if collinear:
            axis = max(range(3), key=lambda k: max(p[k] for p in uniq) - min(p[k] for p in uniq))
            selected = [min(uniq, key=lambda p: p[axis]), max(uniq, key=lambda p: p[axis])]
            dimension = "segment"
            if selected[0] == selected[1]:
                dimension = "point"
                selected = [selected[0]]
                point_contact_count += 1
        else:
            dimension = "area"
            selected = uniq
            area_contact_pairs.append([sf, bone_stable, len(uniq)])
    length_m = None
    if dimension == "segment":
        length_m = math.sqrt(sum(float(selected[1][k] - selected[0][k]) ** 2 for k in range(3)))
    pair_status.append({"skin_face": sf, "bone_triangle": bone_stable,
                        "dimension": dimension, "exact_intersection_vertex_count": len(uniq),
                        "exact_local_endpoints_m": [[fraction_string(x) for x in p] for p in selected],
                        "segment_length_m_for_reporting_only": length_m})
    if dimension == "area":
        continue
    face_nodes = [int(v) for v in F[sf]]
    pair_endpoint_ids = []
    for endpoint_ordinal, p in enumerate(selected):
        sb = barycentric(p, tri_s, exact)
        bb = barycentric(p, tri_b, exact)
        if min(sb) < 0 or max(sb) > 1 or min(bb) < 0 or max(bb) > 1:
            raise RuntimeError(f"exact intersection point outside input triangles {(sf, bone_stable)}")
        fs = feature_for(sb, face_nodes, p, None, "skin", 65, 99)
        fb = feature_for(bb, None, p, bank_vertex_keys[bi], "bone", 65, 99)
        # Exact world location uses source skin triangle barycentrics and emitted F32 metres.
        wp = tuple(sum(sb[j] * frac(P32[face_nodes[j], k]) for j in range(3)) for k in range(3))
        rec_id = len(endpoint_rows)
        endpoint_rows.append({
            "id": rec_id, "skin_face": sf, "bone_triangle": bone_stable,
            "kind": dimension, "endpoint_ordinal": endpoint_ordinal, "local_point": p,
            "skin_bary": sb, "bone_bary": bb, "skin_feature": fs, "bone_feature": fb,
            "world_exact": wp,
        })
        pair_endpoint_ids.append(rec_id)
    if dimension == "segment":
        segment_rows.append((sf, bone_stable, pair_endpoint_ids))
# Each segment pair contributes two endpoint rows; point-contact pairs contribute one.
if area_contact_pairs:
    raise RuntimeError(f"exact area-overlap cases require a separate 2D patch treatment: {area_contact_pairs[:6]}")
if any(row["dimension"] == "empty-after-exact-reconstruction" for row in pair_status):
    raise RuntimeError("pinned intersecting pairs did not reconstruct exact points")

parent = list(range(len(endpoint_rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
feature_owner = {}
for r in endpoint_rows:
    for feat in (r["skin_feature"], r["bone_feature"]):
        if feat is None:
            continue
        if feat in feature_owner:
            union(r["id"], feature_owner[feat])
        else:
            feature_owner[feat] = r["id"]
components = {}
for r in endpoint_rows:
    components.setdefault(find(r["id"]), []).append(r["id"])
coord_conflicts = []
component_world_conflicts = []
for root, members in components.items():
    local_points = {endpoint_rows[i]["local_point"] for i in members}
    world_points = {endpoint_rows[i]["world_exact"] for i in members}
    if len(local_points) != 1:
        coord_conflicts.append({"root": root, "record_ids": members, "exact_local_coordinate_count": len(local_points)})
    if len(world_points) != 1:
        component_world_conflicts.append({"root": root, "record_ids": members, "exact_world_coordinate_count": len(world_points)})

# Graph incidence using exact canonical source-feature identities, never a coordinate bin.
degree = Counter()
edge_mult = Counter()
collapsed_segments = []
for sf, bone_stable, pair_endpoint_ids in segment_rows:
    # Pair-scoped row order is the exact parameter order chosen from exact coordinates.
    # There are exactly two selected rows for a segment pair.
    if len(pair_endpoint_ids) != 2:
        raise RuntimeError(f"expected two segment endpoints for pair {(sf, bone_stable)}")
    ra, rb = find(pair_endpoint_ids[0]), find(pair_endpoint_ids[1])
    if ra == rb:
        collapsed_segments.append([sf, bone_stable])
        continue
    e = tuple(sorted((ra, rb)))
    edge_mult[e] += 1
    degree[ra] += 1
    degree[rb] += 1
degree_hist = Counter(degree.values())
adj = {root: set() for root in degree}
for e in edge_mult:
    a, b = e
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)
graph_components = []
unseen = set(adj)
while unseen:
    seed = min(unseen)
    unseen.remove(seed)
    stack = [seed]
    comp = []
    while stack:
        v = stack.pop()
        comp.append(v)
        for w in adj[v]:
            if w in unseen:
                unseen.remove(w)
                stack.append(w)
    graph_components.append(sorted(comp))
point_roots = sorted({find(r["id"]) for r in endpoint_rows if r["kind"] == "point"})
for root in point_roots:
    degree.setdefault(root, 0)

# Canonical Float32 world emission per identity component, followed by exact
# point-on-triangle checks in the same bank-local metre frame as the audit.
emitted_by_root = {}
emission_inconsistent_roots = []
rounded_on_all_incident_pairs = {}
round_fail_skin = round_fail_bone = 0
for root, members in components.items():
    worlds = {endpoint_rows[i]["world_exact"] for i in members}
    if len(worlds) != 1:
        emission_inconsistent_roots.append(root)
        continue
    world_exact = next(iter(worlds))
    rounded = tuple(rounded_f32(x) for x in world_exact)
    emitted_by_root[root] = rounded
    all_ok = True
    for i in members:
        r = endpoint_rows[i]
        world_emitted64 = np.asarray(rounded, dtype=np.float64)
        local_emitted64 = (world_emitted64 - pos[65]) @ M[65].T - off[65]
        p_local = tuple(frac(x) for x in local_emitted64)
        s_tri = exact_tri(skin_local[F[r["skin_face"]]])
        b_tri = bank_tri_exact[bone_index[r["bone_triangle"]]]
        s_ok = point_on_triangle(p_local, s_tri, exact)
        b_ok = point_on_triangle(p_local, b_tri, exact)
        if not s_ok:
            round_fail_skin += 1
            all_ok = False
        if not b_ok:
            round_fail_bone += 1
            all_ok = False
    rounded_on_all_incident_pairs[root] = all_ok
rounded_key_to_roots = {}
for root, p in emitted_by_root.items():
    bits = tuple(struct.unpack("<I", struct.pack("<f", x))[0] for x in p)
    rounded_key_to_roots.setdefault(bits, []).append(root)
float32_collapsed_distinct_roots = [roots for roots in rounded_key_to_roots.values() if len(roots) > 1]

def serialize_feature(x):
    if isinstance(x, Fraction):
        return fraction_string(x)
    if isinstance(x, tuple):
        return [serialize_feature(y) for y in x]
    if isinstance(x, list):
        return [serialize_feature(y) for y in x]
    return x

exact_pair_witnesses = []
for sf, bone_stable in pairs:
    rows = [r for r in endpoint_rows if r["skin_face"] == sf and r["bone_triangle"] == bone_stable]
    exact_pair_witnesses.append({
        "skin_face": sf, "bone_triangle": bone_stable,
        "exact_local_endpoints_m": [[fraction_string(x) for x in r["local_point"]] for r in rows],
        "exact_world_endpoints_m": [[fraction_string(x) for x in r["world_exact"]] for r in rows],
        "skin_features": [serialize_feature(r["skin_feature"]) for r in rows],
        "bone_emitted_edge_features": [serialize_feature(r["bone_feature"]) for r in rows],
    })

receipt = {
    "schema": "numi.human.bank65_99.exact-endpoint-identity-check.v1",
    "script_path": str(Path(__file__)), "script_sha256": sha(Path(__file__)),
    "status": "EXACT_ENDPOINT_GRAPH_CHECKED",
    "candidate_path": str(C), "candidate_sha256": sha(C),
    "audit_report_path": str(A), "audit_report_sha256": sha(A),
    "prior_patch_report_path": str(REPORT_OLD), "prior_patch_report_sha256": sha(REPORT_OLD),
    "prior_contour_report_path": str(CONTOUR_OLD), "prior_contour_report_sha256": sha(CONTOUR_OLD),
    "exact_predicate_path": str(EXACT), "exact_predicate_sha256": sha(EXACT),
    "bank_helper_path": str(HELPER), "bank_helper_sha256": sha(HELPER),
    "segment_helper_path": str(SEGMENT_HELPER), "segment_helper_sha256": sha(SEGMENT_HELPER),
    "bank_input_pins": bankpins,
    "scope": {"body": 65, "stable_bone_id": 99, "pinned_exact_triangle_pair_count": len(pairs),
              "candidate_surface_face_count": int(len(F)), "crossing_skin_faces": sorted({sf for sf, _ in pairs})},
    "method": {
        "predicate": "exact.triangle_intersection_points on the same Float32-in-metres skin and bank coordinates after the pinned audit's Float64 body-local transform; no millimetre conversion before rationalization",
        "skin_feature": "source_O_faces vertex IDs plus exact Fraction edge parameter ordered by vertex ID",
        "bone_feature": "exact Float32 body-local emitted vertex coordinate keys plus exact Fraction edge parameter; triangle CSV has no explicit bone vertex IDs",
        "helper_review": "exact_segment_triangle_hits filters candidates with floating arithmetic; root_exact_segment_witness returns exact-sign classifications but serializes t and barycentrics as floats; neither public result preserves rational shared-edge parameters",
        "coplanar_and_point_cases": "exact points deduplicated by Fraction; isolated points retained separately; non-collinear polygon overlap fails closed",
    },
    "exact_intersections": {
        "pair_dimension_counts": dict(Counter(x["dimension"] for x in pair_status)),
        "pair_status_and_exact_local_witnesses": pair_status,
        "exact_pair_endpoint_feature_witnesses": exact_pair_witnesses,
        "point_contact_pair_count": point_contact_count,
        "segment_pair_count": sum(x["dimension"] == "segment" for x in pair_status),
        "area_overlap_pair_count": len(area_contact_pairs),
        "duplicate_exact_vertices_removed": duplicate_exact_points,
        "endpoint_record_count": len(endpoint_rows),
        "endpoint_components_by_exact_feature_union": len(components),
        "exact_local_coordinate_conflict_components": coord_conflicts,
        "exact_world_coordinate_conflict_components": component_world_conflicts,
        "endpoints_missing_both_skin_and_bone_edge_vertex_features": sum(
            r["skin_feature"] is None and r["bone_feature"] is None for r in endpoint_rows
        ),
        "skin_edge_or_vertex_feature_count": sum(r["skin_feature"] is not None for r in endpoint_rows),
        "bone_edge_or_vertex_feature_count": sum(r["bone_feature"] is not None for r in endpoint_rows),
        "degree_histogram": {str(k): int(v) for k, v in sorted(degree_hist.items())},
        "graph_edge_count": int(sum(edge_mult.values())),
        "graph_vertex_count": len(adj),
        "graph_connected_component_count": len(graph_components),
        "graph_connected_component_vertex_counts": sorted(Counter(len(c) for c in graph_components).items()),
        "zero_degree_point_contact_components": sum(degree.get(r, 0) == 0 for r in point_roots),
        "collapsed_segments_after_exact_feature_union": collapsed_segments,
        "duplicate_canonical_segment_multiplicities": [
            {"endpoint_roots": list(e), "multiplicity": int(n)} for e, n in edge_mult.items() if n > 1
        ],
    },
    "float32_emission_recheck": {
        "scope": "rounded exact endpoint coordinates in world metres only; no patch interior vertices emitted",
        "canonical_components_with_consistent_exact_world_position": len(emitted_by_root),
        "components_with_conflicting_exact_world_positions": emission_inconsistent_roots,
        "distinct_exact_components_collapsed_to_one_Float32_point": float32_collapsed_distinct_roots,
        "endpoint_pair_checks_failing_skin_triangle": round_fail_skin,
        "endpoint_pair_checks_failing_bone_triangle": round_fail_bone,
        "components_still_on_every_incident_skin_and_bone_triangle": sum(rounded_on_all_incident_pairs.values()),
        "components_checked_after_emission": len(rounded_on_all_incident_pairs),
    },
    "local_patch_boundary_check": "not run; this bounded check constructs the exact endpoint identity graph and Float32 endpoint rechecks only",
    "geometry_or_native_admission": False,
    "limitations": [
        "No candidate or shared file was modified.",
        "No full mesh, all-bank, skin self-intersection, or native admission audit was run.",
        "Bone source vertex indices are absent from the pinned triangle CSV; bone shared-edge identity is exact emitted-coordinate identity, not a separately proven indexed source edge.",
        "Endpoint checks do not establish a conforming patch triangulation, topology admission, physical validity, or native physical admission."
    ],
    "elapsed_seconds": time.monotonic() - started,
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
print(json.dumps({"status": receipt["status"], "pair_dimension_counts": receipt["exact_intersections"]["pair_dimension_counts"],
                  "endpoint_records": len(endpoint_rows), "components": len(components),
                  "degree_histogram": receipt["exact_intersections"]["degree_histogram"],
                  "graph_components": receipt["exact_intersections"]["graph_connected_component_count"],
                  "graph_vertex_count": receipt["exact_intersections"]["graph_vertex_count"],
                  "coord_conflicts": len(coord_conflicts), "world_conflicts": len(component_world_conflicts),
                  "f32_collisions": len(float32_collapsed_distinct_roots),
                  "rounded_skin_fail": round_fail_skin, "rounded_bone_fail": round_fail_bone,
                  "receipt": str(OUT / "receipt.json")}, indent=2))
