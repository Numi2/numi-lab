from pathlib import Path
import hashlib, json
import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123")
INPUT = BASE / "run-027-global-hybrid/hybrid-O-surface-133.npz"
INPUT_REPORT = BASE / "run-027-global-hybrid/hybrid-O-surface-133-report.json"
OUT = BASE / "run-028-paired-ear-retriangulation"
EXPECTED_INPUT_SHA = "cc11c68d58ec492922c3c86247197e8d83bbaf1fec9f55cc1a6fc3fedbad5273"
PAIR_FACE, EAR_FACE = 10761, 10762
PAIR = np.array([5847, 2870, 2871], dtype=np.int64)
EAR = np.array([2871, 2870, 2302], dtype=np.int64)
NEW = np.array([[5847, 2870, 2302], [5847, 2302, 2871]], dtype=np.int64)
SOURCE_PARENTS = np.array([14642, 14650], dtype=np.int32)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def area_normal(points, tri):
    a, b, c = points[np.asarray(tri, dtype=np.int64)]
    cross = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(cross))
    return 0.5 * norm, cross / norm if norm else np.zeros(3)

def edge_incidence(faces):
    edges = {}
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (int(min(a, b)), int(max(a, b)))
            edges[e] = edges.get(e, 0) + 1
    return edges

assert sha(INPUT) == EXPECTED_INPUT_SHA, "input candidate hash mismatch"
assert sha(INPUT_REPORT) == "aba83b696f8fe0d8ee967ebd1077f94a49f291aedec94ff0d47ea44af5f54cda", "input report pin mismatch"
with np.load(INPUT, allow_pickle=False) as z:
    data = {k: z[k].copy() for k in z.files}
X = np.asarray(data["points_world_m"], dtype=np.float64)
F = np.asarray(data["source_O_faces"], dtype=np.int64)
assert F.shape == (17507, 3) and X.shape == (8776, 3)
assert np.array_equal(F[PAIR_FACE], PAIR), (F[PAIR_FACE], PAIR)
assert np.array_equal(F[EAR_FACE], EAR), (F[EAR_FACE], EAR)
markers = np.asarray(data["parent_source_face_marker"], dtype=np.int64)
assert markers[EAR_FACE] == SOURCE_PARENTS[1]
assert markers[PAIR_FACE] == SOURCE_PARENTS[0]
# Verify the third ear vertex lies strictly inside the shared edge.
a, b, p = X[EAR]
d = b - a
alpha = float(np.dot(p - a, d) / np.dot(d, d))
residual = float(np.linalg.norm(p - (a + alpha * d)))
assert 0.0 < alpha < 1.0 and residual < 1e-12, (alpha, residual)
oldareas = [area_normal(X, F[i])[0] for i in (PAIR_FACE, EAR_FACE)]
oldnormal = area_normal(X, F[PAIR_FACE])[1]
newareas, newnormaldots = [], []
for tri in NEW:
    ar, n = area_normal(X, tri)
    newareas.append(ar)
    newnormaldots.append(float(np.dot(n, oldnormal)))
assert min(newareas) > 1e-12, newareas
assert min(newnormaldots) > 0.999999, newnormaldots
Fnew = F.copy()
Fnew[PAIR_FACE] = NEW[0]
Fnew[EAR_FACE] = NEW[1]
old_edges = edge_incidence(F)
new_edges = edge_incidence(Fnew)
old_boundary = {e for e, count in old_edges.items() if count == 1}
new_boundary = {e for e, count in new_edges.items() if count == 1}
assert old_boundary == new_boundary, "paired retriangulation changed boundary"
assert max(new_edges.values()) <= 2, "paired retriangulation created nonmanifold edge"
assert set(new_edges) - set(old_edges) == {(2302, 5847)}
assert set(old_edges) - set(new_edges) == {(2870, 2871)}
# Complete row-aligned face ancestry CSR. Ordinary rows keep their one source
# marker; both new rows carry the union of both original parent source faces.
ptr = np.zeros(len(Fnew) + 1, dtype=np.int64)
ancestry = []
for i, marker in enumerate(markers):
    if i in (PAIR_FACE, EAR_FACE):
        ancestry.extend(SOURCE_PARENTS.tolist())
    elif marker >= 0:
        ancestry.append(int(marker))
    ptr[i + 1] = len(ancestry)
for key in ("parent_coarse_face_row", "parent_fine_detail_face_row", "parent_source_face_marker"):
    arr = np.asarray(data[key]).copy()
    arr[[PAIR_FACE, EAR_FACE]] = -1
    data[key] = arr
data["source_O_faces"] = Fnew.astype(np.int32)
data["face_source_ancestry_indptr"] = ptr
data["face_source_ancestry_face_rows"] = np.asarray(ancestry, dtype=np.int32)
data["retriangulation_output_face_rows"] = np.array([PAIR_FACE, EAR_FACE], dtype=np.int32)
data["retriangulation_original_source_face_rows"] = SOURCE_PARENTS.copy()
data["retriangulation_shared_edge_vertex_rows"] = np.array([2878, 2879], dtype=np.int32)
data["retriangulation_alternate_diagonal_vertex_rows"] = np.array([2321, 5808], dtype=np.int32)
# Geometry was unchanged; only the diagonal and explicit ancestry representation changed.
areas = np.empty(len(Fnew), dtype=np.float64)
for i, tri in enumerate(Fnew):
    areas[i] = area_normal(X, tri)[0]
assert float(areas.min()) > 1e-12, float(areas.min())
assert np.all(np.isfinite(X)) and np.all(np.isfinite(areas))
npz_path = OUT / "hybrid-O-surface-134.npz"
report_path = OUT / "hybrid-O-surface-134-report.json"
with npz_path.open("xb") as f:
    np.savez_compressed(f, **data)
report = {
    "status": "paired-ear-retriangulation-geometry-only",
    "input_candidate": str(INPUT), "input_sha256": EXPECTED_INPUT_SHA,
    "input_report_sha256": sha(INPUT_REPORT),
    "output_candidate": str(npz_path), "output_candidate_sha256": sha(npz_path),
    "face_rows": {"paired_neighbor": int(PAIR_FACE), "ear": int(EAR_FACE)},
    "original_faces": [PAIR.tolist(), EAR.tolist()],
    "replacement_faces": NEW.tolist(),
    "source_face_ancestry_rows_for_each_replacement": [SOURCE_PARENTS.tolist(), SOURCE_PARENTS.tolist()],
    "singular_parent_fields_for_replacement_rows": "-1; use face_source_ancestry_indptr/face_source_ancestry_face_rows",
    "collinear_vertex": {"output_node_row": 2302, "edge_endpoint_rows": [2871, 2870], "parameter_t": alpha, "residual_m": residual},
    "old_pair_areas_m2": oldareas, "new_pair_areas_m2": newareas,
    "new_normals_dot_original_neighbor_normal": newnormaldots,
    "boundary_edge_count_before_after": [len(old_boundary), len(new_boundary)],
    "boundary_edge_set_unchanged": True,
    "nonmanifold_edge_count_after": sum(count > 2 for count in new_edges.values()),
    "output_min_triangle_area_m2": float(areas.min()),
    "vertices_changed": 0, "faces_changed": 2,
    "selected_coarse_face_rows_sha256": hashlib.sha256(np.asarray(data["selected_coarse_face_rows"], dtype="<i4").tobytes()).hexdigest(),
    "limits": ["No exact whole-mesh self-intersection or bone-containment audit in this step.", "All 187/187 original seed faces and 35/35 groups are represented in the input hybrid candidate; full self/bone gates remain pending.", "No I/D layer conformity, tetrahedral quality, mass transfer, or native acceptance claimed."],
}
with report_path.open("x") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({"report_path": str(report_path), "report_sha256": sha(report_path), **{k: report[k] for k in ("output_candidate_sha256", "old_pair_areas_m2", "new_pair_areas_m2", "output_min_triangle_area_m2", "boundary_edge_count_before_after", "collinear_vertex", "limits")}}, indent=2))
