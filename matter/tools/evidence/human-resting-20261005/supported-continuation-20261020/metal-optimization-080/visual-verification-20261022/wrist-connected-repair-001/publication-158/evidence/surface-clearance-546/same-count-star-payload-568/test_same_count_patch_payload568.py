import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import same_count_patch_payload568 as payload

OWNER405 = ROOT / "D/local-two-parent-retriangulation-405/cross_seam_search_405.py"
ROUND039_REPORT = ROOT / "D/surface-clearance-546/topology/round-039-root-policy-v6/round-001/report.json"
PINNED_REPORT_SHA = "46e022684866257a1782c512fbd24adce962dce948244f56aadf239f47621606"
PINNED_CANDIDATE_SHA = "b033848a0893ff45f216d150c3847e21e7db99b3a228719efbd9d31c6c8f912a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pack(rows, dtype=np.int64):
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    vals = []
    for i, row in enumerate(rows):
        vals.extend(row)
        ptr[i + 1] = len(vals)
    return ptr, np.asarray(vals, dtype=dtype)


def synthetic_candidate():
    # CCW pentagon fan around retained center vertex 0.
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1]], dtype=np.int64)
    c = {
        "points_world_m": np.zeros((6, 3), dtype=np.float32),
        "source_O_faces": faces.copy(),
        "area_reference_037_source_O_faces": faces.copy(),
        "face_kind": np.ones(5, dtype=np.int8),
        "parent_fine_detail_face_row": np.arange(10, 15, dtype=np.int64),
        "parent_fine_subface_row": np.arange(20, 25, dtype=np.int64),
        "parent_coarse_face_row": np.arange(30, 35, dtype=np.int64),
        "parent_source_face_marker": np.arange(40, 45, dtype=np.int64),
        "refinement_face_parent_face_row": np.arange(50, 55, dtype=np.int64),
        "refinement_face_parent_barycentric": np.zeros((5, 3, 3), dtype=np.float64),
        "unrelated_domain_payload": np.asarray([b"unchanged", b"source-p"]),
        "typed_coarse_anchor_output_rows_018": np.asarray([4], dtype=np.int64),
    }
    families = (
        ("face_source_ancestry_indptr", "face_source_ancestry_face_rows", [[100+i] for i in range(5)]),
        ("face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows", [[200+i] for i in range(5)]),
        ("refinement_face_parent_ancestry_indptr", "refinement_face_parent_ancestry_face_rows", [[300+i] for i in range(5)]),
        ("face_fine_detail_ancestry_indptr", "face_fine_detail_ancestry_rows", [[400+i] for i in range(5)]),
    )
    for ptr_key, val_key, rows in families:
        c[ptr_key], c[val_key] = pack(rows)
    c["refinement_face_parent_ancestry_semantics"] = np.asarray(["old parent semantics"])
    c["face_fine_detail_ancestry_semantics"] = np.asarray(["old detail semantics"])
    c["refinement_face_parent_barycentric_semantics"] = np.asarray(["old barycentric semantics"])
    return c


def pentagon_retriangulation():
    # Triangulate the boundary pentagon into 3 triangles, then split one
    # boundary triangle at the existing center to make 5 faces total.
    return np.asarray([
        [1, 2, 3],
        [1, 3, 0],
        [3, 4, 0],
        [4, 1, 0],
        [1, 4, 5],
    ], dtype=np.int64)


def assert_array_equal(a, b):
    if np.asarray(a).dtype.kind in "fc" or np.asarray(b).dtype.kind in "fc":
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    else:
        assert np.array_equal(a, b)


def test_same_count_star_patch_preserves_boundary_and_unions_four_csrs_with_row_permutation():
    c = synthetic_candidate()
    repl = pentagon_retriangulation()
    rows = np.asarray([3, 0, 4, 1, 2], dtype=np.int64)
    permuted_repl = repl[[3, 0, 4, 1, 2]]
    out = payload.apply_same_count_patch(c, rows, permuted_repl)

    assert np.array_equal(out["source_O_faces"][rows], permuted_repl)
    assert np.array_equal(out["area_reference_037_source_O_faces"], out["source_O_faces"])
    assert np.array_equal(out["points_world_m"], c["points_world_m"])
    assert np.array_equal(out["typed_coarse_anchor_output_rows_018"], c["typed_coarse_anchor_output_rows_018"])
    assert np.array_equal(out["unrelated_domain_payload"], c["unrelated_domain_payload"])

    expected = {
        "face_source_ancestry_face_rows": list(range(100, 105)),
        "face_fine_subface_ancestry_rows": list(range(200, 205)),
        "refinement_face_parent_ancestry_face_rows": list(range(300, 305)),
        "face_fine_detail_ancestry_rows": list(range(400, 405)),
    }
    for _ptr_key, val_key in payload.CSR_FAMILIES:
        rows_after = payload.csr_rows(out[_ptr_key], out[val_key], 5)
        for row in rows:
            assert rows_after[int(row)] == expected[val_key]
    assert np.all(out["refinement_face_parent_face_row"][rows] == -1)
    assert np.isnan(out["refinement_face_parent_barycentric"][rows]).all()
    assert np.all(out["parent_fine_detail_face_row"][rows] == -1)
    assert np.all(out["parent_fine_subface_row"][rows] == -1)
    assert np.all(out["parent_coarse_face_row"][rows] == -1)
    assert np.all(out["parent_source_face_marker"][rows] == -1)
    for key in set(c) - payload.MUTATED_KEYS:
        assert_array_equal(out[key], c[key])


def test_rejects_changed_oriented_boundary_and_mixed_material():
    c = synthetic_candidate()
    repl = pentagon_retriangulation()
    bad_boundary = np.asarray([[1, 0, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 0]], dtype=np.int64)
    with pytest.raises(payload.PatchPayloadError, match="boundary"):
        payload.apply_same_count_patch(c, np.arange(5), bad_boundary)

    mixed = synthetic_candidate()
    mixed["face_kind"][3] = 2
    with pytest.raises(payload.PatchPayloadError, match="mixed face_kind"):
        payload.apply_same_count_patch(mixed, np.arange(5), repl)

    repeated = repl.copy()
    repeated[0] = [1, 1, 3]
    with pytest.raises(payload.PatchPayloadError, match="repeated vertex"):
        payload.apply_same_count_patch(c, np.arange(5), repeated)

    missing_vertex = repl.copy()
    missing_vertex[-1] = [1, 4, 2]
    with pytest.raises(payload.PatchPayloadError, match="different point sets"):
        payload.apply_same_count_patch(c, np.arange(5), missing_vertex)


def test_exact_two_face_payload_parity_with_pinned_405_accepted039_operation():
    assert sha(OWNER405) == "e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967"
    assert sha(ROUND039_REPORT) == PINNED_REPORT_SHA
    report = json.loads(ROUND039_REPORT.read_text())
    candidate_path = Path(report["candidate_path"])
    assert sha(candidate_path) == PINNED_CANDIDATE_SHA
    op = report["accepted_operations"][0]

    with np.load(candidate_path, allow_pickle=False) as z:
        candidate = {k: z[k] for k in z.files}
    owner = load_module("cross_seam_search_405_parity_fixture", OWNER405)
    expected = owner.apply_op(candidate, op)
    actual = payload.apply_patch(candidate, op)
    assert set(expected) == set(actual)
    for key in expected:
        assert_array_equal(actual[key], expected[key])

