from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_saved_field602_v2 import (
    EXPECTED_580_CHANGED_FIELDS,
    FACE_DEPENDENT_580_FIELDS,
    load_payload,
    merge,
    read_state,
    validate_state_documents,
)

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
ACTUAL_STATES = (
    D / "connected-topology-580/run-004-from1149/final/state.json",
    D / "component-field-selection-627/run-002/final/state.json",
    D / "contained-field-retention-629/from1093-001/final/state.json",
)


def synthetic_payloads():
    points = np.zeros((240, 3), dtype=np.float32)
    anchors = np.arange(237, dtype=np.int64)
    old = {
        "points_world_m": points,
        "typed_coarse_anchor_output_rows_018": anchors,
        "typed_anchor_source_ids": np.arange(237, dtype=np.int64),
        "source_prolongation_indptr": np.arange(241, dtype=np.int64),
        "source_prolongation_indices": np.arange(240, dtype=np.int64),
        "source_prolongation_weights": np.ones(240, dtype=np.float64),
        "parent_coarse_face_row": np.asarray([2], dtype=np.int64),
        "refinement_face_parent_barycentric_semantics": np.asarray(
            ["barycentric rows indexed by immediate parent face"], dtype="U64"),
        "source_O_faces": np.asarray([[0, 1, 2]], dtype=np.int64),
        "area_reference_037_source_O_faces": np.asarray([[0, 1, 2]], dtype=np.int64),
    }
    for i, key in enumerate(sorted(FACE_DEPENDENT_580_FIELDS - {
        "source_O_faces", "area_reference_037_source_O_faces"
    })):
        if key.endswith("indptr"):
            old[key] = np.asarray([0, 1], dtype=np.int64)
        elif "barycentric" in key:
            old[key] = np.eye(3, dtype=np.float64)[None, :, :]
        else:
            old[key] = np.asarray([i + 1], dtype=np.int64)
    baseline = {k: v.copy() for k, v in old.items()}
    current = {k: v.copy() for k, v in old.items()}
    current["source_O_faces"] = np.asarray([[0, 2, 1]], dtype=np.int64)
    current["area_reference_037_source_O_faces"] = np.asarray([[0, 2, 1]], dtype=np.int64)
    for key in sorted(FACE_DEPENDENT_580_FIELDS - {
        "source_O_faces", "area_reference_037_source_O_faces"
    }):
        current[key] = current[key].copy()
        current[key].flat[0] += (0.01 if current[key].dtype.kind == "f" else 1)
    proposal = {k: v.copy() for k, v in baseline.items()}
    proposal["points_world_m"][239, 0] = np.float32(0.001)
    return current, baseline, proposal


class SavedFieldReplay602V2Tests(unittest.TestCase):
    def test_legitimate_face_ancestry_changes_are_allowlisted(self):
        current, baseline, proposal = synthetic_payloads()
        merged, changed = merge(current, baseline, proposal)
        self.assertEqual(set(changed), FACE_DEPENDENT_580_FIELDS)
        self.assertTrue(np.array_equal(merged["points_world_m"], proposal["points_world_m"]))
        for key in current:
            if key != "points_world_m":
                self.assertTrue(np.array_equal(merged[key], current[key]), key)

    def test_unobserved_face_marker_or_semantics_change_rejected(self):
        current, baseline, proposal = synthetic_payloads()
        current["parent_coarse_face_row"][0] += 1
        with self.assertRaisesRegex(ValueError, "non-face field"):
            merge(current, baseline, proposal)
        current, baseline, proposal = synthetic_payloads()
        current["refinement_face_parent_barycentric_semantics"][0] = "different"
        with self.assertRaisesRegex(ValueError, "non-face field"):
            merge(current, baseline, proposal)

    def test_changed_point_provenance_rejected(self):
        current, baseline, proposal = synthetic_payloads()
        current["source_prolongation_weights"][2] += 0.25
        with self.assertRaisesRegex(ValueError, "non-face field"):
            merge(current, baseline, proposal)

    def test_bad_baseline_coordinates_rejected(self):
        current, baseline, proposal = synthetic_payloads()
        current["points_world_m"][239, 1] = np.float32(0.001)
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            merge(current, baseline, proposal)

    def test_anchor_motion_rejected(self):
        current, baseline, proposal = synthetic_payloads()
        proposal["points_world_m"][0, 0] = np.float32(0.001)
        with self.assertRaisesRegex(ValueError, "moved typed anchors"):
            merge(current, baseline, proposal)

    def test_bad_candidate_bound_report_rejected(self):
        state = {"candidate_sha256": "c" * 64, "accepted_physical_steps": 0, "native_admission": False}
        point = {"candidate_sha256": "c" * 64, "status": "OUTER_NODE_CLEARANCE_DIAGNOSTIC",
                 "complete_surface_audit": True, "inside_node_bank_pairs": 0, "inside_nodes": 0,
                 "accepted_physical_steps": 0, "geometry_admitted": False}
        surface = {"candidate_sha256": "c" * 64, "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT",
                   "complete": True, "bank_count": 185, "expected_bank_count": 185,
                   "geometry_or_native_admission": False}
        final = {"candidate_sha256": "c" * 64, "status": "COMPLETE_EXACT_CONSTRUCTION_CHECKS",
                 "surface_pairs": 1, "self_pairs": 0, "contained_pairs": 0, "complete_banks": 185,
                 "accepted_physical_steps": 0, "native_admission": False}
        validate_state_documents(state, "c" * 64, point, surface, final)
        final["self_pairs"] = 1
        with self.assertRaisesRegex(ValueError, "self-intersection"):
            validate_state_documents(state, "c" * 64, point, surface, final)
        final["self_pairs"] = 0
        surface["candidate_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "surface report candidate hash"):
            validate_state_documents(state, "c" * 64, point, surface, final)

    def test_actual_1095_1149_1088_states_pure_merge(self):
        current_bundle, baseline_bundle, proposal_bundle = [read_state(p) for p in ACTUAL_STATES]
        self.assertEqual(current_bundle["counts"]["surface_pairs"], 1095)
        self.assertEqual(baseline_bundle["counts"]["surface_pairs"], 1149)
        self.assertEqual(proposal_bundle["counts"]["surface_pairs"], 1088)
        for bundle in (current_bundle, baseline_bundle, proposal_bundle):
            self.assertEqual(bundle["counts"]["inside_pairs"], 0)
            self.assertEqual(bundle["counts"]["self_pairs"], 0)
            self.assertEqual(bundle["counts"]["complete_banks"], 185)
        current = load_payload(current_bundle)
        baseline = load_payload(baseline_bundle)
        proposal = load_payload(proposal_bundle)
        merged, changed = merge(current, baseline, proposal)
        self.assertEqual(set(changed), EXPECTED_580_CHANGED_FIELDS)
        self.assertEqual(merged["points_world_m"].tobytes(), proposal["points_world_m"].tobytes())
        for key in current:
            if key != "points_world_m":
                self.assertEqual(merged[key].dtype, current[key].dtype, key)
                self.assertEqual(merged[key].shape, current[key].shape, key)
                self.assertEqual(merged[key].tobytes(), current[key].tobytes(), key)

    def test_actual_state_candidate_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = json.loads(ACTUAL_STATES[0].read_text())
            source["candidate_sha256"] = "0" * 64
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "candidate hash does not match"):
                read_state(path)


if __name__ == "__main__":
    unittest.main()
