"""Receipt identity writer checks; these do not qualify anatomy or physiology."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("passive_viscera_geometry_rebase.py")
SPEC = importlib.util.spec_from_file_location("passive_viscera_geometry_rebase", MODULE_PATH)
rebase = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rebase)


class CurrentPayloadIdentityTests(unittest.TestCase):
    OLD = "a" * 64
    NEW = "b" * 64

    @classmethod
    def receipt(cls, with_wall=True):
        cardiac = {
            "output_anatomy_payload_sha256": cls.OLD,
            "method": "exact_source_face_arrangement_with_RA_priority",
            "source_payload_sha256": "c" * 64,
            "source_face_sha256": "d" * 64,
            "ventricular_wall_binding": {
                "output_anatomy_payload_sha256": cls.OLD,
                "source_anatomy_payload_sha256": "e" * 64,
                "source_sha256": "f" * 64,
                "wall_map": {"source_payload_sha256": "1" * 64},
            } if with_wall else None,
        }
        if not with_wall:
            cardiac.pop("ventricular_wall_binding")
        return {
            "payload": {"path": "/input.nhanatomy", "sha256": cls.OLD,
                        "input_payload_sha256": "2" * 64},
            "functional_bindings": {"anatomy_payload_sha256": cls.OLD},
            "provenance": {
                "cardiac_geometry_binding": cardiac,
                "passive_viscera_geometry_rebase": {
                    "source_payload_sha256": "3" * 64,
                    "cardiac_prior_output_anatomy_payload_sha256": "4" * 64,
                },
                # This history-shaped field deliberately resembles an identity
                # pointer. The explicit writer must leave it untouched.
                "retained_history": {"output_anatomy_payload_sha256": cls.OLD},
            },
        }

    def test_refreshes_exact_current_paths_and_preserves_source_history(self):
        value = self.receipt()
        before = copy.deepcopy(value)
        result = rebase.refresh_current_payload_identities(value, self.OLD, self.NEW)

        self.assertEqual(result["updated_current_payload_identity_count"], 4)
        self.assertEqual(set(result["updated_current_payload_identity_paths"]), {
            "payload.sha256",
            "functional_bindings.anatomy_payload_sha256",
            "provenance.cardiac_geometry_binding.output_anatomy_payload_sha256",
            "provenance.cardiac_geometry_binding.ventricular_wall_binding.output_anatomy_payload_sha256",
        })
        self.assertEqual(value["payload"]["sha256"], self.NEW)
        self.assertEqual(value["functional_bindings"]["anatomy_payload_sha256"], self.NEW)
        cardiac = value["provenance"]["cardiac_geometry_binding"]
        self.assertEqual(cardiac["output_anatomy_payload_sha256"], self.NEW)
        self.assertEqual(cardiac["ventricular_wall_binding"]["output_anatomy_payload_sha256"], self.NEW)
        self.assertEqual(cardiac["source_payload_sha256"], before["provenance"]["cardiac_geometry_binding"]["source_payload_sha256"])
        self.assertEqual(cardiac["source_face_sha256"], before["provenance"]["cardiac_geometry_binding"]["source_face_sha256"])
        self.assertEqual(cardiac["ventricular_wall_binding"]["source_anatomy_payload_sha256"],
                         before["provenance"]["cardiac_geometry_binding"]["ventricular_wall_binding"]["source_anatomy_payload_sha256"])
        self.assertEqual(cardiac["ventricular_wall_binding"]["source_sha256"],
                         before["provenance"]["cardiac_geometry_binding"]["ventricular_wall_binding"]["source_sha256"])
        self.assertEqual(value["provenance"]["retained_history"], before["provenance"]["retained_history"])
        self.assertEqual(value["provenance"]["passive_viscera_geometry_rebase"],
                         before["provenance"]["passive_viscera_geometry_rebase"])
        self.assertEqual(result["cardiac_binding_proof_sha256_before"],
                         result["cardiac_binding_proof_sha256_after_resetting_current_output_pointers"])

    def test_legacy_receipt_without_wall_has_three_explicit_current_paths(self):
        value = self.receipt(with_wall=False)
        result = rebase.refresh_current_payload_identities(value, self.OLD, self.NEW)
        self.assertEqual(result["updated_current_payload_identity_count"], 3)
        self.assertNotIn("ventricular_wall_binding",
                         value["provenance"]["cardiac_geometry_binding"])
        self.assertEqual(value["provenance"]["retained_history"]["output_anatomy_payload_sha256"], self.OLD)

    def test_stale_wall_output_is_rejected_before_any_pointer_changes(self):
        value = self.receipt()
        value["provenance"]["cardiac_geometry_binding"]["ventricular_wall_binding"][
            "output_anatomy_payload_sha256"] = "9" * 64
        before = copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError, "ventricular wall output identity"):
            rebase.refresh_current_payload_identities(value, self.OLD, self.NEW)
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
