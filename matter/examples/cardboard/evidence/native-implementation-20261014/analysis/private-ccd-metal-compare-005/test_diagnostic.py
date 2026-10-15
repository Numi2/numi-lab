from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("/Users/home/cardboard-implementation-20261014")
SOURCE = BASE / "snapshots/source-center-action-twoenv-restart-099"
CAPTURE_DIR = BASE / "native"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / "analysis/private-trial-reader-001"))
import generate_metal
import prepare_capture_cases


class SourceExtractionTests(unittest.TestCase):
    def test_exact_frozen_helper_and_bounded_instrumentation(self):
        source_bytes = (SOURCE / "matter/src/metal/contact.metalinc").read_bytes()
        self.assertEqual(hashlib.sha256(source_bytes).hexdigest(),
                         generate_metal.EXPECTED_CONTACT_SHA256)
        original = generate_metal.extract_edge_ee(source_bytes.decode())
        clone16 = generate_metal.make_instrumented_clone(original, 16)
        clone17 = generate_metal.make_instrumented_clone(original, 17)
        self.assertIn("for (uint iteration = 0u; iteration < 16u; ++iteration)", clone16)
        self.assertIn("for (uint iteration = 0u; iteration < 17u; ++iteration)", clone17)
        self.assertNotIn("iterationBudget", clone16 + clone17)
        for clone in (clone16, clone17):
            self.assertIn("iterationsExecuted = iteration + 1u", clone)
            self.assertIn("float4(time, distance, effectiveThickness, speedBound)", clone)
            self.assertIn("witnessP = pointP", clone)
            self.assertIn("witnessQ = pointQ", clone)
        self.assertEqual(original.count("closestPointsOnSegments("), 1)

    def test_generated_source_keeps_production_source_as_external_include(self):
        output = HERE / "build/private_ee_ccd_diagnostic.metal"
        manifest_path = HERE / "build/build-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["production_function_modified"], False)
        self.assertEqual(manifest["contact_source_sha256"], generate_metal.EXPECTED_CONTACT_SHA256)
        self.assertEqual(manifest["frozen_source_manifest_sha256"],
                         generate_metal.EXPECTED_FROZEN_MANIFEST_SHA256)
        self.assertEqual(manifest["frozen_source_file_count"], 775)
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                         manifest["generated_metal_sha256"])
        text = output.read_text()
        self.assertIn("#include \"" + str(SOURCE / "matter/src/matter_kernels.metal") + "\"", text)
        self.assertIn("edgeEdgeCCD16Diagnostic", text)
        self.assertIn("edgeEdgeCCD17Diagnostic", text)
        self.assertIn("nm_contact_ee_ccd_budget_diagnostic", text)


class CapturedInputTests(unittest.TestCase):
    def test_cases_are_exact_333_334_capture_words_with_two_budgets(self):
        identities = []
        blobs = []
        for case_id in (333, 334):
            path = CAPTURE_DIR / f"center-action-feasibility-bracket-{case_id}/private-trial.coupled.json"
            _, rows, blob, identity = prepare_capture_cases.source_case(path, case_id)
            identities.append(identity)
            blobs.append(blob)
            self.assertEqual([row["budget"] for row in rows], [16, 17])
            self.assertEqual([row["capture_case"] for row in rows], [case_id, case_id])
        packed = b"".join(blobs)
        self.assertEqual(len(packed), 4 * 9 * 16)
        self.assertEqual(hashlib.sha256(packed).hexdigest(),
                         "80eaab55f3d471934382eaec6165c2c5a717abafb94cf08dd75f7bb69c42d3f9")
        cases = np.frombuffer(packed, dtype="<f4").reshape(4, 9, 4)
        self.assertEqual(cases[:, 0, 0].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(cases[:, 0, 2].tolist(), [16.0, 17.0, 16.0, 17.0])
        self.assertEqual(cases[:, 0, 3].tolist(), [333.0, 333.0, 334.0, 334.0])
        self.assertEqual(identities[0]["candidate_identity"], [88, 127, 331, 1])
        self.assertEqual(identities[1]["candidate_identity"], [88, 127, 331, 1])


if __name__ == "__main__":
    unittest.main()
