import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostic_batch", ROOT / "tools" / "compose_diagnostic_batch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NativeAdmissionGate(unittest.TestCase):
    def _receipt(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_missing_receipts_are_blocked(self):
        admitted, reasons = m.native_admission_decision("abc")
        self.assertFalse(admitted)
        self.assertEqual(len(reasons), 2)

    def test_candidate_hash_and_each_gate_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            component = d / "component.json"
            mechanical = d / "mechanical.json"
            self._receipt(component, {
                "candidate_sha256": "abc", "exact_self_pass": True,
                "exact_bone_pass": True, "exact_tissue_pass": True,
                "all_required_poses_pass": True, "source_outside_exact": True,
                "native_admission": True})
            self._receipt(mechanical, {
                "candidate_sha256": "abc", "positive_cells_pass": True,
                "material_mass_closure_pass": True, "all_required_poses_pass": True})
            self.assertEqual(m.native_admission_decision("abc", component, mechanical), (True, []))
            self.assertFalse(m.native_admission_decision("def", component, mechanical)[0])
            bad = json.loads(mechanical.read_text())
            bad["positive_cells_pass"] = False
            self._receipt(mechanical, bad)
            admitted, reasons = m.native_admission_decision("abc", component, mechanical)
            self.assertFalse(admitted)
            self.assertIn("mechanical map positive_cells_pass is not true", reasons)


if __name__ == "__main__":
    unittest.main()
