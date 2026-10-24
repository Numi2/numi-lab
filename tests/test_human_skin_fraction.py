import hashlib, importlib.util, json, os, tempfile, unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "human_skin_fraction", Path(__file__).resolve().parents[1] / "tools/materialize_human_skin_fraction.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class HumanSkinFractionTests(unittest.TestCase):
    def test_changed_source_pin_produces_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "changed.nhskin"
            source.write_bytes(b"changed source")
            output = root / "output.nhskin"
            params = {}
            for key in ("base", "displacement", "ancestry", "report", "rest_pose", "rebind_owner", "emission_owner"):
                params[key + "_path"] = source
                params[key + "_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "base hash mismatch"):
                owner.materialize(**params, fraction=.25, output_path=output)
            self.assertFalse(output.exists())

    def inputs(self):
        path = os.environ.get("NUMI_HUMAN_SKIN_FRACTION_TEST_INPUTS")
        if not path:
            self.skipTest("Set NUMI_HUMAN_SKIN_FRACTION_TEST_INPUTS to pinned retained replay inputs.")
        params = json.loads(Path(path).read_text())
        return {k: Path(v) if k.endswith("_path") else v for k, v in params.items()}

    def test_retained_regional_candidate_exact_replay(self):
        params = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fraction.nhskin"
            receipt = owner.materialize(**params, output_path=output)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                "946a473f804ce83ebf97a7b3dc6b5e68a7712346de7196288ac1dced68deb02a")
            self.assertEqual(receipt["clone_count"], 175)
            self.assertEqual(receipt["fixed_boundary_edge_count"], 134)
            self.assertFalse(receipt["native_or_mass_admission"])

    def test_failed_emission_gate_produces_no_output(self):
        params = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = root / "reject.py"
            rejected.write_text('def validate_float32_surface_emission(*args):\n    raise ValueError("emission failed")\n')
            params["emission_owner_path"] = rejected
            params["emission_owner_sha256"] = hashlib.sha256(rejected.read_bytes()).hexdigest()
            output = root / "output.nhskin"
            with self.assertRaisesRegex(ValueError, "emission failed"):
                owner.materialize(**params, output_path=output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
