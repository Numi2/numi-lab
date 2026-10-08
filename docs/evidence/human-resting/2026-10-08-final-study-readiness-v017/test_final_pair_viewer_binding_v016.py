import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pair_viewer_binding_v016", ROOT / "analyze_final_pair.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FinalPairViewerBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.invocation = analysis.audit.read_json(analysis.prepare.V014_PREFLIGHT_INVOCATION)
        cls.log = analysis.prepare.V014_PREFLIGHT_LOG.read_text()

    def test_actual_viewer018_invocation_passes_final_binary_gate(self):
        self.assertEqual(analysis.verify_native_scene_binary(self.invocation),
                         str(analysis.prepare.SCENE_NATIVE_BINARY.resolve()))

    def test_old_capture017_is_rejected_as_final_scene_binary(self):
        invocation = copy.deepcopy(self.invocation)
        invocation["argv"][0] = str(analysis.prepare.NATIVE_BINARY)
        with self.assertRaisesRegex(ValueError, "pinned viewer018"):
            analysis.verify_native_scene_binary(invocation)

    def test_changed_binary_binding_is_rejected(self):
        invocation = copy.deepcopy(self.invocation)
        invocation["asset_sha256"][str(analysis.prepare.SCENE_NATIVE_BINARY.resolve())] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binary hash"):
            analysis.verify_native_scene_binary(invocation)

    def test_parse_actual_native_program_identity(self):
        self.assertEqual(analysis.native_program_identities(self.log),
                         (8317316107528819481, 17997488920282699923))

    def test_missing_duplicate_and_overflow_identities_rejected(self):
        line = "resting_body_source_fingerprint=1 coupled_program_fingerprint=2"
        for text in ("", line + "\n" + line,
                     "resting_body_source_fingerprint=18446744073709551616 coupled_program_fingerprint=2"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    analysis.native_program_identities(text)


if __name__ == "__main__":
    unittest.main()
