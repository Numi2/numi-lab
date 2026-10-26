import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("local_backtracking_config_419", HERE / "local_backtracking_config_419.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class ConfigBindingTests(unittest.TestCase):
    def setUp(self):
        self.local = {
            "helper": str(owner.LOCAL_HELPER), "helper_sha256": owner.LOCAL_SHA256,
            "helper_entry": owner.LOCAL_ENTRY, "uniform_failure_backtracking": False,
        }
        self.uniform = {
            "helper": str(owner.GLOBAL_HELPER), "helper_sha256": owner.GLOBAL_SHA256,
            "helper_entry": owner.GLOBAL_ENTRY, "uniform_failure_backtracking": True,
        }

    def test_418_global_helper_is_rejected_for_local_mode(self):
        bad = dict(self.uniform, uniform_failure_backtracking=False)
        with self.assertRaisesRegex(ValueError, "local mode requires helper path"):
            owner.validate_mode_helper_419(bad)

    def test_local_binding_passes_and_reports_exact_owner(self):
        r = owner.validate_mode_helper_419(self.local)
        self.assertEqual(r["status"], "PASS_BACKTRACKING_MODE_HELPER_BINDING")
        self.assertEqual(r["mode"], "local")
        self.assertEqual(r["helper_sha256"], owner.LOCAL_SHA256)
        self.assertEqual(r["helper_entry"], "reduce_witness_node_scales")
        self.assertIn("step_scales", r["signature"])

    def test_global_binding_passes_only_with_explicit_uniform_mode(self):
        r = owner.validate_mode_helper_419(self.uniform)
        self.assertEqual(r["mode"], "uniform")
        self.assertEqual(r["helper_entry"], "reduce_global_step_scales")

    def test_entry_hash_and_boolean_mismatches_fail_closed(self):
        for field, value in (("helper_entry", "reduce_global_step_scales"),
                             ("helper_sha256", "0" * 64),
                             ("uniform_failure_backtracking", 0)):
            c = dict(self.local); c[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                owner.validate_mode_helper_419(c)

    def test_local_configurator_copies_and_preserves_unrelated_fields(self):
        source = dict(self.uniform, initial_scale=.25, zero_below=1/64,
                      trial="trial.npz", out="out")
        original = dict(source)
        c, r = owner.configure_local_mode_419(source)
        self.assertEqual(source, original)
        self.assertEqual(c["initial_scale"], .25)
        self.assertEqual(c["zero_below"], 1/64)
        self.assertEqual(c["trial"], "trial.npz")
        self.assertEqual(c["out"], "out")
        self.assertFalse(c["uniform_failure_backtracking"])
        self.assertEqual(r["mode"], "local")

    def test_actual_local_helper_changes_only_witness_incident_scales(self):
        module = owner._load_helper(owner.LOCAL_HELPER)
        faces = np.array([[0,1,2],[3,4,5],[6,7,8]], dtype=np.int64)
        scales = np.array([.8,.6,.4,.9,.7,.3,.11,.22,.33], dtype=np.float64)
        anchors = np.array([1], dtype=np.int64)
        out, report = module.reduce_witness_node_scales(
            faces, scales, np.array([0], dtype=np.int64), np.empty((0,2), dtype=np.int64),
            anchors=anchors, zero_below=1/128)
        np.testing.assert_array_equal(out[[0,2]], scales[[0,2]] * .5)
        np.testing.assert_array_equal(out[3:], scales[3:])
        self.assertEqual(out[1], 0.0)
        self.assertTrue(report["unaffected_scales_bitwise_unchanged"])

    def test_actual_local_helper_reduces_both_self_pair_faces_only(self):
        module = owner._load_helper(owner.LOCAL_HELPER)
        faces = np.array([[0,1,2],[3,4,5],[6,7,8]], dtype=np.int64)
        scales = np.ones(9, dtype=np.float64)
        out, _ = module.reduce_witness_node_scales(
            faces, scales, np.empty(0,dtype=np.int64), np.array([[0,1]],dtype=np.int64))
        np.testing.assert_array_equal(out[:6], np.full(6,.5))
        np.testing.assert_array_equal(out[6:], np.ones(3))

if __name__ == "__main__":
    unittest.main()
