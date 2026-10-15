import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("shipping_audit", HERE / "audit_shipping_admission.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
SNAPSHOT = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-dynamic-grips-coupled-damage-and-fp64-contact-074")
PREFLIGHT = HERE / "preflight-074.json"
PREBUILD = HERE / "prebuild-074.json"

class ShippingArenaAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(PREFLIGHT.read_text())

    def temp_preflight(self, mutate):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            value = json.loads(json.dumps(self.source))
            mutate(value)
            path.write_text(json.dumps(value))
            return mod.audit(SNAPSHOT, path, PREBUILD)

    def test_frozen_source_admission_reports_first_blocker(self):
        result = mod.audit(SNAPSHOT, PREFLIGHT, PREBUILD)
        self.assertEqual(result["topology"]["prepared_nodes"], 528444)
        self.assertEqual(result["topology"]["tetrahedra"], 1477728)
        self.assertEqual(result["first_executable_blocker"]["pair_limit"], 65536)
        self.assertGreater(result["prebuild_exact_cpu_probe"]["pair_upper_bound"], 65536)
        self.assertIn("do not carry the sparse request", result["first_executable_blocker"]["sparse_prebuild_propagation"])
        self.assertEqual(result["first_executable_blocker"]["pair_count"], 6335568)
        self.assertEqual(result["next_blockers_after_bounded_sparse_rigid_scratch"][0]["first_scope_reason"], "fem_node_cap_8192")
        self.assertEqual(result["qualification"], False)

    def test_frozen_topology_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            self.temp_preflight(lambda d: d["topology"].update(prepared_nodes=8192))

    def test_dependency_hash_mismatch_fails_closed(self):
        value = json.loads(json.dumps(self.source))
        first = next(iter(value["provenance"]["hashed_non_system_translation_dependencies"]))
        value["provenance"]["hashed_non_system_translation_dependencies"][first] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "source binding"):
                mod.audit(SNAPSHOT, path, PREBUILD)

    def test_best_case_under_budget_is_still_not_admitted(self):
        result = mod.audit(SNAPSHOT, PREFLIGHT, PREBUILD)
        scenarios = result["modeled_scenarios"]
        self.assertLess(scenarios["131072_compact_surface_best_case"]["modeled_peak_domain_bytes"], 16 * 1024**3)
        self.assertFalse(scenarios["131072_compact_surface_best_case"]["private_arenas_admitted"])
        self.assertGreater(result["next_blockers_after_bounded_sparse_rigid_scratch"][4]["minimum_anchor_patches_at_stride_128"], 128)

if __name__ == "__main__":
    unittest.main()
