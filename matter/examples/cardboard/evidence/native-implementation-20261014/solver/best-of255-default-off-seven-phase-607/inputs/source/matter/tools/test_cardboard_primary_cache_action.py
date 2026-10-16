from pathlib import Path
import copy,subprocess,tempfile,unittest
import cardboard_cached_operator_audit as audit

ROOT=Path(__file__).resolve().parents[2]
class PrimaryCacheActionTests(unittest.TestCase):
    def header(self,enabled):
        return {"cached_algorithmic_tangent_capture":{"policy_enabled":enabled},
                "operator_action_policy":{"selection":"matches_primary_FGMRES_cache_mode",
                    "cache_mode":int(enabled),"element_force_source":"same_operator_dispatch_as_nodal_action",
                    "local_stress_tangent_source":"independent_direct_constitutive_probe"}}
    def test_enabled_and_disabled_primary_policies(self):
        for enabled in (False,True):
            self.assertTrue(audit.capture_operator_action_policy(self.header(enabled))["production_operator_action_bound"])
    def test_legacy_has_no_production_cache_claim(self):
        self.assertFalse(audit.capture_operator_action_policy({})["production_operator_action_bound"])
    def test_mismatch_and_boolean_mode_rejected(self):
        for mode in (0,True,2,"1"):
            h=self.header(True);h["operator_action_policy"]["cache_mode"]=mode
            with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
    def test_unknown_or_missing_fields_rejected(self):
        h=self.header(True);del h["operator_action_policy"]["element_force_source"]
        with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
        h=self.header(True);h["operator_action_policy"]["unknown"]=1
        with self.assertRaises(ValueError):audit.capture_operator_action_policy(h)
    def test_actual_fd_dispatch_uses_same_primary_helper(self):
        runtime=(ROOT/"matter/src/runtime.mm").read_text()
        fd=runtime.split("const auto encodeFEMOperatorFDTrace =",1)[1].split("const auto capturedNodeCaptureSlots",1)[0]
        self.assertIn("numi::matter::detail::femPrimaryOperatorCacheMode(",fd)
        self.assertIn("state.femAlgorithmicTangentCacheValid : state.dummy",fd)
        self.assertNotIn("const std::uint32_t useCache = 0u;",fd)
        self.assertEqual(runtime.count("numi::matter::detail::femPrimaryOperatorCacheMode("),6)
    def test_compiled_policy_preserves_primary_semantics(self):
        code='#include "numi/matter/detail/fem_primary_operator_cache_mode.hpp"\nusing numi::matter::detail::femPrimaryOperatorCacheMode;\nstatic_assert(femPrimaryOperatorCacheMode(0)==0);\nstatic_assert(femPrimaryOperatorCacheMode(1)==1);\nstatic_assert(femPrimaryOperatorCacheMode(2)==0);\nstatic_assert(femPrimaryOperatorCacheMode(0xffffffffu)==0);\nint main(){}\n'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"check.cpp";p.write_text(code)
            subprocess.run(["/usr/bin/clang++","-std=c++20","-I",str(ROOT/"matter/include"),str(p),"-o",str(Path(d)/"check")],check=True)
            subprocess.run([str(Path(d)/"check")],check=True)
if __name__=="__main__":unittest.main()
