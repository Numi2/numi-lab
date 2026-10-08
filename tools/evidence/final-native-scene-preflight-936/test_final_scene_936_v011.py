import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

SCRIPT=Path(__file__).with_name("final_scene_936-v011.py")
spec=importlib.util.spec_from_file_location("final_scene_936_v011",SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CompositionBranchTests(unittest.TestCase):
    def test_legacy_fixture_composes_against_003_but_uses_existing_924_receipt(self):
        compose=Mock(return_value={"status":"fixture-composition"})
        out=Path("/evidence/legacy-fixture")
        skin=Path("/fixture/907.nhskin"); manifest=Path("/fixture/907.manifest.json")
        base, directory, emitted, native, result=module.compose_for_scene(compose,True,skin,manifest,out)
        compose.assert_called_once_with(module.BASE003,skin,manifest,out/"anatomy")
        self.assertEqual(base,module.BASE003)
        self.assertEqual(emitted,out/"anatomy/resting-anatomy-receipt.json")
        self.assertEqual(native,module.AR)
        self.assertFalse(native==emitted)
        self.assertEqual(result["status"],"fixture-composition")

    def test_final_candidate_composes_onto_924_and_uses_composed_receipt(self):
        compose=Mock(return_value={"status":"final-composition"})
        out=Path("/evidence/final-candidate")
        skin=Path("/candidate/final.nhskin"); manifest=Path("/candidate/final.manifest.json")
        base, directory, emitted, native, result=module.compose_for_scene(compose,False,skin,manifest,out)
        compose.assert_called_once_with(module.AR,skin,manifest,out/"anatomy")
        self.assertEqual(base,module.AR)
        self.assertEqual(native,emitted)
        self.assertEqual(native,out/"anatomy/resting-anatomy-receipt.json")
        self.assertEqual(result["status"],"final-composition")

class CompareContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime="/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib"
        self.dyld="/Users/n/numi-human-performance-build-014/lib:/build017/matter"
        self.assets={"/a/nha":"sha-a","/a/receipt":"sha-r","/a/native":"sha-n","/a/resp.metallib":"sha-m","/a/libmetalrobo.dylib":"sha-l"}
        self.expected_assets=dict(self.assets)
        self.argv=["native","--muscle-step-count","10000","--muscle-step-seconds","0.002"]
        self.env={"NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT":"0",
          "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS":"8",
          "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS":"0,4991,5375,5759,6111,6495,7743,10000",
          "NUMI_HUMAN_RESTING_INSPECTION_TOUR":"1",
          "NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS":"2.5",
          "DYLD_LIBRARY_PATH":self.dyld}
        self.assembly={"owner_cli":{"resolved_native_argv":self.argv,"asset_sha256":self.assets,
          "requested":{"inspection_tour":True,"inspection_period_s":2.5}}}
        self.invocation={"argv":self.argv,"asset_sha256":self.assets,"environment":self.env}
        loaded={"verified":True,"expected_path":self.runtime,"expected_sha256":"sha-l",
          "observed_images":[{"path":self.runtime,"dyld_uuid":"00000000-0000-0000-0000-000000000000"}]}
        self.metadata={**self.invocation,"loaded_metal_runtime":loaded}

    def validate(self):
        return module.validate_compare_contract(self.invocation,self.metadata,self.assembly,self.expected_assets,
            self.dyld,self.runtime,"sha-l")

    def test_exact_owner_assets_tour_and_runtime_pass(self):
        result=self.validate()
        self.assertTrue(result["assembly_invocation_equal"])
        self.assertEqual(result["critical_assets_verified"],5)
        self.assertTrue(result["loaded_runtime_verified"])

    def test_wrong_tour_fails(self):
        self.invocation["environment"]["NUMI_HUMAN_RESTING_INSPECTION_TOUR"]="0"
        self.metadata["environment"]=self.invocation["environment"]
        with self.assertRaisesRegex(RuntimeError,"inspection-tour"):
            self.validate()

    def test_wrong_runtime_fails(self):
        self.metadata["loaded_metal_runtime"]["expected_sha256"]="wrong"
        with self.assertRaisesRegex(RuntimeError,"loaded native Metal runtime"):
            self.validate()

    def test_missing_critical_asset_fails(self):
        self.invocation["asset_sha256"].pop("/a/receipt")
        self.metadata["asset_sha256"]=self.invocation["asset_sha256"]
        self.assembly["owner_cli"]["asset_sha256"]=self.invocation["asset_sha256"]
        with self.assertRaisesRegex(RuntimeError,"critical runtime/anatomy"):
            self.validate()

if __name__=="__main__":
    unittest.main(verbosity=2)
