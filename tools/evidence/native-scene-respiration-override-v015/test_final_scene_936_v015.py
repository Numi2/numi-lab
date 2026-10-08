import importlib.util
import hashlib
import subprocess
import os
import tempfile
import json
import argparse
from pathlib import Path
import unittest
from unittest.mock import Mock

SCRIPT=Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/v015/final_scene_936-v015.py")
spec=importlib.util.spec_from_file_location("final_scene_936_v015",SCRIPT)
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
        self.dyld="/Users/n/numi-human-performance-build-014/lib:/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/matter"
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


class AnatomyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.nha=self.root/"anatomy.nha";self.nha.write_bytes(b"corrected-payload")
        self.receipt=self.root/"receipt.json"
        self.receipt.write_text(json.dumps({"payload":{"path":str(self.nha),"sha256":module.sha(self.nha)}}))
        self.args=argparse.Namespace(anatomy=self.nha,anatomy_sha=module.sha(self.nha),
            anatomy_receipt=self.receipt,anatomy_receipt_sha=module.sha(self.receipt),fixture_legacy_907=False)
    def test_selected_candidate_requires_exact_receipt_binding(self):
        self.assertEqual(module.select_anatomy(self.args),(self.nha,self.args.anatomy_sha,self.receipt,self.args.anatomy_receipt_sha))
        compose=Mock(return_value={})
        base,_,_,native,_=module.compose_for_scene(compose,False,Path("/skin"),Path("/manifest"),self.root,self.receipt)
        self.assertEqual(base,self.receipt)
        self.assertEqual(native,self.root/"anatomy/resting-anatomy-receipt.json")
        compose.assert_called_once_with(self.receipt,Path("/skin"),Path("/manifest"),self.root/"anatomy")
    def test_partial_override_rejected(self):
        self.args.anatomy_sha=None
        with self.assertRaisesRegex(RuntimeError,"both paths"):module.select_anatomy(self.args)
    def test_fixture_override_rejected(self):
        self.args.fixture_legacy_907=True
        with self.assertRaisesRegex(RuntimeError,"legacy fixture"):module.select_anatomy(self.args)
    def test_receipt_alias_path_rejected_even_with_same_bytes(self):
        other=self.root/"alias.nha";other.write_bytes(self.nha.read_bytes())
        self.receipt.write_text(json.dumps({"payload":{"path":str(other),"sha256":self.args.anatomy_sha}}))
        self.args.anatomy_receipt_sha=module.sha(self.receipt)
        with self.assertRaisesRegex(RuntimeError,"exact selected"):module.select_anatomy(self.args)
    def test_payload_mutation_rejected(self):
        self.nha.write_bytes(b"mutated")
        with self.assertRaisesRegex(RuntimeError,"hash mismatch"):module.select_anatomy(self.args)
    def test_comparator_binds_assembled_anatomy_and_rejects_missing_identity(self):
        a={"native_receipt":{"path":str(self.receipt),"sha256":self.args.anatomy_receipt_sha,
            "NHA_sha256":self.args.anatomy_sha},"base_nha_sha256":self.args.anatomy_sha,
            "source_sha256":{str(self.nha):self.args.anatomy_sha}}
        self.assertEqual(module.anatomy_assets_from_assembly(a),{str(self.nha):self.args.anatomy_sha,str(self.receipt):self.args.anatomy_receipt_sha})
        a["source_sha256"]={}
        with self.assertRaisesRegex(RuntimeError,"immutable source"):module.anatomy_assets_from_assembly(a)


class RetiredAliasVisibilityLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)
        (self.run/"native.log").write_text(
            "resting_retired_inspection_source semantic=51010 stable_id=22 layer_mask=0 payload_retained=true geometry_retained=true\n")

    def test_log_proves_zero_mask_and_retained_payload(self):
        proof=module.verify_retired_alias_visibility_log(self.run)
        self.assertEqual(proof["retired_organ_stable_ids"],[22])
        self.assertTrue(proof["zero_inspection_mask"])
        self.assertTrue(proof["payload_retained"])
        self.assertTrue(proof["geometry_retained"])

    def test_exact_organ_semantic_token_is_required(self):
        proof=module.verify_retired_alias_visibility_log(self.run)
        self.assertEqual(proof["retired_semantic"],51010)
        self.assertTrue(proof["zero_inspection_mask"])

    def test_missing_or_wrong_semantic_fails(self):
        for semantic in ("", "semantic=51011 "):
            (self.run/"native.log").write_text(
                "resting_retired_inspection_source "+semantic+"stable_id=22 layer_mask=0 payload_retained=true geometry_retained=true\n")
            with self.assertRaisesRegex(RuntimeError,"organ semantic 51010"):
                module.verify_retired_alias_visibility_log(self.run)


    def test_wrong_or_ambiguous_stable_id_fails(self):
        (self.run/"native.log").write_text(
            "resting_retired_inspection_source semantic=51010 stable_id=220 layer_mask=0 payload_retained=true geometry_retained=true\n")
        with self.assertRaisesRegex(RuntimeError,"zero inspection mask"):
            module.verify_retired_alias_visibility_log(self.run)

    def test_nonzero_mask_fails(self):
        (self.run/"native.log").write_text(
            "resting_retired_inspection_source semantic=51010 stable_id=22 layer_mask=8 payload_retained=true geometry_retained=true\n")
        with self.assertRaisesRegex(RuntimeError,"zero inspection mask"):
            module.verify_retired_alias_visibility_log(self.run)

    def test_missing_or_duplicate_retirement_record_fails(self):
        (self.run/"native.log").write_text("native initialized\n")
        with self.assertRaisesRegex(RuntimeError,"lacks one retired-alias"):
            module.verify_retired_alias_visibility_log(self.run)
        (self.run/"native.log").write_text(
            "resting_retired_inspection_source stable_id=22 layer_mask=0 payload_retained=true geometry_retained=true\n"*2)
        with self.assertRaisesRegex(RuntimeError,"lacks one retired-alias"):
            module.verify_retired_alias_visibility_log(self.run)



class RespirationConfigTests(unittest.TestCase):
    def test_default_is_exact_frozen_761_config(self):
        args=argparse.Namespace(respiration_config=None,respiration_config_sha=None)
        path,digest=module.select_respiration(args)
        self.assertEqual(path,module.R.resolve())
        self.assertEqual(digest,module.P["resp"])

    def test_custom_respiration_requires_matching_path_hash_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/"resp.json";config.write_text('{"diaphragm_area_m2":0.02}')
            digest=module.sha(config)
            args=argparse.Namespace(respiration_config=config,respiration_config_sha=digest)
            self.assertEqual(module.select_respiration(args),(config.resolve(),digest))
            args.respiration_config_sha="0"*64
            with self.assertRaisesRegex(RuntimeError,"hash mismatch"):
                module.select_respiration(args)

    def test_partial_custom_pair_rejected(self):
        args=argparse.Namespace(respiration_config=Path("/tmp/resp.json"),respiration_config_sha=None)
        with self.assertRaisesRegex(RuntimeError,"both path and exact hash"):
            module.select_respiration(args)

    def test_comparator_derives_actual_config_from_owner_argv_and_source_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/"resp.json";config.write_text('{"diaphragm_area_m2":0.02}')
            digest=module.sha(config);path=str(config.resolve())
            assembly={"owner_cli":{"resolved_native_argv":["native","--resting-scene","/tmp/circ.json",path]},
                      "source_sha256":{path:digest},
                      "respiration_config":{"path":path,"sha256":digest}}
            self.assertEqual(module.respiration_asset_from_assembly(assembly),{path:digest})
            legacy={"owner_cli":assembly["owner_cli"],"source_sha256":{path:digest}}
            self.assertEqual(module.respiration_asset_from_assembly(legacy),{path:digest})
            bad={**assembly,"respiration_config":{"path":path,"sha256":"0"*64}}
            with self.assertRaisesRegex(RuntimeError,"differs from immutable source"):
                module.respiration_asset_from_assembly(bad)
            badargv={"owner_cli":{"resolved_native_argv":["native","--resting-scene","/tmp/circ.json","/other/resp.json"]},
                     "source_sha256":{path:digest},"respiration_config":{"path":path,"sha256":digest}}
            with self.assertRaisesRegex(RuntimeError,"differs from assembled identity"):
                module.respiration_asset_from_assembly(badargv)

    def test_owner_asset_contract_requires_selected_config_hash(self):
        old=CompareContractTests()
        old.setUp()
        old.assets["/a/resp.json"]="sha-rsp"
        old.expected_assets["/a/resp.json"]="sha-rsp"
        old.invocation["asset_sha256"]["/a/resp.json"]="sha-rsp"
        old.assembly["owner_cli"]["asset_sha256"]["/a/resp.json"]="sha-rsp"
        old.metadata["asset_sha256"]=old.invocation["asset_sha256"]
        result=module.validate_compare_contract(old.invocation,old.metadata,old.assembly,old.expected_assets,
            old.dyld,old.runtime,"sha-l")
        self.assertTrue(result["assembly_invocation_equal"])
        old.invocation["asset_sha256"].pop("/a/resp.json")
        old.metadata["asset_sha256"]=old.invocation["asset_sha256"]
        old.assembly["owner_cli"]["asset_sha256"]=old.invocation["asset_sha256"]
        with self.assertRaisesRegex(RuntimeError,"critical runtime/anatomy"):
            module.validate_compare_contract(old.invocation,old.metadata,old.assembly,old.expected_assets,
                old.dyld,old.runtime,"sha-l")

if __name__=="__main__":
    unittest.main(verbosity=2)


class SourceBindingTests(unittest.TestCase):
    def test_build_brain_abi_directory_hash_uses_pinned_manifest_algorithm(self):
        path=Path("/Users/n/numi-human-resting-integration-20261005/numi-brain/Sources/NumiBrainABI/include")
        expected="7bcac3a30f08ca552b6ab1c7b76d58f9c026e5a070a99bae80c1bc567f87b816"
        self.assertEqual(module.hash_path(path),expected)
        self.assertEqual(module.ck(path,expected),expected)

    def test_current_viewer_source_checkout_is_exact_commit_and_clean(self):
        checkout=module.repo(module.V,module.P["viewer_commit"])
        self.assertEqual(checkout["head"],module.P["viewer_commit"])
        self.assertTrue(checkout["clean"])

    def test_viewer_build_lineage_verifies_original_dirty_build_and_committed_bytes(self):
        build=json.loads((module.B/"evidence/build-pins.json").read_text())
        source=json.loads((module.B/"evidence/source-pins.json").read_text())
        proof=module.verify_viewer_build_lineage(build,source)
        self.assertEqual(proof["current_source_checkout"]["head"],module.P["viewer_commit"])
        self.assertGreaterEqual(proof["build_source_file_and_directory_pins_verified"],22)
        self.assertIn("apps/NumiHumanRestingInspectionLayers.hpp",proof["compiled_source_byte_equivalence"])
        self.assertEqual(proof["build_source_provenance"]["source_tree_clean_at_build"],False)

    def test_tree_hash_includes_every_regular_file_and_relative_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"b.txt").write_bytes(b"b")
            (root/"sub").mkdir()
            (root/"sub/a.txt").write_bytes(b"a")
            paths=sorted((root/"b.txt",root/"sub/a.txt"))
            manifest="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in paths).encode()
            self.assertEqual(module.hash_path(root),hashlib.sha256(manifest).hexdigest())

