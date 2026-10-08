#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "prepare_final_plan.py"
spec = importlib.util.spec_from_file_location("readiness_917_v007", str(MODULE))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path.resolve()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalSceneAssemblyLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene_dir = self.root / "candidate-scene" / "native-run"
        self.scene_dir.mkdir(parents=True)
        self.assembly_path = self.scene_dir.parent / "assembly-preflight.json"
        self.skin = put(self.root / "candidate" / "body.nhskin", b"new candidate NHSKIN")
        self.manifest = put(
            self.root / "candidate" / "registration.manifest.json",
            json.dumps({
                "schema": "numi.human.common-atlas-skin-geometry-registration.v1",
                "output_payload": {"path": str(self.skin), "sha256": file_sha(self.skin)}
            }))
        self.nha = put(self.root / "candidate" / "anatomy-v999.nhanatomy", b"dynamic candidate NHA v999")
        self.receipt = put(
            self.root / "candidate" / "resting-anatomy-receipt.json",
            json.dumps({"schema": "numi.human.resting-anatomy-receipt.v1",
                        "payload": {"path": str(self.nha), "sha256": file_sha(self.nha)}}))
        self.base_receipt = put(
            self.root / "candidate" / "base-anatomy-receipt.json",
            json.dumps({"schema": "numi.human.resting-anatomy-receipt.v1",
                        "payload": {"path": str(self.nha), "sha256": file_sha(self.nha)}}))
        self.scene_manifest = put(self.root / "candidate" / "resting-scene.manifest.json", b"scene manifest")
        self.support = put(self.root / "candidate" / "support.nhcnt", b"support")
        # The assembly source map hashes preparation inputs. The final runtime
        # scene/support outputs are bound separately through scene fields and
        # owner_cli.asset_sha256, matching the actual 936 report contract.
        self.source_hashes = {
            str(path): file_sha(path) for path in (
                self.skin, self.manifest, self.nha, self.receipt, self.base_receipt)
        }
        self.argv = [
            "/test/bin/numi-human-native", "/test/rigid.nhrigid", "/test/muscle.nhmyo",
            "/test/bones.nhbones", str(self.scene_dir), "--persistent-metal-stand",
            "--muscle-step-seconds", "0.002", "--muscle-step-count", "10000",
            "--support-contact-payload", str(self.support), "--root-pose",
            "0.0", "1.0", "2.0", "0.0", "0.0", "0.0", "1.0",
            "--torso-anatomy-payload", str(self.nha),
            "--resting-anatomy-receipt", str(self.receipt)
        ]
        asset_hashes = {
            str(self.skin): file_sha(self.skin),
            str(self.nha): file_sha(self.nha),
            str(self.receipt): file_sha(self.receipt),
            str(self.scene_manifest): file_sha(self.scene_manifest),
            str(self.support): file_sha(self.support),
        }
        self.invocation = {
            "argv": self.argv,
            "asset_sha256": asset_hashes,
            "environment": {"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS": "0,10000"}
        }
        self.report = {
            "schema": "numi.human.final-native-scene-preflight.v1",
            "status": "assembled_owner_cli_validated_native_not_run",
            "native_run": False,
            "native_command_executable": True,
            "asset_role": "caller-pinned corrected NHSKIN",
            "owner_checkouts": [
                {"path": str(prepare.FINAL_SCENE_936_HUMAN),
                 "head": prepare.FINAL_SCENE_936_HUMAN_REV, "clean": True},
                {"path": str(prepare.FINAL_SCENE_936_LAB),
                 "head": prepare.FINAL_SCENE_936_LAB_REV, "clean": True},
            ],
            "source_sha256": self.source_hashes,
            "base_nha_sha256": file_sha(self.nha),
            "skin_composition": {
                "used_by_native_cli": True,
                "base_receipt_path": str(self.base_receipt),
                "base_receipt_sha256": file_sha(self.base_receipt),
                "composed_receipt_path": str(self.receipt),
                "composed_receipt_sha256": file_sha(self.receipt),
                "composition_base_nha_sha256": file_sha(self.nha),
                "composed_skin_sha256": file_sha(self.skin),
                "owner_result": {
                    "anatomy_payload_unchanged": True,
                    "functional_bindings_unchanged": True,
                    "candidate_skin_path": str(self.skin),
                    "candidate_skin_sha256": file_sha(self.skin),
                    "candidate_manifest_path": str(self.manifest),
                    "candidate_manifest_sha256": file_sha(self.manifest),
                    "receipt_path": str(self.receipt),
                    "receipt_sha256": file_sha(self.receipt),
                    "base_receipt_path": str(self.base_receipt),
                    "base_receipt_sha256": file_sha(self.base_receipt),
                }
            },
            "native_receipt": {
                "path": str(self.receipt), "sha256": file_sha(self.receipt),
                "NHA_sha256": file_sha(self.nha)
            },
            "scene": {
                "manifest": str(self.scene_manifest),
                "manifest_sha256": file_sha(self.scene_manifest),
                "support_payload": str(self.support),
                "support_sha256": file_sha(self.support),
                "root_pose": {"root_translation_xyz_m": [0.0, 1.0, 2.0],
                              "root_delta_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
            },
            "owner_cli": {
                "resolved_native_argv": self.argv,
                "asset_sha256": asset_hashes,
                "requested": {
                    "nominal_seconds": 20, "dt": 0.002, "steps": 10000,
                    "q_audit": 0, "COM_segment_steps": 8,
                    "inspection_tour": True, "inspection_period_s": 2.5,
                    "captures": [0, 10000]
                }
            }
        }
        self.write_report()

    def tearDown(self):
        self.tmp.cleanup()

    def write_report(self):
        put(self.assembly_path, json.dumps(self.report))

    def test_accepts_candidate_with_dynamic_non924_nha_and_receipt(self):
        proof = prepare.verify_final_scene_assembly(
            self.scene_dir, self.invocation, self.receipt)
        self.assertEqual(proof["nha_path"], str(self.nha))
        self.assertEqual(proof["nha_sha256"], file_sha(self.nha))
        self.assertEqual(proof["skin_sha256"], file_sha(self.skin))
        self.assertEqual(proof["manifest_sha256"], file_sha(self.manifest))
        self.assertEqual(proof["native_receipt_path"], str(self.receipt))

    def test_rejects_missing_adjacent_assembly_report(self):
        self.assembly_path.unlink()
        with self.assertRaisesRegex(ValueError, "adjacent 936 assembly preflight must be a regular"):
            prepare.verify_final_scene_assembly(self.scene_dir, self.invocation, self.receipt)

    def test_rejects_legacy_907_fixture_report(self):
        fixture = Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/legacy-907-cpu-fixture-v011/assembly-preflight.json")
        shutil.copyfile(fixture, self.assembly_path)
        with self.assertRaisesRegex(ValueError, "non-fixture corrected 936 scene assembly"):
            prepare.verify_final_scene_assembly(self.scene_dir, self.invocation, self.receipt)

    def test_rejects_receipt_not_selected_by_936_report(self):
        other = put(self.root / "candidate" / "other-receipt.json", b"other receipt")
        with self.assertRaisesRegex(ValueError, "supplied anatomy receipt differs"):
            prepare.verify_final_scene_assembly(self.scene_dir, self.invocation, other)

    def test_rejects_invocation_asset_mismatch(self):
        changed = dict(self.invocation)
        changed["asset_sha256"] = dict(changed["asset_sha256"])
        changed["asset_sha256"][str(self.skin)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native invocation differs"):
            prepare.verify_final_scene_assembly(self.scene_dir, changed, self.receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
