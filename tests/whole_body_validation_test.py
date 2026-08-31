import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY_ROOT / "tools" / "validate_numilab_human.py"


def scope(report, identifier):
    return next(item for item in report["scopes"] if item["id"] == identifier)


class WholeBodyValidationTest(unittest.TestCase):
    def run_validator(self, *arguments, expected_code=0):
        completed = subprocess.run(
            [sys.executable, str(VALIDATOR), *arguments],
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, expected_code, completed.stderr)
        return json.loads(completed.stdout)

    def test_repository_manifest_reports_strength_aware_gaps(self):
        report = self.run_validator("--allow-incomplete", "--quiet")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["summary"]["scope_count"], 21)
        self.assertEqual(report["summary"]["qualified_scope_count"], 0)
        self.assertEqual(
            report["evidence"]["fully_distributed_exact_entheses"]["status"],
            "contradicted",
        )
        self.assertEqual(
            report["evidence"]["complete_soft_tissue_geometry"]["status"],
            "contradicted",
        )

        left_knee = scope(report, "left_knee")["requirements"]
        self.assertEqual(left_knee["active_tendon_to_bone"]["status"], "contradicted")
        self.assertEqual(left_knee["passive_joint_tissue"]["status"], "contradicted")
        self.assertEqual(left_knee["articular_contact"]["status"], "insufficient")
        self.assertEqual(left_knee["sustained_loaded_motion"]["status"], "missing")

        self.assertEqual(
            report["evidence"]["bilateral_achilles_force_transfer"]["status"],
            "verified",
        )
        for identifier in (
            "left_ankle_hindfoot_midfoot",
            "right_ankle_hindfoot_midfoot",
        ):
            ankle = scope(report, identifier)["requirements"]
            self.assertEqual(ankle["source_muscle_actuation"]["status"], "verified")
            self.assertEqual(ankle["active_tendon_to_bone"]["status"], "verified")
            self.assertEqual(ankle["deterministic_transaction"]["status"], "verified")
            self.assertEqual(ankle["passive_joint_tissue"]["status"], "missing")
            self.assertEqual(ankle["articular_contact"]["status"], "missing")

        shoulder = scope(report, "left_shoulder")["requirements"]
        self.assertEqual(shoulder["geometry_registration"]["status"], "verified")
        self.assertEqual(shoulder["active_tendon_to_bone"]["status"], "contradicted")

        costal = scope(report, "costal_cartilage")["requirements"]
        self.assertEqual(costal["anatomical_geometry"]["status"], "verified")
        self.assertEqual(costal["deformable_mechanics"]["status"], "insufficient")

        toes = scope(report, "left_toes_compound")["requirements"]
        self.assertEqual(toes["compound_dof_policy"]["status"], "verified")

    def test_default_exit_fails_closed_while_incomplete(self):
        report = self.run_validator("--quiet", expected_code=1)
        self.assertEqual(report["status"], "incomplete")

    def test_missing_evidence_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            manifest = {
                "schema": "numilab.human.whole-body-validation-manifest.v1",
                "evidence_levels": ["structural", "one_step"],
                "evidence_boundary": "test fixture",
                "evidence": {
                    "missing": {
                        "path": "missing.json",
                        "format": "json",
                        "level": "one_step",
                        "checks": [],
                    }
                },
                "profiles": {},
                "regions": [
                    {
                        "id": "fixture",
                        "requirements": {
                            "mechanics": {
                                "minimum_level": "one_step",
                                "evidence": ["missing"],
                            }
                        },
                    }
                ],
                "continuum_layers": [],
            }
            manifest_path = root / "docs" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--quiet",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["evidence"]["missing"]["status"], "missing")
            self.assertEqual(
                scope(report, "fixture")["requirements"]["mechanics"]["status"],
                "missing",
            )


if __name__ == "__main__":
    unittest.main()
