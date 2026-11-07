import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


COMMAND = Path(__file__).resolve().parents[1] / "numi/commands/crow"


class EmbodiedCommandContractTest(unittest.TestCase):
    def invoke(self, arguments):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tools").mkdir()
            dispatcher = root / "tools/numi"
            dispatcher.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            dispatcher.chmod(0o755)
            return subprocess.run([str(COMMAND), *arguments], text=True, capture_output=True,
                                  env=os.environ | {"NUMI_LAB_ROOT": str(root)})

    def test_new_embodiment_starts_without_teacher_or_legacy_visuals(self):
        result = self.invoke(["journey", "train", "--variant", "v11-embodied", "--milestone", "standing",
                              "--envs", "8", "--steps", "128", "--updates", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(result.stdout)
        self.assertIn("v11-embodied", arguments)
        self.assertIn("--initialize-policy", arguments)
        self.assertIn("--zero-actor-output", arguments)
        self.assertNotIn("--birdflow-journey-teacher", arguments)
        self.assertNotIn("--visual-observation-config", arguments)

    def test_explicit_teacher_and_unmatched_presentations_are_rejected(self):
        for action, extra in (("train", ["--birdflow-journey-teacher"]), ("capture", []), ("window", [])):
            with self.subTest(action=action):
                result = self.invoke(["journey", action, "--variant", "v11-embodied", *extra])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("v11", result.stderr)

    def test_legacy_teacher_control_stays_explicit(self):
        result = self.invoke(["journey", "train", "--variant", "v9-visual-neural"])
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads(result.stdout)
        self.assertIn("--birdflow-journey-teacher", arguments)
        self.assertIn("--visual-observation-config", arguments)

    def test_residual_training_is_rejected_for_both_contracts(self):
        for variant in ("v9-visual-neural", "v11-embodied"):
            with self.subTest(variant=variant):
                result = self.invoke(["journey", "train", "--variant", variant,
                                      "--base-policy-pack", "unadmitted.policypack"])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("does not accept a residual base policy", result.stderr)


if __name__ == "__main__":
    unittest.main()
