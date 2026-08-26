#!/usr/bin/env python3

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
QUALIFIER = REPOSITORY / "tools" / "crow_journey_qualification.sh"


FAKE_NUMI = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
def value(name):
    return args[args.index(name) + 1]

milestones = [
    "standing", "walking", "takeoff", "cruise", "takeoff-cruise",
    "turn-left", "turn-right", "approach", "touchdown", "landed-hold",
    "full-journey",
]
band = milestones.index(value("--milestone"))
seed = int(value("--seed"))
task = "birdflow_american_crow_journey_v9_visual_neural"
run = Path(os.environ["NUMI_RUN_DIR"])
run.mkdir(parents=True)
record = {
    "task": task,
    "world_source": task,
    "minimum_sampled_difficulty_band": band,
    "maximum_sampled_difficulty_band": band,
    "benchmark_seed": seed,
    "action_source": "policy_pack",
    "birdflow_journey_teacher": False,
    "failed_environment_steps": 0,
    "termination_count": 32,
    "timeout_count": 32,
    "mean_tracking_score": 0.99,
    "mean_tilt": 0.01,
    "maximum_tilt": 0.02,
    "maximum_root_height": 1.2,
    "outcomes": {
        "approach_pitch_warning_fraction": {"mean": 0.0},
        "approach_pitch_full_envelope_fraction": {"mean": 0.0},
    },
}
(run / "evidence.json").write_text(json.dumps(record))
if "--crow-replay-pack" in args:
    replay = Path(value("--crow-replay-pack"))
    replay.write_text(json.dumps({
        "schema": "numi.crow-replay.v1",
        "payload_sha256": "a" * 64,
        "payload": {
            "classification": "simulated accepted-state replay",
            "task": task,
            "frames": [{"step": 0}, {"step": 1}],
        },
    }))
'''


class CrowJourneyQualificationTest(unittest.TestCase):
    def test_builds_complete_matrix_and_rejects_retained_bad_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "root"
            build = temporary / "build"
            runs = temporary / "runs"
            (root / "tools").mkdir(parents=True)
            (build / "bin").mkdir(parents=True)
            fake_numi = root / "tools" / "numi"
            fake_numi.write_text(FAKE_NUMI)
            fake_numi.chmod(fake_numi.stat().st_mode | stat.S_IXUSR)
            rollout = build / "bin" / "metalrobo_task_rollout"
            rollout.write_text("fixture\n")
            rollout.chmod(rollout.stat().st_mode | stat.S_IXUSR)
            policy = temporary / "deployment.policypack"
            policy.write_text("promoted fixture\n")
            environment = os.environ | {
                "NUMI_CROW_QUALIFICATION_ROOT": str(root),
                "NUMI_CROW_QUALIFICATION_BUILD": str(build),
                "NUMI_CROW_QUALIFICATION_POLICY": str(policy),
                "NUMI_CROW_QUALIFICATION_RUNS": str(runs),
            }

            subprocess.run([str(QUALIFIER)], env=environment, check=True)
            summary = json.loads((runs / "qualification.json").read_text())
            self.assertEqual(summary["run_count"], 33)
            self.assertEqual(summary["environment_count"], 1056)
            self.assertEqual(summary["benchmark_seeds"], [2650443581, 2650443582, 2650443583])
            self.assertTrue((runs / "accepted-full-journey.crowreplay.json").is_file())

            bad = runs / "band2-takeoff-seed2650443581" / "evidence.json"
            record = json.loads(bad.read_text())
            record["mean_tracking_score"] = 0.2
            bad.write_text(json.dumps(record))
            failed = subprocess.run(
                [str(QUALIFIER)], env=environment, capture_output=True, text=True
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("tracking below milestone floor", failed.stderr)


if __name__ == "__main__":
    unittest.main()
