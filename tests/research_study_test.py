import tempfile
import unittest
from pathlib import Path

from metalrobo.research_study import expand_runs, protocol_hash, validate_protocol


def protocol():
    return {
        "schema": "numi.research-study.v1", "id": "test", "status": "preregistered",
        "branch": "numisolver", "learner_seeds": [1, 2, 3, 4, 5],
        "methods": [
            {"id": "baseline", "teacher_disconnected_evaluation": False},
            {"id": "teacher", "teacher_disconnected_evaluation": True,
             "train_args": ["--teacher"]},
        ],
        "tasks": [{"id": "balance", "train_args": ["--task", "velocity"]}],
        "primary_outcomes": ["success"], "claim_gates": ["complete"],
    }


class ResearchStudyTest(unittest.TestCase):
    def test_expansion_is_paired_and_deterministic(self):
        value = protocol()
        with tempfile.TemporaryDirectory() as temporary:
            runs = expand_runs(value, Path(temporary))
        self.assertEqual(len(runs), 10)
        self.assertEqual({run["seed"] for run in runs}, {1, 2, 3, 4, 5})
        self.assertEqual({run["method"] for run in runs}, {"baseline", "teacher"})
        self.assertTrue(all(run["protocol_sha256"] == protocol_hash(value) for run in runs))

    def test_fewer_than_five_seeds_is_rejected(self):
        value = protocol()
        value["learner_seeds"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "five"):
            validate_protocol(value)

    def test_teacher_disconnection_is_required(self):
        value = protocol()
        value["methods"][1]["teacher_disconnected_evaluation"] = False
        with self.assertRaisesRegex(ValueError, "teacher-disconnected"):
            validate_protocol(value)


if __name__ == "__main__":
    unittest.main()
