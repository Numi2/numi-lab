from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import importlib.util, sys

import numpy as np

spec = importlib.util.spec_from_file_location("run_bulk_remesh_386", Path(__file__).with_name("run_bulk_remesh_386.py"))
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def op(edge, rows, parent=7, new_faces=None):
    return {
        "edge": list(edge), "face_rows": list(rows), "parent_row": parent,
        "new_faces": new_faces or [[1, 2, 3], [2, 4, 3]],
    }


def report(counts):
    return {
        "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT",
        "complete": True, "bank_count": 185, "expected_bank_count": 185,
        "banks": [
            {"body": i, "stable_bone_id": 1000 + i,
             "skin_aabb_active": 0 if counts[i] == 0 else 1,
             "bank_aabb_active": 0 if counts[i] == 0 else 1,
             "result": {"count": counts[i], "triangle_pairs": [[0, j] for j in range(counts[i])]}
                      if counts[i] else None}
            for i in range(185)
        ],
    }


class Bulk386Tests(unittest.TestCase):
    def test_accepted_checkpoint_copies_only_hash_pinned_complete_candidate(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"immutable candidate bytes")
            audit130 = root / "130"
            audit130.mkdir()
            summary_path = audit130 / "summary.json"
            pairs_path = audit130 / "surface-self-pairs.npy"
            pairs = np.empty((0, 2), dtype=np.int64)
            np.save(pairs_path, pairs)
            summary = {
                "candidate_sha256": runner.sha256_file(candidate),
                "complete_surface_audit": True,
                "exact_PLC_surface_pairs": 0,
            }
            summary_path.write_text(json.dumps(summary))
            audit159 = root / "159.json"
            r = report([0] * 185)
            r["candidate_sha256"] = runner.sha256_file(candidate)
            audit159.write_text(json.dumps(r))
            strict = {"before_total": 1, "after_total": 0,
                      "strict_reduction": 1, "typed_bank_increases": []}
            checkpoint = runner.persist_accepted_checkpoint(
                root / "run", 1, candidate, summary_path, pairs_path, audit159, strict,
            )
            saved = root / "run/checkpoints/round-001/candidate.npz"
            self.assertEqual(runner.sha256_file(saved), runner.sha256_file(candidate))
            self.assertEqual(checkpoint["self_pair_count"], 0)
            self.assertEqual(checkpoint["strict_159_reduction"]["strict_reduction"], 1)

    def test_checkpoint_fails_closed_for_hash_or_nonprogress(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"candidate")
            audit130 = root / "130"
            audit130.mkdir()
            pairs_path = audit130 / "surface-self-pairs.npy"
            np.save(pairs_path, np.empty((0, 2), dtype=np.int64))
            summary_path = audit130 / "summary.json"
            summary_path.write_text(json.dumps({
                "candidate_sha256": "0" * 64,
                "complete_surface_audit": True,
                "exact_PLC_surface_pairs": 0,
            }))
            audit159 = root / "159.json"
            audit159.write_text(json.dumps(report([0] * 185)))
            with self.assertRaises(runner.PipelineError):
                runner.persist_accepted_checkpoint(
                    root / "run", 1, candidate, summary_path, pairs_path, audit159,
                    {"before_total": 1, "after_total": 0, "strict_reduction": 1},
                )

    def test_terminal_receipt_points_to_latest_accepted_checkpoint(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoints/round-001"
            cp.mkdir(parents=True)
            payload = {"status": "RECOVERED_ACCEPTED_ROUND_NO_RERUN", "round": 1}
            (cp / "checkpoint.json").write_text(json.dumps(payload))
            runner.write_terminal_receipt(root, "NO_PROGRESS")
            receipt = json.loads((root / "terminal-receipt.json").read_text())
            self.assertEqual(receipt["status"], "NO_PROGRESS")
            self.assertEqual(receipt["latest_accepted_checkpoint"]["checkpoint"]["round"], 1)

    def test_terminal_failure_retains_latest_checkpoint_and_failed_artifact_hashes(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoints/round-001"
            cp.mkdir(parents=True)
            payload = {"status": "ACCEPTED_CHECKPOINT", "round": 1}
            (cp / "checkpoint.json").write_text(json.dumps(payload))
            failed = root / "round-002-report.json"
            failed.write_text("failure witness")
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"candidate")
            baseline = root / "baseline.json"
            baseline.write_text("{}")
            runner.write_terminal_failure(
                root, runner.PipelineError("strict decrease zero"), candidate, baseline,
            )
            receipt = json.loads((root / "terminal-failure.json").read_text())
            self.assertEqual(receipt["latest_accepted_checkpoint"]["checkpoint"]["round"], 1)
            paths = {item["path"] for item in receipt["failed_output_artifacts"]}
            self.assertIn("round-002-report.json", paths)
            self.assertNotIn("terminal-failure.json", paths)

    def test_filter_removes_every_witness_operation_without_fill(self):
        operations = [
            op((1, 2), (10, 11)),
            op((3, 4), (12, 13)),
            op((5, 6), (14, 15)),
        ]
        kept, removed = runner.filter_implicated_operations(operations, [10, 14])
        self.assertEqual([x["edge"] for x in kept], [[3, 4]])
        self.assertEqual([x["edge"] for x in removed], [[1, 2], [5, 6]])
        self.assertEqual(len(kept) + len(removed), len(operations))

    def test_filter_rejects_duplicate_operations_and_bad_witness_rows(self):
        duplicate = op((1, 2), (10, 11))
        with self.assertRaises(runner.PipelineError):
            runner.filter_implicated_operations([duplicate, duplicate], [])
        with self.assertRaises(runner.PipelineError):
            runner.filter_implicated_operations([duplicate], [-1])

    def test_whitelist_replay_is_exact_and_keeps_points_and_matched_faces(self):
        expected = op((1, 2), (10, 11))
        data = {
            "points_world_m": np.zeros((5, 3), dtype=np.float32),
            "source_O_faces": np.array([[0, 1, 2]], dtype=np.int32),
            "area_reference_037_source_O_faces": np.array([[0, 1, 2]], dtype=np.int32),
        }

        class FakeFlip:
            def __init__(self):
                self.edges = None

            def propose_flip_batch(self, candidate, seeds, **kwargs):
                self.edges = kwargs["allowed_edges"]
                return dict(candidate), {"operations": [expected]}

        fake = FakeFlip()
        out, replay_report = runner.replay_exact_whitelist(data, [expected], [10], fake)
        self.assertEqual(fake.edges, [[1, 2]])
        self.assertEqual(replay_report["operations"], [expected])
        np.testing.assert_array_equal(out["points_world_m"], data["points_world_m"])
        np.testing.assert_array_equal(out["source_O_faces"], data["area_reference_037_source_O_faces"])

    def test_whitelist_replay_fails_closed_if_operator_fills_or_drops(self):
        expected = op((1, 2), (10, 11))
        data = {
            "points_world_m": np.zeros((5, 3), dtype=np.float32),
            "source_O_faces": np.array([[0, 1, 2]], dtype=np.int32),
            "area_reference_037_source_O_faces": np.array([[0, 1, 2]], dtype=np.int32),
        }

        class WrongFlip:
            def propose_flip_batch(self, candidate, seeds, **kwargs):
                replacement = op((2, 3), (12, 13))
                return dict(candidate), {"operations": [replacement]}

        with self.assertRaises(runner.PipelineError):
            runner.replay_exact_whitelist(data, [expected], [10], WrongFlip())

    def test_pair_decrease_requires_all185_nonincreasing_and_strict_total(self):
        before = report([1] + [0] * 184)
        after = report([0] * 185)
        self.assertEqual(runner.require_strict_pair_decrease(before, after)["strict_reduction"], 1)
        with self.assertRaises(runner.PipelineError):
            runner.require_strict_pair_decrease(before, before)
        increased = report([2] + [0] * 184)
        with self.assertRaises(runner.PipelineError):
            runner.require_strict_pair_decrease(before, increased)

    def test_130_receipt_requires_saved_pair_count_and_exact_candidate_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"candidate")
            report_dir = root / "audit"
            report_dir.mkdir()
            (report_dir / "surface-self-pairs.npy").write_bytes(b"")
            pairs = np.empty((0, 2), dtype=np.int64)
            np.save(report_dir / "surface-self-pairs.npy", pairs)
            import hashlib
            summary = {
                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "complete_surface_audit": True, "exact_PLC_surface_pairs": 0,
            }
            (report_dir / "summary.json").write_text(__import__("json").dumps(summary))
            checked, got = runner.checked_130_summary(report_dir / "summary.json", candidate)
            self.assertEqual(checked["exact_PLC_surface_pairs"], 0)
            self.assertEqual(got.shape, (0, 2))
            summary["exact_PLC_surface_pairs"] = 1
            (report_dir / "summary.json").write_text(__import__("json").dumps(summary))
            with self.assertRaises(runner.PipelineError):
                runner.checked_130_summary(report_dir / "summary.json", candidate)


if __name__ == "__main__":
    unittest.main()
