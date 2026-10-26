import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).with_name("refresh_point_depth_rows.py")
spec = importlib.util.spec_from_file_location("refresh_point_depth_rows_tested", P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PointDepthRefreshTests(unittest.TestCase):
    def test_candidate_summary_requires_exact_complete_zero_step_binding(self):
        good = {"candidate_sha256": "a" * 64, "points": 2, "faces": 1,
                "complete_surface_audit": True, "exact_PLC_surface_pairs": 0,
                "geometry_admitted": False, "accepted_physical_steps": 0}
        module.validate_point_summary(good, candidate_sha="a" * 64,
                                      point_count=2, face_count=1)
        bad = dict(good, candidate_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            module.validate_point_summary(bad, candidate_sha="a" * 64,
                                          point_count=2, face_count=1)
        bad = dict(good, exact_PLC_surface_pairs=1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            module.validate_point_summary(bad, candidate_sha="a" * 64,
                                          point_count=2, face_count=1)

    def test_surface_report_requires_all_unique_typed_banks(self):
        banks = [{"body": i, "stable_bone_id": 1000 + i, "result": {"count": 0}}
                 for i in range(185)]
        good = {"candidate_sha256": "c" * 64,
                "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT",
                "complete": True, "bank_count": 185, "expected_bank_count": 185,
                "surface_point_count": 2, "surface_face_count": 1, "banks": banks}
        keys = module.validate_surface_report(good, candidate_sha="c" * 64,
                                              point_count=2, face_count=1)
        self.assertEqual(len(keys), 185)
        with self.assertRaisesRegex(ValueError, "unique typed banks"):
            module.validate_surface_report(dict(good, banks=banks[:-1]),
                                           candidate_sha="c" * 64,
                                           point_count=2, face_count=1)

    def test_query_receipt_requires_complete_mechanics_node_domain(self):
        import tempfile, hashlib
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp)
            payloads = {"points.csv": b"p", "pairs.csv": b"x", "nodes.csv": b"n"}
            for name, data in payloads.items():
                (q / name).write_bytes(data)
            receipt = {"query_rows": 2, "physical_steps": 0,
                       "identity_domain": "mechanics_node_row",
                       "argv": ["bin", "tri.csv", "body.csv", "points.csv", "pairs.csv", "nodes.csv", "185"],
                       "points_sha256": hashlib.sha256(b"p").hexdigest(),
                       "pairs_sha256": hashlib.sha256(b"x").hexdigest(),
                       "nodes_sha256": hashlib.sha256(b"n").hexdigest()}
            (q / "query-receipt.json").write_text("{}")
            body, files = module.validate_query_receipt(receipt, point_count=2, query_dir=q)
            self.assertEqual(body.name, "body.csv")
            self.assertEqual(set(files), {"points", "pairs", "nodes", "receipt"})
            bad = dict(receipt, physical_steps=1)
            with self.assertRaisesRegex(ValueError, "receipt"):
                module.validate_query_receipt(bad, point_count=2, query_dir=q)
            bad = dict(receipt, argv=receipt["argv"][:-1] + ["184"])
            with self.assertRaisesRegex(ValueError, "all expected banks"):
                module.validate_query_receipt(bad, point_count=2, query_dir=q)

    def test_sparse_pairs_are_complete_when_node_rows_account_for_aabb_pruning(self):
        import csv, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            nodes = Path(tmp) / "nodes.csv"
            fields = ["query_row", "total_bone_banks", "aabb_proven_outside_banks",
                      "parity_outside_banks", "inside_banks", "ambiguous_banks",
                      "invalid_banks", "parity_query_banks"]
            rows = [
                [0, 185, 184, 0, 1, 0, 0, 1],
                [1, 185, 185, 0, 0, 0, 0, 0],
            ]
            with nodes.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerows(rows)
            parsed = module.validate_node_accounting(nodes, point_count=2, bank_count=185)
            self.assertEqual(len(parsed), 2)
            bank_keys = [(7, 101 + i) for i in range(185)]
            sparse_pairs = [{"query_row": "0", "bank_body": "7", "bank_stable_id": "101"}]
            self.assertLess(len(sparse_pairs), 2 * 185)
            module.validate_sparse_pair_rows(sparse_pairs, point_count=2, bank_keys=bank_keys)

    def test_node_accounting_rejects_missing_or_inconsistent_bank_coverage(self):
        import csv, tempfile
        fields = ["query_row", "total_bone_banks", "aabb_proven_outside_banks",
                  "parity_outside_banks", "inside_banks", "ambiguous_banks",
                  "invalid_banks", "parity_query_banks"]
        valid = [[0, 185, 184, 0, 1, 0, 0, 1], [1, 185, 185, 0, 0, 0, 0, 0]]
        with tempfile.TemporaryDirectory() as tmp:
            nodes = Path(tmp) / "nodes.csv"
            def write(rows):
                with nodes.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerows(rows)
            write(valid[:-1])
            with self.assertRaisesRegex(ValueError, "row count"):
                module.validate_node_accounting(nodes, point_count=2, bank_count=185)
            bad = [r[:] for r in valid]
            bad[0][2] = 183
            write(bad)
            with self.assertRaisesRegex(ValueError, "complete valid bank accounting"):
                module.validate_node_accounting(nodes, point_count=2, bank_count=185)
            bad = [r[:] for r in valid]
            bad[0][5] = 1
            bad[0][2] = 183
            write(bad)
            with self.assertRaisesRegex(ValueError, "complete valid bank accounting"):
                module.validate_node_accounting(nodes, point_count=2, bank_count=185)

    def test_sparse_pair_rows_reject_duplicate_and_unknown_bank(self):
        bank_keys = [(7, 101)]
        row = {"query_row": "0", "bank_body": "7", "bank_stable_id": "101"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.validate_sparse_pair_rows([row, dict(row)], point_count=2, bank_keys=bank_keys)
        with self.assertRaisesRegex(ValueError, "out-of-domain"):
            module.validate_sparse_pair_rows([dict(row, bank_stable_id="999")],
                                             point_count=1, bank_keys=bank_keys)

    def test_zero_feasibility_gate_accepts_exact_zero_and_rejects_positive(self):
        src = [{"bank_count": 185, "query_rows": 2,
                "complete_bank_accounting_verified": True,
                "complete_bank_accounting_rows": 2,
                "ambiguous_rows": 0, "invalid_rows": 0}]
        good = {"maximum_zero_step_violation_m": 0.0,
                "complete_bank_accounting_rows": 2,
                "complete_bank_accounting_verified": True,
                "query_sources": src}
        module.validate_zero_feasible_builder_result(good, point_count=2)
        for badvalue in (1e-15, float("nan"), -1e-15):
            bad = dict(good, maximum_zero_step_violation_m=badvalue)
            with self.subTest(value=badvalue), self.assertRaisesRegex(ValueError, "zero-step"):
                module.validate_zero_feasible_builder_result(bad, point_count=2)

    def test_existing_owner_source_receipts_are_complete_without_redundant_top_level_fields(self):
        src = [{"bank_count": 185, "query_rows": 2,
                "complete_bank_accounting_verified": True,
                "complete_bank_accounting_rows": 2,
                "ambiguous_rows": 0, "invalid_rows": 0,
                "aabb_excluded_rows": 1}]
        report = {"maximum_zero_step_violation_m": 0.0, "query_sources": src}
        module.validate_zero_feasible_builder_result(report, point_count=2)
        with self.assertRaisesRegex(ValueError, "complete all-bank accounting"):
            module.validate_zero_feasible_builder_result(
                dict(report, complete_bank_accounting_rows=1,
                     complete_bank_accounting_verified=True), point_count=2)

    def test_zero_feasibility_gate_rejects_incomplete_source_coverage(self):
        report = {"maximum_zero_step_violation_m": 0.0,
                  "complete_bank_accounting_rows": 2,
                  "complete_bank_accounting_verified": True,
                  "query_sources": [{"bank_count": 184, "query_rows": 2,
                                     "complete_bank_accounting_verified": True,
                                     "complete_bank_accounting_rows": 2,
                                     "ambiguous_rows": 0, "invalid_rows": 0}]}
        with self.assertRaisesRegex(ValueError, "incomplete bank coverage"):
            module.validate_zero_feasible_builder_result(report, point_count=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
