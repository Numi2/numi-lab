from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("select_bone_reducing_flips_370.py")
spec = importlib.util.spec_from_file_location("select_bone_reducing_flips_370", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StrictPairReductionTests(unittest.TestCase):
    def test_requires_strict_total_decrease_without_per_bank_regression(self):
        old = {(1, 10): 2, (2, 20): 1}
        self.assertEqual(
            module._strict_decrease(old, {(1, 10): 1, (2, 20): 1}),
            (True, "strict_exact_pair_decrease_all_banks_nonincreasing", 1),
        )
        self.assertEqual(
            module._strict_decrease(old, {(1, 10): 1, (2, 20): 2}),
            (False, "typed_bank_pair_count_increased", 0),
        )
        self.assertEqual(
            module._strict_decrease(old, {(1, 10): 1, (2, 20): 0}),
            (True, "strict_exact_pair_decrease_all_banks_nonincreasing", 2),
        )

    def test_equal_total_is_rejected_even_when_pairs_move_between_banks(self):
        old = {(1, 10): 2, (2, 20): 0}
        new = {(1, 10): 1, (2, 20): 1}
        self.assertEqual(
            module._strict_decrease(old, new),
            (False, "typed_bank_pair_count_increased", 0),
        )

    def test_old_new_domains_must_match(self):
        with self.assertRaises(module.SelectorError):
            module._strict_decrease({(1, 10): 1}, {(2, 20): 0})


class TypedBankAnd159CounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.surface_owner, _, cls.exact, _ = module.load_pinned_runtime()

    def test_exact_159_patch_counter_uses_transformed_typed_bank(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        # A non-orthonormal, nonsymmetric transform catches accidental use of
        # transpose-as-inverse or an extra Float32 cast. Pick local bank coords
        # by applying the same declared world->body transform to a world hit.
        matrix = np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])
        position = np.asarray([0.1, -0.2, 0.3])
        offset = np.asarray([0.01, 0.02, -0.03])
        world_hit = np.asarray([
            [0.2, 0.2, -0.5],
            [0.7, 0.2, 0.5],
            [0.2, 0.7, 0.5],
        ])
        local_hit = (world_hit - position) @ matrix.T - offset
        bank = module.TypedBank(
            7, 91, local_hit.reshape(1, 3, 3), np.asarray([1234], dtype=np.int64),
            matrix, position, offset,
        )
        counter = module.Exact159PatchCounter(
            points, [bank], self.exact, self.surface_owner.records
        )
        result = counter.count_faces([17], np.asarray([[0, 1, 2]], dtype=np.int64))
        self.assertEqual(set(result), {(7, 91)})
        self.assertEqual(result[(7, 91)]["count"], 1)
        self.assertEqual(result[(7, 91)]["triangle_pairs"], [(17, 1234)])

    def test_exact_159_counter_reports_zero_for_aabb_disjoint_bank(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        bank = module.TypedBank(
            7, 91,
            np.asarray([[[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]]]),
            np.asarray([5], dtype=np.int64),
            np.eye(3), np.zeros(3), np.zeros(3),
        )
        counter = module.Exact159PatchCounter(
            points, [bank], self.exact, self.surface_owner.records
        )
        result = counter.count_faces([0], np.asarray([[0, 1, 2]], dtype=np.int64))
        self.assertEqual(result[(7, 91)]["count"], 0)
        self.assertEqual(result[(7, 91)]["triangle_pairs"], [])

    def test_bank_inventory_rejects_missing_or_duplicate_keys(self):
        keys = np.asarray([[1, 10]] + [[i + 2, i + 30] for i in range(184)], dtype=np.int64)
        triangles = np.zeros((len(keys), 3, 3))
        ids = np.asarray([[int(body), int(bone), i, i + 1000]
                          for i, (body, bone) in enumerate(keys)], dtype=np.int64)
        body_ids = sorted(set(map(int, keys[:, 0])))
        matrices = {body: np.eye(3) for body in body_ids}
        offsets = {body: np.zeros(3) for body in body_ids}
        positions = {body: np.zeros(3) for body in body_ids}
        raw = (triangles, ids, None, None, matrices, offsets, positions, keys, {}, {})
        banks, keys = module.typed_banks_from_159_inputs(raw)
        self.assertEqual(len(banks), 185)
        self.assertEqual(set(keys), {bank.key for bank in banks})
        bad = list(raw)
        bad[7] = np.vstack([raw[7][:-1], raw[7][0]])
        with self.assertRaises(module.SelectorError):
            module.typed_banks_from_159_inputs(tuple(bad))

    def test_complete_baseline_report_rejects_silent_null_bank(self):
        bank = module.TypedBank(
            1, 10, np.asarray([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]),
            np.asarray([8]), np.eye(3), np.zeros(3), np.zeros(3)
        )
        banks = [bank]
        report = {
            "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT",
            "complete": True, "geometry_or_native_admission": False,
            "candidate_sha256": "pinned", "bank_count": 185,
            "expected_bank_count": 185,
            "banks": [],
        }
        for i in range(185):
            body, bone = (1, 10) if i == 0 else (i + 1, i + 100)
            if i == 0:
                report["banks"].append({
                    "body": body, "stable_bone_id": bone, "skin_faces": 1, "bank_faces": 1,
                    "skin_aabb_active": 0, "bank_aabb_active": 0, "result": None,
                })
            else:
                report["banks"].append({
                    "body": body, "stable_bone_id": bone, "skin_faces": 1, "bank_faces": 1,
                    "skin_aabb_active": 0, "bank_aabb_active": 0, "result": None,
                })
                banks.append(module.TypedBank(body, bone, bank.triangles_local_m, bank.stable_triangle_ids,
                                               np.eye(3), np.zeros(3), np.zeros(3)))
        keys = [item.key for item in banks]
        _pairs, targets, counts = module._pair_rows_and_counts(
            report, candidate_sha256="pinned", face_count=1, banks=banks,
            expected_bank_keys=keys,
        )
        self.assertEqual(targets, set())
        self.assertEqual(len(counts), 185)
        report["banks"][0]["skin_aabb_active"] = 1
        with self.assertRaises(module.SelectorError):
            module._pair_rows_and_counts(
                report, candidate_sha256="pinned", face_count=1, banks=banks,
                expected_bank_keys=keys,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
