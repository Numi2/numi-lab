from pathlib import Path
import importlib.util
import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("prepare_swept_backtrack_seeds558.py")
spec = importlib.util.spec_from_file_location("prep558", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class ValidOwner:
    @staticmethod
    def build_pair_rows(points_mm, faces, pair_ids):
        rows = [{"lower": -0.1} for _ in pair_ids]
        records = [{"pair": tuple(p)} for p in pair_ids]
        return rows, records


class PositiveLowerOwner:
    @staticmethod
    def build_pair_rows(points_mm, faces, pair_ids):
        return [{"lower": 0.01} for _ in pair_ids], [{"pair": tuple(p)} for p in pair_ids]


def test_union_canonicalizes_deduplicates_and_keeps_all_witnesses():
    merged, proof = prep.union_seed_pairs(
        np.array([[2, 1], [4, 3]], dtype=np.int64),
        np.array([[1, 2], [6, 5]], dtype=np.int64), 8)
    assert merged.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert proof["all_exact_witnesses_retained"]
    assert proof["new_exact_witness_count"] == 1


def test_bad_pair_index_rejected():
    with pytest.raises(ValueError, match="invalid face rows"):
        prep.union_seed_pairs(np.array([[0, 9]], dtype=np.int64), np.empty((0, 2), dtype=np.int64), 4)


def test_capped_sweep_is_retained_as_incomplete_and_pins_required():
    receipt = {
        "status": "BLOCKED_CLASSIFICATION_INCOMPLETE",
        "input_pins": {"base_candidate": "b", "raw_trial_0": "t", "seed_pairs_0": "s", "witness_pairs_0": "w"},
        "pair_row_eligibility": {"classification_stop_reason": "candidate_pair_cap",
                                 "unclassified_pair_count": 12, "eligible_pair_count": 0,
                                 "rejected_pair_count": 0},
        "potential_pair_array_shape": [12, 2], "swept_aabb_pair_count": 11,
    }
    result = prep.validate_incomplete_sweep(
        receipt, expected_base_sha="b", expected_trial_sha="t",
        expected_seed_sha="s", expected_witness_sha="w")
    assert result["swept_coverage_complete"] is False
    assert result["potential_pair_count"] == 12
    with pytest.raises(ValueError, match="input pin mismatch"):
        prep.validate_incomplete_sweep(
            receipt, expected_base_sha="stale", expected_trial_sha="t",
            expected_seed_sha="s", expected_witness_sha="w")


def test_only_requested_witness_pairs_are_certified_and_bounded():
    faces = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64)
    points = np.zeros((7, 3), dtype=np.float64)
    pairs = np.array([[0, 2], [1, 2]], dtype=np.int64)
    result = prep.certify_new_pairs_zero_step(
        ValidOwner(), points, faces, pairs, max_pairs=3, batch_size=1,
        timeout_seconds=2.0)
    assert result["pair_count"] == 2
    assert result["owner_calls"] == 2
    assert result["swept_pair_coverage"] is False
    with pytest.raises(ValueError, match="pair cap exceeded"):
        prep.certify_new_pairs_zero_step(ValidOwner(), points, faces, pairs,
                                         max_pairs=1, timeout_seconds=2.0)


def test_positive_zero_step_lower_is_rejected():
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    points = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="failed 026 zero-step eligibility"):
        prep.certify_new_pairs_zero_step(
            PositiveLowerOwner(), points, faces, np.array([[0, 1]], dtype=np.int64),
            timeout_seconds=2.0)



def test_cap_blocked_sweep_pair_arrays_must_cover_old_and_exact_sets():
    available = np.array([[0, 2], [1, 3], [4, 5]], dtype=np.int64)
    assert prep.require_pairs_present(np.array([[2, 0], [3, 1]], dtype=np.int64), available, 6) == 2
    with pytest.raises(ValueError, match="missing"):
        prep.require_pairs_present(np.array([[0, 4]], dtype=np.int64), available, 6)
