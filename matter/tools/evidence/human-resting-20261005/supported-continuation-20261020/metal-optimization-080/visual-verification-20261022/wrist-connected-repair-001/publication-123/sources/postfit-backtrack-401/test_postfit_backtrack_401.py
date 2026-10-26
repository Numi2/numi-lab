import importlib.util
from pathlib import Path
import numpy as np
import pytest

OWNER = Path(__file__).with_name("postfit_backtrack_401.py")
spec = importlib.util.spec_from_file_location("postfit_backtrack_401_tested", OWNER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_actual_402_union_bundle_is_complete_and_pinned():
    state = mod.verify_baseline()
    receipt = mod.verify_seed_bundle(state)
    assert receipt["old_pair_count"] == 16527
    assert receipt["source_pair_counts"] == [888, 361, 164, 81, 51, 31, 14]
    assert receipt["merged_pair_count"] == 17643
    assert receipt["new_unique_pair_count"] == 1116
    assert receipt["rebuilt_026_row_count"] == 99538
    assert receipt["rebuilt_026_max_rhs_mm"] <= 2e-8
    assert all(x["complete_surface_audit"] for x in receipt["source_proofs"])

def test_pair_union_accepts_exact_old_plus_all_sources():
    old = np.array([[0, 1], [2, 4]], dtype=np.int64)
    sources = [np.array([[1, 3], [2, 4]], dtype=np.int64), np.array([[4, 5]], dtype=np.int64)]
    merged = np.array([[0, 1], [1, 3], [2, 4], [4, 5]], dtype=np.int64)
    result = mod.validate_pair_union(merged, old, sources)
    assert result["new_unique_pair_count"] == 2
    assert result["merged_pair_count"] == 4

def test_pair_union_rejects_missing_source_witness():
    old = np.array([[0, 1]], dtype=np.int64)
    src = [np.array([[2, 3]], dtype=np.int64)]
    with pytest.raises(ValueError, match="omit pass witness"):
        mod.validate_pair_union(old, old, src)

def test_pair_union_rejects_missing_baseline_seed():
    old = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="omit baseline-retained"):
        mod.validate_pair_union(np.array([[2, 3]], dtype=np.int64), old, [])

def test_pair_union_rejects_duplicate_or_noncanonical_pairs():
    old = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        mod.validate_pair_union(np.array([[0, 1], [0, 1]], dtype=np.int64), old, [])
    with pytest.raises(ValueError, match="noncanonical"):
        mod.validate_pair_union(np.array([[1, 0]], dtype=np.int64), np.empty((0, 2), dtype=np.int64), [])
