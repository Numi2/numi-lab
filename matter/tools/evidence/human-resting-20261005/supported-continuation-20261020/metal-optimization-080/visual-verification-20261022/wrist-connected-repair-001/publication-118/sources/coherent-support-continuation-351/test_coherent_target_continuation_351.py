from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/coherent-support-continuation-351/run_coherent_target_continuation_351.py")
REAL_TRIAL = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/coherent-support-continuation-348/round-001/attempt-001/trial-halfstep.npz")
spec = importlib.util.spec_from_file_location("continuation_351_tested", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_actual_348_trial_uses_written_346_support_key():
    assert REAL_TRIAL.is_file()
    with np.load(REAL_TRIAL, allow_pickle=False) as z:
        assert "continuation_346_fit_support_rows" in z.files
        assert "continuation_333_fit_support_rows" not in z.files
        expected = np.asarray(z["continuation_346_fit_support_rows"])
    rows = mod.trial_support_rows(REAL_TRIAL)
    assert rows.dtype == np.int64
    assert rows.shape == (6418,)
    assert np.array_equal(rows, expected)


def test_legacy_333_key_remains_supported(tmp_path):
    p = tmp_path / "legacy.npz"
    expected = np.array([2, 7, 9], dtype=np.int32)
    np.savez(p, continuation_333_fit_support_rows=expected)
    assert np.array_equal(mod.trial_support_rows(p), expected.astype(np.int64))


def test_current_346_key_is_supported(tmp_path):
    p = tmp_path / "current.npz"
    expected = np.array([1, 4, 8], dtype=np.int64)
    np.savez(p, continuation_346_fit_support_rows=expected)
    assert np.array_equal(mod.trial_support_rows(p), expected)


def test_equal_legacy_and_current_aliases_are_accepted(tmp_path):
    p = tmp_path / "both-equal.npz"
    expected = np.array([0, 3, 6], dtype=np.int64)
    np.savez(p, continuation_333_fit_support_rows=expected,
             continuation_346_fit_support_rows=expected.copy())
    assert np.array_equal(mod.trial_support_rows(p), expected)


def test_conflicting_aliases_fail_closed(tmp_path):
    p = tmp_path / "both-conflict.npz"
    np.savez(p, continuation_333_fit_support_rows=np.array([0, 3]),
             continuation_346_fit_support_rows=np.array([0, 4]))
    with pytest.raises(ValueError, match="disagree"):
        mod.trial_support_rows(p)


@pytest.mark.parametrize("payload", [
    {},
    {"continuation_346_fit_support_rows": np.array([[1, 2]], dtype=np.int64)},
    {"continuation_346_fit_support_rows": np.array([1.0, 2.0])},
    {"continuation_333_fit_support_rows": np.array([1, 1], dtype=np.int64)},
])
def test_missing_or_malformed_support_fails_closed(tmp_path, payload):
    p = tmp_path / "bad.npz"
    np.savez(p, **payload)
    with pytest.raises(ValueError):
        mod.trial_support_rows(p)
