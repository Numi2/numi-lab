from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/no-progress-backtracker-306/run_local_witness_backtracking_306.py")


def load_no_progress_comparator():
    tree = ast.parse(SOURCE.read_text())
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "emitted_f32_bitwise_equal")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["emitted_f32_bitwise_equal"]


def test_guard_uses_exact_float32_bytes_including_signed_zero():
    equal = load_no_progress_comparator()
    start = np.array([[-0.0, 0.0, 1.0]], dtype=np.float32)
    same = start.copy()
    numerically_equal_different_bits = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    assert np.array_equal(start, numerically_equal_different_bits)
    assert equal(same, start) is True
    assert equal(numerically_equal_different_bits, start) is False


def test_one_float32_ulp_change_is_not_classified_as_no_progress():
    equal = load_no_progress_comparator()
    start = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    changed = start.copy()
    changed[0, 0] = np.nextafter(start[0, 0], np.float32(np.inf))
    assert changed[0, 0] != start[0, 0]
    assert equal(changed, start) is False


@pytest.mark.parametrize(
    "proposal,start",
    [
        (np.zeros((1, 3), dtype=np.float64), np.zeros((1, 3), dtype=np.float32)),
        (np.zeros((2, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)),
    ],
    ids=["wrong-dtype", "wrong-shape"],
)
def test_no_progress_comparator_rejects_invalid_domains(proposal, start):
    equal = load_no_progress_comparator()
    with pytest.raises(ValueError):
        equal(proposal, start)


def test_actual_guard_precedes_candidate_write_and_all_exact_audits():
    source = SOURCE.read_text()
    guard = source.index("if emitted_f32_bitwise_equal(xp,np.asarray(z['points_world_m'],dtype=np.float32)):")
    candidate_write = source.index("p=out/'candidate.npz'", guard)
    point_audit = source.index("run(D/'audit_connected_outer_geometry_130.py'", guard)
    surface_audit = source.index("run(D/'audit_skin_bone_surfaces_159.py'", guard)
    assert guard < candidate_write < point_audit < surface_audit
    guard_block = source[guard:candidate_write]
    assert "UNCHANGED_FLOAT32_CANDIDATE_NO_PROGRESS" in guard_block
    assert "accepted_physical_steps=0" in guard_block
    assert "write(out/'result.json',rec)" in guard_block
    assert "write(OUT/'result.json',rec)" in guard_block
    assert "break" in guard_block
