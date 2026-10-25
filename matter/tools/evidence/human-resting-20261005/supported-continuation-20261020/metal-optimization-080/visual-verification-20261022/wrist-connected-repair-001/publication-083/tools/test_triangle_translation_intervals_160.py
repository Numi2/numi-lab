import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sat160", ROOT / "triangle_translation_intervals_160.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def static_sat(a, b):
    ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], -ea[2]), np.cross(eb[0], -eb[2])
    axes = [na, nb] + [np.cross(x, y) for x in ea for y in eb]
    axes += [np.cross(na, x) for x in ea] + [np.cross(nb, x) for x in eb]
    for axis in axes:
        n = np.linalg.norm(axis)
        if n == 0:
            continue
        p, q = a @ (axis / n), b @ (axis / n)
        if p.max() < q.min() or q.max() < p.min():
            return False
    return True


def test_coplanar_separated_and_hit_interval():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    b = np.array([[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
    assert MOD.triangle_translation_interval(a, b, [1, 0, 0], .9) is None
    hit = MOD.triangle_translation_interval(a, b, [1, 0, 0], 3.1)
    assert hit is not None
    assert abs(hit["entry_distance_m"] - 1.) < 1e-12
    assert abs(hit["exit_distance_m"] - 3.) < 1e-12


def test_zero_translation_intersection_and_separation():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    hit = MOD.triangle_translation_interval(a, a, [1, 0, 0], 0.)
    assert hit is not None and hit["entry_distance_m"] == hit["exit_distance_m"] == 0.
    assert MOD.triangle_translation_interval(a, a + [0, 0, 1], [1, 0, 0], 0.) is None


def test_nearly_parallel_interval_and_sampled_intersections():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    b = np.array([[.2, .2, .5], [.8, .2, .5 + 1e-8], [.2, .8, .5]])
    hit = MOD.triangle_translation_interval(a, b, [0, 0, 1], 1.)
    assert hit is not None and .49 < hit["entry_distance_m"] < .51
    assert .49 < hit["exit_distance_m"] < .51
    assert not static_sat(a, b)
    for s in np.linspace(hit["entry_distance_m"], hit["exit_distance_m"], 9):
        assert static_sat(a + [0, 0, s], b)


def test_union_returns_first_current_component_not_later_hit():
    merged = MOD.union_triangle_translation_intervals([
        {"pair_id": "a", "interval": {"entry_distance_m": 0., "exit_distance_m": .2}},
        {"pair_id": "b", "interval": {"entry_distance_m": .15, "exit_distance_m": .4}},
        {"pair_id": "later", "interval": {"entry_distance_m": .7, "exit_distance_m": .9}},
    ])
    assert len(merged) == 2
    first = MOD.first_current_overlap_exit(merged)
    assert first["candidate_first_exit_distance_m"] == .4
    assert first["pair_ids"] == ["a", "b"]


def test_invalid_inputs_fail_closed():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    for direction, cap in (([0, 0, 0], 1.), ([1, 0, 0], -1.)):
        try:
            MOD.triangle_translation_interval(a, a, direction, cap)
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")



def test_direction_input_is_not_mutated():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    direction = np.array([2., 0., 0.], dtype=np.float64)
    before = direction.copy()
    MOD.triangle_translation_interval(a, a + [2., 0., 0.], direction, 3.)
    assert np.array_equal(direction, before)
