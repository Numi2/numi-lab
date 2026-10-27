import importlib.util
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).with_name("connected_witness_damping588.py")
spec = importlib.util.spec_from_file_location("connected_witness_damping588_test", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_disconnected_patches_damp_only_the_witness_component():
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    mutable = np.ones(6, dtype=bool)
    labels, components = mod.mutable_components588(faces, 6, mutable)
    assert [len(c) for c in components] == [3, 3]
    before = np.ones(6, dtype=np.float64)
    after, changed, selected = mod.damp_connected588(
        before, np.asarray([1], dtype=np.int64), mutable, faces,
        component_labels=labels, components=components)
    assert changed.tolist() == [0, 1, 2]
    assert selected.tolist() == [0]
    assert np.array_equal(after[:3], np.full(3, 0.5))
    assert np.array_equal(after[3:], np.ones(3))


def test_connected_component_gets_uniform_xyz_scale_and_immutable_anchor_stays_fixed():
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    mutable = np.asarray([True, True, True, False])
    labels, components = mod.mutable_components588(faces, 4, mutable)
    assert [c.tolist() for c in components] == [[0, 1, 2]]
    alpha = np.ones(4, dtype=np.float64)
    delta = np.asarray([[0.1, -0.3, 0.4], [0.2, 0.7, -0.5], [-0.8, 0.1, 0.9], [0, 0, 0]])
    after, changed, selected = mod.damp_connected588(
        alpha, np.asarray([0, 3], dtype=np.int64), mutable, faces,
        component_labels=labels, components=components)
    assert changed.tolist() == [0, 1, 2]
    assert selected.tolist() == [0]
    assert np.array_equal(after[:3], np.full(3, 0.5))
    assert after[3] == alpha[3] == 1.0
    moved = after[:, None] * delta
    assert np.allclose(moved[:3], 0.5 * delta[:3], atol=0, rtol=0)
    assert np.allclose(moved[0] - moved[1], 0.5 * (delta[0] - delta[1]), atol=1e-15, rtol=0)
    assert np.array_equal(moved[3], np.zeros(3))


def test_zero_boundary_targets_are_required_and_checked_bitwise():
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    base = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    target = base.copy()
    boundary = mod.validate_zero_boundary588(base, target, faces)
    assert boundary.tolist() == [0, 1, 2, 3]
    target[2, 1] = np.float32(1e-6)
    try:
        mod.validate_zero_boundary588(base, target, faces)
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero boundary target must fail closed")


def test_bad_witness_domain_and_immutable_only_witness_reject():
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mutable = np.asarray([True, True, False])
    labels, components = mod.mutable_components588(faces, 3, mutable)
    alpha = np.ones(3, dtype=np.float64)
    for bad in (np.asarray([3], dtype=np.int64), np.asarray([2], dtype=np.int64)):
        try:
            mod.damp_connected588(alpha, bad, mutable, faces,
                                  component_labels=labels, components=components)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or immutable-only witness must reject")


def test_install_hook_uses_component_expansion_and_records_it():
    class Core:
        pass
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    mutable = np.ones(6, dtype=bool)
    core = Core()
    receipt = mod.install_damper588(core, faces, mutable)
    alpha, changed = core.damp(np.ones(6), np.asarray([4]), mutable)
    assert changed.tolist() == [3, 4, 5]
    assert len(receipt["events"]) == 1
    assert receipt["events"][0]["damped_component_ids"] == [1]
    assert receipt["events"][0]["component_alpha_before"] == [1.0]
    assert receipt["events"][0]["component_alpha_after"] == [0.5]
