import ast
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from plane_filter_085 import (
    INSIDE_TOL, filtered_tetra_intersection_volume, separated_by_owner_planes,
    tetra_intersection_volume,
)

OWNER_073 = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/"
    "connected-skin-parent-mass-transfer-067/native-parent-transfer-073/"
    "build_native_parent_transfer_073.py"
)
OWNER_SHA = "167c176edef02cbb2360d68e3c011809722030bd00009e20afa80d2b8ad59586"


def owner_clip():
    assert hashlib.sha256(OWNER_073.read_bytes()).hexdigest() == OWNER_SHA
    tree = ast.parse(OWNER_073.read_text())
    wanted = {"halfspaces", "tetra_intersection_volume"}
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in wanted]
    assert {n.name for n in defs} == wanted
    ns = {"np": np, "itertools": itertools,
          "ConvexHull": ConvexHull, "QhullError": QhullError}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(OWNER_073), "exec"), ns)
    return ns["tetra_intersection_volume"]


@pytest.fixture(scope="module")
def exact_073():
    return owner_clip()


def unit_tet(offset=(0.0, 0.0, 0.0)):
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) + np.asarray(offset)


@pytest.mark.parametrize("offset,expected_reject", [
    ((2.0, 0.0, 0.0), True),
    ((1.0, 0.0, 0.0), False),                 # exact point contact
    ((1.0 + INSIDE_TOL * 0.25, 0.0, 0.0), False),  # within owner tolerance
    ((1.0 - 1e-7, 0.0, 0.0), False),         # thin positive overlap
    ((0.2, 0.2, 0.2), False),                # proper overlap
])
def test_filtered_matches_frozen_owner(offset, expected_reject, exact_073):
    a, b = unit_tet(), unit_tet(offset)
    value, rejected = filtered_tetra_intersection_volume(a, b)
    reference = exact_073(a, b)
    assert rejected is expected_reject
    assert value == reference
    if rejected:
        assert reference == 0.0


def test_nearly_coplanar_rotated_and_sliver_cases_match_owner(exact_073):
    a = unit_tet()
    # Both are non-axis-aligned but valid tetrahedra; the final pair is a thin sliver.
    rot = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cases = [
        (a, (a @ rot.T) + [1.0, 0.0, 0.0]),
        (a, (a @ rot.T) + [0.0, 1.0 + 0.5 * INSIDE_TOL, 0.0]),
        (a, (a @ rot.T) + [0.0, 1.0 - 1e-8, 0.0]),
    ]
    for x, y in cases:
        v, rejected = filtered_tetra_intersection_volume(x, y)
        ref = exact_073(x, y)
        assert v == ref
        assert not rejected or ref == 0.0


def test_randomized_rejections_are_owner_zero_and_survivors_exact(exact_073):
    rng = np.random.default_rng(85073)
    base = unit_tet()
    for _ in range(40):
        shift = rng.uniform(-0.25, 2.0, size=3)
        a = base + rng.normal(0.0, 0.03, size=(4, 3))
        b = (base + shift) @ np.asarray(
            [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
        ).T
        v, rejected = filtered_tetra_intersection_volume(a, b)
        ref = exact_073(a, b)
        assert v == ref
        assert not rejected or ref == 0.0


def test_invalid_tolerance_fails_closed():
    a = unit_tet()
    with pytest.raises(ValueError):
        separated_by_owner_planes(a, a, float("nan"))
    with pytest.raises(ValueError):
        separated_by_owner_planes(a, a, -1.0)



def _prepared_arrays(rows):
    from plane_filter_085 import prepare_tetra
    prepared = [prepare_tetra(v) for v in rows]
    normals = np.zeros((len(rows), 4, 3), dtype=np.float64)
    offsets = np.zeros((len(rows), 4), dtype=np.float64)
    expanded = np.zeros((len(rows), 4, 3), dtype=np.float64)
    valid = np.zeros(len(rows), dtype=bool)
    for i, item in enumerate(prepared):
        if item is not None:
            normals[i], offsets[i], expanded[i] = item
            valid[i] = True
    return normals, offsets, expanded, valid


def test_vectorized_batch_matches_frozen_owner(exact_073):
    from plane_filter_085 import separated_pairs_prepared
    base = unit_tet()
    a = np.asarray([base, base, base, base, base])
    offsets = [(2., 0., 0.), (1., 0., 0.), (1. - 1e-7, 0., 0.),
               (.2, .2, .2), (1. + INSIDE_TOL * .25, 0., 0.)]
    b = np.asarray([unit_tet(x) for x in offsets])
    pa = _prepared_arrays(a); pb = _prepared_arrays(b)
    reject = separated_pairs_prepared(*pa, *pb, np.arange(5), np.arange(5))
    for i in range(5):
        ref = exact_073(a[i], b[i])
        assert not reject[i] or ref == 0.0
        filtered = 0.0 if reject[i] else tetra_intersection_volume(a[i], b[i])
        assert filtered == ref


def test_vectorized_batch_empty_and_invalid_rows():
    from plane_filter_085 import separated_pairs_prepared
    empty3 = np.zeros((0, 4, 3)); empty4 = np.zeros((0, 4)); emptyb = np.zeros(0, dtype=bool)
    assert separated_pairs_prepared(empty3, empty4, empty3, emptyb,
                                    empty3, empty4, empty3, emptyb,
                                    np.asarray([], dtype=np.int64),
                                    np.asarray([], dtype=np.int64)).size == 0
    a = unit_tet()[None, ...]
    bad = np.zeros((1, 4, 3))
    pa = _prepared_arrays(a)
    pb = (np.zeros((1, 4, 3)), np.zeros((1, 4)), np.zeros((1, 4, 3)), np.asarray([False]))
    out = separated_pairs_prepared(*pa, *pb, np.asarray([0]), np.asarray([0]))
    assert out.tolist() == [False]

