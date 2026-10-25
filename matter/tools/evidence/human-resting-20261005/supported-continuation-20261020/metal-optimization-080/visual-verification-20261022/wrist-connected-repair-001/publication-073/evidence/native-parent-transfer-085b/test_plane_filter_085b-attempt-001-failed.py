import ast
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from plane_filter_085 import INSIDE_TOL, prepare_tetra
from plane_filter_085b import tetra_intersection_volume_stacked, filtered_tetra_intersection_volume_stacked

OWNER_073 = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/"
    "connected-skin-parent-mass-transfer-067/native-parent-transfer-073/"
    "build_native_parent_transfer_073.py"
)
OWNER_SHA = "167c176edef02cbb2360d68e3c011809722030bd00009e20afa80d2b8ad59586"


def exact_owner():
    assert hashlib.sha256(OWNER_073.read_bytes()).hexdigest() == OWNER_SHA
    tree = ast.parse(OWNER_073.read_text())
    wanted = {"halfspaces", "tetra_intersection_volume"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in defs} == wanted
    ns = {"np": np, "itertools": itertools, "ConvexHull": ConvexHull, "QhullError": QhullError}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(OWNER_073), "exec"), ns)
    return ns["tetra_intersection_volume"]


def tet(offset=(0., 0., 0.), height=1.):
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., height]]) + np.asarray(offset)


@pytest.mark.parametrize("offset,height", [
    ((2., 0., 0.), 1.), ((1., 0., 0.), 1.),
    ((1. + INSIDE_TOL * .25, 0., 0.), 1.),
    ((1. - 1e-7, 0., 0.), 1.), ((.2, .2, .2), 1.),
])
def test_stacked_matches_pinned_owner_cases(offset, height):
    clip = exact_owner()
    a, b = tet(height=height), tet(offset, height=height)
    assert tetra_intersection_volume_stacked(a, b) == clip(a, b)
    value, rejected = filtered_tetra_intersection_volume_stacked(a, b)
    assert value == clip(a, b)
    assert not rejected or clip(a, b) == 0.


@pytest.mark.parametrize("height", [0.9e-10, 1.0e-10, 1.1e-10, 2.0e-10, 1e-8, 1e-5])
def test_determinant_threshold_slivers_match_owner(height):
    clip = exact_owner()
    a = tet(height=height)
    for dz in (0., .25 * INSIDE_TOL, -0.25 * INSIDE_TOL):
        b = tet((.1, .1, dz), height=height)
        got = tetra_intersection_volume_stacked(a, b)
        ref = clip(a, b)
        assert np.isclose(got, ref, rtol=2e-13, atol=1e-28)


def test_stacked_filter_reuses_prepared_planes_and_matches_owner():
    clip = exact_owner()
    a, b = tet(), tet((2., 0., 0.))
    pa, pb = prepare_tetra(a), prepare_tetra(b)
    got, rejected = filtered_tetra_intersection_volume_stacked(
        a, b, prepared_a=pa, prepared_b=pb)
    assert rejected and got == 0. and got == clip(a, b)


def test_many_deterministic_orientations_match_owner():
    clip = exact_owner()
    rng = np.random.default_rng(85074)
    R = np.asarray([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    for _ in range(32):
        a = tet() + rng.normal(0., .03, (4, 3))
        b = (tet(rng.uniform(-.25, 2., 3)) @ R.T)
        assert tetra_intersection_volume_stacked(a, b) == clip(a, b)

