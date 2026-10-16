"""Portable CPU regressions for the compensated cardboard reference helpers.

The test loads the equation module adjacent to these tools and uses only its
sibling fixture. No external checkout path, absolute machine path, or native runtime is required.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cardboard_bilinear_fp64_reference.py"
FIXTURE = HERE / "cardboard_compensated_fp64_fixture.py"
EXPECTED_SOURCE_SHA256 = "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995"
EXPECTED_FIXTURE_SHA256 = "8d9d82cde0b9cb2158dbb7faf86ff47eb9b87258cb0e303d30aa2ae7be7467cd"

sys.path.insert(0, str(HERE))
import cardboard_compensated_fp64_base as helpers
import cardboard_compensated_fp64_reference as reference_v2
import cardboard_compensated_fp64_fixture as fixture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_equation(path: Path):
    spec = importlib.util.spec_from_file_location("cardboard_sibling_equation_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling equation source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompensatedReferencePortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file() or _sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("sibling equation source is absent or does not match this source173 test fixture")
        if not FIXTURE.is_file() or _sha256(FIXTURE) != EXPECTED_FIXTURE_SHA256:
            raise RuntimeError("sibling captured fixture is absent or changed")
        cls.parameters = fixture.PARAMETERS_F32.astype(np.float64)
        cls.old = fixture.ACCEPTED_HISTORY.astype(np.float64)
        cls.F = fixture.F_CENTER
        cls.root = fixture.MP_ROOT_STATE
        cls.branch_states = fixture.FB_BRANCH_STATES
        cls.base = _load_equation(SOURCE)
        cls.ref = reference_v2.Reference(cls.base)

    def test_mixed_fb_scatter_in_both_retained_implementations(self):
        p = np.repeat(self.parameters, 2, axis=0)
        old = np.repeat(self.old, 2, axis=0)
        z = self.branch_states.copy()
        values_v1, _, meta_v1 = helpers._fb_row(p, old, z)
        values_v2, _, meta_v2 = reference_v2.fb_row(p, old, z)
        self.assertEqual(meta_v1["fb_quotient_branch"].tolist(), [True, False])
        self.assertEqual(meta_v2["fb_quotient_branch"].tolist(), [True, False])
        np.testing.assert_array_equal(values_v1, values_v2)
        self.assertTrue(np.isfinite(values_v1).all())

    def test_legacy_residual_and_jacobian_entrypoints_retain_mixed_rows(self):
        p = np.repeat(self.parameters, 2, axis=0)
        old = np.repeat(self.old, 2, axis=0)
        F = np.repeat(self.F, 2, axis=0)
        z = self.branch_states.copy()
        residual = helpers.residual_dd(self.base, p, F, old, z)
        jacobian = helpers.jacobian_dd(self.base, p, F, old, z)
        expected_fb, _, _ = reference_v2.fb_row(p, old, z)
        expected_jacobian = reference_v2.jacobian(self.base, p, F, old, z)
        np.testing.assert_array_equal(residual[:, 12], expected_fb)
        np.testing.assert_array_equal(jacobian[:, 12, :], expected_jacobian[:, 12, :])
        self.assertTrue(np.isfinite(residual).all())
        self.assertTrue(np.isfinite(jacobian).all())

    def test_legacy_reference_project_path(self):
        legacy = helpers.Reference(self.base)
        p, F, old = self.parameters, self.F, self.old
        state, iterations, scaled_residual = helpers.project(legacy, p, F, old)
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(scaled_residual).all())
        self.assertLessEqual(float(np.max(scaled_residual)), 2.0e-10)
        self.assertGreaterEqual(int(iterations[0]), 0)

    def test_v2_projection_and_tangent_paths(self):
        state, iterations, scaled_residual = self.ref.project(self.parameters, self.F, self.old)
        self.assertTrue(np.isfinite(state).all())
        self.assertLessEqual(float(np.max(scaled_residual)), 2.0e-10)
        tangent = reference_v2.tangent(
            self.base, self.parameters, self.F, self.old, self.root,
            np.eye(3, dtype=np.float64)[None, :, :])
        self.assertEqual(tangent.shape, (1, 3, 3))
        self.assertTrue(np.isfinite(tangent).all())
        self.assertGreaterEqual(int(iterations[0]), 0)

    def test_explicit_loader_rejects_missing_source(self):
        with self.assertRaises(ValueError):
            helpers.import_reference(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
