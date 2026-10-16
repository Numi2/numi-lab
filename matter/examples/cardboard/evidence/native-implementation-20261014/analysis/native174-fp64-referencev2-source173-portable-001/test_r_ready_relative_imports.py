from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE / "r-ready" / "matter" / "tools"
SOURCE173 = pathlib.Path("/Users/home/cardboard-implementation-20261014/snapshots/source-checkpoint-warm-breakdown-173/matter/tools/cardboard_bilinear_fp64_reference.py")
SOURCE173_SHA256 = "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995"
sys.path.insert(0, str(TOOLS))
import cardboard_compensated_fp64_base as base_loader
import cardboard_compensated_fp64_reference as v2


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RelativeToolLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sha(SOURCE173) != SOURCE173_SHA256:
            raise RuntimeError("source173 equation base changed")
        x = np.load(HERE / "fixture_tet750.npz")
        cls.p = x["parameters_f32"].astype(np.float64)
        cls.old = x["accepted_history_f32"].astype(np.float64)
        cls.F = x["F_center_fp64"]
        cls.states = x["fb_branch_states_fp64"]
        cls.base = base_loader.import_reference(SOURCE173)
        cls.ref = v2.Reference(cls.base)

    def test_equation_base_is_explicit_and_modules_are_sibling_imports(self):
        self.assertEqual(sha(SOURCE173), SOURCE173_SHA256)
        for path in TOOLS.glob("cardboard_*.py"):
            source = path.read_text()
            self.assertNotIn("/Users/home/cardboard-implementation-20261014/analysis", source)
        with self.assertRaises(ValueError):
            base_loader.import_reference(None)

    def test_mixed_fb_representation_branches_and_finite_outputs(self):
        p = np.repeat(self.p, 2, axis=0)
        old = np.repeat(self.old, 2, axis=0)
        values, _, meta = v2.fb_row(p, old, self.states)
        self.assertEqual(meta["fb_quotient_branch"].tolist(), [True, False])
        self.assertTrue(np.isfinite(values).all())

    def test_tangent_is_finite_for_captured_state(self):
        z = self.states[:1]
        h = np.zeros_like(self.F)
        h[:, 0, 0] = 1.0
        result = v2.tangent(self.base, self.p, self.F, self.old, z, h)
        self.assertEqual(result.shape, (1, 3, 3))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
