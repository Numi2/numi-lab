#!/usr/bin/env python3
"""Source-bound checks for async coupled-capture manifest ownership."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "source/matter/src/runtime.mm"
HEADER = ROOT / "source/matter/src/coupled_capture_manifest_context.hpp"


def completion_block(source: str) -> str:
    snapshot = source.index("const detail::CoupledCaptureManifestContext capturedManifestContext{")
    start = source.index("[commandBuffer addCompletedHandler:", snapshot)
    end = source.index("                }];", start) + len("                }];")
    return source[start:end]


class CallbackCaptureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNTIME.read_text()
        cls.block = completion_block(cls.source)

    def test_completion_block_uses_owned_manifest_snapshot(self):
        import re
        for pattern, label in (
            (r"\bstate\.", "state member read"),
            (r"\brequest\.", "request member read"),
            (r"\bfemCoupledCaptureRHSReady\b", "RHS flag read"),
            (r"\bfemCoupledCaptureCapBytes\b", "capture cap read"),
            (r"\bomittedOptional\b", "optional-list read"),
            (r"\bsnapshot\.contents", "snapshot object read"),
        ):
            self.assertIsNone(re.search(pattern, self.block), f"callback retains outer {label}")
        self.assertIn("capturedManifestContext.rhsPreservedBeforeFGMRES", self.block)
        self.assertIn("capturedManifestContext.cachedOperatorUsed", self.block)
        self.assertIn("capturedManifestContext.maxPayloadBytes", self.block)
        self.assertIn("capturedManifestContext.omittedOptionalBuffers", self.block)
        self.assertIn("capturedSnapshot.contents", self.block)

    def test_snapshot_is_built_before_async_block_without_normalization(self):
        snapshot = self.source.index("const detail::CoupledCaptureManifestContext capturedManifestContext{")
        block = self.source.index("[commandBuffer addCompletedHandler:", snapshot)
        self.assertLess(snapshot, block)
        prefix = self.source[snapshot:block]
        self.assertIn("femCoupledCaptureRHSReady,", prefix)
        self.assertIn("state.mixedSolverValue.executionBudgets.w == 1u,", prefix)
        self.assertIn("femCoupledCaptureCapBytes,", prefix)
        self.assertIn("omittedOptional", prefix)
        self.assertIn("!femCoupledCaptureRHSReady", self.source)
        self.assertIn("femCoupledCaptureRHSReady = true;", self.source)

    def test_context_owns_vector_and_scalar_values(self):
        header = HEADER.read_text()
        self.assertIn("bool rhsPreservedBeforeFGMRES", header)
        self.assertIn("bool cachedOperatorUsed", header)
        self.assertIn("std::uint64_t maxPayloadBytes", header)
        self.assertIn("std::vector<std::string> omittedOptionalBuffers", header)


if __name__ == "__main__":
    unittest.main(verbosity=2)
