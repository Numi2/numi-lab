from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-primary-policy-143")
MANIFEST = json.loads((ROOT / "overlay-manifest.json").read_text())
CANDIDATE = ROOT / MANIFEST["candidate_file"]
SOURCE = BASE / "matter/src/runtime.mm"
METAL = BASE / "matter/src/metal/fem.metalinc"
COMMON = BASE / "matter/src/metal/common.metalinc"


class SourceContractTests(unittest.TestCase):
    def test_base_and_candidate_bindings(self) -> None:
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), MANIFEST["base_sha256"])
        self.assertEqual(hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(), MANIFEST["candidate_sha256"])
        self.assertEqual(hashlib.sha256((ROOT / MANIFEST["patch_file"]).read_bytes()).hexdigest(),
                         MANIFEST["patch_sha256"])
        self.assertFalse(MANIFEST["gpu_or_build_run"])

    def test_capture_precedes_both_direction_construction_paths(self) -> None:
        text = CANDIDATE.read_text()
        self.assertEqual(text.count("const auto encodeBaseCGLSStateCapture = [&]() -> bool"), 1)
        self.assertEqual(text.count("if (!encodeBaseCGLSStateCapture())"), 2)
        primary = text.index("if (baseCGLSPrimaryPolicySelectedNewton)", text.index("micro.solverIteration = nonlinearIteration"))
        primary_call = text.index("if (!encodeBaseCGLSStateCapture())", primary)
        primary_direction = text.index("encodeBaseCauchyDirection();", primary)
        self.assertLess(primary_call, primary_direction)
        fallback = text.index("if (directionPass == 1u)", primary_direction)
        fallback_call = text.index("if (!encodeBaseCGLSStateCapture())", fallback)
        fallback_direction = text.index("encodeBaseCauchyDirection();", fallback)
        self.assertLess(fallback_call, fallback_direction)

    def test_reader_captures_raw_buffers_without_claiming_projection(self) -> None:
        text = CANDIDATE.read_text()
        for name in (
            "accepted_nodes", "candidate_nodes", "accepted_tetrahedra",
            "candidate_tetrahedra", "accepted_material_history",
            "candidate_material_state_buffer", "original_newton_correction",
            "base_residual_rhs", "algorithmic_tangent_cache",
            "algorithmic_tangent_cache_valid", "fgmres_state", "mixed_solver",
            "production_statuses", "fem_line_search", "environment_line_search",
        ):
            self.assertIn(f'"{name}"', text)
        self.assertIn("current_newton_status", text)
        self.assertIn("unavailable_not_persisted", text)
        self.assertIn("candidate_buffer_is_current_projection", text)
        self.assertIn("nm_fem_build_algorithmic_tangent_cache projects state in local temporaries", text)

    def test_opt_in_is_strict_and_disabled_by_default(self) -> None:
        text = CANDIDATE.read_text()
        self.assertIn('std::getenv(\n                "NM_FEM_BASE_CGLS_STATE_CAPTURE")', text)
        self.assertIn('std::strcmp(cglsStateCaptureText, "0") != 0 &&\n                    std::strcmp(cglsStateCaptureText, "1") != 0', text)
        self.assertIn("!femBaseCGLSStateCaptureRequested", text)
        self.assertIn("selected CGLS capture path, Newton selector, and admitted scratch", text)

    def test_copy_is_read_only_blit_and_manifest_is_exclusive_complete_marker(self) -> None:
        text = CANDIDATE.read_text()
        body = text[text.index("const auto encodeBaseCGLSStateCapture = [&]() -> bool"):
                    text.index("const auto encodeBaseCauchyDirectionTrace = [&](\n")]
        self.assertIn("blitCommandEncoder", body)
        self.assertIn("copyFromBuffer:", body)
        self.assertNotIn('dispatchThreads("nm_', body)
        self.assertNotIn("setBuffer:", body)
        writer = text[text.index("void writeBaseCGLSStateCapture("):
                      text.index("class UploadPlan {")]
        self.assertIn('"complete\\":true', writer)
        self.assertGreaterEqual(writer.count("O_EXCL"), 2)
        self.assertGreaterEqual(writer.count("fsync("), 2)

    def test_source_material_projection_lineage(self) -> None:
        metal = METAL.read_text()
        tangent_start = metal.index("kernel void nm_fem_build_algorithmic_tangent_cache(")
        tangent_end = metal.index("kernel void nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic(", tangent_start)
        tangent = metal[tangent_start:tangent_end]
        self.assertIn("device const float* acceptedMaterialState", tangent)
        self.assertIn("tangentCache", tangent)
        self.assertNotIn("candidateMaterialState", tangent)
        common = COMMON.read_text()
        tangent_helper_start = common.index("inline void evaluateAlgorithmicStressTangent(")
        next_helper = re.search(r"^(?:inline|kernel)\s+", common[tangent_helper_start + 20:], re.M)
        self.assertIsNotNone(next_helper)
        tangent_helper_end = tangent_helper_start + 20 + next_helper.start()
        tangent_helper = common[tangent_helper_start:tangent_helper_end]
        self.assertIn("float projectedState[NM_MAX_MATERIAL_STATE]", tangent_helper)
        self.assertIn("projectMaterialState(", tangent_helper)
        self.assertIn("returnedProjectedState", tangent_helper)
        validate_start = metal.index("kernel void nm_fem_validate(")
        validate = metal[validate_start:validate_start + 9000]
        self.assertIn("device float* candidateMaterialState", validate)
        self.assertIn("candidateMaterialState[materialStateBase + state] =\n            acceptedMaterialState", validate)


if __name__ == "__main__":
    unittest.main()
