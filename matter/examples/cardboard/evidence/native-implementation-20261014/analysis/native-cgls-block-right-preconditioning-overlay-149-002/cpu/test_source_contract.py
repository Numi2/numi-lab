#!/usr/bin/env python3
"""Static ABI/admission contracts for the opt-in source overlay."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (ROOT / "source/matter/src/runtime.mm").read_text()
FGMRES = (ROOT / "source/matter/src/metal/fgmres.metalinc").read_text()
CGLS = (ROOT / "source/matter/src/metal/base_cgls.metalinc").read_text()

class SourceContractTests(unittest.TestCase):
    def test_exact_opt_in_and_default_off(self):
        self.assertIn('std::getenv("NM_FEM_BASE_CGLS_RIGHT_SCALE")', RUNTIME)
        self.assertIn('std::strcmp(rightScale, "block3") != 0', RUNTIME)
        self.assertIn('femBaseCGLSIterations == 0u', RUNTIME)
        self.assertIn('if (femBaseCGLSRightScaleEnabled)', RUNTIME)
        self.assertIn('action(4u, state.femBaseCGLSP, state.femBaseCGLSQ)', RUNTIME)
        self.assertIn('action(4u, state.femBaseCGLSX, state.femBaseCGLSQ)', RUNTIME)

    def test_newton_local_fresh_selected_block_build(self):
        self.assertIn('"nm_fgmres_build_tangent_fem_block_preconditioner"', RUNTIME)
        self.assertIn('state.femCandidate', RUNTIME)
        self.assertIn('state.femMaterialStateAccepted', RUNTIME)
        self.assertIn('state.femAlgorithmicTangentCache', RUNTIME)
        self.assertIn('state.femBaseCGLSSelected', RUNTIME)
        self.assertIn('const std::uint32_t onlySelected = 1u;', RUNTIME)
        self.assertIn('selectedEnvironments[environment] != 1u', FGMRES)
        self.assertIn('device const float* acceptedMaterialState [[buffer(12)]]', FGMRES)

    def test_right_scale_builder_uses_private_scratch_not_fgmres_arena(self):
        start = RUNTIME.index('if (femBaseCGLSRightScaleEnabled) {', RUNTIME.index('nm_fgmres_base_cgls_initialize_state'))
        end = RUNTIME.index('"nm_fgmres_base_cgls_validate_right_scale"', start)
        build_span = RUNTIME[start:end]
        self.assertIn('state.femBaseCGLSRightScalePhysical', build_span)
        self.assertIn('atIndex:13u', build_span)
        self.assertNotIn('state.femPreconditioned', build_span)

    def test_unselected_block_outputs_are_initialized_before_skip(self):
        start = FGMRES.index('kernel void nm_fgmres_build_tangent_fem_block_preconditioner(')
        end = FGMRES.index('\nkernel void nm_fgmres_export_rigid(', start)
        body = FGMRES[start:end]
        zero = body.index('inverseBlockRows[packedBase + 2u] = float4(0.0f);')
        selected = body.index('if (onlySelectedEnvironments != 0u &&')
        self.assertLess(zero, selected)

    def test_s_and_transpose_are_applied_on_correct_sides(self):
        self.assertIn('applyRightScale(state.femBaseCGLSP,', RUNTIME)
        self.assertIn('action(4u, state.femBaseCGLSRightScalePhysical,', RUNTIME)
        self.assertIn('action(3u, state.femBaseCGLSR, state.femBaseCGLSS);', RUNTIME)
        self.assertIn('applyRightScale(state.femBaseCGLSS,\n                                state.femBaseCGLSS, 1u)', RUNTIME)
        self.assertIn('applyRightScale(state.femBaseCGLSX,\n                            state.femBaseCGLSX, 0u)', RUNTIME)
        self.assertIn('float3(row0.x * value.x + row1.x * value.y + row2.x * value.z', CGLS)
        self.assertIn('float3(dot(row0.xyz, value.xyz)', CGLS)

    def test_fail_closed_row_and_block_guards(self):
        self.assertIn('kBaseCGLSErrorUnsupportedConstraint', CGLS)
        self.assertIn('tag != 0.0f && tag != 1.0f && tag != 3.0f', CGLS)
        self.assertIn('node.positionAndMass.w <= 0.0f', CGLS)
        self.assertIn('row0.w != 1.0f', CGLS)
        self.assertIn('atomic_load_explicit(&errors[environment]', CGLS)
        self.assertIn('kBaseCGLSErrorNonzeroConstrainedRow', CGLS)

    def test_physical_install_and_certificate_kernel_unchanged(self):
        # Scaled CGLS only reaches the existing final action/install; no kernel
        # source writes acceptance or the original FGMRES certificate.
        self.assertIn('"nm_fgmres_install_base_cgls_direction"', RUNTIME)
        self.assertIn('states[environment].diagnostics.z = 0.0f;',
                      (ROOT / "source/matter/src/metal/base_cgls_runtime.metalinc").read_text())
        self.assertIn('original_newton_certificate_applies_to_direction\\":false', RUNTIME)

    def test_capture_version_disambiguates_scaled_coordinate_rows(self):
        self.assertIn('femBaseCGLSRightScaleEnabled ? 2u : 1u', RUNTIME)
        self.assertIn('femBaseCGLSRightScaleEnabled ? 13u : 12u', RUNTIME)
        self.assertIn('cgls_right_scale', RUNTIME)

if __name__ == "__main__":
    unittest.main(verbosity=2)
