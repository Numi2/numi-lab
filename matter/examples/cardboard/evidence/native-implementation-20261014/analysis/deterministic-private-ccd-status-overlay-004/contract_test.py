#!/usr/bin/env python3
"""CPU contracts for source109 private narrowphase status selection."""
from __future__ import annotations
import pathlib
import re
import struct
import unittest

PKG = pathlib.Path(__file__).resolve().parent
CONTACT = (PKG / "overlay/matter/src/metal/contact.metalinc").read_text()
RUNTIME = (PKG / "overlay/matter/src/runtime.mm").read_text()
NATIVE = (PKG / "overlay/matter/tools/cardboard_contact_ccd_native_check.mm").read_text()

SUCCESS = 0
CONTACT_FAILURE = 6


def reduce_status(earlier_status, candidate_count, slot_statuses):
    """Model the reducer's ascending candidate-slot scan and preserve precedence."""
    if earlier_status["code"] != SUCCESS:
        return earlier_status
    for slot in range(min(candidate_count, len(slot_statuses))):
        candidate = slot_statuses[slot]
        if candidate["code"] != SUCCESS:
            return candidate
    return earlier_status


class StatusReducerContract(unittest.TestCase):
    def test_minimum_failed_candidate_slot_wins_independent_of_arrival(self):
        fail_124 = {"code": CONTACT_FAILURE, "failing_index": 124,
                    "diagnostics": (24, 127, 1, 0)}
        fail_283 = {"code": CONTACT_FAILURE, "failing_index": 283,
                    "diagnostics": (92, 127, 0, 0)}
        zero = {"code": SUCCESS}
        slots_a = [zero.copy() for _ in range(284)]
        slots_a[124], slots_a[283] = fail_124, fail_283
        slots_b = [zero.copy() for _ in range(284)]
        # Different simulated thread completion order cannot change the indexed records.
        slots_b[283], slots_b[124] = fail_283, fail_124
        ok = {"code": SUCCESS}
        self.assertEqual(reduce_status(ok, 284, slots_a), fail_124)
        self.assertEqual(reduce_status(ok, 284, slots_b), fail_124)

    def test_earlier_phase_failure_is_preserved(self):
        earlier = {"code": 3, "failing_index": 17, "diagnostics": (1, 2, 3, 4)}
        later = {"code": CONTACT_FAILURE, "failing_index": 0,
                 "diagnostics": (92, 127, 0, 0)}
        self.assertIs(reduce_status(earlier, 1, [later]), earlier)

    def test_no_private_failure_is_success_passthrough(self):
        ok = {"code": SUCCESS}
        self.assertIs(reduce_status(ok, 3, [{"code": SUCCESS}, {"code": SUCCESS}, {"code": SUCCESS}]), ok)

    def test_unused_stale_tail_status_is_not_selected(self):
        ok = {"code": SUCCESS}
        stale = {"code": CONTACT_FAILURE, "failing_index": 9,
                 "diagnostics": (9, 9, 0, 0)}
        self.assertIs(reduce_status(ok, 2, [{"code": SUCCESS}, {"code": SUCCESS}, stale]), ok)

    def test_empty_candidate_list_is_noop(self):
        ok = {"code": SUCCESS}
        self.assertIs(reduce_status(ok, 0, []), ok)

    def test_only_private_path_enables_candidate_status_mode(self):
        self.assertIn("narrowphaseDeterministicFailureSelection = 0u", RUNTIME)
        self.assertIn("narrowphaseDeterministicFailureSelection = 1u", RUNTIME)
        self.assertIn("femArmijoDeformableCandidateStatuses =", RUNTIME)
        self.assertIn("privateScratch<NMMatterStatusGPU>", RUNTIME)

    def test_reducer_runs_after_narrowphase_before_compaction(self):
        narrow = RUNTIME.index('dispatchThreads("nm_contact_narrowphase_deformable",')
        reduce = RUNTIME.index('dispatchThreads("nm_contact_reduce_deformable_narrowphase_failures",', narrow)
        compact = RUNTIME.index('dispatchGroups32("nm_contact_compact_deformable",', reduce)
        self.assertLess(narrow, reduce)
        self.assertLess(reduce, compact)
        self.assertIn("atIndex:17u", RUNTIME[narrow:reduce])

    def test_kernel_has_one_per_slot_status_writer_path(self):
        start = CONTACT.index("kernel void nm_contact_narrowphase_deformable(")
        end = CONTACT.index("kernel void nm_contact_reduce_deformable_narrowphase_failures(", start)
        body = CONTACT[start:end]
        self.assertEqual(body.count("recordDeformableNarrowphaseFailure("), 7)
        self.assertIn("candidateStatuses[global] = {};", body)
        self.assertIn("candidateStatuses[global]", body)
        self.assertNotIn("recordFailure(statuses[environment]", body)

    def test_reducer_preserves_status_precedence_and_scans_lowest_slot(self):
        start = CONTACT.index("kernel void nm_contact_reduce_deformable_narrowphase_failures(")
        end = CONTACT.index("kernel void nm_contact_compact_deformable(", start)
        body = CONTACT[start:end]
        self.assertIn("statuses[environment].code != NM_STATUS_SUCCESS", body)
        self.assertRegex(body, r"for \(uint slot = 0u; slot < count; \+\+slot\)")
        self.assertIn("recordFailure(statuses[environment], failure.code", body)

    def test_native_checker_dispatches_both_production_kernels(self):
        self.assertIn("numi_matter_metal::nm_contact_narrowphase_deformable", NATIVE)
        self.assertIn("numi_matter_metal::nm_contact_reduce_deformable_narrowphase_failures", NATIVE)
        self.assertIn("checkProductionNarrowphaseClearsCandidateStatuses", NATIVE)
        self.assertIn("checkProductionStatusReducer", NATIVE)

    def test_runtime_candidate_status_scratch_is_environment_major_and_accounted(self):
        self.assertIn("privateScratch<NMMatterStatusGPU>(candidate->device,", RUNTIME)
        self.assertIn("multiplied(static_cast<std::size_t>(capacity))", RUNTIME)
        self.assertIn("valid, candidate->residentBytes", RUNTIME)
        self.assertIn("environments * perEnvironment", RUNTIME)

    def test_native_producer_clear_uses_private_selector_and_full_extent(self):
        self.assertIn("const std::uint32_t deterministicSelection = 1u", NATIVE)
        self.assertIn("candidateStatuses(total)", NATIVE)
        self.assertIn("dispatch.environmentCount = 2u", NATIVE)
        self.assertIn("dispatch.deformableContactCapacity = 3u", NATIVE)
        self.assertIn("atIndex:17u", NATIVE)
        self.assertIn("production narrowphase did not clear a stale private status slot", NATIVE)

    def test_native_reducer_covers_multienvironment_precedence_and_prefixes(self):
        self.assertIn("candidateCounts{4u, 3u, 0u, 2u}", NATIVE)
        self.assertIn("NM_STATUS_NONFINITE_INPUT", NATIVE)
        self.assertIn("empty active candidate prefix selected stale candidate status", NATIVE)
        self.assertIn("healthy environment selected a stale failure beyond its active prefix", NATIVE)
        self.assertIn("status reducer overwrote completed-microstep/FGMRES/contact/event counters", NATIVE)

    def test_native_output_labels_status_instrument_scope(self):
        self.assertIn("contact-ccd-native-regression.v3", NATIVE)
        self.assertIn(r'\"native_device_action\":true', NATIVE)
        self.assertIn(r'\"producer_clear_verified\":true', NATIVE)

    def test_geometric_failure_specimen_has_finite_fp32_inputs_and_overflowing_products(self):
        # The native fixture uses finite coordinates/velocities whose relative
        # square exceeds FLT_MAX, driving the real point-point CCD arithmetic
        # to an uncertified result without the kernel's nonfinite-input return.
        f32 = lambda value: struct.unpack("=f", struct.pack("=f", value))[0]
        coordinate = f32(1.0e20)
        relative_position = f32(coordinate - f32(-coordinate))
        relative_velocity = f32(f32(-2.0e20) - f32(2.0e20))
        self.assertTrue(all(abs(f32(v)) < 3.4028235e38 for v in
                            (coordinate, -coordinate, -2.0e20, 2.0e20)))
        self.assertGreater(float(relative_position) ** 2, 3.4028235e38)
        self.assertGreater(float(relative_velocity) ** 2, 3.4028235e38)
        self.assertIn("finite_point_point_inputs_with_overflowing_relative_dot_products", NATIVE)
        self.assertIn("gridNodes[0].positionAndMass = nm_float4{1.0e20f", NATIVE)
        self.assertIn("gridNodes[0].candidateVelocity = nm_float4{-2.0e20f", NATIVE)
        self.assertIn("gridNodes[1].candidateVelocity = nm_float4{2.0e20f", NATIVE)

    def test_native_geometric_producer_private_and_ordinary_modes_are_both_dispatched(self):
        self.assertIn("checkProductionNarrowphaseGeometricFailure", NATIVE)
        start = NATIVE.index("void checkProductionNarrowphaseGeometricFailure(")
        end = NATIVE.index("void checkDirect(", start)
        body = NATIVE[start:end]
        self.assertIn("dispatchNarrowphase(1u)", body)
        self.assertIn("dispatchNarrowphase(0u)", body)
        self.assertIn("private geometric failure changed environment status before reducer", body)
        self.assertIn("ordinary geometric failure wrote or cleared private candidate status", body)
        self.assertIn("checkStatusTuple(reduced[0]", body)

    def test_geometric_specimen_reaches_real_pointpoint_uncertified_status_branch(self):
        start = CONTACT.index("kernel void nm_contact_narrowphase_deformable(")
        end = CONTACT.index("kernel void nm_contact_reduce_deformable_narrowphase_failures(", start)
        body = CONTACT[start:end]
        point_pair = body.index("if (leftKind == NM_CONTINUUM_SURFACE_POINT &&")
        branch = body.index("if (!certified)", point_pair)
        call = body.index("recordDeformableNarrowphaseFailure(", branch)
        self.assertLess(branch, call)
        self.assertIn("environment, global", body[call:call + 400])
        self.assertIn("float(candidate.identity.x)", body[call:call + 400])
        self.assertIn("float(candidate.identity.y)", body[call:call + 400])
        self.assertIn("expected_failure_path", NATIVE)
        self.assertIn("pointPointCCD_uncertified_then_recordDeformableNarrowphaseFailure", NATIVE)

    def test_geometric_failure_control_is_not_a_physics_qualification(self):
        self.assertIn(r'\"physical_qualification\":false', NATIVE)
        self.assertIn("physical qualification is false", (PKG / "README.md").read_text().lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
