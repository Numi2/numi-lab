#!/usr/bin/env python3
"""CPU contract checks for the proposed source109 narrowphase status reducer."""
from __future__ import annotations
import pathlib
import re
import unittest

PKG = pathlib.Path(__file__).resolve().parent
CONTACT = (PKG / "overlay/matter/src/metal/contact.metalinc").read_text()
RUNTIME = (PKG / "overlay/matter/src/runtime.mm").read_text()

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
