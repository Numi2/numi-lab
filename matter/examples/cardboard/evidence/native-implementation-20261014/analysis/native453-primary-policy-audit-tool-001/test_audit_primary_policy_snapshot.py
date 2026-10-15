import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("audit_primary_policy_snapshot.py")
spec = importlib.util.spec_from_file_location("audit_primary_policy_snapshot", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def armijo(iteration, kind, env_rows):
    return {"root": 1, "newton_iteration": iteration,
            "direction_kind": kind, "environments": env_rows}


def direction(iteration, eligibility, start=48):
    return {"root": 1, "newton_iteration": iteration,
            "direction_kind": "truncated_unregularized_CGLS",
            "selection_policy": "base_cgls_primary_policy",
            "start_newton_inclusive": start, "iteration_budget": 64,
            "original_newton_certificate_applies_to_direction": False,
            "environments": [
                {"environment": e, "primary_policy_eligible": selected,
                 "candidate_l2": 0.125 if selected else 0.0}
                for e, selected in enumerate(eligibility)]}


class PrimarySelectionConsumerTests(unittest.TestCase):
    def test_selected_cgls_update_does_not_inherit_newton_certificate(self):
        marker = direction(48, [True, False])
        row = armijo(48, "base_cgls_primary_policy", [
            {"accepted": True, "certified_noop": False, "selected_alpha": 0.25,
             "trial_alpha": 0.25, "trial_outcome": "selected_trial_accepted"},
            {"accepted": True, "certified_noop": True, "selected_alpha": 1.0,
             "trial_alpha": 0.0, "trial_outcome": "certified_noop"},
        ])
        selected = audit.summarize_primary_armijo_row(marker, row, 0)
        retained_noop = audit.summarize_primary_armijo_row(marker, row, 1)
        self.assertTrue(selected["cgls_direction_selected"])
        self.assertFalse(selected["original_newton_certificate_transferred_to_CGLS"])
        self.assertEqual(selected["armijo_outcome"], "accepted_state_update")
        self.assertFalse(retained_noop["cgls_direction_selected"])
        self.assertIsNone(retained_noop["original_newton_certificate_transferred_to_CGLS"])
        self.assertEqual(retained_noop["armijo_outcome"], "certified_noop_no_state_change")

    def test_eligible_but_rejected_is_not_reported_as_accepted(self):
        marker = direction(49, [True])
        record = armijo(49, "base_cgls_primary_policy", [
            {"accepted": False, "certified_noop": False, "selected_alpha": 0.0,
             "trial_alpha": 1.0, "trial_outcome": "no_trial_accepted"},
        ])
        result = audit.summarize_primary_armijo_row(marker, record, 0)
        self.assertTrue(result["cgls_direction_selected"])
        self.assertEqual(result["armijo_outcome"], "rejected_or_blocked_no_accepted_update")

    def test_primary_trace_requires_complete_per_iteration_markers(self):
        rows = [armijo(48, "base_cgls_primary_policy", [
            {"accepted": True, "certified_noop": False, "selected_alpha": 1.0}])]
        markers = [direction(48, [True])]
        complete = audit.summarize_later_primary_trials(rows, markers, 1, 48, 0, 64)
        self.assertTrue(complete["complete_observed_primary_trace"])
        missing = audit.summarize_later_primary_trials(rows, [], 1, 48, 0, 64)
        self.assertFalse(missing["complete_observed_primary_trace"])
        self.assertEqual(missing["missing_primary_direction_iterations"], [48])

    def test_pretarget_comparison_lists_only_proven_inert_fallbacks(self):
        newton = armijo(0, "newton", [{"accepted": True, "certified_noop": False}])
        baseline = [newton]
        fallback = armijo(0, "base_cgls", [{
            "accepted": True, "certified_noop": False,
            "last_trial_status_role": "unused_tail_after_selected_accept"}])
        cgls_marker = {"root": 1, "newton_iteration": 0,
                       "direction_kind": "truncated_unregularized_CGLS",
                       "original_newton_certificate_applies_to_direction": False,
                       "environments": [{"environment": 0, "candidate_l2": 0.0}]}
        got = audit.summarize_pretarget_trace(
            [newton, fallback], baseline, [cgls_marker], [], 1, 1, 0)
        self.assertTrue(got["pretarget_trace_parity_with_inert_fallbacks_discarded"])
        self.assertEqual(len(got["inert_fallback_observations_discarded_from_newton_comparison"]), 1)

        active_fallback = {**fallback, "environments": [{
            "accepted": False, "certified_noop": False,
            "last_trial_status_role": "last_tested_trial_status"}]}
        active_marker = {**cgls_marker, "environments": [{"environment": 0, "candidate_l2": 0.25}]}
        not_inert = audit.summarize_pretarget_trace(
            [newton, active_fallback], baseline, [active_marker], [], 1, 1, 0)
        self.assertFalse(not_inert["pretarget_trace_parity_with_inert_fallbacks_discarded"])
        self.assertEqual(len(not_inert["active_or_unresolved_fallback_observations"]), 1)

        # An unused-tail status does not by itself prove the whole fallback
        # inert: with a nonzero CGLS direction keep the observation unresolved.
        nonzero_candidate = audit.summarize_pretarget_trace(
            [newton, fallback], baseline,
            [{**cgls_marker, "environments": [{"environment": 0, "candidate_l2": 0.5}]}],
            [], 1, 1, 0)
        self.assertFalse(nonzero_candidate["pretarget_trace_parity_with_inert_fallbacks_discarded"])
        self.assertEqual(len(nonzero_candidate["active_or_unresolved_fallback_observations"]), 1)

    def test_newton_rejection_cannot_make_later_fallback_inert(self):
        primary = armijo(4, "newton", [{"accepted": False, "certified_noop": False}])
        fallback = armijo(4, "base_cgls", [{
            "accepted": True, "certified_noop": False,
            "last_trial_status_role": "unused_tail_after_selected_accept"}])
        marker = {"root": 1, "newton_iteration": 4,
                  "direction_kind": "truncated_unregularized_CGLS",
                  "original_newton_certificate_applies_to_direction": False,
                  "environments": [{"environment": 0, "candidate_l2": 0.0}]}
        got = audit.summarize_pretarget_trace(
            [primary, fallback], [primary], [marker], [], 1, 5, 0)
        self.assertFalse(got["pretarget_trace_parity_with_inert_fallbacks_discarded"])
        self.assertEqual(len(got["active_or_unresolved_fallback_observations"]), 1)

    def test_receipt_requires_ended_unchanged_execution(self):
        class EmptyAuditModule:
            @staticmethod
            def _manifest_file(*_args):
                return None, None

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snap = root / "snapshot"
            snap.mkdir()
            (root / "prediction.json").write_text(json.dumps({
                "source_snapshot": str(snap), "inputs_sha256": {}}))
            (root / "stderr.log").write_text("")
            receipt = {"schema": "numi.cardboard.native-run-receipt.v2",
                       "execution_started": True, "source_inputs_unchanged": True,
                       "changed_inputs": [], "exit_code": 2, "timed_out": False,
                       "termination_errors": [], "finished_at_utc": "done",
                       "wall_seconds": 1.0}
            (root / "receipt.json").write_text(json.dumps(receipt))
            _, _, _, checked = audit.bind_run(root, snap, EmptyAuditModule)
            self.assertTrue(checked["all_prediction_inputs_verified"])
            receipt["timed_out"] = True
            (root / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "timed out"):
                audit.bind_run(root, snap, EmptyAuditModule)


if __name__ == "__main__":
    unittest.main()
