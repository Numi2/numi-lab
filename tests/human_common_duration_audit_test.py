#!/usr/bin/env python3
"""CPU-only contract tests for the physical Human trace runner."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_human_common_duration as audit  # noqa: E402


def sample(step: int, time_seconds: float) -> dict:
    return {
        "step": step,
        "time_seconds": time_seconds,
        "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 122,
        "v": [0.0] * 128,
        "normal_impulse": 0.0,
        "contact_normal_impulse_work_j": 0.0,
        "contact_tangential_impulse_work_j": 0.0,
        "equality_impulse_work_j": 0.0,
        "source_limit_impulse_work_j": 0.0,
        "contact_normal_absolute_impulse_work_j": 0.0,
        "contact_tangential_absolute_impulse_work_j": 0.0,
        "equality_absolute_impulse_work_j": 0.0,
        "source_limit_absolute_impulse_work_j": 0.0,
    }


def trace() -> dict:
    return {
        "schema": audit.TRACE_SCHEMA,
        "endpoint_equivalent": "bitwise",
        "endpoint_max_q_delta": 0,
        "endpoint_max_v_delta": 0,
        "work_scope": (
            "production_constraint_impulse_work_by_family;"
            "exact_coordinate_projection_is_an_unowned_overwrite_not_impulse_work"
        ),
        "samples": [sample(0, 0.0), sample(1, 0.0000125)],
        **{total: 0.0 for total in audit.IMPULSE_WORK_TOTALS.values()},
    }


class HumanCommonDurationAuditTests(unittest.TestCase):
    def test_valid_trace_contract(self) -> None:
        audit.validate_trace(
            trace(),
            {
                "step_count": 1,
                "timestep_seconds": 0.0000125,
                "root_assistance": False,
            },
            12_500,
            1,
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(audit.AuditError, "duplicate JSON key"):
            audit.strict_json_object('{"schema":"one","schema":"two"}', "case")

    def test_absolute_work_cannot_hide_signed_work(self) -> None:
        value = trace()
        value["samples"][1]["equality_impulse_work_j"] = -0.25
        with self.assertRaisesRegex(audit.AuditError, "hides signed work"):
            audit.validate_trace(
                value,
                {
                    "step_count": 1,
                    "timestep_seconds": 0.0000125,
                    "root_assistance": False,
                },
                12_500,
                1,
            )

    def test_trace_timestamp_must_match_exact_clock(self) -> None:
        value = trace()
        value["samples"][1]["time_seconds"] = 0.000013
        with self.assertRaisesRegex(audit.AuditError, "wrong physical time"):
            audit.validate_trace(
                value,
                {
                    "step_count": 1,
                    "timestep_seconds": 0.0000125,
                    "root_assistance": False,
                },
                12_500,
                1,
            )


if __name__ == "__main__":
    unittest.main()
