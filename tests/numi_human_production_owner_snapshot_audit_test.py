#!/usr/bin/env python3
"""CPU-only tests for the production-owner snapshot evidence auditor."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY_ROOT / "tools" / "audit_numi_human_production_owner_snapshot.py"
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
import audit_numi_human_production_owner_snapshot as audit  # noqa: E402


def words(data: bytes) -> list[str]:
    assert len(data) % 4 == 0
    return [
        f"{struct.unpack('<I', data[offset:offset + 4])[0]:08x}"
        for offset in range(0, len(data), 4)
    ]


def floats(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def captured(count: int, width: int, data: bytes | None = None) -> dict:
    raw = bytes(count * width) if data is None else data
    assert len(raw) == count * width
    return {
        "available": True,
        "expected_elements": count,
        "captured_elements": count,
        "element_bytes": width,
        "words": words(raw),
    }


def equality_row() -> bytes:
    return (
        struct.pack("<4I", 7, 6, audit.INVALID_INDEX, audit.INVALID_INDEX) +
        floats(
            [0.0, 0.0, 0.0, 0.0] +  # references and a0/a1
            [0.0, 0.0, 0.0, 0.0] +  # a2/a3/a4/reserved
            [0.0, 0.0, 0.0, 0.0] +  # solref/reserved
            [0.5, 0.5, 0.0, 0.5] +  # solimp0
            [1.0, 0.0, 0.0, 0.0] +  # solimp1/reserved
            [1.0, 0.0, 0.0, 0.0]    # dependent/master inverse weights/reserved
        )
    )


def limit_row() -> bytes:
    return (
        struct.pack("<4I", 7, 6, 123, 0) +
        floats(
            [-1.0, 1.0, 2.0, 1.0] +  # lower/upper/margin/inverse weight
            [0.0, 0.0, 0.0, 0.0] +  # solref/reserved
            [0.5, 0.5, 0.0, 0.5] +  # solimp0
            [1.0, 0.0, 0.0, 0.0]    # solimp1/reserved
        )
    )


def factor_storage(*, upper01: float = 1.0, diagonal0: float = 2.0) -> bytes:
    matrix = [0.0] * 49
    for index in range(7):
        matrix[index * 7 + index] = 1.0
    matrix[0] = diagonal0
    matrix[1] = upper01
    matrix[7] = 0.5
    matrix[8] = 3.0
    return floats(matrix)


def payload() -> dict:
    result = {
        "schema": audit.PAYLOAD_SCHEMA,
        "format_version": 1,
        "comparison_identity_algorithm": "fnv1a64-domain-v1",
        "comparison_identity_is_cryptographic_proof": False,
        "rhs_bias_origin": "host-reconstructed-from-captured-A0-v0-vfree-tau",
        "constraint_witness_origin": "host-reconstructed-from-captured-production-inputs",
        "inertial_operator_capture": "source-effective-tangent-factor-storage",
        "effective_tangent_factor_storage_layout":
            "row-major-lower-cholesky-upper-source-A0",
        "rollback_identity_scope":
            "lifecycle-terminal-disposition-only-not-byte-restoration-proof",
        "terminal_matter_state_role": "published-attempt-accepted-state",
        "work_energy_scope": "solver-components-not-physical-energy-closure",
        "treatment": "cold",
        "disposition": "published",
        "base_state_fingerprint": "0000000000000010",
        "treatment_history_fingerprint": "0000000000000011",
        "human_source_fingerprint": "0000000000000012",
        "matter_source_physics_fingerprint": "0000000000000013",
        "matter_device_program_fingerprint": "0000000000000014",
        "owner_program_fingerprint": "0000000000000015",
        "transaction_fingerprint": "0000000000000016",
        "previous_transaction_fingerprint": "0000000000000000",
        "linearization_epoch": "0000000000000017",
        "slot_generation": "0000000000000018",
        "physics_generation": "0000000000000019",
        "previous_physics_generation": "0000000000000018",
        "brain_generation": "000000000000001a",
        "sensor_generation": "000000000000001b",
        "human_io_program_fingerprint": "000000000000001c",
        "sensor_fingerprint": "000000000000001d",
        "transaction_instance_fingerprint": "000000000000001e",
        "joint_fence_fingerprint": "000000000000001f",
        "equality_program_fingerprint": "0000000000000020",
        "limit_program_fingerprint": "0000000000000021",
        "coverage_mask": "00000000007bffff",
        "required_coverage_mask": "00000000007bffff",
        "truncation_mask": "0000000000000000",
        "control_step": 0,
        "candidate_timestamp_nanoseconds": 500_000_000,
        "publication_epoch": 1,
        "timestep_nanoseconds": 500_000_000,
        "support_payload_byte_count": 84,
        "support_payload_abi": 1,
        "support_payload_sha256": "01" * 32,
        "matter_control_step": 0,
        "q_coordinate_count": 8,
        "dof_count": 7,
        "muscle_count": 1,
        "muscle_site_count": 0,
        "muscle_wrap_count": 0,
        "muscle_route_node_count": 0,
        "support_row_count": 0,
        "equality_row_count": 1,
        "limit_row_count": 1,
        "tendon_row_count": 0,
        "contact_sample_count": 0,
        "terminal_accepted_matter_rigid_generalized_state_count": 0,
        "terminal_accepted_matter_rigid_reaction_count": 0,
        "publication_identity_available": True,
        "rollback_identity_available": False,
    }
    arrays = {}
    for name, (count_for, width) in audit.ARRAY_SPECS.items():
        arrays[name] = captured(count_for(result), width)
    arrays["checkpoint_v"] = captured(
        7, 4, floats([1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["source_predicted_velocity"] = captured(
        7, 4, floats([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["source_generalized_force"] = captured(
        7, 4, floats([20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["matter_generalized_reaction"] = captured(
        7, 4, floats([-4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["candidate_v"] = captured(
        7, 4, floats([2.5, -1.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    )
    arrays["effective_tangent_factor_storage"] = captured(
        49, 4, factor_storage()
    )
    arrays["acceleration"] = captured(
        7, 4, floats([2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["source_rhs"] = captured(
        7, 4, floats([12.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["source_bias"] = captured(
        7, 4, floats([8.0, -9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    arrays["work_energy_components"] = captured(
        3, 4, floats([0.0, -5.75, 60.625])
    )
    arrays["equality_rows"] = captured(1, 112, equality_row())
    arrays["limit_rows"] = captured(1, 80, limit_row())
    arrays["equality_reconstructed_linearization_impulses"] = captured(
        1, 32, floats([0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 1.0, 0.0])
    )
    arrays["limit_reconstructed_linearization_impulses"] = captured(
        2,
        32,
        floats(
            [1.0, 0.0, 1.0, -1.0, 0.0, 10.0, 1.0, 0.0] +
            [-1.0, 0.0, 1.0, -1.0, -10.0, -10.0, 1.0, 0.0]
        ),
    )
    result["arrays"] = arrays
    return result


def envelope(value: dict, *, indent: int | None = None) -> str:
    payload_text = json.dumps(value, separators=(",", ":") if indent is None else None,
                              indent=indent, allow_nan=False)
    digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    return (
        '{"evidence_schema":"' + audit.EVIDENCE_SCHEMA +
        '","payload_sha256":"' + digest + '","payload":' + payload_text + "}"
    )


def envelope_digest(value: dict) -> str:
    payload_text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def rejected_payload() -> dict:
    result = payload()
    result["disposition"] = "rejected"
    result["terminal_matter_state_role"] = (
        "prior-accepted-state-context-after-rejection"
    )
    result["publication_identity_available"] = False
    result["rollback_identity_available"] = True
    result["publication_epoch"] = 0
    result["joint_fence_fingerprint"] = "0000000000000000"
    result["coverage_mask"] = f"{audit.REJECTED_REQUIRED_COVERAGE_MASK:016x}"
    result["required_coverage_mask"] = f"{audit.REJECTED_REQUIRED_COVERAGE_MASK:016x}"
    return result


class ProductionOwnerSnapshotAuditTests(unittest.TestCase):
    def test_valid_snapshot_recomputes_dynamics_and_reports_dominant_rows(self) -> None:
        report = audit.audit_text(envelope(payload()))
        self.assertTrue(report["verified"]["fp64_dynamics_to_exact_fp32"])
        self.assertTrue(
            report["verified"]["positive_cholesky_and_strict_upper_source_a0"]
        )
        self.assertEqual(report["dominant_dofs"]["acceleration"], {"dof": 1, "value": 4.0})
        self.assertEqual(report["dominant_dofs"]["source_rhs"], {"dof": 1, "value": 39.0})
        self.assertEqual(report["dominant_dofs"]["matter_reaction"], {"dof": 1, "value": 5.0})
        self.assertEqual(
            report["constraint_witnesses"]["equality"]["largest_impulse"],
            {"row": 0, "value": 10.0},
        )
        self.assertEqual(
            report["constraint_witnesses"]["equality"]["largest_violation"],
            {"row": 0, "value": 10.0},
        )
        self.assertEqual(
            report["constraint_witnesses"]["limit"]["largest_violation"],
            {"row": 0, "side": "upper", "value": -10.0},
        )
        self.assertFalse(report["payload_digest"]["authenticated_provenance_claimed"])

    def test_hash_uses_exact_pretty_printed_payload_substring(self) -> None:
        report = audit.audit_text(envelope(payload(), indent=1))
        self.assertTrue(report["verified"]["exact_embedded_payload_self_digest"])

    def test_payload_byte_change_without_digest_update_fails(self) -> None:
        source = envelope(payload())
        changed = source.replace('"format_version":1', '"format_version": 1')
        with self.assertRaisesRegex(audit.AuditError, "exact embedded payload bytes"):
            audit.audit_text(changed)

    def test_nonzero_truncation_and_reduced_required_coverage_fail(self) -> None:
        truncated = payload()
        truncated["truncation_mask"] = "0000000000000001"
        with self.assertRaisesRegex(audit.AuditError, "truncation_mask"):
            audit.audit_text(envelope(truncated))
        reduced = payload()
        reduced["required_coverage_mask"] = "00000000007bfffe"
        with self.assertRaisesRegex(audit.AuditError, "exact disposition contract"):
            audit.audit_text(envelope(reduced))

    def test_rejected_mask_is_exact_and_matter_step_must_match(self) -> None:
        report = audit.audit_text(envelope(rejected_payload()))
        self.assertEqual(
            report["coverage"]["required_mask"],
            f"{audit.REJECTED_REQUIRED_COVERAGE_MASK:016x}",
        )
        wrong_mask = rejected_payload()
        wrong_mask["required_coverage_mask"] = (
            f"{audit.PUBLISHED_REQUIRED_COVERAGE_MASK:016x}"
        )
        with self.assertRaisesRegex(audit.AuditError, "exact disposition contract"):
            audit.audit_text(envelope(wrong_mask))
        wrong_step = payload()
        wrong_step["matter_control_step"] = 1
        with self.assertRaisesRegex(audit.AuditError, "does not equal control_step"):
            audit.audit_text(envelope(wrong_step))

    def test_word_spelling_and_array_shape_fail_closed(self) -> None:
        bad_word = payload()
        bad_word["arrays"]["checkpoint_v"]["words"][0] = "3F800000"
        with self.assertRaisesRegex(audit.AuditError, "lowercase hex"):
            audit.audit_text(envelope(bad_word))
        bad_shape = payload()
        bad_shape["arrays"]["candidate_v"]["captured_elements"] = 1
        with self.assertRaisesRegex(audit.AuditError, "partially captured"):
            audit.audit_text(envelope(bad_shape))

    def test_fp32_witness_tampering_fails_exact_comparison(self) -> None:
        changed = payload()
        changed["arrays"]["source_rhs"] = captured(
            7, 4, floats([12.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        with self.assertRaisesRegex(audit.AuditError, r"source_rhs\[1\]"):
            audit.audit_text(envelope(changed))

    def test_cholesky_pivot_and_retained_upper_a0_tampering_fail(self) -> None:
        upper = payload()
        upper["arrays"]["effective_tangent_factor_storage"] = captured(
            49, 4, factor_storage(upper01=1.25)
        )
        with self.assertRaisesRegex(audit.AuditError, "strict-upper source A0"):
            audit.audit_text(envelope(upper))
        pivot = payload()
        pivot["arrays"]["effective_tangent_factor_storage"] = captured(
            49, 4, factor_storage(diagonal0=0.0)
        )
        with self.assertRaisesRegex(audit.AuditError, "diagonal 0 is not positive"):
            audit.audit_text(envelope(pivot))

    def test_constraint_identity_and_raw_positive_zero_tampering_fail(self) -> None:
        witness = payload()
        bad_limit = floats(
            [1.0, 0.0, 1.0, -1.0, 0.0, 10.0, 1.0, 0.0] +
            [-1.0, 0.0, 1.0, -1.0, -9.0, -9.0, 1.0, 0.0]
        )
        witness["arrays"]["limit_reconstructed_linearization_impulses"] = captured(
            2, 32, bad_limit
        )
        with self.assertRaisesRegex(audit.AuditError, "limit witness 1 violation"):
            audit.audit_text(envelope(witness))

        negative_zero = payload()
        equality = [0.0, 0.0, 1.0, 0.0, 10.0, 10.0, 1.0, -0.0]
        negative_zero["arrays"]["equality_reconstructed_linearization_impulses"] = (
            captured(1, 32, floats(equality))
        )
        with self.assertRaisesRegex(audit.AuditError, r"not raw FP32 \+0"):
            audit.audit_text(envelope(negative_zero))

        row_negative_zero = payload()
        bad_row = bytearray(equality_row())
        bad_row[11 * 4:12 * 4] = struct.pack("<I", 0x80000000)
        row_negative_zero["arrays"]["equality_rows"] = captured(
            1, 112, bytes(bad_row)
        )
        with self.assertRaisesRegex(audit.AuditError, r"reserved word 11.*raw FP32 \+0"):
            audit.audit_text(envelope(row_negative_zero))

        bad_identity = payload()
        bad_equality = bytearray(equality_row())
        bad_equality[:16] = struct.pack(
            "<4I", 7, 5, audit.INVALID_INDEX, audit.INVALID_INDEX
        )
        bad_identity["arrays"]["equality_rows"] = captured(
            1, 112, bytes(bad_equality)
        )
        with self.assertRaisesRegex(audit.AuditError, "dependent coordinate identity"):
            audit.audit_text(envelope(bad_identity))

    def test_duplicate_key_and_wrong_schema_fail_closed(self) -> None:
        source = envelope(payload())
        duplicate = source.replace(
            '"evidence_schema":', '"evidence_schema":"duplicate","evidence_schema":', 1
        )
        with self.assertRaisesRegex(audit.AuditError, "duplicate JSON key"):
            audit.audit_text(duplicate)
        wrong = payload()
        wrong["schema"] = "persistent-production-owner-snapshot.v2"
        with self.assertRaisesRegex(audit.AuditError, "payload schema"):
            audit.audit_text(envelope(wrong))
        with self.assertRaisesRegex(audit.AuditError, "not strict JSON"):
            audit.audit_text('{"evidence_schema":')

    def test_bounded_read_rejects_one_byte_past_limit(self) -> None:
        prior = audit.MAX_INPUT_BYTES
        self.addCleanup(setattr, audit, "MAX_INPUT_BYTES", prior)
        audit.MAX_INPUT_BYTES = 32
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.json"
            path.write_bytes(b"{" + b" " * 32)
            with self.assertRaisesRegex(audit.AuditError, "exceeds"):
                audit.audit_path(path)

    def test_detached_expected_digest_is_separate_and_fail_closed(self) -> None:
        source = payload()
        expected = envelope_digest(source)
        report = audit.audit_text(
            envelope(source), expected_payload_sha256=expected
        )
        self.assertTrue(
            report["verified"]["detached_expected_payload_digest"]
        )
        self.assertEqual(
            report["payload_digest"]["detached_expected_sha256"], expected
        )
        with self.assertRaisesRegex(audit.AuditError, "detached expected SHA-256"):
            audit.audit_text(
                envelope(source), expected_payload_sha256="0" * 64
            )

    def test_cli_reports_success_and_clear_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(envelope(payload()) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    str(path),
                    "--expected-payload-sha256",
                    envelope_digest(payload()),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cli_report = json.loads(completed.stdout)
            self.assertEqual(cli_report["audit_schema"], audit.AUDIT_SCHEMA)
            self.assertTrue(
                cli_report["payload_digest"]["detached_expected_match"]
            )
            changed = copy.deepcopy(payload())
            changed["arrays"]["acceleration"] = captured(
                7, 4, floats([2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            path.write_text(envelope(changed), encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, str(TOOL), str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, "")
            self.assertIn("does not exactly match FP64 recomputation", failed.stderr)


if __name__ == "__main__":
    unittest.main()
