#!/usr/bin/env python3
"""Portable integrity and FP32-widening tests for the capture reader."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coupled_capture_reader import Capture, CaptureError, compare_history_dirs


CAPTURE_POINT = "after_fgmres_measure_and_human_direction_map_before_armijo_trial"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_capture(root: Path, rows: list[tuple[str, str, bytes, int]]) -> Path:
    payload = bytearray()
    entries = []
    for name, dtype, data, element_bytes in rows:
        count = len(data) // element_bytes
        entries.append({
            "name": name,
            "source": "host_snapshot",
            "dtype": dtype,
            "count": count,
            "element_bytes": element_bytes,
            "offset": len(payload),
            "byte_count": len(data),
            "capacity_snapshot": True,
            "sha256": _sha(data),
        })
        payload.extend(data)
    payload_bytes = bytes(payload)
    (root / "payload.bin").write_bytes(payload_bytes)
    manifest = {
        "schema": "numi.matter.coupled_newton_capture.v1",
        "byte_order": "little_endian",
        "capture_point": CAPTURE_POINT,
        "capture_kind": "raw_transaction_candidate_not_checkpoint",
        "control_step": 1201,
        "iteration": 12,
        "environment_count": 1,
        "node_count": 1,
        "tetrahedron_count": 0,
        "dt_seconds": 0.0001,
        "max_payload_bytes": 32 * 1024 * 1024,
        "max_arrays": 192,
        "array_count": len(entries),
        "payload": {"file": "payload.bin", "byte_count": len(payload_bytes),
                    "sha256": _sha(payload_bytes)},
        "arrays": entries,
    }
    path = root / "capture.coupled.json"
    path.write_text(json.dumps(manifest))
    return path


class CaptureReaderTests(unittest.TestCase):
    def test_fp32_words_are_promoted_then_high_low_added_in_fp64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            high = np.array([1.0, -1.0], dtype="<f4")
            low = np.array([2.0 ** -24, -(2.0 ** -24)], dtype="<f4")
            path = make_capture(root, [
                ("high", "float32", high.tobytes(), 4),
                ("low", "float32", low.tobytes(), 4),
            ])
            captured = Capture(path)
            combined = captured.high_low_f64("high", "low")
            np.testing.assert_array_equal(combined, np.array([1.0 + 2.0 ** -24,
                                                              -1.0 - 2.0 ** -24]))

    def test_node_record_decodes_the_frozen_80_byte_word_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            words = np.arange(1, 21, dtype="<f4")
            path = make_capture(root, [
                ("fem_candidate_nodes", "record:NMFEMNodeStateGPU",
                 words.tobytes(), 80),
            ])
            fields = Capture(path).fem_nodes_f64("fem_candidate_nodes")
            np.testing.assert_array_equal(fields["position_and_mass"], [[1, 2, 3, 4]])
            np.testing.assert_array_equal(
                fields["reference_displacement_and_mode"], [[17, 18, 19, 20]])

    def test_nonstandard_json_constants_are_rejected(self) -> None:
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = make_capture(root, [("x", "float32", b"\0\0\x80?", 4)])
                raw = path.read_text().replace('"dt_seconds": 0.0001',
                                               f'"dt_seconds": {token}')
                path.write_text(raw)
                with self.assertRaisesRegex(CaptureError,
                                            "non-standard JSON numeric token"):
                    Capture(path)

    def test_array_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_capture(root, [("x", "float32", b"\0\0\x80?", 4)])
            value = json.loads(path.read_text())
            value["arrays"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(CaptureError, "SHA-256 mismatch for array"):
                Capture(path)

    def test_payload_gap_is_rejected_even_with_valid_individual_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_capture(root, [
                ("a", "float32", b"\0\0\x80?", 4),
                ("b", "float32", b"\0\0\0@", 4),
            ])
            value = json.loads(path.read_text())
            payload_path = root / "payload.bin"
            payload = payload_path.read_bytes()
            payload = payload[:4] + b"\0" + payload[4:]
            payload_path.write_bytes(payload)
            value["payload"]["byte_count"] = len(payload)
            value["payload"]["sha256"] = _sha(payload)
            value["arrays"][1]["offset"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(CaptureError, "gap or overlap"):
                Capture(path)

    def test_byte_identical_failed_attempt_stays_unqualified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            on = root / "on"
            off = root / "off"
            on.mkdir(); off.mkdir()
            (on / "accepted.nmcb").write_bytes(b"raw-state")
            (off / "accepted.nmcb").write_bytes(b"raw-state")
            (on / "newton12.coupled.bin").write_bytes(b"diagnostic-only")
            report = compare_history_dirs(on, off, ["*.nmcb"])
            self.assertTrue(report["raw_files_identical"])
            self.assertEqual(report["physical_convergence"],
                             "not_assessed_by_byte_comparison")

    def test_history_mismatch_fails_and_capture_payload_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            on = root / "on"; off = root / "off"
            on.mkdir(); off.mkdir()
            (on / "histories.csv").write_text("step,accepted\n1,1\n")
            (off / "histories.csv").write_text("step,accepted\n1,0\n")
            (on / "candidate.coupled.bin").write_bytes(b"different diagnostic")
            report = compare_history_dirs(on, off, ["*.csv"])
            self.assertFalse(report["raw_files_identical"])
            self.assertFalse(report["files_compared"][0]["identical"])


if __name__ == "__main__":
    unittest.main()
