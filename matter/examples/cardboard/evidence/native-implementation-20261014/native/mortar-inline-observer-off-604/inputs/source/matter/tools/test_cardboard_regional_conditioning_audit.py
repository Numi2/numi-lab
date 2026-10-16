#!/usr/bin/env python3
"""Stdlib regression tests for bounded rejected-conditioning audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from cardboard_regional_conditioning_audit import AuditError, audit_capture


def _write_floats(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack("<" + "f" * len(values), *values))


def _write_uints(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack("<" + "I" * len(values), *values))


class ConditioningAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.capture = self.run / "output" / "rejected-conditioning-000002"
        self.capture.mkdir(parents=True)
        source = self.root / "source"
        runtime = self.root / "runtime"
        source.mkdir()
        runtime.mkdir()
        (source / "owned.cpp").write_text("same sealed source\n", encoding="utf-8")
        status_header = source / "matter" / "include" / "numi" / "matter" / "shared.h"
        status_header.parent.mkdir(parents=True)
        status_header.write_text("NM_STATUS_NONLINEAR_SOLVER_FAILURE = 10u\n", encoding="utf-8")
        (runtime / "probe").write_text("same sealed executable\n", encoding="utf-8")
        external = self.root / "build-source.json"
        external.write_text("{}\n", encoding="utf-8")

        files: dict[str, tuple[bytes, int, int]] = {}
        files["candidate_nodes.bin"] = (bytes(2 * 80), 2, 80)
        # Two orthonormal columns over the 2-node x 4-float layout.
        p = [0.0] * 16
        p[0] = 1.0
        p[13] = 1.0
        files["P.bin"] = (struct.pack("<16f", *p), 4, 16)
        files["effective_P.bin"] = (struct.pack("<16f", *p), 4, 16)
        ap = [0.0] * 16
        ap[0] = 2.0
        ap[13] = 3.0
        files["AP.bin"] = (struct.pack("<16f", *ap), 4, 16)
        files["pivoted_RAP.bin"] = (struct.pack("<4f", 2.0, 0.0, 0.0, 3.0), 4, 4)
        files["pivots.bin"] = (struct.pack("<2I", 0, 1), 2, 4)
        files["factor_valid.bin"] = (struct.pack("<I", 1), 1, 4)
        block = [0.0] * (2 * 3 * 4)
        for node in range(2):
            block[(node * 3) * 4 + 3] = 1.0
        files["node_block_inverse.bin"] = (struct.pack("<24f", *block), 6, 16)

        rows = []
        for name, (blob, count, stride) in files.items():
            (self.capture / name).write_bytes(blob)
            rows.append({
                "name": name,
                "count": count,
                "stride_bytes": stride,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            })
        manifest = {
            "schema": "numi.cardboard.regional-conditioning.v1",
            "accepted_physical_state": False,
            "control_step": 1,
            "source_physics_fingerprint": "123",
            "device_program_fingerprint": "456",
            "environment_count": 1,
            "fem_node_count": 2,
            "mode_count": 2,
            "byte_order": "native_little_endian",
            "layout": "P mode-node; effective_P/AP environment-mode-node; RAP pivoted LU environment-row-column; block_inverse environment-node-row",
            "files": rows,
        }
        (self.capture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.run / "output" / "result.json").write_text(json.dumps({
            "status": "failed", "accepted_steps": 1,
            "compiled_world_fingerprint": 123,
        }), encoding="utf-8")
        (self.run / "receipt.json").write_text(json.dumps({
            "execution_started": True, "source_inputs_unchanged": True,
            "timed_out": False, "exit_code": 2,
        }), encoding="utf-8")
        (self.run / "output" / "observations.csv").write_text(
            "step,environment,status_code,step_accepted,certificate_raw_accepted_flag,fgmres_iterations,"
            "certificate_residual,certificate_correction,status_diagnostic_x,status_diagnostic_y,"
            "free_force_imbalance_l2_N\n"
            "1,0,10,0,0,256,0.00004,0.0138,0.00004,0.0138,0.41\n",
            encoding="utf-8")
        argv = ["probe", "--preset", "literature2009", "--line-search-steps", "24",
                "--residual-tolerance", "1e-6", "--newton-iterations", "20",
                "--fgmres-restart", "16", "--fgmres-iterations", "256",
                "--fgmres-preconditioner", "regional-galerkin-block", "--steps", "12",
                "--hold-steps", "3", "--unload-steps", "6", "--relax-steps", "3",
                "--release-steps", "3",
                "--nx", "8", "--ny", "2", "--thickness-slices", "1",
                "--bend-angle", ".375", "--dt", ".0001", "--material-iterations", "16",
                "--accurate-paper-projection", "--scaled-paper-residuals", "--without-self-contact",
                "--reference-displacement-gradient", "--compensated-paper-arithmetic",
                "--safeguarded-paper-initializer"]
        input_hashes = {
            "source/owned.cpp": hashlib.sha256((source / "owned.cpp").read_bytes()).hexdigest(),
            "source/matter/include/numi/matter/shared.h": hashlib.sha256(status_header.read_bytes()).hexdigest(),
            "runtime/probe": hashlib.sha256((runtime / "probe").read_bytes()).hexdigest(),
            "external/build-source.json": hashlib.sha256(external.read_bytes()).hexdigest(),
        }
        prediction = {
            "argv": argv,
            "environment": {"NM_FEM_RESIDUAL_ARMIJO_ROOT": "all"},
            "source_snapshot": str(source),
            "runtime_dir": str(runtime),
            "external_inputs": [str(external)],
            "inputs_sha256": input_hashes,
        }
        (self.run / "prediction.json").write_text(json.dumps(prediction), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_capture_reports_coarse_and_one_sided_metrics(self) -> None:
        result = audit_capture(self.capture)
        self.assertEqual(result["classification"], "rejected_step_diagnostic_only")
        self.assertFalse(result["accepted_physical_state"])
        self.assertEqual(result["outer_binding"]["verified_input_count"], 4)
        self.assertEqual(result["outer_binding"]["rejected_observation"]["status_name_from_sealed_source"],
                         "NM_STATUS_NONLINEAR_SOLVER_FAILURE")
        env = result["environments"][0]
        self.assertAlmostEqual(env["symmetric_RAP_condition_number"], 1.5, places=10)
        self.assertLess(env["pivoted_LU_relative_reconstruction_error_against_unpivoted_FP64_reduction"], 1e-12)
        self.assertEqual(env["valid_node_block_count"], 2)
        self.assertIn("full_space_A_invariance_residual_relative_to_Aq", env["smallest_Ritz_mode"])

    def test_raw_hash_mismatch_fails_closed(self) -> None:
        path = self.capture / "AP.bin"
        path.write_bytes(path.read_bytes()[:-1] + b"\xff")
        with self.assertRaisesRegex(AuditError, "SHA-256 mismatch"):
            audit_capture(self.capture)

    def test_dimension_metadata_mismatch_fails_closed(self) -> None:
        path = self.capture / "manifest.json"
        manifest = json.loads(path.read_text())
        next(row for row in manifest["files"] if row["name"] == "effective_P.bin")["count"] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AuditError, "count/stride mismatch"):
            audit_capture(self.capture)

    def test_invalid_factor_fails_closed_even_with_valid_file_hash(self) -> None:
        path = self.capture / "factor_valid.bin"
        blob = struct.pack("<I", 0)
        path.write_bytes(blob)
        manifest_path = self.capture / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        row = next(row for row in manifest["files"] if row["name"] == "factor_valid.bin")
        row["sha256"] = hashlib.sha256(blob).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(AuditError, "factors were invalid"):
            audit_capture(self.capture)

    def test_outer_source_hash_mismatch_fails_closed(self) -> None:
        prediction_path = self.run / "prediction.json"
        prediction = json.loads(prediction_path.read_text())
        prediction["inputs_sha256"]["source/owned.cpp"] = "0" * 64
        prediction_path.write_text(json.dumps(prediction))
        with self.assertRaisesRegex(AuditError, "sealed input hash mismatch"):
            audit_capture(self.capture)


if __name__ == "__main__":
    unittest.main()
