#!/usr/bin/env python3
"""Verify exported captured419 direction payloads and exact CSR predictions."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy import sparse

BASE = Path("/Users/home/cardboard-implementation-20261014")
HERE = Path(__file__).resolve().parent
CAPTURE = BASE / "solver/two-stage-first-failure-rich-fd-419/operator-capture"
SOURCE = BASE / "snapshots/source-status5-vtccd-diagnostics-125"
ASSEMBLY = SOURCE / "matter/tools/cardboard_cached_operator_audit.py"
CAPTURE_PACKAGE = BASE / "analysis/captured-base-state-cauchy-419-001"
HEADER = struct.Struct("<8s6IQfI")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dot(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return math.fsum(float(x) * float(y) for x, y in zip(aa, bb))


def norm(a) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def close(observed: float, recorded: float, label: str) -> None:
    if not math.isclose(observed, recorded, rel_tol=2.0e-12, abs_tol=1.0e-24):
        raise AssertionError(f"{label} differs: {observed:.17g} vs {recorded:.17g}")


def import_assembly():
    spec = importlib.util.spec_from_file_location("source125_verify_operator", ASSEMBLY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bound source125 operator auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    manifest_path = HERE / "manifest.json"
    report_path = HERE / "report.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if manifest.get("schema") != "numi.cardboard.captured419-regularized-directions-package.v1":
        raise AssertionError("unexpected package manifest schema")
    if report.get("schema") != "numi.cardboard.captured419-regularized-directions.v1":
        raise AssertionError("unexpected report schema")
    if report.get("classification", "").find("CPU-only") < 0:
        raise AssertionError("package is not classified as CPU-only")
    if report["execution"].get("gpu_or_native_run") is not False:
        raise AssertionError("report claims a native/GPU run")
    if report["source_capture"].get("capture_sha256") != manifest.get("capture_sha256"):
        raise AssertionError("manifest/report capture identity mismatch")

    for item in manifest["artifacts"]:
        path = HERE / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise AssertionError(f"artifact hash/size mismatch: {item['path']}")
    if sha(report_path) != manifest.get("report_sha256"):
        raise AssertionError("report hash differs from manifest")
    if sha(HERE / "generate.py") != manifest.get("generator_sha256"):
        raise AssertionError("generator hash differs from manifest")

    # Recheck all source-bound input hashes, including the 424 base payload.
    for item in manifest["source_input_bindings"].values():
        source = Path(item["path"])
        if not source.is_file() or source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]:
            raise AssertionError(f"source input changed: {source}")
    if sha(CAPTURE) != manifest["capture_sha256"]:
        raise AssertionError("selected source125 capture changed")

    base = (HERE / "base-replay-input.bin").read_bytes()
    if sha(HERE / "base-replay-input.bin") != manifest["base_replay_input_sha256"]:
        raise AssertionError("packaged 424 replay payload is not byte-identical to bound base")
    magic, version, envs, nodes, tets, state_stride, params, fingerprint, dt, reserved = HEADER.unpack_from(base)
    layout = manifest["payload_format"]
    if (magic.decode("ascii"), version, envs, nodes, tets, state_stride, params) != (
        layout["magic"], layout["version"], layout["environments"],
        layout["nodes_per_environment"], layout["tetrahedra_per_environment"],
        layout["state_stride"], layout["parameter_count"]
    ):
        raise AssertionError("replay input header dimensions differ from manifest")
    direction_offset = int(layout["direction_offset_bytes"])
    direction_bytes = int(layout["direction_bytes"])
    if direction_offset != HEADER.size + envs * nodes * 80 or direction_bytes != envs * nodes * 16:
        raise AssertionError("native replay direction offsets violate NMFDSWP1 layout")
    if len(base) <= direction_offset + direction_bytes:
        raise AssertionError("base replay payload is truncated")

    coa = import_assembly()
    header, node_rows, tet_rows, integrity = coa.capture_records(CAPTURE)
    _, _, reconstructed, _, metadata = coa.validate_and_assemble(header, node_rows, tet_rows, integrity, 0)
    A = sparse.load_npz(CAPTURE_PACKAGE / "A_free_csr.npz").tocsr()
    difference = (A - reconstructed).tocsr()
    difference.eliminate_zeros()
    if difference.nnz or A.shape != reconstructed.shape:
        raise AssertionError("source125 reconstructed CSR differs from bound matrix artifact")
    b = np.fromfile(CAPTURE_PACKAGE / "base_rhs.float64le.bin", dtype="<f8")
    free_nodes = np.asarray(metadata["free_nodes"], dtype=np.int64)
    if A.shape != (2538, 2538) or len(free_nodes) != 846 or b.shape != (2538,):
        raise AssertionError("captured419 CSR/free-node dimensions changed")
    g = np.asarray(A.T @ b).reshape(-1)
    captured_delta = np.fromfile(CAPTURE_PACKAGE / "captured_newton_delta.float64le.bin", dtype="<f8")
    delta_norm = norm(captured_delta)
    base_norm = norm(b)

    checked = 0
    for name, recorded in report["results"].items():
        direction_file = HERE / recorded["direction_file"]
        payload_file = HERE / recorded["replay_payload_file"]
        direction = np.fromfile(direction_file, dtype="<f4")
        if direction.size != envs * nodes * 4:
            raise AssertionError(f"wrong full E*N float4 direction size: {name}")
        full = direction.reshape(envs, nodes, 4)
        if np.any(full[1]) or np.any(full[:, :, 3]):
            raise AssertionError(f"quiet environment or w lane is not zero: {name}")
        fixed = np.setdiff1d(np.arange(nodes), free_nodes)
        if np.any(full[0, fixed, :3]):
            raise AssertionError(f"fixed node direction is nonzero: {name}")
        payload = payload_file.read_bytes()
        if len(payload) != len(base):
            raise AssertionError(f"payload size changed: {name}")
        if payload[:direction_offset] != base[:direction_offset] or \
           payload[direction_offset + direction_bytes:] != base[direction_offset + direction_bytes:]:
            raise AssertionError(f"non-direction payload bytes changed: {name}")
        if payload[direction_offset:direction_offset + direction_bytes] != direction_file.read_bytes():
            raise AssertionError(f"payload/exported direction bytes differ: {name}")
        x = full[0, free_nodes, :3].reshape(-1).astype(np.float64)
        residual = b - np.asarray(A @ x).reshape(-1)
        residual_norm = norm(residual)
        phi0, phi1 = 0.5 * dot(b, b), 0.5 * dot(residual, residual)
        gp = dot(g, x)
        close(norm(x), recorded["free_correction_l2_m_per_s_after_fp32_export"], name + " correction L2")
        close(float(np.max(np.abs(x))), recorded["free_correction_linf_m_per_s_after_fp32_export"], name + " correction Linf")
        close(residual_norm, recorded["predicted_free_csr_residual_l2_Ns_after_fp32_export"], name + " residual L2")
        close(float(np.max(np.abs(residual))), recorded["predicted_free_csr_residual_linf_Ns_after_fp32_export"], name + " residual Linf")
        close(residual_norm / base_norm, recorded["predicted_residual_ratio_after_fp32_export"], name + " residual ratio")
        close((phi0 - phi1) / phi0, recorded["predicted_residual_fractional_merit_drop_after_fp32_export"], name + " merit drop")
        close(gp, recorded["g_dot_p_after_fp32_export"], name + " g dot p")
        close(-gp, recorded["least_squares_merit_directional_derivative_minus_g_dot_p"], name + " merit derivative")
        close(norm(x) / delta_norm, recorded["free_correction_over_captured_newton_delta_norm"], name + " correction ratio")
        lam = recorded["lambda"] or 0.0
        stationarity = np.asarray(A.T @ residual).reshape(-1) - float(lam) ** 2 * x
        close(norm(stationarity), recorded["regularized_normal_equation_residual_l2_after_fp32_export"], name + " stationarity")
        if lam > 0.0:
            if recorded["linear_solver"].get("istop") != 2:
                raise AssertionError(f"Tikhonov solve did not reach LSMR stop code 2: {name}")
            if not math.isclose(float(recorded["lambda_over_sigma_max"]),
                                float(recorded["lambda"] / report["proposal_definition"]["sigma_max_source"]["value"]),
                                rel_tol=1e-15, abs_tol=0.0):
                raise AssertionError(f"lambda scale mismatch: {name}")
        if gp <= 0.0:
            raise AssertionError(f"candidate is not descent for captured least-squares merit: {name}")
        checked += 1

    if checked != 4:
        raise AssertionError(f"expected four candidate payloads, checked {checked}")
    print(json.dumps({
        "result": "PASS_CPU_EXPORT_AND_CSR_RECOMPUTE",
        "candidates": checked,
        "csr_shape": list(A.shape),
        "csr_nnz": int(A.nnz),
        "free_nodes": int(len(free_nodes)),
        "base_payload_sha256": sha(HERE / "base-replay-input.bin"),
        "native_or_gpu_replay": False,
        "physical_transactions": 0,
        "certificate_scope": "captured original Newton certificate does not apply to alternative proposals",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
