#!/usr/bin/env python3
"""Create source-bound FP32 proposal directions for captured source125 case 419.

This is a CPU-only linear diagnostic. It creates replay payloads but does not
launch Metal, mutate production state, or certify any proposal as a Newton
step. See README.md for the exact bound inputs and interpretation limits.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse.linalg import lsmr

BASE = Path("/Users/home/cardboard-implementation-20261014")
OUT = Path(__file__).resolve().parent
SNAPSHOT = BASE / "snapshots/source-status5-vtccd-diagnostics-125"
ASSEMBLY = SNAPSHOT / "matter/tools/cardboard_cached_operator_audit.py"
CAPTURE_RUN = BASE / "solver/two-stage-first-failure-rich-fd-419"
CAPTURE = CAPTURE_RUN / "operator-capture"
CAPTURE_PACKAGE = BASE / "analysis/captured-base-state-cauchy-419-001"
CONDITIONING = BASE / "analysis/captured419-linear-conditioning-001"
SWEEP_PACKAGE = BASE / "analysis/native-residual-fd-sweep-419-001"
SWEEP_RESULT = BASE / "analysis/native-residual-fd-sweep-result-424"
DIRECTION_PACKAGE = BASE / "analysis/native-residual-direction-replay-419-001"
COMPILED = CAPTURE_RUN / "output/compiled.nmatterpack"
SOURCE_BUILD = Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json")

EXPECTED = {
    "capture_sha256": "f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a",
    "conditioning_manifest_sha256": "b40e54b12b5c8b08a1f3f6e246233da3daba2d760e9529d3d27fd24726dee784",
    "capture_package_manifest_sha256": "03c713a1aeeb57544d47430c7bd519b6128ea4b8243142d9e798cd114f1e0348",
    "sweep_manifest_sha256": "8fdc04c247e4fb2b8b28cd43b9ca139c345765ac68ea9888f7dc6745de6be024",
    "direction_manifest_sha256": "cbc2a5f20af28f082a381e6dc88f6df91a4d910b10e4301c9bfbd345b96d2c4b",
    "source125_build_sha256": "35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6",
    "base_replay_input_sha256": "30c472a120d7ff091d81c04e342939d571a392e02c249c8bfef8402f120852a8",
    "compiled_pack_sha256": "cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c",
}
HEADER = struct.Struct("<8s6IQfI")
ENVIRONMENTS, NODES, TETS, STATE_STRIDE, PARAMETER_COUNT = 2, 882, 1920, 15, 59
DIRECTION_OFFSET = HEADER.size + ENVIRONMENTS * NODES * 80
DIRECTION_BYTES = ENVIRONMENTS * NODES * 4 * 4
SIGMA_RATIOS = (0.01, 0.001, 0.0001)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dot(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError("dot shape mismatch")
    return math.fsum(float(x) * float(y) for x, y in zip(aa, bb))


def norm(a: np.ndarray) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("source125_cardboard_operator_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import bound source125 assembly implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def artifact(path: Path, relative_to: Path = OUT) -> dict:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def main() -> int:
    start = time.monotonic()
    if (OUT / "manifest.json").exists() or (OUT / "report.json").exists():
        raise SystemExit("refusing to overwrite a sealed result directory")

    input_paths = {
        "capture": CAPTURE,
        "source125_assembly_module": ASSEMBLY,
        "source125_build_manifest": SOURCE_BUILD,
        "capture_prediction": CAPTURE_RUN / "prediction.json",
        "capture_receipt": CAPTURE_RUN / "receipt.json",
        "capture_stderr": CAPTURE_RUN / "stderr.log",
        "capture_package_manifest": CAPTURE_PACKAGE / "manifest.json",
        "capture_package_report": CAPTURE_PACKAGE / "report.json",
        "capture_csr": CAPTURE_PACKAGE / "A_free_csr.npz",
        "capture_rhs": CAPTURE_PACKAGE / "base_rhs.float64le.bin",
        "capture_direction_reference": CAPTURE_PACKAGE / "base_raw_cauchy.full-node-delta-v.float32x4.bin",
        "conditioning_manifest": CONDITIONING / "manifest.json",
        "conditioning_report": CONDITIONING / "report.json",
        "base_sweep_manifest": SWEEP_PACKAGE / "manifest.json",
        "base_sweep_replay_input": SWEEP_PACKAGE / "replay-input.bin",
        "424_prediction": SWEEP_RESULT / "prediction.json",
        "424_receipt": SWEEP_RESULT / "receipt.json",
        "native_direction_manifest": DIRECTION_PACKAGE / "manifest.json",
        "compiled_package": COMPILED,
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing bound input {label}: {path}")
    hashes = {label: sha(path) for label, path in input_paths.items()}
    if hashes["capture"] != EXPECTED["capture_sha256"]:
        raise ValueError("operator capture no longer matches frozen capture419 identity")
    if hashes["conditioning_manifest"] != EXPECTED["conditioning_manifest_sha256"]:
        raise ValueError("prior sealed conditioning package differs from expected identity")
    if hashes["capture_package_manifest"] != EXPECTED["capture_package_manifest_sha256"]:
        raise ValueError("captured Cauchy package differs from expected identity")
    if hashes["base_sweep_manifest"] != EXPECTED["sweep_manifest_sha256"]:
        raise ValueError("native 424 sweep payload package differs from expected identity")
    if hashes["native_direction_manifest"] != EXPECTED["direction_manifest_sha256"]:
        raise ValueError("native 429 direction replay package differs from expected identity")
    if hashes["base_sweep_replay_input"] != EXPECTED["base_replay_input_sha256"]:
        raise ValueError("native 424 base replay payload changed")
    if hashes["compiled_package"] != EXPECTED["compiled_pack_sha256"]:
        raise ValueError("compiled source125 package changed")
    source_build = json.loads(SOURCE_BUILD.read_text())
    if source_build.get("schema") != "numi.cardboard.source-build.v1":
        raise ValueError("unexpected source-build manifest schema")
    source_build_sha = sha(SOURCE_BUILD)
    if source_build_sha != EXPECTED["source125_build_sha256"]:
        raise ValueError("source125 build manifest differs from compiled replay identity")
    if source_build.get("source_snapshot") != str(SNAPSHOT):
        raise ValueError("source-build manifest does not bind source125 snapshot")

    conditioning = json.loads((CONDITIONING / "report.json").read_text())
    if conditioning.get("source_capture_bindings", {}).get("capture_sha256") != hashes["capture"]:
        raise ValueError("spectral scale is not bound to selected capture")
    if conditioning.get("source_capture_bindings", {}).get("input_hashes", {}).get("A_free_csr.npz") != hashes["capture_csr"]:
        raise ValueError("spectral scale report references a different CSR operator")
    sigma_max = float(conditioning["spectrum"]["sigma_max"])
    if not math.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("invalid source-bound largest singular value")

    coa = load_module(ASSEMBLY)
    capture_binding = coa.bind_inputs(
        CAPTURE,
        CAPTURE_RUN / "prediction.json",
        CAPTURE_RUN / "receipt.json",
        CAPTURE_RUN / "stderr.log",
        0,
        source_snapshot_override=SNAPSHOT,
    )
    header, nodes, tets, integrity = coa.capture_records(CAPTURE)
    if (int(header.get("root", -1)), int(header.get("newton_iteration", -1))) != (1, 48):
        raise ValueError("wrong source125 capture selection")
    _, _, operator, captured_delta, metadata = coa.validate_and_assemble(
        header, nodes, tets, integrity, 0
    )
    A = operator.tocsr()
    b = np.asarray(metadata["native_rhs_free_impulse_Ns"], dtype=np.float64).reshape(-1)
    free_nodes = np.asarray(metadata["free_nodes"], dtype=np.int64)
    node_count = int(metadata["node_count"])
    if A.shape != (2538, 2538) or b.shape != (2538,) or len(free_nodes) != 846:
        raise ValueError("captured 419 free operator dimensions changed")
    if A.nnz != 71046:
        raise ValueError("captured 419 CSR nonzero count changed")
    b_source = np.fromfile(CAPTURE_PACKAGE / "base_rhs.float64le.bin", dtype="<f8")
    if b_source.shape != b.shape or not np.array_equal(b_source, b):
        raise ValueError("source-bound CSR RHS differs from captured package artifact")
    if not np.array_equal(captured_delta, np.fromfile(
        CAPTURE_PACKAGE / "captured_newton_delta.float64le.bin", dtype="<f8")):
        raise ValueError("captured Newton delta differs from bound package")

    g = np.asarray(A.T @ b).reshape(-1)
    if not np.isfinite(g).all() or norm(g) == 0.0:
        raise ValueError("captured RHS gradient is zero or nonfinite")

    proposals: dict[str, dict] = {}
    # Raw Cauchy control from the same unregularized least-squares merit.
    Ag = np.asarray(A @ g).reshape(-1)
    denom = dot(Ag, Ag)
    if denom <= 0.0 or not math.isfinite(denom):
        raise ValueError("raw Cauchy denominator is not positive finite")
    raw_alpha = dot(g, g) / denom
    proposals["raw_cauchy"] = {
        "formula": "p = (g^T g / (A g)^T(A g)) g, g=A^T b",
        "lambda": None,
        "solve": {"method": "closed-form Cauchy slope", "iterations": 1},
        "proposal_f64": raw_alpha * g,
    }

    for ratio in SIGMA_RATIOS:
        lam = ratio * sigma_max
        # SciPy LSMR with damp=lambda solves min ||A p-b||^2 + lambda^2||p||^2
        # directly on the exact unsymmetrized CSR operator, without forming A^T A.
        result = lsmr(A, b, damp=lam, atol=1.0e-13, btol=1.0e-13,
                      conlim=1.0e16, maxiter=20000, show=False)
        p64 = np.asarray(result[0], dtype=np.float64)
        if p64.shape != b.shape or not np.isfinite(p64).all():
            raise ValueError(f"Tikhonov solve failed for lambda/sigma={ratio:g}")
        proposals[f"tikhonov_sigma_{ratio:g}"] = {
            "formula": "argmin_p ||A p-b||_2^2 + lambda^2 ||p||_2^2; lambda=ratio*sigma_max",
            "lambda": lam,
            "solve": {
                "method": "scipy.sparse.linalg.lsmr on exact unsymmetrized CSR with damp=lambda",
                "atol": 1.0e-13,
                "btol": 1.0e-13,
                "conlim": 1.0e16,
                "maxiter": 20000,
                "istop": int(result[1]),
                "iterations": int(result[2]),
                "reported_residual_norm": float(result[3]),
                "reported_normal_residual_norm": float(result[4]),
                "reported_operator_norm_estimate": float(result[5]),
                "reported_condition_estimate": float(result[6]),
                "reported_solution_norm": float(result[7]),
            },
            "proposal_f64": p64,
        }

    base_payload = (SWEEP_PACKAGE / "replay-input.bin").read_bytes()
    if len(base_payload) != 2_195_272:
        raise ValueError(f"unexpected 424 replay payload size: {len(base_payload)}")
    magic, version, envs, nodes_count, tets_count, stride, params, fingerprint, dt, reserved = HEADER.unpack_from(base_payload)
    if (magic, version, envs, nodes_count, tets_count, stride, params) != (
        b"NMFDSWP1", 1, ENVIRONMENTS, NODES, TETS, STATE_STRIDE, PARAMETER_COUNT
    ):
        raise ValueError("native residual replay header is not source125 capture419 format")
    if DIRECTION_OFFSET + DIRECTION_BYTES > len(base_payload):
        raise ValueError("direction buffer is outside the bound replay payload")
    original_direction_bytes = base_payload[DIRECTION_OFFSET:DIRECTION_OFFSET + DIRECTION_BYTES]
    if len(original_direction_bytes) != DIRECTION_BYTES:
        raise ValueError("truncated source125 direction payload")

    report_candidates = {}
    output_artifacts = []
    expected_files = [OUT / "base-replay-input.bin"]
    shutil.copyfile(SWEEP_PACKAGE / "replay-input.bin", OUT / "base-replay-input.bin")

    for name, proposal in proposals.items():
        p64 = np.asarray(proposal.pop("proposal_f64"), dtype=np.float64).reshape(-1)
        if p64.shape != (2538,) or not np.isfinite(p64).all():
            raise ValueError(f"proposal {name} has invalid FP64 dimensions or values")
        # Export only the actually replayed binary32 proposal. Re-evaluate CSR
        # metrics from this exact rounded free vector after serialization.
        full = np.zeros((ENVIRONMENTS, NODES, 4), dtype="<f4")
        rounded = p64.astype("<f4")
        if not np.isfinite(rounded).all():
            raise ValueError(f"proposal {name} overflows in FP32 export")
        full[0, free_nodes, :3] = rounded.reshape(-1, 3)
        # Fourth lane and the quiet/reference environment remain exactly zero.
        if np.any(full[1]) or np.any(full[0, :, 3]):
            raise AssertionError("full E*N float4 export contains nonzero reserved lanes")
        if name == "raw_cauchy":
            prior_raw = np.fromfile(
                CAPTURE_PACKAGE / "base_raw_cauchy.full-node-delta-v.float32x4.bin",
                dtype="<f4").reshape(NODES, 4)
            if prior_raw.tobytes() != full[0].tobytes():
                raise AssertionError("regenerated raw Cauchy differs from sealed capture419 proposal")
        direction_path = OUT / "directions" / f"{name}.float32x4.bin"
        full.tofile(direction_path)
        direction_bytes = direction_path.read_bytes()
        if len(direction_bytes) != DIRECTION_BYTES:
            raise AssertionError("exported E*N float4 direction has wrong byte length")

        replay = bytearray(base_payload)
        replay[DIRECTION_OFFSET:DIRECTION_OFFSET + DIRECTION_BYTES] = direction_bytes
        replay_path = OUT / "replay-inputs" / f"{name}.input.bin"
        replay_path.write_bytes(replay)
        # Prove this payload differs from the source 424 replay only in the
        # direction buffer, and that readback equals the exported words.
        reread = replay_path.read_bytes()
        if reread[:DIRECTION_OFFSET] != base_payload[:DIRECTION_OFFSET] or \
           reread[DIRECTION_OFFSET + DIRECTION_BYTES:] != base_payload[DIRECTION_OFFSET + DIRECTION_BYTES:]:
            raise AssertionError("candidate payload changed non-direction source125 bytes")
        if reread[DIRECTION_OFFSET:DIRECTION_OFFSET + DIRECTION_BYTES] != direction_bytes:
            raise AssertionError("candidate payload direction differs from exported FP32 words")
        readback = np.frombuffer(reread[DIRECTION_OFFSET:DIRECTION_OFFSET + DIRECTION_BYTES], dtype="<f4").reshape(ENVIRONMENTS, NODES, 4)
        x32 = readback[0, free_nodes, :3].reshape(-1).astype(np.float64)
        if not np.array_equal(x32.astype("<f4"), rounded):
            raise AssertionError("FP32 free direction did not roundtrip exactly")

        residual64 = b - np.asarray(A @ x32).reshape(-1)
        residual_norm = norm(residual64)
        base_norm = norm(b)
        phi_before = 0.5 * dot(b, b)
        phi_after = 0.5 * dot(residual64, residual64)
        g_dot_p = dot(g, x32)
        grad_norm = norm(g)
        x_norm = norm(x32)
        scale = float(proposal.get("lambda") or 0.0)
        normal_residual = np.asarray(A.T @ residual64).reshape(-1) - scale * scale * x32
        fp64_residual = b - np.asarray(A @ p64).reshape(-1)
        metric = {
            "proposal_formula": proposal["formula"],
            "lambda": proposal["lambda"],
            "lambda_over_sigma_max": None if proposal["lambda"] is None else float(proposal["lambda"] / sigma_max),
            "linear_solver": proposal["solve"],
            "direction_file": str(direction_path.relative_to(OUT)),
            "direction_sha256": sha(direction_path),
            "replay_payload_file": str(replay_path.relative_to(OUT)),
            "replay_payload_sha256": sha(replay_path),
            "free_correction_l2_m_per_s_after_fp32_export": x_norm,
            "free_correction_linf_m_per_s_after_fp32_export": float(np.max(np.abs(x32))),
            "fp64_unrounded_proposal_l2_m_per_s": norm(p64),
            "fp32_export_vs_fp64_proposal_l2_m_per_s": norm(x32 - p64),
            "base_rhs_l2_Ns": base_norm,
            "predicted_free_csr_residual_l2_Ns_after_fp32_export": residual_norm,
            "predicted_free_csr_residual_linf_Ns_after_fp32_export": float(np.max(np.abs(residual64))),
            "predicted_residual_ratio_after_fp32_export": residual_norm / base_norm,
            "predicted_residual_fractional_merit_drop_after_fp32_export": (phi_before - phi_after) / phi_before,
            "g_dot_p_after_fp32_export": g_dot_p,
            "least_squares_merit_directional_derivative_minus_g_dot_p": -g_dot_p,
            "descent_angle_cosine_g_p": g_dot_p / (grad_norm * x_norm) if grad_norm and x_norm else None,
            "regularized_normal_equation_residual_l2_after_fp32_export": norm(normal_residual),
            "fp64_unrounded_predicted_residual_l2_Ns": norm(fp64_residual),
            "fp32_rounding_residual_change_l2_Ns": norm(residual64 - fp64_residual),
            "full_payload_shape": [ENVIRONMENTS, NODES, 4],
            "quiet_environment_direction_all_zero": bool(np.count_nonzero(readback[1]) == 0),
            "fixed_node_xyz_all_zero": bool(np.count_nonzero(readback[0, np.setdiff1d(np.arange(NODES), free_nodes), :3]) == 0),
            "w_lane_all_zero": bool(np.count_nonzero(readback[:, :, 3]) == 0),
            "non_direction_payload_bytes_bitwise_unchanged": True,
        }
        report_candidates[name] = metric
        output_artifacts.extend([artifact(direction_path), artifact(replay_path)])
        expected_files.extend([direction_path, replay_path])

    output_artifacts.append(artifact(OUT / "base-replay-input.bin"))
    source_inputs = {label: {"path": str(path), "sha256": hashes[label], "bytes": path.stat().st_size}
                     for label, path in input_paths.items()}
    report = {
        "schema": "numi.cardboard.captured419-regularized-directions.v1",
        "classification": "CPU-only linear-algebra proposal export; no native replay was run and no production solver change is proposed",
        "source_capture": {
            "capture_sha256": hashes["capture"],
            "selection": {"control_step": 1, "newton_iteration": 48, "environment": 0},
            "source_snapshot": str(SNAPSHOT),
            "source_build_manifest_sha256": source_build_sha,
            "source_build_sha256": EXPECTED["source125_build_sha256"],
            "compiled_nmatterpack_sha256": hashes["compiled_package"],
            "native_replay_format": "NMFDSWP1 v1 from source125 native residual replay 424/429; E=2,N=882,T=1920,stateStride=15,parameterCount=59",
            "base_payload_sha256": hashes["base_sweep_replay_input"],
            "base_replay_manifest_sha256": hashes["base_sweep_manifest"],
            "native_direction_replay_manifest_sha256": hashes["native_direction_manifest"],
            "world_fingerprint_hex": f"{fingerprint:016x}",
            "dt_seconds_float32": float(dt),
            "free_dofs": int(A.shape[0]),
            "free_nodes": int(len(free_nodes)),
            "total_nodes_per_environment": NODES,
            "csr_shape": list(A.shape),
            "csr_nnz": int(A.nnz),
            "operator_definition": metadata["assembly"],
        },
        "proposal_definition": {
            "base_residual": "b is the captured pre-FGMRES free mechanical impulse RHS at the source125 1:48 candidate",
            "proposal_unknown": "p is the total alternative velocity correction at that captured center, not delta_v plus a correction",
            "tikhonov_objective": "minimize ||A p-b||_2^2 + lambda^2 ||p||_2^2 using the exact unsymmetrized captured CSR matrix; lambda ratios are relative to source-bound sigma_max",
            "sigma_max_source": {"value": sigma_max, "report_sha256": hashes["conditioning_report"], "method": conditioning["spectrum"].get("method", "source-bound prior conditioning report")},
            "raw_cauchy_objective": "unregularized least-squares merit 0.5||b-Ap||^2; p=(g^Tg)/(Ag)^T(Ag) g, g=A^T b",
            "metric_precision": "each predicted residual, correction norm, and g dot p below is recomputed from the exact CSR after the exported full payload has been rounded to FP32 and read back",
            "payload_mutation_contract": "Only the E*N float4 direction region differs from exact native 424 replay-input.bin; all candidate/history/parameters/tangent/tetrahedron/accepted-node bytes remain bitwise unchanged",
            "no_physical_claim": "These proposals are not accepted Newton corrections; the original captured Newton residual/certificate does not certify any proposal. Native production residual replay, nonlinear acceptance, CCD, and terminal certification remain unrun for these directions.",
        },
        "input_bindings": {"source_files": source_inputs, "source125_capture_binding": capture_binding},
        "results": report_candidates,
        "limits": [
            "A is a reconstructed source125 capture CSR at one rejected Newton candidate, not the current operator after changing the nonlinear state.",
            "The Tikhonov and Cauchy vectors are diagnostic alternative directions only; no production regularization or fallback path was enabled.",
            "The exact captured Newton certificate applies only to its original direction and candidate. It does not certify these new vectors or any nonlinear/CCD trial.",
            "FP32 CSR metrics are mathematical predictions on the exported rounded vector; they do not model native kernel summation, material reprojection, Armijo, contact, or rollback behavior.",
            "No GPU/native replay was executed by this package.",
        ],
        "execution": {"gpu_or_native_run": False, "source_modified": False, "blas_thread_limit": 1,
                      "elapsed_seconds": time.monotonic() - start},
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_artifacts.append(artifact(report_path))
    script_artifact = artifact(Path(__file__))
    manifest = {
        "schema": "numi.cardboard.captured419-regularized-directions-package.v1",
        "package": OUT.name,
        "classification": report["classification"],
        "report_sha256": sha(report_path),
        "generator_sha256": script_artifact["sha256"],
        "source125_build_sha256": EXPECTED["source125_build_sha256"],
        "capture_sha256": hashes["capture"],
        "base_replay_input_sha256": hashes["base_sweep_replay_input"],
        "compiled_package_sha256": hashes["compiled_package"],
        "proposal_ratios": list(SIGMA_RATIOS),
        "payload_format": {"magic": "NMFDSWP1", "version": 1, "environments": ENVIRONMENTS,
                           "nodes_per_environment": NODES, "tetrahedra_per_environment": TETS,
                           "state_stride": STATE_STRIDE, "parameter_count": PARAMETER_COUNT,
                           "direction_offset_bytes": DIRECTION_OFFSET,
                           "direction_bytes": DIRECTION_BYTES,
                           "direction_dtype": "little-endian float32 [environment,node,xyzw]"},
        "physical_transactions": 0,
        "native_replay_executed": False,
        "artifacts": sorted(output_artifacts, key=lambda item: item["path"]),
        "source_input_bindings": source_inputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": "EXPORTED_CPU_PROPOSALS_UNQUALIFIED", "report": str(report_path),
                      "report_sha256": sha(report_path), "manifest_sha256": sha(OUT / "manifest.json"),
                      "metrics": {k: {"residual_ratio": v["predicted_residual_ratio_after_fp32_export"],
                                      "g_dot_p": v["g_dot_p_after_fp32_export"],
                                      "correction_l2": v["free_correction_l2_m_per_s_after_fp32_export"]}
                                  for k, v in report_candidates.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
