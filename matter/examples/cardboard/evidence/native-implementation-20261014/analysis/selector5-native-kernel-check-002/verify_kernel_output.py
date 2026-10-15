#!/usr/bin/env python3
"""Complete, warning-free CPU audit for selector-5 kernel output bundles.

Numerical findings are accumulated per patch instead of raising at the first
failed threshold. Dot products and vector norms use math.fsum; no NumPy
matmul/dot is used. SciPy is used only for a separate FP64 direct solve and
condition estimate that help classify a local residual failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm as scipy_norm
from scipy.linalg.lapack import get_lapack_funcs

INPUT_MAGIC = b"NMSGKC1\0"
OUTPUT_MAGIC = b"NMGKO001"
HEADER_FMT = "<11I"
ARRAY_SPECS = (
    ("rap_lu", "<f4", lambda h: h[3] * h[3]),
    ("rap_pivots", "<u4", lambda h: h[3]),
    ("rap_valid", "<u4", lambda h: 1),
    ("basis", "<f4", lambda h: h[1] * 4),
    ("regional_basis", "<f4", lambda h: h[3] * h[1] * 4),
    ("regional_ap", "<f4", lambda h: h[3] * h[1] * 4),
    ("patches", "<u4", lambda h: h[4] * 4),
    ("patch_nodes", "<u4", lambda h: h[5]),
    ("node_patch_offsets", "<u4", lambda h: h[1] + 1),
    ("node_patch_values", "<u4", lambda h: h[5]),
    ("node_coverage", "<u4", lambda h: h[1]),
    ("patch_matrices", "<f4", lambda h: h[7]),
)
OUTPUT_SPECS = (
    ("coarse_rhs", "<f4", lambda h: h[3]),
    ("coefficients", "<f4", lambda h: h[3]),
    ("patch_lu", "<f4", lambda h: h[7]),
    ("patch_pivots", "<u4", lambda h: h[8]),
    ("factor_valid", "<u4", lambda h: h[4]),
    ("corrections", "<f4", lambda h: h[5] * 4),
    ("preconditioned", "<f4", lambda h: h[1] * 4),
    ("status", "<u4", lambda h: 12),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _norm_fsum(values: Any) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    return math.sqrt(math.fsum(float(x) * float(x) for x in vals))


def _relative_error_fsum(actual: Any, expected: Any) -> float:
    a = np.asarray(actual, dtype=np.float64).ravel()
    e = np.asarray(expected, dtype=np.float64).ravel()
    require(a.shape == e.shape, "relative-error shape mismatch")
    return _norm_fsum(a - e) / max(_norm_fsum(e), 1e-300)


def _dot_fsum(a: Any, b: Any) -> float:
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def _factor_solve_fsum(lu: np.ndarray, pivots: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = int(rhs.size)
    require(lu.shape == (n, n), "LU/RHS shape mismatch")
    require(sorted(int(x) for x in pivots) == list(range(n)),
            "pivot array is not a permutation")
    values = np.asarray(rhs, dtype=np.float64)[pivots.astype(np.int64)].copy()
    for row in range(n):
        values[row] -= _dot_fsum(lu[row, :row], values[:row])
    for row in range(n - 1, -1, -1):
        values[row] -= _dot_fsum(lu[row, row + 1:], values[row + 1:])
        diagonal = float(lu[row, row])
        require(math.isfinite(diagonal) and diagonal != 0.0,
                "LU diagonal is invalid")
        values[row] /= diagonal
    return values


def _decode_arrays(raw: bytes, header: tuple[int, ...], specs: tuple) -> dict[str, np.ndarray]:
    cursor = 8 + struct.calcsize(HEADER_FMT)
    result: dict[str, np.ndarray] = {}
    for name, dtype, count_fn in specs:
        count = int(count_fn(header))
        byte_count = count * np.dtype(dtype).itemsize
        require(cursor + byte_count <= len(raw), f"truncated array: {name}")
        result[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor).copy()
        cursor += byte_count
    require(cursor == len(raw), "bundle has trailing or unaccounted bytes")
    return result


def validate_bundle_manifest(input_raw: bytes, bundle: dict) -> dict:
    require(bundle.get("bundle_sha256") == sha256(input_raw),
            "input bytes do not match source-bound bundle manifest")
    require(bundle.get("schema") == "numi.matter.selector5-kernel-check-input.v1" and
            bundle.get("source_capture_selector") == 3 and
            bundle.get("native_selector5_capture_present") is False,
            "bundle scope/identity differs from checker contract")
    require(input_raw[:8] == INPUT_MAGIC and len(input_raw) >= 8 + struct.calcsize(HEADER_FMT),
            "input header is invalid")
    h = struct.unpack_from(HEADER_FMT, input_raw, 8)
    _, nodes, tets, modes, patches, patch_nodes, dofs, matrix_floats, pivots, envs, selected = h
    expected_dims = {
        "node_count": nodes, "tetrahedron_count": tets, "mode_count": modes,
        "patch_count": patches, "patch_node_count": patch_nodes, "dof_count": dofs,
        "matrix_float_count": matrix_floats, "pivot_count": pivots,
        "source_environment_count": envs, "selected_environment": selected,
    }
    require(bundle.get("dimensions") == expected_dims,
            "binary dimensions differ from bundle manifest")
    descriptors = bundle.get("arrays")
    require(isinstance(descriptors, list) and len(descriptors) == len(ARRAY_SPECS),
            "bundle array descriptors are missing or incomplete")
    expected_shapes = {
        "rap_lu": [modes, modes], "rap_pivots": [modes], "rap_valid": [1],
        "basis": [nodes, 4], "regional_basis": [modes, nodes, 4],
        "regional_ap": [modes, nodes, 4], "patches": [patches, 4],
        "patch_nodes": [patch_nodes], "node_patch_offsets": [nodes + 1],
        "node_patch_values": [patch_nodes], "node_coverage": [nodes],
        "patch_matrices": [matrix_floats],
    }
    cursor = 8 + struct.calcsize(HEADER_FMT)
    for descriptor, (name, dtype, count_fn) in zip(descriptors, ARRAY_SPECS):
        byte_count = int(count_fn(h)) * np.dtype(dtype).itemsize
        require(descriptor.get("name") == name and descriptor.get("dtype") == dtype,
                f"bundle descriptor order/type mismatch at {name}")
        require(descriptor.get("shape") == expected_shapes[name] and
                descriptor.get("offset") == cursor and
                descriptor.get("byte_count") == byte_count,
                f"bundle descriptor bounds/shape mismatch at {name}")
        end = cursor + byte_count
        require(end <= len(input_raw) and
                descriptor.get("sha256") == sha256(input_raw[cursor:end]),
                f"bundle descriptor SHA-256 mismatch at {name}")
        cursor = end
    require(cursor == len(input_raw), "bundle descriptors do not consume exact input bytes")
    return expected_dims


def _check(result: dict, name: str, value: float | bool, tolerance: float,
           details: str | None = None) -> None:
    passed = bool(value) if isinstance(value, (bool, np.bool_)) else (
        math.isfinite(float(value)) and float(value) <= tolerance)
    result["checks"].append({
        "name": name, "passed": passed,
        "value": bool(value) if isinstance(value, (bool, np.bool_)) else float(value),
        "tolerance": tolerance,
        **({"details": details} if details else {}),
    })


def _as_fp32_rhs(nodes: np.ndarray, arrays: dict[str, np.ndarray],
                 coefficients: np.ndarray, mode_count: int) -> np.ndarray:
    node_count = arrays["basis"].size // 4
    basis = arrays["basis"].reshape(node_count, 4)
    regional_ap = arrays["regional_ap"].reshape(mode_count, node_count, 4)
    result: list[float] = []
    for node in nodes:
        for axis in range(3):
            value = np.float32(basis[int(node), axis])
            for mode in range(mode_count):
                product = np.float32(coefficients[mode] * regional_ap[mode, int(node), axis])
                value = np.float32(value - product)
            result.append(float(value))
    return np.asarray(result, dtype=np.float64)


def _right_hand_side(nodes: np.ndarray, arrays: dict[str, np.ndarray],
                     coefficients: np.ndarray, mode_count: int) -> np.ndarray:
    node_count = arrays["basis"].size // 4
    basis = arrays["basis"].reshape(node_count, 4).astype(np.float64)
    regional_ap = arrays["regional_ap"].reshape(mode_count, node_count, 4).astype(np.float64)
    coeff = np.asarray(coefficients, dtype=np.float64)
    return np.asarray([
        math.fsum([float(basis[int(node), axis]), *(
            -float(coeff[mode]) * float(regional_ap[mode, int(node), axis])
            for mode in range(mode_count))])
        for node in nodes for axis in range(3)
    ], dtype=np.float64)


def _residual_fsum(matrix: np.ndarray, x: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.asarray([
        _dot_fsum(row, x) - float(rhs[i])
        for i, row in enumerate(matrix)
    ], dtype=np.float64)


def _factor_reconstruction(lu: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    n = int(pivots.size)
    lower = np.tril(lu, -1)
    lower[np.diag_indices(n)] = 1.0
    upper = np.triu(lu)
    # einsum's unoptimized contraction avoids the platform BLAS matmul path.
    # All values are FP64 here, and the result is checked for finiteness.
    permuted = np.einsum("ik,kj->ij", lower, upper, optimize=False)
    require(np.isfinite(permuted).all(), "LU reconstruction produced nonfinite values")
    matrix = np.empty_like(permuted)
    matrix[pivots.astype(np.int64), :] = permuted
    return matrix


def verify(input_raw: bytes, output_raw: bytes, tolerance: float = 1e-3) -> dict:
    require(input_raw[:8] == INPUT_MAGIC, "input magic mismatch")
    require(output_raw[:8] == OUTPUT_MAGIC, "output magic mismatch")
    header_bytes = struct.calcsize(HEADER_FMT)
    require(len(input_raw) >= 8 + header_bytes and len(output_raw) >= 8 + header_bytes,
            "bundle header truncated")
    ih = struct.unpack_from(HEADER_FMT, input_raw, 8)
    oh = struct.unpack_from(HEADER_FMT, output_raw, 8)
    require(ih == oh, "output dimensions differ from input")
    version, node_count, tet_count, mode_count, patch_count, patch_node_count, dof_count, matrix_count, pivot_count, source_env_count, selected_env = ih
    require(version == 1 and node_count > 0 and tet_count > 0 and mode_count <= 32 and
            patch_count > 0 and patch_node_count > 0 and dof_count == patch_node_count * 3 and
            pivot_count == dof_count and source_env_count == 2 and selected_env == 0,
            "unsupported or inconsistent input dimensions")

    inputs = _decode_arrays(input_raw, ih, ARRAY_SPECS)
    outputs = _decode_arrays(output_raw, oh, OUTPUT_SPECS)
    require(all(np.isfinite(a).all() for a in inputs.values() if a.dtype.kind == "f"),
            "input contains nonfinite floating-point values")
    require(all(np.isfinite(a).all() for a in outputs.values() if a.dtype.kind == "f"),
            "output contains nonfinite floating-point values")

    report: dict[str, Any] = {
        "schema": "numi.matter.selector5_kernel_output_audit.v2",
        "status": "arithmetic_checks_failed",
        "source_scope": "source182 selector3 RAP/operator inputs plus actual-topology graph layout; source067 production Metal selector5 point kernels",
        "dimensions": {"node_count": node_count, "tetrahedron_count": tet_count,
                       "mode_count": mode_count, "patch_count": patch_count,
                       "patch_node_count": patch_node_count, "dof_count": dof_count,
                       "matrix_float_count": matrix_count},
        "input_sha256": sha256(input_raw), "output_sha256": sha256(output_raw),
        "checks": [], "patch_results": [], "diagnostics": {},
        "limits": [
            "This is prepared-kernel arithmetic, not an end-to-end FGMRES or full-selector-5 FEM run.",
            "Patch matrices are exact principal blocks from the source182 audited FP32 CSR operator; native FEM patch assembly is bypassed.",
            "Direct FP64 patch solves are CPU diagnostic references, not part of the native production operator.",
        ],
    }
    status_code = int(outputs["status"][0])
    _check(report, "native_status_zero", status_code == 0, tolerance,
           f"status_code={status_code}")
    factor_valid = outputs["factor_valid"]
    for patch_index, valid in enumerate(factor_valid):
        _check(report, f"patch_{patch_index}_factor_valid", int(valid) == 1, tolerance)
    _check(report, "captured_rap_valid", int(inputs["rap_valid"][0]) == 1, tolerance)

    M, N = mode_count, node_count
    rap_lu = inputs["rap_lu"].reshape(M, M).astype(np.float64)
    rap_pivots = inputs["rap_pivots"].astype(np.int64)
    rap = _factor_reconstruction(rap_lu, rap_pivots)
    P = inputs["regional_basis"].reshape(M, N, 4)[:, :, :3].astype(np.float64)
    AP = inputs["regional_ap"].reshape(M, N, 4)[:, :, :3].astype(np.float64)
    basis = inputs["basis"].reshape(N, 4)[:, :3].astype(np.float64)
    expected_rhs = np.asarray([
        math.fsum(float(P[m, node, axis]) * float(basis[node, axis])
                  for node in range(N) for axis in range(3))
        for m in range(M)
    ])
    actual_rhs = outputs["coarse_rhs"].astype(np.float64)
    expected_coeff = _factor_solve_fsum(rap_lu, rap_pivots, expected_rhs)
    actual_coeff = outputs["coefficients"].astype(np.float64)
    expected_rap = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            expected_rap[i, j] = math.fsum(
                float(P[i, node, axis]) * float(AP[j, node, axis])
                for node in range(N) for axis in range(3))
    rap_err = _relative_error_fsum(rap, expected_rap)
    coarse_err = _relative_error_fsum(actual_rhs, expected_rhs)
    coefficient_err = _relative_error_fsum(actual_coeff, expected_coeff)
    _check(report, "RAP_LU_reconstruction_vs_PtAP", rap_err, tolerance)
    _check(report, "coarse_projection", coarse_err, tolerance)
    _check(report, "RAP_solve", coefficient_err, tolerance)

    patches = inputs["patches"].reshape(patch_count, 4).astype(np.int64)
    patch_nodes = inputs["patch_nodes"].astype(np.int64)
    original_matrices = inputs["patch_matrices"].astype(np.float64)
    local_lus = outputs["patch_lu"].astype(np.float64)
    local_pivots = outputs["patch_pivots"].astype(np.int64)
    corrections = outputs["corrections"].reshape(patch_node_count, 4).astype(np.float64)
    maximum_factor_error = 0.0
    maximum_native_residual = 0.0
    maximum_fp64lu_residual = 0.0
    maximum_native_vs_fp64lu = 0.0
    maximum_native_vs_fp64matrix = 0.0
    for patch_index, (node_offset, nodes_count, matrix_offset, pivot_offset) in enumerate(patches):
        n = int(nodes_count) * 3
        matrix_count_local = n * n
        local_nodes = patch_nodes[int(node_offset):int(node_offset + nodes_count)]
        original = original_matrices[int(matrix_offset):int(matrix_offset + matrix_count_local)].reshape(n, n)
        lu = local_lus[int(matrix_offset):int(matrix_offset + matrix_count_local)].reshape(n, n)
        pivots = local_pivots[int(pivot_offset):int(pivot_offset + n)]
        native_x = corrections[int(node_offset):int(node_offset + nodes_count), :3].reshape(n)
        finite = bool(np.isfinite(original).all() and np.isfinite(lu).all() and
                      np.isfinite(native_x).all())
        if not finite:
            detail = {"patch_index": patch_index, "node_count": int(nodes_count),
                      "finite": False, "factor_reconstruction_relative": None,
                      "native_solve_residual_relative": None}
            report["patch_results"].append(detail)
            _check(report, f"patch_{patch_index}_finite", False, tolerance)
            continue
        factor_error = _relative_error_fsum(_factor_reconstruction(lu, pivots), original)
        maximum_factor_error = max(maximum_factor_error, factor_error)
        rhs = _right_hand_side(local_nodes, inputs, actual_coeff, mode_count)
        rhs_fp32 = _as_fp32_rhs(local_nodes, inputs, outputs["coefficients"], mode_count)
        native_residual = _residual_fsum(original, native_x, rhs)
        native_residual_rel = _norm_fsum(native_residual) / max(_norm_fsum(rhs), 1e-300)
        lu_solution = _factor_solve_fsum(lu, pivots, rhs)
        lu_residual_rel = _norm_fsum(_residual_fsum(original, lu_solution, rhs)) / max(_norm_fsum(rhs), 1e-300)
        native_vs_lu = _relative_error_fsum(native_x, lu_solution)
        fp32_solve = _factor_solve_fsum(lu, pivots, rhs_fp32)
        fp32_residual_rel = _norm_fsum(_residual_fsum(original, fp32_solve, rhs_fp32)) / max(_norm_fsum(rhs_fp32), 1e-300)
        fp32_rhs_delta = _relative_error_fsum(rhs_fp32, rhs)
        native_vs_fp32_solve = _relative_error_fsum(native_x, fp32_solve)
        # Independent FP64 direct solve of the original principal block
        # separates FP32 factorization error from triangular-solve rounding.
        try:
            direct_lu, direct_pivots = lu_factor(original, check_finite=True)
            direct_x = lu_solve((direct_lu, direct_pivots), rhs, check_finite=True)
            direct_residual_rel = _norm_fsum(_residual_fsum(original, direct_x, rhs)) / max(_norm_fsum(rhs), 1e-300)
            rcond, info = get_lapack_funcs("gecon", (direct_lu,))(
                direct_lu, scipy_norm(original, 1), norm="1")
            cond1 = (1.0 / float(rcond)) if info == 0 and float(rcond) > 0 else None
            native_vs_direct = _relative_error_fsum(native_x, direct_x)
            lu_vs_direct = _relative_error_fsum(lu_solution, direct_x)
        except Exception as exc:  # preserve the rest of the patch record
            direct_residual_rel = None
            cond1 = None
            native_vs_direct = None
            lu_vs_direct = None
            direct_error = f"{type(exc).__name__}: {exc}"
        else:
            direct_error = None

        detail = {
            "patch_index": patch_index, "node_count": int(nodes_count), "dof_count": n,
            "finite": finite, "factor_valid": int(factor_valid[patch_index]) == 1,
            "rhs_l2": _norm_fsum(rhs),
            "factor_reconstruction_relative": factor_error,
            "native_solve_residual_relative": native_residual_rel,
            "fp64_solve_of_native_lu_residual_relative": lu_residual_rel,
            "fp32_emulated_solve_residual_relative": fp32_residual_rel,
            "native_vs_fp64_solve_of_native_lu_relative": native_vs_lu,
            "native_vs_fp32_emulated_solve_relative": native_vs_fp32_solve,
            "fp32_rhs_vs_fp64_rhs_relative": fp32_rhs_delta,
            "fp64_direct_original_matrix_residual_relative": direct_residual_rel,
            "native_vs_fp64_direct_solution_relative": native_vs_direct,
            "fp64_native_lu_solution_vs_fp64_direct_relative": lu_vs_direct,
            "fp64_direct_matrix_condition_1_estimate": cond1,
            "direct_solve_error": direct_error,
        }
        report["patch_results"].append(detail)
        _check(report, f"patch_{patch_index}_finite", finite, tolerance)
        _check(report, f"patch_{patch_index}_factor_reconstruction", factor_error, tolerance)
        _check(report, f"patch_{patch_index}_native_solve_residual", native_residual_rel, tolerance)
        _check(report, f"patch_{patch_index}_w_lane_zero", bool(np.max(np.abs(corrections[int(node_offset):int(node_offset + nodes_count), 3])) <= 1e-7), tolerance)
        maximum_native_residual = max(maximum_native_residual, native_residual_rel)
        maximum_fp64lu_residual = max(maximum_fp64lu_residual, lu_residual_rel)
        maximum_native_vs_fp64lu = max(maximum_native_vs_fp64lu, native_vs_lu)
        if native_vs_direct is not None:
            maximum_native_vs_fp64matrix = max(maximum_native_vs_fp64matrix, native_vs_direct)

    offsets = inputs["node_patch_offsets"].astype(np.int64)
    values = inputs["node_patch_values"].astype(np.int64)
    coverage = inputs["node_coverage"].astype(np.int64)
    offsets_valid = bool(offsets.shape == (N + 1,) and offsets[0] == 0 and
                         offsets[-1] == patch_node_count and
                         np.all(offsets[1:] >= offsets[:-1]))
    coverage_valid = bool(np.all(coverage >= 0) and
                          np.array_equal(np.diff(offsets), coverage)) if offsets_valid else False
    map_valid = offsets_valid and coverage_valid
    expected_pre = np.zeros((N, 3), dtype=np.float64)
    for node in range(N):
        for axis in range(3):
            expected_pre[node, axis] = math.fsum(
                float(actual_coeff[mode]) * float(P[mode, node, axis])
                for mode in range(M))
        if map_valid and coverage[node]:
            mapped = values[offsets[node]:offsets[node + 1]]
            if not (np.all(mapped >= 0) and np.all(mapped < patch_node_count) and
                    np.all(patch_nodes[mapped] == node)):
                map_valid = False
                continue
            for axis in range(3):
                expected_pre[node, axis] += math.fsum(
                    float(corrections[idx, axis]) for idx in mapped) / float(coverage[node])
    actual_pre = outputs["preconditioned"].reshape(N, 4).astype(np.float64)
    gather_err = _relative_error_fsum(actual_pre[:, :3], expected_pre)
    pre_w_zero = bool(np.max(np.abs(actual_pre[:, 3])) <= 1e-7)
    _check(report, "node_patch_maps_valid", map_valid, tolerance)
    _check(report, "global_overlap_gather", gather_err, max(tolerance, 3e-6))
    _check(report, "global_w_lane_zero", pre_w_zero, tolerance)

    report["relative_errors"] = {
        "RAP_factor_vs_PtAP": rap_err, "coarse_rhs": coarse_err,
        "RAP_coefficients": coefficient_err,
        "maximum_patch_LU_reconstruction": maximum_factor_error,
        "maximum_patch_native_solve_residual": maximum_native_residual,
        "maximum_patch_fp64_solve_of_native_LU_residual": maximum_fp64lu_residual,
        "maximum_native_vs_fp64_solve_of_native_LU": maximum_native_vs_fp64lu,
        "maximum_native_vs_fp64_direct_solution": maximum_native_vs_fp64matrix,
        "global_gather": gather_err,
    }
    report["diagnostics"] = {
        "all_patch_results_retained": len(report["patch_results"]) == patch_count,
        "failed_patch_indices": [p["patch_index"] for p in report["patch_results"]
                                 if p.get("native_solve_residual_relative") is None or
                                 p["native_solve_residual_relative"] > tolerance],
        "max_native_residual_patch": max(
            report["patch_results"],
            key=lambda p: p.get("native_solve_residual_relative") or -1.0)["patch_index"],
        "factorization_vs_solve_classification": (
            "FP32 factorization error is amplified by ill-conditioned local blocks"
            if maximum_fp64lu_residual > tolerance and
               maximum_native_vs_fp64lu < max(tolerance, 1e-4)
            else "see per-patch FP64 direct and native-factor solve comparisons"),
        "no_numpy_matmul_or_dot_used": True,
        "summation_policy": "math.fsum for scalar products, residual rows, norms, and coarse projections; FP64 einsum only for LU reconstruction",
    }
    failed = [check for check in report["checks"] if not check["passed"]]
    report["failed_checks"] = failed
    report["status"] = "arithmetic_checks_passed" if not failed else "arithmetic_checks_failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    input_raw = args.input.read_bytes()
    output_raw = args.output.read_bytes()
    bundle_raw = args.bundle_manifest.read_bytes()
    bundle = json.loads(bundle_raw)
    validate_bundle_manifest(input_raw, bundle)
    report = verify(input_raw, output_raw, args.tolerance)
    report.update({
        "tolerance": args.tolerance,
        "input_path": str(args.input), "output_path": str(args.output),
        "bundle_manifest_sha256": sha256(bundle_raw),
        "source_capture_selector": bundle["source_capture_selector"],
        "native_selector5_capture_present": bundle["native_selector5_capture_present"],
        "operator_audit_sha256": bundle.get("operator_audit_sha256"),
        "topology_npz_sha256": bundle.get("topology_npz_sha256"),
        "regional_capture_manifest_sha256": bundle.get("regional_capture_manifest_sha256"),
        "regional_capture_payload_sha256": bundle.get("regional_capture_payload_sha256"),
        "source_physics_fingerprint": bundle.get("source_physics_fingerprint"),
        "device_program_fingerprint": bundle.get("device_program_fingerprint"),
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "failed_checks": report["failed_checks"],
        "relative_errors": report["relative_errors"],
        "patches_retained": len(report["patch_results"]),
        "output_sha256": report["output_sha256"],
    }, indent=2))
    return 0 if report["status"] == "arithmetic_checks_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
