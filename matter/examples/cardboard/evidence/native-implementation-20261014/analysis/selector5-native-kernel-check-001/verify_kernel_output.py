#!/usr/bin/env python3
"""Independently verify the selector-5 production-kernel check result bundle.

This verifies arithmetic and array integrity for the captured source-bound input
bundle. It cannot qualify a native selector-5 preconditioner because the source
182 input is a selector-3 RAP capture and graph matrices are reconstructed from
the audited CSR operator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np

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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _factor_solve(lu: np.ndarray, pivots: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(rhs)
    require(lu.shape == (n, n), "LU/RHS shape mismatch")
    require(sorted(int(x) for x in pivots) == list(range(n)), "pivot array is not a permutation")
    values = np.asarray(rhs[pivots.astype(np.int64)], dtype=np.float64).copy()
    for row in range(n):
        values[row] -= float(np.dot(lu[row, :row].astype(np.float64), values[:row]))
    for row in range(n - 1, -1, -1):
        values[row] -= float(np.dot(lu[row, row + 1:].astype(np.float64), values[row + 1:]))
        diagonal = float(lu[row, row])
        require(math.isfinite(diagonal) and diagonal != 0.0, "LU diagonal is invalid")
        values[row] /= diagonal
    return values


def _unpivoted_matrix(lu: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    n = len(pivots)
    require(lu.shape == (n, n), "LU shape mismatch")
    require(sorted(int(x) for x in pivots) == list(range(n)), "pivot array is not a permutation")
    lower = np.tril(lu, -1) + np.eye(n, dtype=np.float64)
    upper = np.triu(lu)
    permuted = lower @ upper
    result = np.empty_like(permuted)
    result[pivots.astype(np.int64), :] = permuted
    return result


def _rel_error(actual: np.ndarray, expected: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    denom = float(np.linalg.norm(e.ravel()))
    numer = float(np.linalg.norm((a - e).ravel()))
    return numer / max(denom, 1e-30)


def validate_bundle_manifest(input_raw: bytes, bundle: dict) -> dict:
    """Verify the binary bundle and its source-preparation manifest agree."""
    require(bundle.get("bundle_sha256") == sha256(input_raw),
            "input bundle bytes do not match the source-bound bundle manifest")
    require(bundle.get("schema") == "numi.matter.selector5-kernel-check-input.v1" and
            bundle.get("source_capture_selector") == 3 and
            bundle.get("native_selector5_capture_present") is False,
            "bundle manifest scope/identity differs from this checker contract")
    require(input_raw[:8] == INPUT_MAGIC and len(input_raw) >= 8 + struct.calcsize(HEADER_FMT),
            "input bundle header is invalid")
    h = struct.unpack_from(HEADER_FMT, input_raw, 8)
    _, nodes, tets, modes, patches, patch_nodes, dofs, matrix_floats, pivots, source_envs, selected_env = h
    expected_dims = {
        "node_count": nodes, "tetrahedron_count": tets, "mode_count": modes,
        "patch_count": patches, "patch_node_count": patch_nodes, "dof_count": dofs,
        "matrix_float_count": matrix_floats, "pivot_count": pivots,
        "source_environment_count": source_envs, "selected_environment": selected_env,
    }
    require(bundle.get("dimensions") == expected_dims,
            "binary dimensions differ from bundle manifest")
    descriptors = bundle.get("arrays")
    require(isinstance(descriptors, list) and len(descriptors) == len(ARRAY_SPECS),
            "bundle array descriptors are missing or incomplete")
    cursor = 8 + struct.calcsize(HEADER_FMT)
    expected_shapes = {
        "rap_lu": [modes, modes], "rap_pivots": [modes], "rap_valid": [1],
        "basis": [nodes, 4], "regional_basis": [modes, nodes, 4],
        "regional_ap": [modes, nodes, 4], "patches": [patches, 4],
        "patch_nodes": [patch_nodes], "node_patch_offsets": [nodes + 1],
        "node_patch_values": [patch_nodes], "node_coverage": [nodes],
        "patch_matrices": [matrix_floats],
    }
    for descriptor, (name, dtype, count_fn) in zip(descriptors, ARRAY_SPECS):
        count = int(count_fn(h))
        byte_count = count * np.dtype(dtype).itemsize
        require(descriptor.get("name") == name and descriptor.get("dtype") == dtype,
                f"bundle descriptor order/type mismatch at {name}")
        require(descriptor.get("offset") == cursor and descriptor.get("byte_count") == byte_count,
                f"bundle descriptor bounds mismatch at {name}")
        require(descriptor.get("shape") == expected_shapes[name],
                f"bundle descriptor shape mismatch at {name}")
        end = cursor + byte_count
        require(end <= len(input_raw) and descriptor.get("sha256") == sha256(input_raw[cursor:end]),
                f"bundle descriptor SHA-256 mismatch at {name}")
        cursor = end
    require(cursor == len(input_raw), "bundle descriptors do not consume exact input bytes")
    return expected_dims


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

    cursor = 8 + header_bytes
    inputs = {}
    for name, dtype, count_fn in ARRAY_SPECS:
        count = int(count_fn(ih))
        size = count * np.dtype(dtype).itemsize
        require(cursor + size <= len(input_raw), f"truncated input array: {name}")
        inputs[name] = np.frombuffer(input_raw, dtype=dtype, count=count, offset=cursor).copy()
        cursor += size
    require(cursor == len(input_raw), "input bundle has trailing bytes")

    # Output arrays are emitted in this exact order by selector5_graph_kernels.mm.
    output_specs = (
        ("coarse_rhs", "<f4", mode_count),
        ("coefficients", "<f4", mode_count),
        ("patch_lu", "<f4", matrix_count),
        ("patch_pivots", "<u4", pivot_count),
        ("factor_valid", "<u4", patch_count),
        ("corrections", "<f4", patch_node_count * 4),
        ("preconditioned", "<f4", node_count * 4),
        ("status", "<u4", 12),
    )
    cursor = 8 + header_bytes
    outputs = {}
    for name, dtype, count in output_specs:
        size = count * np.dtype(dtype).itemsize
        require(cursor + size <= len(output_raw), f"truncated output array: {name}")
        outputs[name] = np.frombuffer(output_raw, dtype=dtype, count=count, offset=cursor).copy()
        cursor += size
    require(cursor == len(output_raw), "output bundle has trailing bytes")
    require(int(outputs["status"][0]) == 0, f"native Matter status code={int(outputs['status'][0])}")
    require(np.all(outputs["factor_valid"] == 1), "one or more native graph patch factors are invalid")
    require(int(inputs["rap_valid"][0]) == 1, "captured RAP factor is invalid")

    M, N = mode_count, node_count
    rap_lu = inputs["rap_lu"].reshape(M, M).astype(np.float64)
    rap_pivots = inputs["rap_pivots"].astype(np.int64)
    rap = _unpivoted_matrix(rap_lu, rap_pivots)
    P = inputs["regional_basis"].reshape(M, N, 4)[:, :, :3].astype(np.float64)
    AP = inputs["regional_ap"].reshape(M, N, 4)[:, :, :3].astype(np.float64)
    b = inputs["basis"].reshape(N, 4)[:, :3].astype(np.float64)
    expected_coarse_rhs = np.einsum("mni,ni->m", P, b, optimize=False)
    actual_coarse_rhs = outputs["coarse_rhs"].astype(np.float64)
    expected_coefficients = _factor_solve(rap_lu, rap_pivots, expected_coarse_rhs)
    actual_coefficients = outputs["coefficients"].astype(np.float64)

    # P^T AP independently checks that the source factor represents the source
    # regional operator, with only ordinary FP32 factorization/capture rounding.
    expected_rap = np.einsum("mni,kni->mk", P, AP, optimize=False)
    rap_factor_error = _rel_error(rap, expected_rap)
    require(rap_factor_error <= tolerance,
            f"captured RAP LU does not reconstruct P^T AP: rel={rap_factor_error:.6g}")

    patches = inputs["patches"].reshape(patch_count, 4).astype(np.int64)
    patch_nodes = inputs["patch_nodes"].astype(np.int64)
    patch_matrices = inputs["patch_matrices"].astype(np.float64)
    patch_lu_all = outputs["patch_lu"].astype(np.float64)
    pivots_all = outputs["patch_pivots"].astype(np.int64)
    corrections = outputs["corrections"].reshape(patch_node_count, 4).astype(np.float64)
    correction_xyz = corrections[:, :3]
    max_factor_error = 0.0
    max_local_residual = 0.0
    for patch_index, (node_offset, nodes_count, matrix_offset, pivot_offset) in enumerate(patches):
        scalar_count = int(nodes_count) * 3
        matrix_count_local = scalar_count * scalar_count
        original = patch_matrices[matrix_offset:matrix_offset + matrix_count_local].reshape(scalar_count, scalar_count)
        lu = patch_lu_all[matrix_offset:matrix_offset + matrix_count_local].reshape(scalar_count, scalar_count)
        pivots = pivots_all[pivot_offset:pivot_offset + scalar_count]
        require(np.isfinite(original).all() and np.isfinite(lu).all(), f"patch {patch_index} has nonfinite matrix")
        reconstructed = _unpivoted_matrix(lu, pivots)
        factor_error = _rel_error(reconstructed, original)
        max_factor_error = max(max_factor_error, factor_error)
        require(factor_error <= tolerance, f"patch {patch_index} LU reconstruction error={factor_error:.6g}")
        nodes = patch_nodes[node_offset:node_offset + nodes_count]
        rhs = np.empty(scalar_count, dtype=np.float64)
        for local_node, node in enumerate(nodes):
            rhs[local_node * 3:local_node * 3 + 3] = b[node]
            for mode in range(M):
                rhs[local_node * 3:local_node * 3 + 3] -= actual_coefficients[mode] * AP[mode, node]
        local_correction = correction_xyz[node_offset:node_offset + nodes_count].reshape(-1)
        solve_residual = original @ local_correction - rhs
        local_residual = float(np.linalg.norm(solve_residual)) / max(float(np.linalg.norm(rhs)), 1e-30)
        max_local_residual = max(max_local_residual, local_residual)
        require(local_residual <= tolerance,
                f"patch {patch_index} solve residual={local_residual:.6g}")
        require(np.max(np.abs(corrections[node_offset:node_offset + nodes_count, 3])) <= 1e-7,
                f"patch {patch_index} correction w lane is not zero")

    offsets = inputs["node_patch_offsets"].astype(np.int64)
    values = inputs["node_patch_values"].astype(np.int64)
    coverage = inputs["node_coverage"].astype(np.int64)
    require(offsets.shape == (N + 1,) and offsets[0] == 0 and offsets[-1] == patch_node_count and
            np.all(offsets[1:] >= offsets[:-1]), "node-patch CSR offsets invalid")
    require(np.all(coverage >= 0) and np.array_equal(np.diff(offsets), coverage),
            "node-patch coverage and offsets disagree")
    expected_preconditioned = np.zeros((N, 3), dtype=np.float64)
    for node in range(N):
        expected_preconditioned[node] = np.einsum("m,mi->i", actual_coefficients, P[:, node, :], optimize=False)
        if coverage[node]:
            mapped = values[offsets[node]:offsets[node + 1]]
            require(np.all(mapped < patch_node_count), f"node {node} maps out of correction range")
            require(np.all(patch_nodes[mapped] == node), f"node {node} maps to another node's correction")
            expected_preconditioned[node] += np.sum(correction_xyz[mapped], axis=0) / float(coverage[node])
    actual_preconditioned = outputs["preconditioned"].reshape(N, 4).astype(np.float64)
    gather_error = _rel_error(actual_preconditioned[:, :3], expected_preconditioned)
    require(gather_error <= max(tolerance, 3e-6), f"native graph gather mismatch={gather_error:.6g}")
    require(np.max(np.abs(actual_preconditioned[:, 3])) <= 1e-7, "preconditioned w lane is not zero")
    coarse_rhs_error = _rel_error(actual_coarse_rhs, expected_coarse_rhs)
    coefficient_error = _rel_error(actual_coefficients, expected_coefficients)
    require(coarse_rhs_error <= tolerance, f"native coarse projection mismatch={coarse_rhs_error:.6g}")
    require(coefficient_error <= tolerance, f"native RAP solve mismatch={coefficient_error:.6g}")

    return {
        "schema": "numi.matter.selector5_kernel_output_audit.v1",
        "status": "arithmetic_checks_passed",
        "source_scope": "source182 selector3 RAP/operator capture plus graph layout rebuilt from actual topology; production selector5 point kernels only",
        "dimensions": {"node_count": N, "tetrahedron_count": tet_count, "mode_count": M,
                       "patch_count": patch_count, "patch_node_count": patch_node_count,
                       "dof_count": dof_count, "matrix_float_count": matrix_count},
        "input_sha256": sha256(input_raw), "output_sha256": sha256(output_raw),
        "relative_errors": {
            "RAP_factor_vs_PtAP": rap_factor_error,
            "coarse_rhs": coarse_rhs_error,
            "RAP_coefficients": coefficient_error,
            "maximum_patch_LU_reconstruction": max_factor_error,
            "maximum_patch_solve_residual": max_local_residual,
            "global_gather": gather_error,
        },
        "limits": [
            "No native selector-5 same-generation capture was supplied.",
            "Patch principal matrices were formed from the independently audited FP32 CSR operator, not production native graph assembly.",
            "Passing verifies production factor/solve/gather kernel arithmetic for this prepared input only; it is not full solver or preconditioner qualification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    parser.add_argument("--checker-source", type=Path,
                        default=Path(__file__).with_name("selector5_graph_kernels.mm"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    input_raw = args.input.read_bytes()
    output_raw = args.output.read_bytes()
    bundle = json.loads(args.bundle_manifest.read_text())
    validate_bundle_manifest(input_raw, bundle)
    report = verify(input_raw, output_raw, args.tolerance)
    report["tolerance"] = args.tolerance
    report["input_path"] = str(args.input)
    report["output_path"] = str(args.output)
    report["bundle_manifest_sha256"] = sha256(args.bundle_manifest.read_bytes())
    report["metallib_sha256"] = sha256(args.metallib.read_bytes())
    report["checker_source_sha256"] = sha256(args.checker_source.read_bytes())
    report["source_capture_selector"] = bundle["source_capture_selector"]
    report["native_selector5_capture_present"] = bundle["native_selector5_capture_present"]
    report["operator_audit_sha256"] = bundle.get("operator_audit_sha256")
    report["topology_npz_sha256"] = bundle.get("topology_npz_sha256")
    report["regional_capture_manifest_sha256"] = bundle.get("regional_capture_manifest_sha256")
    report["regional_capture_payload_sha256"] = bundle.get("regional_capture_payload_sha256")
    report["source_physics_fingerprint"] = bundle.get("source_physics_fingerprint")
    report["device_program_fingerprint"] = bundle.get("device_program_fingerprint")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "relative_errors": report["relative_errors"],
                      "input_sha256": report["input_sha256"], "output_sha256": report["output_sha256"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
