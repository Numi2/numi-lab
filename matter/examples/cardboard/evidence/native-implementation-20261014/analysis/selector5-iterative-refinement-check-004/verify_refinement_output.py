#!/usr/bin/env python3
"""Audit native selector-5 solve/refinement branch outputs with FP64 row sums."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

MAGIC_IN = b"NMSGKC1\0"
MAGIC_OUT = b"NMSRC001"
IN_HEADER = struct.Struct("<11I")
OUT_HEADER = struct.Struct("<8I")
STATUS_BYTES = 48


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Cursor:
    def __init__(self, raw: bytes, offset: int):
        self.raw = raw
        self.offset = offset
        self.arrays: dict[str, tuple[str, int, int]] = {}

    def take(self, name: str, fmt: str, count: int, item_size: int) -> tuple[str, int, int]:
        size = count * item_size
        require(count >= 0 and self.offset <= len(self.raw) and size <= len(self.raw) - self.offset,
                f"truncated {name}")
        result = (fmt, self.offset, count)
        self.arrays[name] = result
        self.offset += size
        return result

    def values(self, name: str) -> tuple[float | int, ...]:
        fmt, offset, count = self.arrays[name]
        if count == 0:
            return ()
        return struct.unpack_from("<" + fmt * count, self.raw, offset)


def parse_input(raw: bytes) -> tuple[dict, Cursor]:
    require(len(raw) >= 8 + IN_HEADER.size, "input too short")
    require(raw[:8] == MAGIC_IN, "input magic mismatch")
    fields = IN_HEADER.unpack_from(raw, 8)
    (version, n_nodes, n_tets, n_modes, n_patches, n_patch_nodes, n_dof,
     n_matrix, n_pivot, n_env, selected_env) = fields
    require(version == 1 and all(x > 0 for x in fields[1:9]), "invalid input header")
    require(n_env == 2 and selected_env == 0, "input is not selected environment 0 of source pair")
    c = Cursor(raw, 8 + IN_HEADER.size)
    c.take("rap_lu", "f", n_modes * n_modes, 4)
    c.take("rap_pivots", "I", n_modes, 4)
    c.take("rap_valid", "I", 1, 4)
    c.take("basis", "f", n_nodes * 4, 4)
    c.take("regional_basis", "f", n_modes * n_nodes * 4, 4)
    c.take("regional_ap", "f", n_modes * n_nodes * 4, 4)
    c.take("patches", "I", n_patches * 4, 4)
    c.take("patch_nodes", "I", n_patch_nodes, 4)
    c.take("node_patch_offsets", "I", n_nodes + 1, 4)
    c.take("node_patch_values", "I", n_patch_nodes, 4)
    c.take("node_coverage", "I", n_nodes, 4)
    c.take("patch_matrices", "f", n_matrix, 4)
    require(c.offset == len(raw), "input has trailing bytes")
    header = {
        "version": version, "node_count": n_nodes, "tetrahedron_count": n_tets,
        "mode_count": n_modes, "patch_count": n_patches,
        "patch_node_count": n_patch_nodes, "dof_count": n_dof,
        "matrix_float_count": n_matrix, "pivot_count": n_pivot,
        "source_environment_count": n_env, "selected_environment": selected_env,
    }
    return header, c


def parse_output(raw: bytes, header: dict) -> Cursor:
    require(len(raw) >= 8 + OUT_HEADER.size, "output too short")
    require(raw[:8] == MAGIC_OUT, "output magic mismatch")
    out = OUT_HEADER.unpack_from(raw, 8)
    (version, n_nodes, n_patches, n_patch_nodes, n_modes, n_matrix, n_pivot,
     n_status) = out
    require((version, n_nodes, n_patches, n_patch_nodes, n_modes, n_matrix, n_pivot, n_status) ==
            (1, header["node_count"], header["patch_count"], header["patch_node_count"],
             header["mode_count"], header["matrix_float_count"], header["pivot_count"], 10),
            "output dimensions mismatch")
    c = Cursor(raw, 8 + OUT_HEADER.size)
    c.take("coarse_rhs", "f", n_modes, 4)
    c.take("coefficients", "f", n_modes, 4)
    c.take("factor_valid", "I", n_patches, 4)
    c.take("patch_lu", "f", n_matrix, 4)
    c.take("patch_pivots", "I", n_pivot, 4)
    for name, count in (
        ("mode0_corrections", n_patch_nodes * 4),
        ("mode1_corrections", n_patch_nodes * 4),
        ("mode0_preconditioned", n_nodes * 4),
        ("mode1_preconditioned", n_nodes * 4),
        ("zero_rhs_mode0_corrections", n_patch_nodes * 4),
        ("zero_rhs_mode0_preconditioned", n_nodes * 4),
        ("zero_rhs_corrections", n_patch_nodes * 4),
        ("zero_rhs_preconditioned", n_nodes * 4),
        ("completed_tail_corrections", n_patch_nodes * 4),
        ("completed_tail_preconditioned", n_nodes * 4),
        ("invalid_factor_corrections", n_patch_nodes * 4),
        ("invalid_mode_corrections", n_patch_nodes * 4),
    ):
        c.take(name, "f", count, 4)
    c.take("singular_factor_valid", "I", n_patches, 4)
    c.take("singular_lu", "f", n_matrix, 4)
    c.take("singular_pivots", "I", n_pivot, 4)
    c.take("singular_corrections", "f", n_patch_nodes * 4, 4)
    for name in ("factor_status", "mode0_status", "mode1_status", "zero_mode0_status", "zero_status",
                 "completed_status", "invalid_factor_status", "invalid_mode_status",
                 "singular_factor_status", "singular_solve_status"):
        c.take(name, "B", STATUS_BYTES, 1)
    require(c.offset == len(raw), "output has trailing/unaccounted bytes")
    return c


def l2(values) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def residual_metrics(header: dict, inputs: Cursor, outputs: Cursor, correction_name: str) -> list[dict]:
    node_count = header["node_count"]
    mode_count = header["mode_count"]
    patches = inputs.values("patches")
    patch_nodes = inputs.values("patch_nodes")
    basis = inputs.values("basis")
    regional_ap = inputs.values("regional_ap")
    coefficients = outputs.values("coefficients")
    matrices = inputs.values("patch_matrices")
    corrections = outputs.values(correction_name)
    result = []
    for p in range(header["patch_count"]):
        node_offset, node_count_local, matrix_offset, _pivot_offset = patches[p * 4:p * 4 + 4]
        n = node_count_local * 3
        local_nodes = patch_nodes[node_offset:node_offset + node_count_local]
        rhs = []
        for local_node, global_node in enumerate(local_nodes):
            for axis in range(3):
                b = float(basis[global_node * 4 + axis])
                b -= math.fsum(
                    float(coefficients[m]) *
                    float(regional_ap[(m * node_count + global_node) * 4 + axis])
                    for m in range(mode_count)
                )
                rhs.append(b)
        x = [float(corrections[(node_offset + i // 3) * 4 + i % 3]) for i in range(n)]
        residual = []
        for row in range(n):
            ax = math.fsum(float(matrices[matrix_offset + row * n + col]) * x[col]
                           for col in range(n))
            residual.append(rhs[row] - ax)
        rhs_norm = l2(rhs)
        res_norm = l2(residual)
        relative = 0.0 if rhs_norm == 0.0 and res_norm == 0.0 else (
            math.inf if rhs_norm == 0.0 else res_norm / rhs_norm)
        result.append({"patch_index": p, "node_count": node_count_local,
                       "rhs_l2": rhs_norm, "residual_l2": res_norm,
                       "residual_relative": relative})
    return result


def status_code(c: Cursor, name: str) -> int:
    # NMMatterStatusGPU starts with eight uint32 fields.
    raw = c.values(name)
    return struct.unpack_from("<I", bytes(raw), 0)[0]


def bitwise_zero(c: Cursor, name: str) -> bool:
    _, offset, count = c.arrays[name]
    raw = c.raw[offset:offset + count * 4]
    return all(byte == 0 for byte in raw)


def numeric_zero_summary(c: Cursor, name: str) -> dict:
    """Report numeric-zero semantics without hiding signed zero or bad values."""
    _, offset, count = c.arrays[name]
    values = struct.unpack_from("<" + "f" * count, c.raw, offset) if count else ()
    nonfinite = 0
    nonzero = 0
    negative_zero = 0
    first_nonzero_index = None
    first_nonzero_bits = None
    for index, value in enumerate(values):
        bits = struct.unpack_from("<I", c.raw, offset + index * 4)[0]
        if not math.isfinite(value):
            nonfinite += 1
        elif value == 0.0:
            if bits & 0x80000000:
                negative_zero += 1
        else:
            nonzero += 1
            if first_nonzero_index is None:
                first_nonzero_index = index
                first_nonzero_bits = bits
    return {
        "value_count": count,
        "nonfinite_count": nonfinite,
        "nonzero_count": nonzero,
        "negative_zero_count": negative_zero,
        "first_nonzero_index": first_nonzero_index,
        "first_nonzero_bits": first_nonzero_bits,
    }


def verify(input_raw: bytes, output_raw: bytes, tolerance: float = 1.0e-3) -> dict:
    header, inputs = parse_input(input_raw)
    outputs = parse_output(output_raw, header)
    checks = []

    def check(name: str, passed: bool, detail=None):
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    # Status IDs are sourced from NMStatusCode in shared.h: success=0,
    # invalid-dispatch=1, linear-solver-failure=7.
    expected_status = {
        "factor_status": 0, "mode0_status": 0, "mode1_status": 0,
        "zero_mode0_status": 0, "zero_status": 0, "completed_status": 0,
        "invalid_factor_status": 7, "invalid_mode_status": 1,
        "singular_factor_status": 7, "singular_solve_status": 7,
    }
    for key, expected in expected_status.items():
        actual = status_code(outputs, key)
        check(key, actual == expected, {"expected": expected, "actual": actual})
    factor_valid = outputs.values("factor_valid")
    singular_valid = outputs.values("singular_factor_valid")
    check("active_factor_valid", all(v == 1 for v in factor_valid),
          {"valid_patch_count": sum(v == 1 for v in factor_valid),
           "patch_count": len(factor_valid)})
    check("singular_factor_rejected", len(singular_valid) > 0 and singular_valid[0] == 0,
          {"first_patch_factor_valid": singular_valid[0] if singular_valid else None})

    zero_names = ["zero_rhs_mode0_corrections", "zero_rhs_mode0_preconditioned",
                  "zero_rhs_corrections", "zero_rhs_preconditioned"]
    numeric_zero = {}
    for name in zero_names:
        summary = numeric_zero_summary(outputs, name)
        numeric_zero[name] = summary
        check(name + "_numeric_zero",
              summary["nonfinite_count"] == 0 and summary["nonzero_count"] == 0,
              summary)
    for name in ["completed_tail_corrections", "completed_tail_preconditioned",
                 "invalid_factor_corrections", "invalid_mode_corrections"]:
        check(name + "_bitwise_zero", bitwise_zero(outputs, name))
    first_patch = inputs.values("patches")[:4]
    first_patch_node_offset, first_patch_node_count = first_patch[:2]
    _, singular_offset, _ = outputs.arrays["singular_corrections"]
    singular_first_bytes = outputs.raw[
        singular_offset + first_patch_node_offset * 16:
        singular_offset + (first_patch_node_offset + first_patch_node_count) * 16
    ]
    check("singular_patch_zero_correction", all(byte == 0 for byte in singular_first_bytes),
          {"patch_index": 0, "patch_node_count": first_patch_node_count})

    mode0 = residual_metrics(header, inputs, outputs, "mode0_corrections")
    mode1 = residual_metrics(header, inputs, outputs, "mode1_corrections")
    check("mode0_finite", all(math.isfinite(p["residual_relative"]) for p in mode0))
    check("mode1_finite", all(math.isfinite(p["residual_relative"]) for p in mode1))
    max0 = max((p["residual_relative"] for p in mode0), default=0.0)
    max1 = max((p["residual_relative"] for p in mode1), default=0.0)
    failing_unrefined = [i for i, p in enumerate(mode0)
                         if p["residual_relative"] > tolerance]
    check("refined_all_patch_residual_gate", max1 <= tolerance,
          {"limit": tolerance, "maximum_mode0": max0, "maximum_mode1": max1,
           "worst_mode1_patch": max(mode1, key=lambda p: p["residual_relative"])["patch_index"]
           if mode1 else None})
    check("refinement_exercises_unrefined_failure",
          bool(failing_unrefined), {"mode0_patches_over_limit": failing_unrefined})
    check("refinement_improves_each_unrefined_failure",
          all(mode1[i]["residual_relative"] < mode0[i]["residual_relative"]
              for i in failing_unrefined),
          {"patches": [{"patch_index": i,
                        "mode0": mode0[i]["residual_relative"],
                        "mode1": mode1[i]["residual_relative"]}
                       for i in failing_unrefined]})
    mode_ratio = max((b["residual_relative"] / max(a["residual_relative"], 1e-300)
                      for a, b in zip(mode0, mode1)), default=0.0)
    report = {
        "schema": "numi.matter.selector5.iterative_refinement_native_check.v1",
        "input_sha256": sha256(input_raw), "output_sha256": sha256(output_raw),
        "dimensions": header,
        "checker_scope": "prepared source182 operator patches; production Metal factor/solve/apply kernels",
        "qualification": "kernel arithmetic contract only; not full selector-5 FEM/FGMRES qualification",
        "checks": checks,
        "mode0": {"maximum_patch_residual_relative": max0, "patches": mode0},
        "mode1": {"maximum_patch_residual_relative": max1,
                  "maximum_mode1_to_mode0_residual_ratio": mode_ratio,
                  "patches": mode1},
        "zero_rhs_numeric_summary": numeric_zero,
        "limitations": [
            "Input operator is the source182 selected-generation FEM operator with selector-5 graph patches rebuilt from its actual mesh; no selector-5 production FEM transaction is implied.",
            "FP64 row sums independently audit native float32 correction vectors; they do not validate the physical model or full FGMRES convergence.",
        ],
    }
    report["pass"] = all(item["pass"] for item in checks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()
    try:
        report = verify(args.input.read_bytes(), args.output.read_bytes(), args.tolerance)
    except Exception as exc:
        report = {"schema": "numi.matter.selector5.iterative_refinement_native_check.v1",
                  "pass": False, "error": f"{type(exc).__name__}: {exc}"}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    sys.exit(main())
