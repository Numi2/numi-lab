#!/usr/bin/env python3
"""Strict source-bound audit of a native private CGLS snapshot (read-only)."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any
import numpy as np

MAGIC = 0x534C4743
HEADER_WORDS = 8
SEGMENT_COUNT = 12
VECTOR_NAMES = ("x", "r", "s", "p", "Ax", "b", "mask")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def align16(value: int) -> int:
    return (value + 15) & ~15


def parse_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_WORDS * 4:
        raise ValueError("CGLS snapshot is shorter than its eight-word header")
    magic, version, env_count, nodes_per_env, unknown_count, cap, nsegments, reserved = struct.unpack_from("<8I", raw)
    if magic != MAGIC or version != 1 or nsegments != SEGMENT_COUNT or reserved != 0:
        raise ValueError(f"unsupported CGLS header: {(magic, version, nsegments, reserved)}")
    if env_count == 0 or nodes_per_env == 0 or unknown_count != 2 * env_count * nodes_per_env:
        raise ValueError("CGLS header violates U = 2 * E * N")
    if cap not in (32, 64):
        raise ValueError(f"unsupported CGLS cap {cap}")
    specs: list[tuple[str, str, int, tuple[int, ...]]] = []
    for name in VECTOR_NAMES:
        specs.append((name, "<f4", 4 * unknown_count, (unknown_count, 4)))
    specs.extend([
        ("row_environment", "<u4", unknown_count, (unknown_count,)),
        ("scalars", "<f4", 4 * env_count, (env_count, 4)),
        ("control", "<u4", 4 * env_count, (env_count, 4)),
        ("errors", "<u4", env_count, (env_count,)),
        ("selected", "<u4", env_count, (env_count,)),
    ])
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, int | str]] = []
    offset = HEADER_WORDS * 4
    for name, dtype, count, shape in specs:
        start = align16(offset)
        if start > len(raw) or any(raw[offset:start]):
            raise ValueError(f"nonzero/truncated alignment padding before {name}")
        byte_count = count * 4
        end = start + byte_count
        if end > len(raw):
            raise ValueError(f"truncated CGLS segment {name}")
        arr = np.frombuffer(raw, dtype=np.dtype(dtype), count=count, offset=start).reshape(shape).copy()
        arrays[name] = arr
        segments.append({"name": name, "offset": start, "bytes": byte_count})
        offset = end
    if offset != len(raw):
        raise ValueError(f"unexpected trailing bytes: expected {offset}, got {len(raw)}")
    header = {
        "magic_hex": f"0x{magic:08x}", "version": version, "environment_count": env_count,
        "nodes_per_environment": nodes_per_env, "unknown_count": unknown_count,
        "iteration_cap": cap, "segment_count": nsegments, "file_bytes": len(raw),
        "sha256": sha(path), "segments": segments,
    }
    return header, arrays


def finite_summary(a: np.ndarray) -> dict[str, Any]:
    v = np.asarray(a)
    finite = np.isfinite(v)
    if not bool(np.all(finite)):
        return {"finite": False, "nonfinite_count": int(v.size - np.count_nonzero(finite))}
    return {"finite": True, "l2": float(np.linalg.norm(v.astype(np.float64))),
            "linf": float(np.max(np.abs(v))) if v.size else 0.0}


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return {"shape_match": False, "left_shape": list(aa.shape), "right_shape": list(bb.shape)}
    d = aa - bb
    finite = np.isfinite(d)
    if not np.all(finite):
        return {"shape_match": True, "finite": False,
                "nonfinite_count": int(d.size - np.count_nonzero(finite))}
    dn = float(np.linalg.norm(d))
    bn = float(np.linalg.norm(bb))
    return {"shape_match": True, "finite": True, "difference_l2": dn,
            "reference_l2": bn, "relative_l2": dn / bn if bn > 0.0 else None,
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
            "classification": "measured CPU/native-array comparison; no tolerance or noise-floor gate asserted"}


def strict_capture(path: Path) -> tuple[dict, list[dict], list[dict], dict]:
    header = integrity = None
    nodes: list[dict] = []
    tets: list[dict] = []
    def parse_int(token: str):
        return -0.0 if token == "-0" else int(token)
    def reject_constant(token: str):
        raise ValueError(f"nonfinite JSON token {token}")
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line, parse_int=parse_int, parse_constant=reject_constant)
            kind = row.get("record")
            if kind == "header":
                if header is not None: raise ValueError("duplicate operator-capture header")
                header = row
            elif kind == "node": nodes.append(row)
            elif kind == "tetrahedron": tets.append(row)
            elif kind == "integrity":
                if integrity is not None: raise ValueError("duplicate operator-capture integrity row")
                integrity = row
            else: raise ValueError(f"unknown capture record at line {line_no}")
    if header is None or integrity is None:
        raise ValueError("operator capture lacks header or terminal integrity record")
    return header, nodes, tets, integrity


def load_audit_module(snapshot: Path):
    path = snapshot / "matter/tools/cardboard_cached_operator_audit.py"
    if not path.is_file(): raise ValueError(f"missing source-bound audit module {path}")
    spec = importlib.util.spec_from_file_location("source_bound_cardboard_operator_audit", path)
    if spec is None or spec.loader is None: raise ValueError("cannot import source-bound operator audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def marker_rows(stderr: Path, prefix: str) -> list[dict]:
    rows = []
    for line in Path(stderr).read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            rows.append(json.loads(line.split("=", 1)[1]))
    return rows


def raw_zero(a: np.ndarray) -> bool:
    return bool(np.all(np.asarray(a).view(np.uint32) == 0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.binary.resolve()
    run_dir = args.run_dir.resolve()
    capture = args.operator_capture.resolve()
    source_snapshot = args.source_snapshot.resolve()
    prediction_path = run_dir / "prediction.json"
    receipt_path = run_dir / "receipt.json"
    stderr_path = run_dir / "stderr.log"
    for path in (binary, capture, prediction_path, receipt_path, stderr_path):
        if not path.is_file(): raise ValueError(f"required input does not exist: {path}")
    prediction = json.loads(prediction_path.read_text())
    if Path(prediction.get("source_snapshot", "")).resolve() != source_snapshot:
        raise ValueError("prediction source_snapshot does not equal the requested retained snapshot")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_inputs_unchanged") is not True:
        raise ValueError("native run receipt does not attest unchanged source inputs")

    snapshot_header, arrays = parse_snapshot(binary)
    selected = arrays["selected"]
    if np.any(selected > 1) or int(np.count_nonzero(selected == 1)) != 1:
        raise ValueError("snapshot must select exactly one environment")
    env = int(np.flatnonzero(selected == 1)[0])
    module, module_path = load_audit_module(source_snapshot)
    binding = module.bind_inputs(capture, prediction_path, receipt_path, stderr_path,
                                 env, source_snapshot_override=source_snapshot)
    capture_header, nodes, tets, integrity = strict_capture(capture)
    if (snapshot_header["environment_count"] != int(capture_header["environment_count"]) or
        snapshot_header["nodes_per_environment"] != int(capture_header["node_count_per_environment"])):
        raise ValueError("CGLS binary dimensions do not match the exact selected operator capture")
    if snapshot_header["iteration_cap"] != int(prediction["environment"].get("NM_FEM_BASE_CGLS_ITERATIONS", snapshot_header["iteration_cap"])):
        raise ValueError("CGLS binary cap disagrees with the sealed runtime configuration")

    cgls_captures = marker_rows(stderr_path, "fem_base_cgls_capture=")
    cgls_directions = marker_rows(stderr_path, "fem_base_cgls_direction=")
    matching_capture = [r for r in cgls_captures if r.get("root") == capture_header.get("root") and
                        r.get("newton_iteration") == capture_header.get("newton_iteration")]
    matching_direction = [r for r in cgls_directions if r.get("root") == capture_header.get("root") and
                          r.get("newton_iteration") == capture_header.get("newton_iteration")]
    if len(matching_capture) != 1 or len(matching_direction) != 1:
        raise ValueError("stderr must contain one CGLS capture and direction marker matching operator-capture root/Newton")
    cm, dm = matching_capture[0], matching_direction[0]
    if cm.get("complete") is not True or cm.get("physical_state_changed") is not False:
        raise ValueError("CGLS snapshot marker is incomplete or claims physical state mutation")
    if int(cm.get("bytes", -1)) != binary.stat().st_size or int(cm.get("expected_bytes", -1)) != binary.stat().st_size:
        raise ValueError("CGLS snapshot byte counts disagree with the native file")
    if dm.get("original_newton_certificate_applies_to_direction") is not False:
        raise ValueError("runtime marker improperly transfers the original Newton certificate")
    if int(dm.get("iteration_budget", -1)) != snapshot_header["iteration_cap"]:
        raise ValueError("CGLS direction marker cap disagrees with binary header")
    if (capture_header.get("root"), capture_header.get("newton_iteration")) != (cm.get("root"), cm.get("newton_iteration")):
        raise ValueError("CGLS and rich operator captures identify different Newton states")

    bind = binding["prediction_environment"]
    if not isinstance(bind, dict): raise ValueError("missing prediction environment binding")
    operator_env = int(args.operator_environment)
    if operator_env != env:
        raise ValueError("requested operator assembly environment does not match selected CGLS environment")
    mass, dt2_stiffness, A, captured_delta, meta = module.validate_and_assemble(
        capture_header, nodes, tets, integrity, operator_env)
    E, N, U = snapshot_header["environment_count"], snapshot_header["nodes_per_environment"], snapshot_header["unknown_count"]
    mechanical_rows = E * N
    expected_row_env = np.concatenate((np.repeat(np.arange(E, dtype=np.uint32), N),
                                       np.repeat(np.arange(E, dtype=np.uint32), N)))
    if not np.array_equal(arrays["row_environment"], expected_row_env):
        raise ValueError("row-environment map is not the exact mechanics/field E-by-N layout")
    if np.any(arrays["selected"] != np.array([1 if i == env else 0 for i in range(E)], dtype=np.uint32)):
        raise ValueError("selected environment flags disagree with selected index")
    for name in VECTOR_NAMES:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"CGLS vector {name} contains nonfinite values")
        if np.any(arrays[name][:, 3].view(np.uint32) != 0):
            raise ValueError(f"CGLS vector {name} has nonzero w padding")
    if np.any(arrays["mask"][:, 3].view(np.uint32) != 0):
        raise ValueError("row mask has nonzero w padding")

    quiet = [i for i in range(E) if i != env]
    quiet_zero_checks = {}
    for qenv in quiet:
        rows = np.arange(qenv * N, (qenv + 1) * N)
        field_rows = mechanical_rows + rows
        checks = {name: raw_zero(arrays[name][np.r_[rows, field_rows]])
                  for name in ("x", "r", "s", "p", "Ax", "b", "mask")}
        checks["control_idle"] = bool(arrays["control"][qenv, 0] == 0)
        checks["errors_zero"] = bool(arrays["errors"][qenv] == 0)
        checks["scalars_zero"] = raw_zero(arrays["scalars"][qenv])
        if not all(checks.values()): raise ValueError(f"quiet environment {qenv} is not exact-zero/idle: {checks}")
        quiet_zero_checks[str(qenv)] = checks

    field_rows = np.arange(mechanical_rows, U)
    field_checks = {name: raw_zero(arrays[name][field_rows])
                    for name in ("x", "r", "s", "p", "Ax", "b", "mask")}
    if not all(field_checks.values()): raise ValueError(f"field backing rows are not exact zero: {field_checks}")

    selected_control = arrays["control"][env]
    status, completed, cap, selected_word = (int(v) for v in selected_control)
    status_valid = status in (2, 3) and int(arrays["errors"][env]) == 0 and selected_word == 1 and completed <= cap
    if cap != snapshot_header["iteration_cap"]: raise ValueError("control cap differs from binary cap")
    if np.any((arrays["mask"][:mechanical_rows, :3] != 0.0) &
              (arrays["mask"][:mechanical_rows, :3] != 1.0)):
        raise ValueError("mechanical mask contains values outside {0,1}")
    if np.any(arrays["mask"][mechanical_rows:, :3] != 0.0):
        raise ValueError("field backing mask is not zero")
    for row in range(mechanical_rows):
        row_env = int(arrays["row_environment"][row])
        if row_env != env or not np.any(arrays["mask"][row, :3] != 0.0):
            if any(not raw_zero(arrays[name][row:row+1]) for name in ("x", "r", "s", "p", "Ax", "b")):
                raise ValueError(f"fixed/quiet mechanical row {row} contains a nonzero CGLS vector")

    free_nodes = np.asarray(meta["free_nodes"], dtype=np.int64)
    if free_nodes.size != int(meta["free_node_count"]): raise ValueError("operator free-node list is inconsistent")
    selected_mech_indices = env * N + free_nodes
    x_free = arrays["x"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    native_ax_free = arrays["Ax"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    b_free = arrays["b"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    r_free = arrays["r"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    s_free = arrays["s"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    p_free = arrays["p"][selected_mech_indices, :3].astype(np.float64).reshape(-1)
    assembled_ax = np.asarray(A @ x_free).reshape(-1)
    assembled_at_r = np.asarray(A.T @ r_free).reshape(-1)
    recurrence_r_from_native_ax = b_free - native_ax_free
    recurrence_r_from_assembled_ax = b_free - assembled_ax
    capture_rhs = np.asarray(meta["native_rhs_free_impulse_Ns"], dtype=np.float64).reshape(-1)
    if capture_rhs.shape != b_free.shape: raise ValueError("captured RHS and CGLS free rows differ in shape")
    norms = {name: finite_summary(arrays[name]) for name in ("x", "r", "s", "p", "Ax", "b")}
    report = {
        "schema": "numi.cardboard.native_cgls_snapshot_audit.v1",
        "classification": "source-bound native readback and independent captured-operator comparison; not a physical solve or solver certificate",
        "bindings": {
            "binary_sha256": snapshot_header["sha256"], "binary_bytes": snapshot_header["file_bytes"],
            "operator_capture_sha256": sha(capture), "prediction_sha256": sha(prediction_path),
            "receipt_sha256": sha(receipt_path), "stderr_sha256": sha(stderr_path),
            "source_snapshot": str(source_snapshot), "source_module": str(module_path),
            "source_module_sha256": sha(module_path), "operator_environment": operator_env,
            "capture_root": int(capture_header["root"]), "capture_newton_iteration": int(capture_header["newton_iteration"]),
            "run_exit_code": receipt.get("exit_code"), "run_timed_out": receipt.get("timed_out"),
            "selected_environment": env,
        },
        "snapshot_header": snapshot_header,
        "selected_environment_state": {
            "status": status, "status_name": {2:"exhausted",3:"zero_normal_residual"}.get(status,"invalid_or_breakdown"),
            "completed_iterations": completed, "cap": cap, "selected_word": selected_word,
            "error_bits": int(arrays["errors"][env]), "status_valid": bool(status_valid),
            "scalars_gamma_alpha_beta_bnorm2": arrays["scalars"][env].astype(float).tolist(),
            "candidate_l2": float(np.linalg.norm(x_free)), "candidate_linf": float(np.max(np.abs(x_free))),
            "rhs_l2": float(np.linalg.norm(b_free)), "residual_l2": float(np.linalg.norm(r_free)),
            "normal_residual_l2": float(np.linalg.norm(s_free)), "search_direction_l2": float(np.linalg.norm(p_free)),
        },
        "scope_checks": {"quiet_environment_exact_zero_idle": quiet_zero_checks,
                          "field_backing_exact_zero": field_checks,
                          "newton_certificate_applies_to_cgls_direction": False},
        "operator_replay": {
            "operator_assembly": meta["assembly"], "shape": list(A.shape), "nnz": int(A.nnz),
            "dt_seconds": float(meta["dt_seconds"]), "mass_l2": float(np.linalg.norm(mass.diagonal())),
            "dt2_stiffness_l2_frobenius": float(np.sqrt(np.sum(dt2_stiffness.data.astype(np.float64)**2))),
            "candidate_action_native_vs_assembled": compare(native_ax_free, assembled_ax),
            "RHS_native_vs_captured_center_residual": compare(b_free, capture_rhs),
            "recurrence_r_vs_b_minus_native_Ax": compare(r_free, recurrence_r_from_native_ax),
            "recurrence_r_vs_b_minus_assembled_Ax": compare(r_free, recurrence_r_from_assembled_ax),
            "normal_residual_s_vs_assembled_AT_r": compare(s_free, assembled_at_r),
            "assembled_Ax_l2": float(np.linalg.norm(assembled_ax)),
            "native_Ax_l2": float(np.linalg.norm(native_ax_free)),
            "candidate_linear_residual_from_native_Ax_l2": float(np.linalg.norm(recurrence_r_from_native_ax)),
            "candidate_linear_residual_from_assembled_Ax_l2": float(np.linalg.norm(recurrence_r_from_assembled_ax)),
            "residual_ratio_native": (float(np.linalg.norm(recurrence_r_from_native_ax))/float(np.linalg.norm(b_free)) if np.linalg.norm(b_free)>0 else None),
            "residual_ratio_assembled": (float(np.linalg.norm(recurrence_r_from_assembled_ax))/float(np.linalg.norm(b_free)) if np.linalg.norm(b_free)>0 else None),
            "gate": "report-only; no action-parity/noise-floor threshold is asserted",
        },
        "vector_norms_full_layout": norms,
        "limitations": [
            "The captured operator is a read-only independent FP64 assembly from captured source state; action differences are reported without a qualified noise floor.",
            "CGLS status and candidate do not inherit the prior Newton/FGMRES certificate.",
            "This audit does not execute a nonlinear trial, accept a physical transaction, or qualify full-coupon convergence.",
            "A status outside exhausted/zero-normal-residual is retained and reported as invalid rather than coerced to success."
        ]
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="contains prediction.json, receipt.json, stderr.log")
    ap.add_argument("--operator-capture", type=Path, required=True, help="native rich operator-capture JSONL from the same root/Newton")
    ap.add_argument("--source-snapshot", type=Path, required=True)
    ap.add_argument("--operator-environment", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite existing output {out}")
    try:
        report = run(args)
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=False)
        (out / "failure.json").write_text(json.dumps({"schema":"numi.cardboard.native_cgls_snapshot_audit_failure.v1","error":str(exc)},indent=2)+"\n")
        raise
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {"files": {p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in out.iterdir()}}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"result":"AUDIT_WRITTEN","output":str(out),"status_valid":report["selected_environment_state"]["status_valid"],"selected_environment":report["bindings"]["selected_environment"],"capture_newton":report["bindings"]["capture_newton_iteration"]},indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
