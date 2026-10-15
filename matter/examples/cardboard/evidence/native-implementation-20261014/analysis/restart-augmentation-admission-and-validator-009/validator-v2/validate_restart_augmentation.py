#!/usr/bin/env python3
"""Offline checks for same-Newton regional restart augmentation captures.

Inputs are the rich FEM operator JSONL, its regional JSON/binary sidecar, and a
rejected-conditioning manifest with per-environment augmentation arenas. This
checks hashes/layouts, quiet-environment sentinels, independent A*q actions,
and nonsymmetric augmented RAP/LU reconstruction. Feature-off comparisons are
conditional on exact center identity. It does not qualify convergence or
physical behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_REGIONAL = "numi.matter.regional_rap_capture.v1"
SCHEMA_CONDITIONING = "numi.cardboard.regional-conditioning.v1"
SCHEMA_OPERATOR = "numi.cardboard.fem-newton-capture.v1"
SCHEMA_REPORT = "numi.cardboard.restart-augmentation-validation.v1"
MAX_AQ_RELATIVE_LIMIT = 2e-3
MAX_BASE_AP_RELATIVE_LIMIT = 2e-3
MAX_BASIS_RELATIVE_LIMIT = 2e-3
MAX_LU_RELATIVE_LIMIT = 2e-4
UINT32_MAX = 0xFFFFFFFF
DEFAULT_ASSEMBLER = Path(
    "/Users/home/cardboard-implementation-20261014/analysis/"
    "offline-operator-tool-development-058/cardboard_cached_operator_audit.py"
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_thresholds(*values: float) -> None:
    limits = (MAX_AQ_RELATIVE_LIMIT, MAX_BASE_AP_RELATIVE_LIMIT,
              MAX_BASIS_RELATIVE_LIMIT, MAX_LU_RELATIVE_LIMIT)
    if len(values) != len(limits):
        raise ValueError("internal threshold contract mismatch")
    for label, value, maximum in zip(("fresh Aq", "base AP", "basis", "RAP LU"), values, limits):
        if not math.isfinite(value) or value <= 0.0 or value > maximum:
            raise ValueError(f"{label} threshold must be finite, positive, and no looser than {maximum:g}")


def load_operator_assembler(path: Path):
    spec = importlib.util.spec_from_file_location("cardboard_cached_operator_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load independent operator assembler: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class RichCapture:
    path: Path
    header: dict[str, Any]
    nodes: list[dict[str, Any]]
    tets: list[dict[str, Any]]
    integrity: dict[str, Any]


@dataclass
class RegionalPayload:
    manifest_path: Path
    payload_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass
class ConditioningPayload:
    manifest_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, bytes]


def load_rich_capture(path: Path) -> RichCapture:
    header = integrity = None
    nodes: list[dict[str, Any]] = []
    tets: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"operator capture line {line_no} is invalid JSON") from error
            kind = row.get("record")
            if kind == "header":
                if header is not None:
                    raise ValueError("operator capture has multiple headers")
                header = row
            elif kind == "node":
                nodes.append(row)
            elif kind == "tetrahedron":
                tets.append(row)
            elif kind == "integrity":
                if integrity is not None:
                    raise ValueError("operator capture has multiple terminal integrity records")
                integrity = row
            else:
                raise ValueError(f"operator capture line {line_no} has unknown record kind")
    if header is None or integrity is None:
        raise ValueError("operator capture lacks its header or terminal integrity record")
    if header.get("schema") != SCHEMA_OPERATOR:
        raise ValueError("unsupported rich operator capture schema")
    if integrity.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("rich capture lacks unchanged-production-input attestation")
    if integrity.get("cached_tangent_snapshot_bitwise_guard_included") is not True:
        raise ValueError("rich capture lacks cached-tangent snapshot guard")
    envs = int(header.get("environment_count", 0))
    n = int(header.get("node_count_per_environment", 0))
    t = int(header.get("tetrahedron_count_per_environment", 0))
    if min(envs, n, t) <= 0 or len(nodes) != envs * n or len(tets) != envs * t:
        raise ValueError("rich capture dimensions disagree with record counts")
    for env in range(envs):
        enodes = sorted(int(r.get("node_id", -1)) for r in nodes if int(r.get("environment", -1)) == env)
        etets = sorted(int(r.get("tet_id", -1)) for r in tets if int(r.get("environment", -1)) == env)
        if enodes != list(range(n)) or etets != list(range(t)):
            raise ValueError(f"environment {env} node/tet IDs are not dense and unique")
    return RichCapture(path, header, nodes, tets, integrity)


def load_regional(manifest_path: Path, payload_path: Path | None = None) -> RegionalPayload:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA_REGIONAL or manifest.get("byte_order") != "little_endian":
        raise ValueError("unsupported regional capture schema or byte order")
    payload_spec = manifest.get("payload")
    if not isinstance(payload_spec, dict):
        raise ValueError("regional capture lacks payload descriptor")
    payload = payload_path or Path(str(payload_spec.get("file", "")))
    if not payload.is_file():
        raise ValueError(f"regional payload is unavailable: {payload}")
    data = payload.read_bytes()
    if len(data) != int(payload_spec.get("byte_count", -1)) or sha256_bytes(data) != payload_spec.get("sha256"):
        raise ValueError("regional payload length or SHA-256 mismatch")
    arrays: dict[str, np.ndarray] = {}
    spans: list[tuple[int, int, str]] = []
    # The capture descriptor's shape counts float4/uint4 records; the x4
    # dtype suffix contributes a real trailing lane dimension in NumPy.
    type_map = {"<f4": ("<f4", 0), "<f4x4": ("<f4", 4),
                "<u4": ("<u4", 0), "<u4x4": ("<u4", 4)}
    for desc in manifest.get("arrays", []):
        name, dtype, shape = desc.get("name"), desc.get("dtype"), desc.get("shape")
        if not isinstance(name, str) or name in arrays or not isinstance(shape, list) or not shape:
            raise ValueError("regional array descriptor has invalid name/shape")
        dims = tuple(int(d) for d in shape)
        if any(d < 0 for d in dims):
            raise ValueError(f"regional array {name} has negative dimension")
        offset, size, elem = int(desc.get("offset", -1)), int(desc.get("byte_count", -1)), int(desc.get("element_bytes", -1))
        if offset < 0 or size < 0 or elem <= 0 or math.prod(dims) * elem != size or offset + size > len(data):
            raise ValueError(f"regional array {name} has inconsistent byte extent")
        raw = data[offset:offset + size]
        if sha256_bytes(raw) != desc.get("sha256"):
            raise ValueError(f"regional array {name} SHA-256 mismatch")
        spans.append((offset, offset + size, name))
        if dtype in type_map:
            scalar_dtype, lanes = type_map[dtype]
            semantic_shape = dims + ((lanes,) if lanes else ())
            values = np.frombuffer(raw, dtype=scalar_dtype).copy().reshape(semantic_shape)
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                raise ValueError(f"regional array {name} contains nonfinite values")
            arrays[name] = values
        else:
            arrays[name] = np.frombuffer(raw, dtype=np.uint8).copy()
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError(f"regional payload has a gap or overlap before {name}")
        cursor = end
    if cursor != len(data):
        raise ValueError("regional payload has trailing unaccounted bytes")
    return RegionalPayload(manifest_path, payload, manifest, arrays)


def load_conditioning(manifest_path: Path) -> ConditioningPayload:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA_CONDITIONING:
        raise ValueError("unsupported regional-conditioning schema")
    if manifest.get("byte_order") != "native_little_endian":
        raise ValueError("conditioning arrays are not declared little-endian")
    arrays: dict[str, bytes] = {}
    for desc in manifest.get("files", []):
        name = desc.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in arrays:
            raise ValueError("conditioning filename is invalid or duplicated")
        path = manifest_path.parent / name
        if not path.is_file():
            raise ValueError(f"conditioning arena is missing: {name}")
        raw = path.read_bytes()
        count, stride, size = int(desc.get("count", -1)), int(desc.get("stride_bytes", -1)), int(desc.get("bytes", -1))
        if count < 0 or stride <= 0 or count * stride != size or len(raw) != size:
            raise ValueError(f"conditioning arena {name} dimensions/length mismatch")
        if sha256_bytes(raw) != desc.get("sha256"):
            raise ValueError(f"conditioning arena {name} SHA-256 mismatch")
        arrays[name] = raw
    return ConditioningPayload(manifest_path, manifest, arrays)


def _array(cond: ConditioningPayload, basename: str, dtype: str,
           shape: tuple[int, ...]) -> np.ndarray:
    filename = basename if basename.endswith(".bin") else basename + ".bin"
    raw = cond.arrays.get(filename)
    if raw is None:
        raise ValueError(f"missing conditioning arena {filename}")
    values = np.frombuffer(raw, dtype=dtype)
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{filename}: expected {expected} values, got {values.size}")
    if values.dtype.kind == "f" and not np.isfinite(values).all():
        raise ValueError(f"{filename} contains nonfinite values")
    return values.copy().reshape(shape)


def _metric(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d = aa - bb
    return {
        "observed_l2": float(np.linalg.norm(aa)),
        "reference_l2": float(np.linalg.norm(bb)),
        "difference_l2": float(np.linalg.norm(d)),
        "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(aa), np.linalg.norm(bb), 1e-300)),
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
    }


def _fingerprint_u64(value: Any, label: str) -> int:
    """Normalize the capture's hex string and sidecar's uint64 encodings."""
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a fingerprint")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        try:
            number = int(value, 16 if value.lower().startswith("0x") or
                         any(c in "abcdef" for c in value.lower()) else 10)
        except ValueError as error:
            raise ValueError(f"{label} is not a hexadecimal fingerprint") from error
    else:
        raise ValueError(f"{label} has an unsupported fingerprint encoding")
    if not (0 <= number <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError(f"{label} is outside uint64")
    return number


def _restart_conditioning_diagnostics(
    capture: RichCapture,
    conditioning: ConditioningPayload,
    base_ep: np.ndarray,
    base_ap: np.ndarray,
    operators: list[tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]],
    base_ap_limit: float,
) -> dict[str, Any]:
    """Validate restart readback against the same rich-capture operator.

    A UINT32_MAX generation is the attempt latch. factor_valid is deliberately
    not that latch: an environment can retain a base/fallback factor status
    without having formed the additional q mode. In that case only the
    attempt-owned q/Aq and scalar inputs are required to be zero, while the
    factor/RAP/pivot fields are reported as non-authoritative scratch.
    """
    cm = conditioning.manifest
    aug = cm.get("restart_augmentation", {})
    if aug.get("diagnostics_available") is not True:
        return {"status": "unavailable", "message": aug.get("diagnostics_message")}
    names = (
        "restart_generations_by_environment.bin",
        "restart_attempt_counts_by_environment.bin",
        "restart_target_scales_by_environment.bin",
        "restart_raw_norm_squared_by_environment.bin",
        "restart_mode_scales_by_environment.bin",
        "restart_mode_valid_by_environment.bin",
        "restart_factor_valid_by_environment.bin",
        "restart_q_by_environment.bin",
        "restart_Aq_by_environment.bin",
        "restart_pivoted_RAP_by_environment.bin",
        "restart_pivots_by_environment.bin",
    )
    if not all(name in conditioning.arrays for name in names):
        raise ValueError("restart diagnostic metadata says available but per-environment arenas are missing")
    h = capture.header
    envs = int(h["environment_count"])
    nodes = int(h["node_count_per_environment"])
    base_modes = int(cm.get("mode_count", 0))
    stride = int(aug.get("augmented_mode_stride", -1))
    if (int(aug.get("diagnostic_environment_count", -1)),
        int(aug.get("nodes_per_environment", -1)), stride) != (
            envs, nodes, base_modes + 1):
        raise ValueError("restart diagnostics have inconsistent E/N/base-plus-one dimensions")
    generation = _array(conditioning, "restart_generations_by_environment", "<u4", (envs, 4))
    attempts = _array(conditioning, "restart_attempt_counts_by_environment", "<u4", (envs,))
    target = _array(conditioning, "restart_target_scales_by_environment", "<f4", (envs,))
    raw_n2 = _array(conditioning, "restart_raw_norm_squared_by_environment", "<f4", (envs,))
    scale = _array(conditioning, "restart_mode_scales_by_environment", "<f4", (envs,))
    mode_valid = _array(conditioning, "restart_mode_valid_by_environment", "<u4", (envs,))
    factor_valid = _array(conditioning, "restart_factor_valid_by_environment", "<u4", (envs,))
    q = _array(conditioning, "restart_q_by_environment", "<f4", (envs, nodes, 4))
    aq = _array(conditioning, "restart_Aq_by_environment", "<f4", (envs, nodes, 4))
    aug_lu = _array(conditioning, "restart_pivoted_RAP_by_environment", "<f4", (envs, stride, stride))
    aug_pivots = _array(conditioning, "restart_pivots_by_environment", "<u4", (envs, stride))
    results: list[dict[str, Any]] = []
    for env in range(envs):
        words = [int(x) for x in generation[env]]
        op, free, fixed, meta = operators[env]
        if all(word == UINT32_MAX for word in words):
            if attempts[env] != 0 or mode_valid[env] != 0:
                raise ValueError(f"environment {env}: no-attempt generation conflicts with attempt count/mode validity")
            scalars = (target[env], raw_n2[env], scale[env])
            if any(float(x) != 0.0 for x in scalars) or np.any(q[env] != 0.0) or np.any(aq[env] != 0.0):
                raise ValueError(f"environment {env}: no-attempt q/Aq or input scalar payload is nonzero")
            if factor_valid[env] not in (0, 1):
                raise ValueError(f"environment {env}: no-attempt factor status is not a boolean word")
            results.append({
                "environment": env,
                "attempt": False,
                "generation_words": words,
                "attempt_contract": "generation all UINT32_MAX; count/mode_valid zero; q/Aq and attempt scalars exactly zero",
                "non_attempt_factor_metadata": {
                    "factor_valid_word": int(factor_valid[env]),
                    "interpretation": "fallback or scratch status only; not evidence of a formed q mode or valid augmented factor",
                },
                "unattempted_augmented_factor_scratch": {
                    "max_abs_matrix_word": float(np.max(np.abs(aug_lu[env]))) if aug_lu[env].size else 0.0,
                    "nonzero_pivot_count": int(np.count_nonzero(aug_pivots[env])),
                    "interpretation": "reported but not used as an augmentation certificate without a generation",
                },
                "base_AP_max_fresh_action_relative_error": max(
                    (_metric(np.asarray(op @ base_ep[env, mode, free, :3].astype(np.float64).reshape(-1)),
                             base_ap[env, mode, free, :3].astype(np.float64).reshape(-1))["relative_l2"]
                     for mode in range(base_modes)), default=0.0),
            })
            continue

        if attempts[env] == 0 or mode_valid[env] != 1 or factor_valid[env] != 1:
            raise ValueError(f"environment {env}: attempted generation has inconsistent count/mode/factor validity")
        if (words[0] != int(cm.get("control_step", -1)) or
                words[2] != int(h.get("newton_iteration", -1))):
            raise ValueError(f"environment {env}: generation does not match captured control/Newton identity")
        if words[3] == UINT32_MAX:
            raise ValueError(f"environment {env}: attempted generation lacks a restart-cycle index")
        if not (float(target[env]) > 0.0 and float(raw_n2[env]) > 0.0 and float(scale[env]) > 0.0):
            raise ValueError(f"environment {env}: attempted mode has nonpositive scale inputs")
        if (np.any(q[env, fixed, :3] != 0.0) or np.any(aq[env, fixed, :3] != 0.0) or
                np.any(q[env, :, 3] != 0.0) or np.any(aq[env, :, 3] != 0.0)):
            raise ValueError(f"environment {env}: attempted q/Aq has nonzero constrained-node or float4.w values")
        qf = q[env, free, :3].astype(np.float64).reshape(-1)
        aqf = aq[env, free, :3].astype(np.float64).reshape(-1)
        fresh_aq = np.asarray(op @ qf).reshape(-1)
        aq_metric = _metric(fresh_aq, aqf)
        raw_expected = float(raw_n2[env]) * float(scale[env]) ** 2
        qnorm2 = float(qf @ qf)
        qscale_error = abs(qnorm2 - raw_expected) / max(qnorm2, raw_expected, 1e-300)
        pmat = base_ep[env, :, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, base_modes)
        apmat = base_ap[env, :, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, base_modes)
        cross = float(np.linalg.norm(pmat.T @ qf)) / max(float(np.linalg.norm(pmat)) * float(np.linalg.norm(qf)), 1e-300)
        true_rap = np.column_stack((pmat, qf)).T @ np.column_stack((apmat, aqf))
        piv = aug_pivots[env].astype(np.int64)
        if sorted(piv.tolist()) != list(range(stride)):
            raise ValueError(f"environment {env}: attempted augmented pivots are not a permutation")
        lu = aug_lu[env].astype(np.float64)
        reconstructed = (np.tril(lu, -1) + np.eye(stride)) @ np.triu(lu)
        lu_metric = _metric(reconstructed, true_rap[piv, :])
        base_metrics = [
            _metric(np.asarray(op @ base_ep[env, mode, free, :3].astype(np.float64).reshape(-1)),
                    base_ap[env, mode, free, :3].astype(np.float64).reshape(-1))["relative_l2"]
            for mode in range(base_modes)
        ]
        results.append({
            "environment": env, "attempt": True, "attempt_count": int(attempts[env]),
            "generation_words": words, "restart_cycle": int(words[3]),
            "mode_scale": float(scale[env]), "target_scale": float(target[env]),
            "raw_norm_squared": float(raw_n2[env]), "q_norm_squared": qnorm2,
            "q_norm_scale_relative_error": qscale_error,
            "relative_P_transpose_q": cross,
            "fresh_Aq": aq_metric,
            "fresh_Aq_gate_passed": aq_metric["relative_l2"] <= MAX_AQ_RELATIVE_LIMIT,
            "base_AP_max_fresh_action_relative_error": max(base_metrics, default=0.0),
            "base_AP_gate_passed": max(base_metrics, default=0.0) <= base_ap_limit,
            "augmented_RAP_LU_reconstruction": lu_metric,
            "augmented_RAP_gate_passed": lu_metric["relative_l2"] <= MAX_LU_RELATIVE_LIMIT,
            "RAP_singular_values": np.linalg.svd(true_rap, compute_uv=False).tolist(),
            "operator_assembly": meta["assembly"],
        })
    attempts_seen = sum(bool(row["attempt"]) for row in results)
    all_checks = all(
        not row["attempt"] or (
            row["fresh_Aq_gate_passed"] and row["base_AP_gate_passed"] and
            row["augmented_RAP_gate_passed"] and
            row["q_norm_scale_relative_error"] <= MAX_BASIS_RELATIVE_LIMIT and
            row["relative_P_transpose_q"] <= MAX_BASIS_RELATIVE_LIMIT)
        for row in results
    )
    return {
        "status": "augmentation_attempts_consistent_diagnostic" if attempts_seen and all_checks else
                  "augmentation_attempts_inconsistent_diagnostic" if attempts_seen else
                  "no_augmentation_attempt_observed",
        "generation_semantics": "only a non-UINT32_MAX generation with positive attempt count is an attempt; factor validity alone is not an attempt latch",
        "attempted_environment_count": attempts_seen,
        "environments": results,
        "thresholds": {"fresh_Aq_relative_l2": MAX_AQ_RELATIVE_LIMIT,
                       "base_AP_relative_l2": base_ap_limit,
                       "q_scale_and_P_orthogonality_relative": MAX_BASIS_RELATIVE_LIMIT,
                       "RAP_LU_reconstruction_relative_l2": MAX_LU_RELATIVE_LIMIT},
    }


def _center_hash(capture: RichCapture, env: int) -> str:
    h = capture.header
    env_nodes = sorted((r for r in capture.nodes if int(r["environment"]) == env), key=lambda r: int(r["node_id"]))
    env_tets = sorted((r for r in capture.tets if int(r["environment"]) == env), key=lambda r: int(r["tet_id"]))
    data = {
        "header": {k: h.get(k) for k in ("root", "newton_iteration", "world_fingerprint_hex",
                "environment_count", "node_count_per_environment", "tetrahedron_count_per_environment",
                "dt_seconds_by_environment", "materials", "parameters_by_environment")},
        "nodes": [{k: r.get(k) for k in ("node_id", "accepted", "center", "correction_delta_v_m_s",
                "center_residual_impulse_Ns", "production_A_delta_v_impulse_Ns")} for r in env_nodes],
        "tets": [{k: r.get(k) for k in ("tet_id", "material_index", "material_name", "nodes",
                "rest_volume_m3", "material_state_count", "material_parameter_count", "object_index",
                "tet_flags", "topology_generation", "cached_algorithmic_tangent_local",
                "inverse_rest_rows_per_m", "material_frame_quaternion_xyzw", "center")} for r in env_tets],
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256_bytes(canonical)


def compare_centers(candidate: RichCapture, baseline: RichCapture) -> dict[str, Any]:
    fields = ("root", "newton_iteration", "world_fingerprint_hex", "environment_count",
              "node_count_per_environment", "tetrahedron_count_per_environment",
              "dt_seconds_by_environment", "materials", "parameters_by_environment")
    a, b = candidate.header, baseline.header
    mismatched = [field for field in fields if a.get(field) != b.get(field)]
    if mismatched:
        return {"status": "inconclusive_header_identity_mismatch", "different_fields": mismatched}
    envs = int(a["environment_count"])
    ah = [_center_hash(candidate, e) for e in range(envs)]
    bh = [_center_hash(baseline, e) for e in range(envs)]
    if ah != bh:
        return {"status": "inconclusive_linearization_center_mismatch",
                "candidate_center_sha256": ah, "baseline_center_sha256": bh,
                "reason": "A changed Newton trajectory invalidates direct A/RHS parity comparison."}
    return {"status": "exact_center_match", "per_environment_center_sha256": ah}


def _bindings(operator_path: Path, regional: RegionalPayload,
              conditioning: ConditioningPayload) -> dict[str, Any]:
    return {
        "operator_capture": {"path": str(operator_path), "sha256": sha256_file(operator_path)},
        "regional_manifest": {"path": str(regional.manifest_path), "sha256": sha256_file(regional.manifest_path)},
        "regional_payload": {"path": str(regional.payload_path), "sha256": sha256_file(regional.payload_path)},
        "conditioning_manifest": {"path": str(conditioning.manifest_path), "sha256": sha256_file(conditioning.manifest_path)},
        "conditioning_arrays": {name: sha256_bytes(raw) for name, raw in sorted(conditioning.arrays.items())},
    }


def validate_conditioning_only(operator_path: Path, conditioning_path: Path,
                               assembler_path: Path, base_ap_limit: float,
                               regional_path: Path | None = None,
                               run_result_path: Path | None = None,
                               run_receipt_path: Path | None = None,
                               run_manifest_path: Path | None = None,
                               source_build_path: Path | None = None) -> dict[str, Any]:
    """Recompute base AP from the exact rich-capture center and conditioning P.

    This mode is useful when a feature-off run has no regional sidecar. It
    validates only the captured linear operator action and its input binding;
    it never infers that restart augmentation ran or that a failed physical
    step was accepted.
    """
    validate_thresholds(MAX_AQ_RELATIVE_LIMIT, base_ap_limit, MAX_BASIS_RELATIVE_LIMIT,
                        MAX_LU_RELATIVE_LIMIT)
    capture = load_rich_capture(operator_path)
    conditioning = load_conditioning(conditioning_path)
    h, cm = capture.header, conditioning.manifest
    envs = int(h["environment_count"])
    nodes = int(h["node_count_per_environment"])
    tets = int(h["tetrahedron_count_per_environment"])
    modes = int(cm.get("mode_count", 0))
    if modes <= 0:
        raise ValueError("conditioning manifest has no positive base mode count")
    if (int(cm.get("environment_count", -1)), int(cm.get("fem_node_count", -1))) != (envs, nodes):
        raise ValueError("conditioning and rich capture environment/node counts differ")
    if int(cm.get("source_physics_fingerprint", -1)) != int(str(h["world_fingerprint_hex"]), 16):
        raise ValueError("conditioning source fingerprint differs from rich capture world")
    if int(cm.get("device_program_fingerprint", -1)) != int(str(h["execution_fingerprint_hex"]), 16):
        raise ValueError("conditioning device fingerprint differs from rich capture execution")
    if "tetrahedron_count" in cm and int(cm["tetrahedron_count"]) != tets:
        raise ValueError("conditioning and rich capture tetrahedron counts differ")

    p = _array(conditioning, "P", "<f4", (modes, nodes, 4))
    ep = _array(conditioning, "effective_P", "<f4", (envs, modes, nodes, 4))
    ap = _array(conditioning, "AP", "<f4", (envs, modes, nodes, 4))
    node_raw = _array(conditioning, "candidate_nodes", "<f4", (envs, nodes, 5, 4))
    # Require the candidate FEM state to be exactly the rich capture's current
    # center, not merely a mesh or accepted checkpoint from the same run.
    for env in range(envs):
        rows = sorted((r for r in capture.nodes if int(r["environment"]) == env),
                      key=lambda r: int(r["node_id"]))
        reconstructed = np.asarray([r["center"]["raw_state_float4s"] for r in rows], dtype="<f4")
        if reconstructed.shape != (nodes, 5, 4) or not np.array_equal(reconstructed, node_raw[env]):
            raise ValueError(f"environment {env}: conditioning candidate nodes differ from rich capture center")

    assembler = load_operator_assembler(assembler_path)
    environments = []
    operator_records: list[tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]] = []
    all_pass = True
    for env in range(envs):
        _, _, operator, _, meta = assembler.validate_and_assemble(
            h, capture.nodes, capture.tets, capture.integrity, env)
        free = meta["free_nodes"]
        node_rows = sorted((r for r in capture.nodes if int(r["environment"]) == env),
                           key=lambda r: int(r["node_id"]))
        fixed = np.asarray([float(r["center"]["constraint_tag"]) != 0.0 for r in node_rows])
        operator_records.append((operator, free, fixed, meta))
        if np.any(p[:, :, 3] != 0) or np.any(ep[env, :, :, 3] != 0) or np.any(ap[env, :, :, 3] != 0):
            raise ValueError(f"environment {env}: base P/effective_P/AP has a nonzero float4.w lane")
        if np.any(p[:, fixed, :3] != 0) or np.any(ep[env, :, fixed, :3] != 0) or np.any(ap[env, :, fixed, :3] != 0):
            raise ValueError(f"environment {env}: base basis/action has nonzero constrained-node values")
        per_mode = []
        for mode in range(modes):
            direction = ep[env, mode, free, :3].astype(np.float64).reshape(-1)
            observed = ap[env, mode, free, :3].astype(np.float64).reshape(-1)
            expected = np.asarray(operator @ direction).reshape(-1)
            metric = _metric(expected, observed)
            metric["mode"] = mode
            metric["passed"] = metric["relative_l2"] <= base_ap_limit
            per_mode.append(metric)
        max_rel = max((row["relative_l2"] for row in per_mode), default=0.0)
        all_pass = all_pass and max_rel <= base_ap_limit
        environments.append({
            "environment": env,
            "free_node_count": int(len(free)),
            "fixed_node_count": int(np.count_nonzero(fixed)),
            "mode_count": modes,
            "P_vs_effective_P_relative_l2": float(
                np.linalg.norm(p.astype(np.float64) - ep[env].astype(np.float64)) /
                max(np.linalg.norm(p.astype(np.float64)), np.linalg.norm(ep[env].astype(np.float64)), 1e-300)),
            "base_AP_fresh_action_relative_limit": base_ap_limit,
            "base_AP_max_relative_error": max_rel,
            "base_AP_median_relative_error": float(np.median([r["relative_l2"] for r in per_mode])) if per_mode else 0.0,
            "base_AP_worst_mode": max(per_mode, key=lambda r: r["relative_l2"]) if per_mode else None,
            "base_AP_all_modes_passed": max_rel <= base_ap_limit,
            "operator_assembly": meta["assembly"],
        })

    sidecar_comparison = None
    if regional_path is not None:
        regional = load_regional(regional_path)
        rm = regional.manifest
        if (rm.get("root"), rm.get("iteration"), rm.get("environment_count"),
            rm.get("node_count"), rm.get("tetrahedron_count")) != (
                h.get("root"), h.get("newton_iteration"), envs, nodes, tets):
            raise ValueError("regional and rich capture identity/dimensions differ")
        sidecar_comparison = {}
        for name, actual in (("P", p), ("effective_P", ep), ("AP", ap)):
            ref = regional.arrays.get(name)
            sidecar_comparison[name] = {
                "present": ref is not None,
                "exact_match": bool(ref is not None and np.array_equal(ref, actual)),
            }

    run_binding = None
    if run_result_path is not None:
        result = json.loads(run_result_path.read_text())
        if int(result.get("compiled_world_fingerprint", -1)) != int(str(h["world_fingerprint_hex"]), 16):
            raise ValueError("run result fingerprint differs from rich operator capture")
        if (int(result.get("nodes", -1)), int(result.get("tetrahedra", -1))) != (nodes, tets):
            raise ValueError("run result mesh dimensions differ from rich operator capture")
        run_binding = {
            "result": {"path": str(run_result_path), "sha256": sha256_file(run_result_path)},
            "status": result.get("status"),
            "accepted_steps": int(result.get("accepted_steps", -1)),
            "physical_validation": result.get("physical_validation"),
        }
    if run_receipt_path is not None:
        receipt = json.loads(run_receipt_path.read_text())
        if run_binding is None:
            run_binding = {}
        run_binding["receipt"] = {"path": str(run_receipt_path), "sha256": sha256_file(run_receipt_path),
                                   "exit_code": receipt.get("exit_code"),
                                   "execution_started": receipt.get("execution_started")}
    if run_manifest_path is not None:
        run_manifest = json.loads(run_manifest_path.read_text())
        if int(run_manifest.get("compiled_world_fingerprint", -1)) != int(str(h["world_fingerprint_hex"]), 16):
            raise ValueError("run manifest fingerprint differs from rich operator capture")
        mesh = run_manifest.get("mesh", {})
        if (int(mesh.get("nodes", -1)), int(mesh.get("tetrahedra", -1))) != (nodes, tets):
            raise ValueError("run manifest mesh dimensions differ from rich operator capture")
        if run_binding is None:
            run_binding = {}
        run_binding["manifest"] = {"path": str(run_manifest_path), "sha256": sha256_file(run_manifest_path),
                                   "compiled_world_fingerprint": run_manifest.get("compiled_world_fingerprint"),
                                   "accepted_steps": run_manifest.get("accepted_steps"),
                                   "solver_status": run_manifest.get("solver", {}).get("status")}
    source_build_binding = None
    if source_build_path is not None:
        source_build = json.loads(source_build_path.read_text())
        if source_build.get("schema") != "numi.cardboard.source-build.v1" or source_build.get("success") is not True:
            raise ValueError("source build record is not a successful supported source-build receipt")
        if run_manifest_path is not None:
            run_manifest = json.loads(run_manifest_path.read_text())
            source_snapshot = source_build.get("source_snapshot")
            if not source_snapshot or str(source_snapshot) != str(run_manifest.get("source", {}).get("snapshot", source_snapshot)):
                # Some manifest versions omit the source path; when present it
                # must agree with the build receipt, and the exact build file
                # remains hash-bound even when it is absent.
                if run_manifest.get("source", {}).get("snapshot") is not None:
                    raise ValueError("source-build snapshot differs from run manifest")
        source_build_binding = {"path": str(source_build_path), "sha256": sha256_file(source_build_path),
                                "source_snapshot": source_build.get("source_snapshot"),
                                "success": True,
                                "source_sha256_manifest_sha256": sha256_bytes(
                                    json.dumps(source_build.get("source_sha256", {}), sort_keys=True,
                                               separators=(",", ":")).encode())}

    aug = cm.get("restart_augmentation", {})
    restart_validation = _restart_conditioning_diagnostics(
        capture, conditioning, ep, ap, operator_records, base_ap_limit)
    if not all_pass:
        overall_status = "base_operator_inconsistent_diagnostic"
    elif restart_validation["status"] == "augmentation_attempts_inconsistent_diagnostic":
        overall_status = "augmentation_capture_inconsistent_diagnostic"
    else:
        overall_status = "base_operator_consistent_diagnostic"
    return {
        "schema": "numi.cardboard.restart-augmentation-base-operator-check.v1",
        "status": overall_status,
        "bindings": {
            "operator_capture": {"path": str(operator_path), "sha256": sha256_file(operator_path)},
            "conditioning_manifest": {"path": str(conditioning_path), "sha256": sha256_file(conditioning_path)},
            "conditioning_arrays": {name: sha256_bytes(raw) for name, raw in sorted(conditioning.arrays.items())},
            "assembler": {"path": str(assembler_path), "sha256": sha256_file(assembler_path)},
            "regional_capture": ({"path": str(regional_path), "sha256": sha256_file(regional_path)}
                                 if regional_path is not None else None),
        },
        "source_build": source_build_binding,
        "linearization": {"root": h.get("root"), "newton_iteration": h.get("newton_iteration"),
                          "world_fingerprint_hex": h.get("world_fingerprint_hex"),
                          "execution_fingerprint_hex": h.get("execution_fingerprint_hex"),
                          "environment_count": envs, "node_count": nodes,
                          "tetrahedron_count": tets, "base_modes": modes,
                          "candidate_node_bytes_match_rich_center": True},
        "conditioning_generation": {"control_step": cm.get("control_step"),
                                    "regional_diagnostics_available": cm.get("regional_diagnostics_available"),
                                    "restart_diagnostics_available": aug.get("diagnostics_available"),
                                    "restart_diagnostics_message": aug.get("diagnostics_message")},
        "restart_augmentation_validation": restart_validation,
        "run": run_binding,
        "environments": environments,
        "regional_sidecar_exact_comparison": sidecar_comparison,
        "claim_boundary": "This recomputes the captured base A*P action at the exact rich-capture center. It does not establish restart augmentation, solver convergence, accepted physical state, or physical validation.",
    }


def validate(operator_path: Path, regional_path: Path, conditioning_path: Path,
             regional_bin: Path | None, assembler_path: Path,
             require_augmentation: bool, aq_limit: float, lu_limit: float,
             base_ap_limit: float = MAX_BASE_AP_RELATIVE_LIMIT,
             basis_limit: float = MAX_BASIS_RELATIVE_LIMIT) -> dict[str, Any]:
    validate_thresholds(aq_limit, base_ap_limit, basis_limit, lu_limit)
    capture = load_rich_capture(operator_path)
    regional = load_regional(regional_path, regional_bin)
    conditioning = load_conditioning(conditioning_path)
    h, rm, cm = capture.header, regional.manifest, conditioning.manifest
    envs = int(h["environment_count"])
    nodes = int(h["node_count_per_environment"])
    tets = int(h["tetrahedron_count_per_environment"])
    modes = int(rm["mode_count"])
    if (rm.get("root"), rm.get("iteration")) != (h.get("root"), h.get("newton_iteration")):
        raise ValueError("regional and rich capture root/Newton identity mismatch")
    if (rm.get("environment_count"), rm.get("node_count"), rm.get("tetrahedron_count")) != (envs, nodes, tets):
        raise ValueError("regional and rich capture dimensions mismatch")
    if (cm.get("environment_count"), cm.get("fem_node_count")) != (envs, nodes):
        raise ValueError("conditioning and rich capture environment/node counts differ")
    cond_fingerprints = (
        _fingerprint_u64(cm.get("source_physics_fingerprint"), "conditioning source fingerprint"),
        _fingerprint_u64(cm.get("device_program_fingerprint"), "conditioning device fingerprint"))
    regional_fingerprints = (
        _fingerprint_u64(rm.get("source_physics_fingerprint"), "regional source fingerprint"),
        _fingerprint_u64(rm.get("device_program_fingerprint"), "regional device fingerprint"))
    if cond_fingerprints != regional_fingerprints:
        raise ValueError("conditioning and regional captures have different source/device fingerprints")

    # The conditioning readback is an environment-major compact view. Verify
    # exact bytes against rich regional capture before recomputing any action.
    cond_p = _array(conditioning, "P", "<f4", (modes, nodes, 4))
    cond_ep = _array(conditioning, "effective_P", "<f4", (envs, modes, nodes, 4))
    cond_ap = _array(conditioning, "AP", "<f4", (envs, modes, nodes, 4))
    rp, rep, rap = (regional.arrays.get(k) for k in ("P", "effective_P", "AP"))
    if rp is None or not np.array_equal(rp, cond_p):
        raise ValueError("conditioning base P differs from rich regional P")
    if rep is None or not np.array_equal(rep, cond_ep):
        raise ValueError("conditioning compact effective_P differs from rich regional capture")
    if rap is None or not np.array_equal(rap, cond_ap):
        raise ValueError("conditioning compact AP differs from rich regional capture")
    cond_nodes = conditioning.arrays.get("candidate_nodes.bin")
    rich_nodes = regional.arrays.get("candidate_nodes")
    if cond_nodes is None or rich_nodes is None or len(cond_nodes) != envs * nodes * 80 or rich_nodes.tobytes() != cond_nodes:
        raise ValueError("conditioning candidate node bytes differ from rich regional linearization")

    aug = cm.get("restart_augmentation", {})
    available = aug.get("diagnostics_available") is True
    names = ("restart_generations_by_environment.bin", "restart_attempt_counts_by_environment.bin",
             "restart_target_scales_by_environment.bin", "restart_raw_norm_squared_by_environment.bin",
             "restart_mode_scales_by_environment.bin", "restart_mode_valid_by_environment.bin",
             "restart_factor_valid_by_environment.bin", "restart_q_by_environment.bin",
             "restart_Aq_by_environment.bin", "restart_pivoted_RAP_by_environment.bin",
             "restart_pivots_by_environment.bin")
    if available and not all(n in conditioning.arrays for n in names):
        raise ValueError("augmentation diagnostics available but one or more per-environment arenas are absent")
    base_report = {
        "schema": SCHEMA_REPORT,
        "bindings": _bindings(operator_path, regional, conditioning),
        "linearization": {"root": h.get("root"), "newton_iteration": h.get("newton_iteration"),
                          "world_fingerprint_hex": h.get("world_fingerprint_hex"),
                          "environment_count": envs, "node_count": nodes,
                          "tetrahedron_count": tets, "base_modes": modes},
    }
    if not available:
        base_report.update(status=("inconclusive_augmentation_diagnostics_unavailable"
                                  if require_augmentation else "baseline_capture_valid_augmentation_disabled"),
                           reason=aug.get("diagnostics_message", "augmentation diagnostics unavailable"))
        return base_report

    aug_stride = int(aug.get("augmented_mode_stride", -1))
    if (int(aug.get("diagnostic_environment_count", -1)), int(aug.get("nodes_per_environment", -1)),
        aug_stride) != (envs, nodes, modes + 1):
        raise ValueError("augmentation dimensions must be E x N with K=base_modes+1")
    if cm.get("regional_basis_storage_mode_stride") != aug_stride or cm.get("exported_base_modes_compacted") is not True:
        raise ValueError("conditioning manifest lacks the compact-base/augmented-stride contract")

    generations = _array(conditioning, "restart_generations_by_environment", "<u4", (envs, 4))
    attempts = _array(conditioning, "restart_attempt_counts_by_environment", "<u4", (envs,))
    target = _array(conditioning, "restart_target_scales_by_environment", "<f4", (envs,))
    raw_n2 = _array(conditioning, "restart_raw_norm_squared_by_environment", "<f4", (envs,))
    mode_scale = _array(conditioning, "restart_mode_scales_by_environment", "<f4", (envs,))
    mode_valid = _array(conditioning, "restart_mode_valid_by_environment", "<u4", (envs,))
    factor_valid = _array(conditioning, "restart_factor_valid_by_environment", "<u4", (envs,))
    q = _array(conditioning, "restart_q_by_environment", "<f4", (envs, nodes, 4))
    aq = _array(conditioning, "restart_Aq_by_environment", "<f4", (envs, nodes, 4))
    aug_lu = _array(conditioning, "restart_pivoted_RAP_by_environment", "<f4", (envs, aug_stride, aug_stride))
    aug_pivots = _array(conditioning, "restart_pivots_by_environment", "<u4", (envs, aug_stride))

    assembler = load_operator_assembler(assembler_path)
    env_results = []
    center_hashes = []
    for env in range(envs):
        _, _, operator, _, meta = assembler.validate_and_assemble(h, capture.nodes, capture.tets, capture.integrity, env)
        center_hash = _center_hash(capture, env)
        center_hashes.append(center_hash)
        free = meta["free_nodes"]
        base_errors = []
        for mode in range(modes):
            p = cond_ep[env, mode, free, :3].astype(np.float64).reshape(-1)
            ap = cond_ap[env, mode, free, :3].astype(np.float64).reshape(-1)
            fresh = np.asarray(operator @ p).reshape(-1)
            base_errors.append(_metric(fresh, ap))
        base_ap_max = max((x["relative_l2"] for x in base_errors), default=0.0)
        base_ap_passed = base_ap_max <= base_ap_limit
        g = generations[env].astype(np.uint64)
        no_attempt = bool(np.all(g == UINT32_MAX))
        if no_attempt:
            if attempts[env] != 0 or mode_valid[env] != 0:
                raise ValueError(f"environment {env}: UINT32_MAX generation conflicts with nonzero attempt or mode validity")
            # factor_valid is not an augmentation-attempt latch. The runtime
            # may leave a usable identity/base fallback factor in this slot
            # even when the extra q mode was never formed. Likewise, the
            # augmented RAP/pivot arenas are diagnostic scratch, not an
            # accepted mode certificate when the generation is absent. Never
            # promote either field into q validity. The attempt-owned scalar
            # and vector inputs must still be exact zero, which detects stale
            # q/Aq leakage across Newton generations.
            payload_values = (target[env], raw_n2[env], mode_scale[env], q[env], aq[env])
            if any(np.any(v != 0) for v in payload_values):
                raise ValueError(f"environment {env}: quiet no-attempt payload is not exactly zero")
            env_results.append({"environment": env, "attempt": False,
                                "generation_words": [int(x) for x in g],
                                "attempt_contract": "all generation words UINT32_MAX; attempt count and mode-valid flag zero; target/raw norm/mode scale and q/Aq exactly zero",
                                "non_attempt_factor_metadata": {
                                    "factor_valid_word": int(factor_valid[env]),
                                    "interpretation": "fallback or scratch status only; never evidence that q or an augmented RAP factor was valid",
                                },
                                "unattempted_augmented_rap": {
                                    "max_abs": float(np.max(np.abs(aug_lu[env]))) if aug_lu[env].size else 0.0,
                                    "nonzero_pivot_count": int(np.count_nonzero(aug_pivots[env])),
                                    "interpretation": "not evaluated as an augmentation factor because this environment has no attempted mode generation",
                                },
                                "base_AP_max_fresh_action_relative_error": base_ap_max,
                                "base_AP_gate": {"relative_limit": base_ap_limit, "passed": base_ap_passed},
                                "operator_center_sha256": center_hash})
            continue
        if attempts[env] == 0 or g[2] != int(h.get("newton_iteration", -1)) or g[0] != int(cm.get("control_step", -1)):
            raise ValueError(f"environment {env}: attempt count or Newton/control generation does not match capture")
        if g[3] == UINT32_MAX:
            raise ValueError(f"environment {env}: attempted generation has UINT32_MAX restart cycle")
        if mode_valid[env] != 1 or factor_valid[env] != 1:
            raise ValueError(f"environment {env}: attempted mode/factor is invalid")
        if not (1e-4 <= float(mode_scale[env]) <= 1e5) or target[env] <= 0 or raw_n2[env] <= 0:
            raise ValueError(f"environment {env}: mode scaling inputs are out of producer bounds")
        q_env, aq_env = q[env].astype(np.float64), aq[env].astype(np.float64)
        fixed = np.zeros(nodes, dtype=bool)
        for row in capture.nodes:
            if int(row["environment"]) == env:
                fixed[int(row["node_id"])] = float(row["center"]["constraint_tag"]) != 0.0
        if np.any(q_env[fixed, :3] != 0) or np.any(aq_env[fixed, :3] != 0) or np.any(q_env[:, 3] != 0) or np.any(aq_env[:, 3] != 0):
            raise ValueError(f"environment {env}: q/Aq has nonzero fixed-node or float4.w component")
        qf = q_env[free, :3].reshape(-1)
        aqf = aq_env[free, :3].reshape(-1)
        fresh_aq = np.asarray(operator @ qf).reshape(-1)
        aq_metrics = _metric(fresh_aq, aqf)
        raw_scale_rel = abs(float(qf @ qf) - float(raw_n2[env]) * float(mode_scale[env])**2) / max(float(qf @ qf), 1e-300)
        pmat = cond_ep[env, :, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        cross_ratio = float(np.linalg.norm(pmat.T @ qf)) / max(float(np.linalg.norm(pmat)) * float(np.linalg.norm(qf)), 1e-300)
        apmat = cond_ap[env, :, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        p_aug = np.column_stack((pmat, qf))
        ap_aug = np.column_stack((apmat, aqf))
        true_rap = p_aug.T @ ap_aug
        piv = aug_pivots[env].astype(np.int64)
        if sorted(piv.tolist()) != list(range(aug_stride)):
            raise ValueError(f"environment {env}: augmented pivots are not a permutation")
        lu = aug_lu[env].astype(np.float64)
        reconstructed = (np.tril(lu, -1) + np.eye(aug_stride)) @ np.triu(lu)
        lu_metrics = _metric(reconstructed, true_rap[piv, :])
        env_results.append({
            "environment": env, "attempt": True, "attempt_count": int(attempts[env]),
            "generation_words": [int(x) for x in g], "restart_cycle": int(g[3]),
            "mode_scale": float(mode_scale[env]), "target_scale": float(target[env]),
            "raw_norm_squared": float(raw_n2[env]), "q_norm_squared": float(qf @ qf),
            "q_norm_squared_vs_scale_relative_error": raw_scale_rel,
            "q_norm_scale_gate": {"relative_limit": basis_limit, "passed": raw_scale_rel <= basis_limit},
            "relative_P_transpose_q": cross_ratio,
            "basis_orthogonality_gate": {"relative_limit": basis_limit, "passed": cross_ratio <= basis_limit},
            "fresh_Aq": aq_metrics,
            "fresh_Aq_gate": {"relative_limit": aq_limit, "passed": aq_metrics["relative_l2"] <= aq_limit},
            "base_AP_max_fresh_action_relative_error": max((x["relative_l2"] for x in base_errors), default=0.0),
            "base_AP_gate": {"relative_limit": base_ap_limit, "passed": base_ap_passed},
            "augmented_RAP_LU_reconstruction": lu_metrics,
            "augmented_RAP_gate": {"relative_limit": lu_limit, "passed": lu_metrics["relative_l2"] <= lu_limit},
            "RAP_singular_values": np.linalg.svd(true_rap, compute_uv=False).tolist(),
            "operator_center_sha256": center_hash,
            "operator_assembly": meta["assembly"],
        })
    attempted = [r for r in env_results if r["attempt"]]
    if require_augmentation and not attempted:
        raise ValueError("diagnostics are present but no environment recorded an augmentation attempt")
    return {
        **base_report,
        "status": "internally_checked_diagnostic_capture",
        "linearization": {**base_report["linearization"], "per_environment_center_sha256": center_hashes,
                          "augmented_mode_stride": aug_stride},
        "thresholds": {"fresh_Aq_relative_l2": aq_limit, "base_AP_fresh_action_relative_l2": base_ap_limit,
                       "basis_scale_and_orthogonality_relative": basis_limit,
                       "RAP_LU_reconstruction_relative_l2": lu_limit,
                       "meaning": "offline diagnostic consistency thresholds, not native convergence/physical gates"},
        "base_capture_integrity": {
            "base_P_eff_AP_exact_match_between_sidecars": True,
            "candidate_node_bytes_exact_match_between_sidecars": True,
            "max_base_AP_fresh_action_relative_error": max(
                (r["base_AP_max_fresh_action_relative_error"] for r in env_results), default=None),
            "base_AP_passed_all_environments": all(r["base_AP_gate"]["passed"] for r in env_results),
            "fresh_Aq_passed_all_attempted_environments": all(r["fresh_Aq_gate"]["passed"] for r in attempted),
            "q_scale_passed_all_attempted_environments": all(r["q_norm_scale_gate"]["passed"] for r in attempted),
            "basis_orthogonality_passed_all_attempted_environments": all(r["basis_orthogonality_gate"]["passed"] for r in attempted),
            "augmented_RAP_passed_all_attempted_environments": all(r["augmented_RAP_gate"]["passed"] for r in attempted),
        },
        "environments": env_results,
        "qualification_boundary": "A consistent q/Aq and RAP readback verifies this captured diagnostic generation only; it does not establish native solver benefit, certificate acceptance, or physical validity.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-capture", required=True, type=Path)
    p.add_argument("--regional-json", type=Path)
    p.add_argument("--regional-bin", type=Path)
    p.add_argument("--conditioning-manifest", required=True, type=Path)
    p.add_argument("--baseline-operator-capture", type=Path)
    p.add_argument("--assembler", type=Path, default=DEFAULT_ASSEMBLER)
    p.add_argument("--require-augmentation", action="store_true")
    p.add_argument("--conditioning-only", action="store_true",
                   help="recompute base A*P from the rich capture without requiring a regional sidecar")
    p.add_argument("--run-result", type=Path,
                   help="optional source-matched probe result.json for diagnostic-status context")
    p.add_argument("--run-receipt", type=Path,
                   help="optional native receipt.json to bind execution outcome")
    p.add_argument("--run-manifest", type=Path,
                   help="optional explicit-strip run manifest to bind compiled world and mesh")
    p.add_argument("--source-build", type=Path,
                   help="optional successful source-build receipt to bind the compiled snapshot")
    p.add_argument("--fresh-aq-relative-limit", type=float, default=2e-3)
    p.add_argument("--base-ap-relative-limit", type=float, default=2e-3)
    p.add_argument("--basis-relative-limit", type=float, default=2e-3)
    p.add_argument("--rap-lu-relative-limit", type=float, default=2e-4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        if args.conditioning_only:
            report = validate_conditioning_only(
                args.operator_capture, args.conditioning_manifest, args.assembler,
                args.base_ap_relative_limit, args.regional_json, args.run_result,
                args.run_receipt, args.run_manifest, args.source_build)
        else:
            if args.regional_json is None:
                raise ValueError("--regional-json is required unless --conditioning-only is selected")
            report = validate(args.operator_capture, args.regional_json, args.conditioning_manifest,
                              args.regional_bin, args.assembler, args.require_augmentation,
                              args.fresh_aq_relative_limit, args.rap_lu_relative_limit,
                              args.base_ap_relative_limit, args.basis_relative_limit)
        if args.baseline_operator_capture:
            report["feature_off_center_comparison"] = compare_centers(
                load_rich_capture(args.operator_capture), load_rich_capture(args.baseline_operator_capture))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
        if report["status"].startswith("inconclusive"):
            return 3
        if report["status"] == "base_operator_inconsistent_diagnostic":
            return 2
        if report["status"] == "augmentation_capture_inconsistent_diagnostic":
            return 2
        if report["status"] == "internally_checked_diagnostic_capture":
            checks = report["base_capture_integrity"]
            if (not checks["base_AP_passed_all_environments"] or
                    not checks["fresh_Aq_passed_all_attempted_environments"] or
                    not checks["q_scale_passed_all_attempted_environments"] or
                    not checks["basis_orthogonality_passed_all_attempted_environments"] or
                    not checks["augmented_RAP_passed_all_attempted_environments"]):
                return 2
        return 0
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {"schema": SCHEMA_REPORT, "status": "validation_failed",
                   "error": f"{type(error).__name__}: {error}"}
        args.output.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
