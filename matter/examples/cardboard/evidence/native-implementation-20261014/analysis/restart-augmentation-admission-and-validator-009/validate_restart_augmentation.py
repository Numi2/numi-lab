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


def validate(operator_path: Path, regional_path: Path, conditioning_path: Path,
             regional_bin: Path | None, assembler_path: Path,
             require_augmentation: bool, aq_limit: float, lu_limit: float) -> dict[str, Any]:
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
        g = generations[env].astype(np.uint64)
        no_attempt = bool(np.all(g == UINT32_MAX))
        if no_attempt:
            if attempts[env] != 0 or mode_valid[env] != 0 or factor_valid[env] != 0:
                raise ValueError(f"environment {env}: UINT32_MAX generation conflicts with nonzero attempt/validity")
            payload_values = (target[env], raw_n2[env], mode_scale[env], q[env], aq[env], aug_lu[env], aug_pivots[env])
            if any(np.any(v != 0) for v in payload_values):
                raise ValueError(f"environment {env}: quiet no-attempt payload is not exactly zero")
            env_results.append({"environment": env, "attempt": False,
                                "generation_words": [int(x) for x in g],
                                "contract": "all generation words UINT32_MAX; scalar/vector/matrix/pivot payload and flags zero",
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
            "relative_P_transpose_q": cross_ratio,
            "fresh_Aq": aq_metrics,
            "fresh_Aq_gate": {"relative_limit": aq_limit, "passed": aq_metrics["relative_l2"] <= aq_limit},
            "base_AP_max_fresh_action_relative_error": max((x["relative_l2"] for x in base_errors), default=0.0),
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
        "thresholds": {"fresh_Aq_relative_l2": aq_limit, "RAP_LU_reconstruction_relative_l2": lu_limit,
                       "meaning": "offline diagnostic consistency thresholds, not native convergence/physical gates"},
        "base_capture_integrity": {
            "base_P_eff_AP_exact_match_between_sidecars": True,
            "candidate_node_bytes_exact_match_between_sidecars": True,
            "max_base_AP_fresh_action_relative_error": max(
                (r["base_AP_max_fresh_action_relative_error"] for r in attempted), default=None),
            "fresh_Aq_passed_all_attempted_environments": all(r["fresh_Aq_gate"]["passed"] for r in attempted),
            "augmented_RAP_passed_all_attempted_environments": all(r["augmented_RAP_gate"]["passed"] for r in attempted),
        },
        "environments": env_results,
        "qualification_boundary": "A consistent q/Aq and RAP readback verifies this captured diagnostic generation only; it does not establish native solver benefit, certificate acceptance, or physical validity.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-capture", required=True, type=Path)
    p.add_argument("--regional-json", required=True, type=Path)
    p.add_argument("--regional-bin", type=Path)
    p.add_argument("--conditioning-manifest", required=True, type=Path)
    p.add_argument("--baseline-operator-capture", type=Path)
    p.add_argument("--assembler", type=Path, default=DEFAULT_ASSEMBLER)
    p.add_argument("--require-augmentation", action="store_true")
    p.add_argument("--fresh-aq-relative-limit", type=float, default=2e-3)
    p.add_argument("--rap-lu-relative-limit", type=float, default=2e-4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        report = validate(args.operator_capture, args.regional_json, args.conditioning_manifest,
                          args.regional_bin, args.assembler, args.require_augmentation,
                          args.fresh_aq_relative_limit, args.rap_lu_relative_limit)
        if args.baseline_operator_capture:
            report["feature_off_center_comparison"] = compare_centers(
                load_rich_capture(args.operator_capture), load_rich_capture(args.baseline_operator_capture))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
        if report["status"].startswith("inconclusive"):
            return 3
        if report["status"] == "internally_checked_diagnostic_capture":
            checks = report["base_capture_integrity"]
            if not checks["fresh_Aq_passed_all_attempted_environments"] or not checks["augmented_RAP_passed_all_attempted_environments"]:
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
