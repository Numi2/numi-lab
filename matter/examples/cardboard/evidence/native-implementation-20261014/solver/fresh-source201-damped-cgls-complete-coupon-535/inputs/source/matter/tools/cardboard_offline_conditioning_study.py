#!/usr/bin/env python3
"""Compare native balanced RAP and bounded graph-Schwarz on one captured A.

This is an offline diagnostic.  It does not modify Matter, accepted histories,
or the production operator.  It requires an operator-audit report (with the
hashed CSR/vector/topology sidecars written by
cardboard_cached_operator_audit.py) and a regional-conditioning export from
the exact same source-bound candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg, sparse


SCHEMA = "numi.cardboard.offline-conditioning-study.v1"
MAX_HOST_BYTES = 1 << 30
REGIONAL_RAP_SCHEMA = "numi.matter.regional_rap_capture.v1"
REGIONAL_CAPTURE_POINT = "after_regional_rap_factor_barrier_before_fgmres_correction"
REGIONAL_OPERATOR_GENERATION = (
    "candidate state, accepted history, scheduler, deformation words, tangent cache, AP and RAP copied from the same ordered command-buffer generation"
)
NMMATTER_STATUS_GPU_BYTES = 48  # 32 bytes of uints followed by one aligned float4.
REGIONAL_DIGEST_ALGORITHM = (
    "SHA-256 over raw little-endian environment slices concatenated in this exact order: candidate_nodes, material_history, scheduler_exponents, deformation_high, deformation_low, then algorithmic_tangent_cache and algorithmic_tangent_cache_valid when tangent_cache_used=true"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")


def rowwise_fsum_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Accumulate each small dense row with independent scalar fsum operations.

    On the Apple Accelerate/NumPy build used for this study, even harmless
    finite ``matrix @ vector`` calls set divide-by-zero/overflow/invalid FP
    flags and emit warnings.  The rowwise fsum path avoids that backend call;
    it is a numerical implementation change for this offline comparison only.
    """
    lhs = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(vector, dtype=np.float64)
    if lhs.ndim != 2 or rhs.ndim != 1 or lhs.shape[1] != rhs.size:
        raise ValueError("rowwise fsum matvec dimensions are inconsistent")
    require_finite("rowwise fsum matrix", lhs)
    require_finite("rowwise fsum vector", rhs)
    result = np.fromiter(
        (math.fsum(float(lhs[i, j]) * float(rhs[j]) for j in range(rhs.size))
         for i in range(lhs.shape[0])), dtype=np.float64, count=lhs.shape[0])
    require_finite("rowwise fsum result", result)
    return result


def _manifest_files(folder: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "numi.cardboard.regional-conditioning.v1":
        raise ValueError("unsupported native RAP manifest schema")
    if manifest.get("byte_order") != "native_little_endian":
        raise ValueError("RAP export byte order is unsupported")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("RAP manifest lacks its file inventory")
    resolved: dict[str, Path] = {}
    for row in files:
        name = row.get("name")
        if not isinstance(name, str) or Path(name).name != name or name in resolved:
            raise ValueError("RAP manifest has an unsafe or duplicate filename")
        path = folder / name
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise ValueError(f"RAP export hash/file check failed for {name}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"RAP export byte count mismatch for {name}")
        if int(row.get("count", -1)) * int(row.get("stride_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"RAP export count/stride mismatch for {name}")
        resolved[name.removesuffix(".bin")] = path
    required = {"candidate_nodes", "P", "effective_P", "AP", "pivoted_RAP",
                "pivots", "factor_valid", "node_block_inverse"}
    missing = required - resolved.keys()
    if missing:
        raise ValueError(f"RAP export is missing required arrays: {sorted(missing)}")
    return manifest, resolved


def load_hashed_operator(report_path: Path) -> tuple[dict, sparse.csr_matrix, dict, dict, dict]:
    report = json.loads(report_path.read_text())
    if report.get("schema") != "numi.cardboard.cached-operator-audit.v1":
        raise ValueError("unsupported cached-operator report schema")
    artifacts = report.get("matrix_artifacts", {}).get("files_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("operator report does not bind its matrix sidecars")
    base = report_path.parent
    paths = {}
    for name in (report_path.stem + ".operator.npz",
                 report_path.stem + ".vectors.npz",
                 report_path.stem + ".topology.npz"):
        path = base / name
        expected = artifacts.get(name)
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"operator sidecar missing or hash mismatch: {name}")
        paths[name] = path
    matrix = sparse.load_npz(paths[report_path.stem + ".operator.npz"]).tocsr()
    with np.load(paths[report_path.stem + ".vectors.npz"], allow_pickle=False) as z:
        vectors = {key: np.array(z[key]) for key in z.files}
    with np.load(paths[report_path.stem + ".topology.npz"], allow_pickle=False) as z:
        topology = {key: np.array(z[key]) for key in z.files}
    return report, matrix, vectors, topology, {k: sha256(v) for k, v in paths.items()}


def validate_pairing(report: dict, matrix: sparse.csr_matrix, vectors: dict,
                     topology: dict, rap: dict) -> dict:
    selected = report.get("capture_selection", {})
    binding = report.get("binding", {})
    diagnostic = binding.get("operator_fd_iteration", {})
    op_meta = report.get("operator", {})
    if not selected or not diagnostic:
        raise ValueError("operator report lacks sealed capture selection or native fingerprint")
    if rap.get("control_step") != selected.get("root"):
        raise ValueError("native RAP control step does not match captured operator root")
    if rap.get("environment_count", 0) <= selected.get("environment", -1):
        raise ValueError("selected environment is outside native RAP export")
    if rap.get("fem_node_count") != op_meta.get("node_count"):
        raise ValueError("native RAP and operator node counts differ")
    if int(str(rap.get("source_physics_fingerprint")), 10) != int(diagnostic["world_fingerprint"]):
        raise ValueError("native RAP world/source fingerprint differs from operator capture")
    if int(str(rap.get("device_program_fingerprint")), 10) != int(diagnostic["execution_fingerprint"]):
        raise ValueError("native RAP device-program fingerprint differs from operator capture")
    if not binding.get("capture_run_fingerprints_match"):
        raise ValueError("operator capture was not admitted against its source-bound stderr diagnostic")
    parity = op_meta.get("action_parity", {})
    rel = parity.get("relative_l2_error")
    if rel is None or not math.isfinite(float(rel)):
        raise ValueError("operator action parity is absent or nonfinite")
    if float(rel) > 1.0e-3:
        raise ValueError(f"operator/native action parity {rel:.6g} exceeds the fixed 1e-3 admission gate")
    free_nodes = np.asarray(vectors.get("free_node_ids"), dtype=np.int64)
    rhs = np.asarray(vectors.get("native_rhs_free_impulse_Ns"), dtype=np.float64)
    candidate = np.asarray(vectors.get("captured_node_state_f32"), dtype=np.float32)
    if free_nodes.ndim != 1 or rhs.shape != (matrix.shape[0],) or \
            candidate.ndim != 2 or candidate.shape[1] != 20:
        raise ValueError("operator vector/topology sidecars lack native RHS or candidate state")
    require_finite("captured native RHS", rhs)
    if np.linalg.norm(rhs) == 0.0:
        raise ValueError("native center FGMRES RHS is zero; cannot compare preconditioners")
    if candidate.shape[0] != int(rap["fem_node_count"]):
        raise ValueError("candidate-state sidecar differs from RAP node count")
    return {"root_control_step": selected["root"],
            "newton_iteration": selected["newton_iteration"],
            "environment": selected["environment"],
            "world_fingerprint": int(diagnostic["world_fingerprint"]),
            "execution_fingerprint": int(diagnostic["execution_fingerprint"]),
            "native_action_relative_l2_error": float(rel),
            "native_rhs_source": "center_residual_impulse_Ns captured immediately before FGMRES",
            "native_candidate_state_match_required": True}


def _dtype_layout(dtype: str, element_bytes: int) -> tuple[str, int]:
    layouts = {
        "<f4x4": ("<f4", 16),
        "<f4": ("<f4", 4),
        "<u4": ("<u4", 4),
        "struct:NMFEMNodeStateGPU": ("<f4", 80),
        "struct:NMMatterStatusGPU": ("|u1", NMMATTER_STATUS_GPU_BYTES),
    }
    if dtype not in layouts:
        raise ValueError(f"regional sidecar has unsupported dtype {dtype!r}")
    numpy_dtype, expected_bytes = layouts[dtype]
    if int(element_bytes) != expected_bytes:
        raise ValueError(f"regional sidecar dtype {dtype!r} has incorrect element_bytes")
    return numpy_dtype, expected_bytes


def _regional_array_shapes(manifest: dict) -> dict[str, tuple[int, ...]]:
    envs = int(manifest["environment_count"])
    nodes = int(manifest["node_count"])
    tets = int(manifest["tetrahedron_count"])
    modes = int(manifest["mode_count"])
    stride = int(manifest["material_state_stride"])
    tangent = manifest.get("tangent_cache_used") is True
    shapes = {
        "P": (modes, nodes),
        "effective_P": (envs, modes, nodes),
        "AP": (envs, modes, nodes),
        "pivoted_RAP": (envs, modes, modes),
        "pivots": (envs, modes),
        "factor_valid": (envs,),
        "node_block_inverse": (envs, nodes, 3),
        "candidate_nodes": (envs, nodes),
        "material_history": (envs, tets, stride),
        "scheduler_exponents": (envs, tets),
        "deformation_high": (envs, tets, 3),
        "deformation_low": (envs, tets, 3),
        "production_status": (envs,),
    }
    if tangent:
        shapes["algorithmic_tangent_cache"] = (envs, tets, 81)
        shapes["algorithmic_tangent_cache_valid"] = (envs, tets)
    return shapes


def _read_regional_payload(manifest_path: Path) -> tuple[dict, dict[str, np.ndarray], dict, bytes]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != REGIONAL_RAP_SCHEMA:
        raise ValueError("unsupported selected-Newton regional RAP sidecar schema")
    if manifest.get("capture_point") != REGIONAL_CAPTURE_POINT:
        raise ValueError("regional RAP was not captured at the selected pre-FGMRES factor barrier")
    if manifest.get("byte_order") != "little_endian":
        raise ValueError("regional RAP sidecar is not explicitly little-endian")
    if manifest.get("operator_generation") != REGIONAL_OPERATOR_GENERATION:
        raise ValueError("regional RAP does not declare a same-command-buffer operator generation")
    if manifest.get("candidate_linearization_digest_algorithm") != REGIONAL_DIGEST_ALGORITHM:
        raise ValueError("regional RAP candidate digest algorithm is unknown")
    if manifest.get("linearization_generation_encoding") != "(micro_generation << 32) | newton_iteration":
        raise ValueError("regional RAP linearization generation encoding is unsupported")
    if not isinstance(manifest.get("tangent_cache_used"), bool):
        raise ValueError("regional RAP tangent-cache policy is missing or malformed")
    payload_meta = manifest.get("payload")
    if not isinstance(payload_meta, dict):
        raise ValueError("regional sidecar lacks its payload manifest")
    payload_path = Path(str(payload_meta.get("file", "")))
    expected_payload = manifest_path.with_suffix(".bin")
    if not payload_path.is_absolute() or payload_path.resolve() != expected_payload.resolve():
        raise ValueError("regional payload path does not match the selected manifest basename")
    if not payload_path.is_file():
        raise ValueError("regional payload file is missing")
    payload = payload_path.read_bytes()
    if len(payload) > MAX_HOST_BYTES:
        raise ValueError("regional payload exceeds the bounded 1 GiB host analysis cap")
    if len(payload) != int(payload_meta.get("byte_count", -1)):
        raise ValueError("regional payload byte count differs from manifest")
    if sha256(payload_path) != payload_meta.get("sha256"):
        raise ValueError("regional payload SHA-256 does not match manifest")

    envs = int(manifest.get("environment_count", 0))
    nodes = int(manifest.get("node_count", 0))
    tets = int(manifest.get("tetrahedron_count", 0))
    modes = int(manifest.get("mode_count", 0))
    stride = int(manifest.get("material_state_stride", 0))
    if min(envs, nodes, tets, modes, stride) <= 0:
        raise ValueError("regional sidecar dimensions/stride must be positive")
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("regional readback exceeds its authored 32 MiB capture bound")

    inventory = manifest.get("arrays")
    if not isinstance(inventory, list):
        raise ValueError("regional sidecar lacks array inventory")
    required = set(_regional_array_shapes(manifest))
    by_name: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    cursor = 0
    expected_shapes = _regional_array_shapes(manifest)
    for row in inventory:
        if not isinstance(row, dict):
            raise ValueError("regional sidecar array descriptor is malformed")
        name = row.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("regional sidecar has unsafe or duplicate array names")
        if name not in expected_shapes:
            raise ValueError(f"regional sidecar contains unsupported array {name!r}")
        shape = row.get("shape")
        if not isinstance(shape, list) or tuple(int(x) for x in shape) != expected_shapes[name]:
            raise ValueError(f"regional sidecar {name} shape differs from the captured world layout")
        dtype = str(row.get("dtype", ""))
        numpy_dtype, element_bytes = _dtype_layout(dtype, int(row.get("element_bytes", -1)))
        count = math.prod(expected_shapes[name])
        expected_bytes = count * element_bytes
        offset = int(row.get("offset", -1))
        byte_count = int(row.get("byte_count", -1))
        if offset != cursor or byte_count != expected_bytes:
            raise ValueError(f"regional sidecar {name} is not tightly packed at its declared size")
        if offset < 0 or byte_count < 0 or offset + byte_count > len(payload):
            raise ValueError(f"regional sidecar {name} range is out of payload bounds")
        raw = payload[offset:offset + byte_count]
        if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            raise ValueError(f"regional sidecar array hash mismatch for {name}")
        if dtype == "struct:NMMatterStatusGPU":
            value = np.frombuffer(raw, dtype="|u1").reshape(
                expected_shapes[name] + (NMMATTER_STATUS_GPU_BYTES,)).copy()
        elif dtype == "struct:NMFEMNodeStateGPU":
            value = np.frombuffer(raw, dtype="<f4").reshape(expected_shapes[name] + (20,)).copy()
        elif dtype == "<f4x4":
            value = np.frombuffer(raw, dtype="<f4").reshape(expected_shapes[name] + (4,)).copy()
        else:
            value = np.frombuffer(raw, dtype=numpy_dtype).reshape(expected_shapes[name]).copy()
        if dtype != "struct:NMMatterStatusGPU":
            require_finite(f"regional RAP {name}", value)
        by_name[name] = row
        arrays[name] = value
        cursor += byte_count
    missing = required - by_name.keys()
    if missing:
        raise ValueError(f"regional sidecar is missing required arrays: {sorted(missing)}")
    if cursor != len(payload):
        raise ValueError("regional array inventory does not cover the complete payload")
    if len(by_name) != len(inventory):
        raise ValueError("regional sidecar array inventory is inconsistent")
    hashes = {str(manifest_path.name): sha256(manifest_path), payload_path.name: sha256(payload_path)}
    return manifest, arrays, hashes, payload


def _capture_tetrahedra(capture_path: Path, report: dict, manifest: dict,
                        arrays: dict[str, np.ndarray], payload: bytes) -> dict:
    binding = report.get("binding", {})
    expected_sha = binding.get("capture_sha256")
    if not capture_path.is_file() or not isinstance(expected_sha, str) or sha256(capture_path) != expected_sha:
        raise ValueError("rich operator capture is missing or differs from its sealed SHA-256")
    selection = report.get("capture_selection", {})
    diagnostic = binding.get("operator_fd_iteration", {})
    if diagnostic.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("rich operator capture did not preserve its production input guard")
    header = None
    integrity = None
    tet_rows: dict[tuple[int, int], dict] = {}
    with capture_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("record")
            if kind == "header":
                if header is not None:
                    raise ValueError("rich operator capture has duplicate headers")
                header = row
            elif kind == "integrity":
                if integrity is not None:
                    raise ValueError("rich operator capture has duplicate integrity records")
                integrity = row
            elif kind == "tetrahedron":
                key = (int(row.get("environment", -1)), int(row.get("tet_id", -1)))
                if key in tet_rows:
                    raise ValueError("rich operator capture has duplicate tetrahedron IDs")
                tet_rows[key] = row
    if not isinstance(header, dict) or header.get("schema") != "numi.cardboard.fem-newton-capture.v1":
        raise ValueError("unsupported or missing rich operator capture header")
    if not isinstance(integrity, dict) or integrity.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("rich operator capture lacks a passing final integrity record")
    if header.get("root") != selection.get("root") or header.get("newton_iteration") != selection.get("newton_iteration"):
        raise ValueError("rich operator capture root/iteration differs from the sealed operator report")
    if int(header.get("world_fingerprint_hex", "0"), 16) != int(manifest["source_physics_fingerprint"], 16):
        raise ValueError("rich operator capture world fingerprint differs from regional RAP")
    if int(header.get("execution_fingerprint_hex", "0"), 16) != int(manifest["device_program_fingerprint"], 16):
        raise ValueError("rich operator capture device fingerprint differs from regional RAP")
    if int(header.get("environment_count", 0)) != int(manifest["environment_count"]) or \
       int(header.get("node_count_per_environment", 0)) != int(manifest["node_count"]) or \
       int(header.get("tetrahedron_count_per_environment", 0)) != int(manifest["tetrahedron_count"]):
        raise ValueError("rich operator capture dimensions differ from regional RAP")
    if np.float32(header.get("dt_seconds", 0.0)) != np.float32(manifest.get("dt_seconds", float("nan"))):
        raise ValueError("rich operator capture timestep differs from regional RAP")
    selected_env = int(report.get("capture_selection", {}).get("environment", -1))
    by_environment = header.get("dt_seconds_by_environment")
    if not isinstance(by_environment, list) or selected_env < 0 or selected_env >= len(by_environment) or \
       np.float32(by_environment[selected_env]) != np.float32(manifest.get("dt_seconds", float("nan"))):
        raise ValueError("selected rich-capture environment timestep differs from regional RAP")
    if manifest.get("tangent_cache_used") is True:
        cache_capture = header.get("cached_algorithmic_tangent_capture", {})
        if cache_capture.get("available") is not True or \
           cache_capture.get("snapshot_bitwise_guard_included") is not True:
            raise ValueError("regional tangent cache lacks source-capture availability/bitwise guard")

    env = selected_env
    if env < 0 or env >= int(manifest["environment_count"]):
        raise ValueError("selected environment is outside regional candidate state")
    tet_count = int(manifest["tetrahedron_count"])
    expected_ids = {(env, tet) for tet in range(tet_count)}
    if not expected_ids.issubset(tet_rows):
        raise ValueError("rich operator capture omits one or more selected-environment tetrahedra")
    side_high = arrays["deformation_high"][env, :, :, :3]
    side_low = arrays["deformation_low"][env, :, :, :3]
    high_expected = np.empty((tet_count, 3, 3), dtype="<f4")
    low_expected = np.empty((tet_count, 3, 3), dtype="<f4")
    history_expected = np.empty((tet_count, int(manifest["material_state_stride"])), dtype="<f4")
    tangent_used = manifest.get("tangent_cache_used") is True
    if tangent_used:
        tangent_expected = np.empty((tet_count, 81), dtype="<f4")
        tangent_valid_expected = np.empty((tet_count,), dtype="<u4")
    for tet in range(tet_count):
        row = tet_rows[(env, tet)]
        center = row.get("center", {})
        high_expected[tet] = np.asarray(center.get("F_local"), dtype="<f4").reshape(3, 3)
        low_expected[tet] = np.asarray(center.get("F_local_low"), dtype="<f4").reshape(3, 3)
        accepted = np.asarray(center.get("accepted_state"), dtype="<f4")
        if accepted.shape != (history_expected.shape[1],) or int(row.get("material_state_count", -1)) != accepted.size:
            raise ValueError("rich operator capture accepted material history has a different stride")
        history_expected[tet] = accepted
        if tangent_used:
            tangent = row.get("cached_algorithmic_tangent_local", {})
            matrix_values = np.asarray(tangent.get("matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"), dtype="<f4")
            if matrix_values.shape != (81,):
                raise ValueError("rich operator capture lacks the full cached 9x9 tangent")
            tangent_expected[tet] = matrix_values
            tangent_valid_expected[tet] = 1 if tangent.get("valid") is True else 0
    if not np.array_equal(side_high.view("<u4"), high_expected.view("<u4")):
        raise ValueError("regional deformation high words differ from rich capture F_local")
    if not np.array_equal(side_low.view("<u4"), low_expected.view("<u4")):
        raise ValueError("regional deformation low words differ from rich capture F_local_low")
    side_history = arrays["material_history"][env]
    if not np.array_equal(side_history.view("<u4"), history_expected.view("<u4")):
        raise ValueError("regional accepted material history differs from rich capture")
    if tangent_used:
        if not np.array_equal(arrays["algorithmic_tangent_cache"][env].view("<u4"), tangent_expected.view("<u4")):
            raise ValueError("regional tangent cache differs from rich capture cached tangent")
        if not np.array_equal(arrays["algorithmic_tangent_cache_valid"][env], tangent_valid_expected):
            raise ValueError("regional tangent validity differs from rich capture")

    candidates = manifest.get("candidate_linearization_sha256")
    if not isinstance(candidates, list) or len(candidates) != int(manifest["environment_count"]):
        raise ValueError("regional sidecar candidate digest inventory is incomplete")
    candidate_by_env = {}
    for row in candidates:
        i = int(row.get("environment", -1))
        if i in candidate_by_env or i < 0 or i >= int(manifest["environment_count"]):
            raise ValueError("regional sidecar candidate digest has duplicate/out-of-range environment")
        candidate_by_env[i] = row
    if set(candidate_by_env) != set(range(int(manifest["environment_count"]))):
        raise ValueError("regional sidecar candidate digest environment coverage is incomplete")
    canonical = ["candidate_nodes", "material_history", "scheduler_exponents", "deformation_high", "deformation_low"]
    if tangent_used:
        canonical += ["algorithmic_tangent_cache", "algorithmic_tangent_cache_valid"]
    for i, entry in candidate_by_env.items():
        chunks = []
        for name in canonical:
            descriptor = next(x for x in manifest["arrays"] if x["name"] == name)
            per_env_bytes = int(descriptor["byte_count"]) // int(manifest["environment_count"])
            start = int(descriptor["offset"]) + i * per_env_bytes
            chunks.append(payload[start:start + per_env_bytes])
        digest = hashlib.sha256(b"".join(chunks)).hexdigest()
        node_row = next(x for x in manifest["arrays"] if x["name"] == "candidate_nodes")
        node_per_env = int(node_row["byte_count"]) // int(manifest["environment_count"])
        node_start = int(node_row["offset"]) + i * node_per_env
        node_digest = hashlib.sha256(payload[node_start:node_start + node_per_env]).hexdigest()
        if digest != entry.get("sha256") or node_digest != entry.get("candidate_nodes_sha256"):
            raise ValueError(f"regional candidate linearization digest mismatch for environment {i}")
    return {"capture_sha256": expected_sha, "capture_path": str(capture_path),
            "capture_point": manifest["capture_point"], "capture_header": header}


def load_selected_newton_rap(manifest_path: Path, report: dict,
                             matrix: sparse.csr_matrix, vectors: dict) -> tuple[dict, dict, dict]:
    manifest, arrays, hashes, payload = _read_regional_payload(manifest_path)
    selected = report.get("capture_selection", {})
    binding = report.get("binding", {})
    diagnostic = binding.get("operator_fd_iteration", {})
    if not selected or not diagnostic:
        raise ValueError("operator report lacks sealed root/Newton/environment binding")
    source_hashes = binding.get("source_hashes_verified_against_prediction")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("operator report lacks source files verified against its prediction")
    if report.get("capture_integrity", {}).get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("operator report does not certify unchanged selected capture inputs")
    if int(manifest.get("root", -1)) != int(selected.get("root", -2)) or \
       int(manifest.get("iteration", -1)) != int(selected.get("newton_iteration", -2)):
        raise ValueError("regional RAP root/Newton iteration differs from selected operator capture")
    selected_env = int(selected.get("environment", -1))
    if selected_env < 0 or int(manifest.get("environment_count", 0)) <= selected_env:
        raise ValueError("selected environment is outside selected-Newton regional RAP")
    if int(manifest.get("node_count", -1)) != int(report.get("operator", {}).get("node_count", -2)) or \
       int(manifest.get("tetrahedron_count", -1)) != int(report.get("operator", {}).get("tetrahedron_count", -2)):
        raise ValueError("regional RAP node/tetrahedron counts differ from captured operator")
    if int(str(manifest.get("source_physics_fingerprint")), 16) != int(diagnostic["world_fingerprint"]):
        raise ValueError("regional RAP source fingerprint differs from operator capture")
    if int(str(manifest.get("device_program_fingerprint")), 16) != int(diagnostic["execution_fingerprint"]):
        raise ValueError("regional RAP device-program fingerprint differs from operator capture")
    generation = (int(manifest.get("micro_generation", -1)) << 32) | int(manifest.get("iteration", -1))
    if int(manifest.get("micro_generation", -1)) < 0 or \
       int(manifest.get("substep", -1)) < 0 or \
       int(manifest.get("microtick", -1)) < 0 or \
       int(manifest.get("microtick_count", 0)) <= 0 or \
       not math.isfinite(float(manifest.get("dt_seconds", float("nan")))) or \
       float(manifest.get("dt_seconds", 0.0)) <= 0.0 or \
       generation < 0 or generation != int(manifest.get("linearization_generation", -2)):
        raise ValueError("regional RAP linearization generation does not match micro-generation/Newton iteration")
    if np.float32(manifest["dt_seconds"]) != np.float32(report.get("operator", {}).get("dt_seconds", float("nan"))):
        raise ValueError("regional RAP timestep differs from captured operator report")
    capture_text = binding.get("prediction_environment", {}).get("NM_FEM_OPERATOR_FD_CAPTURE_PATH")
    if not isinstance(capture_text, str) or not capture_text:
        raise ValueError("operator prediction lacks the rich capture path needed for exact candidate binding")
    capture_binding = _capture_tetrahedra(Path(capture_text), report, manifest, arrays, payload)
    candidate_nodes = arrays["candidate_nodes"]
    expected_state = np.asarray(vectors.get("captured_node_state_f32"), dtype="<f4")
    env = selected_env
    if candidate_nodes.shape[1:] != expected_state.shape or not np.array_equal(
            candidate_nodes[env].view("<u4"), expected_state.view("<u4")):
        raise ValueError("regional candidate nodes differ bitwise from the source-bound operator capture")

    selected_env = env
    nodes = int(manifest["node_count"])
    modes = int(manifest["mode_count"])
    state_raw = candidate_nodes[selected_env]
    tags = state_raw[:, 11]
    free_nodes = np.asarray(vectors["free_node_ids"], dtype=np.int64)
    expected_free = np.flatnonzero(~np.isin(tags, [1.0, 3.0]))
    if not np.array_equal(free_nodes, expected_free):
        raise ValueError("regional candidate constraints differ from offline free-node mapping")
    free_dofs = (3 * free_nodes[:, None] + np.arange(3, dtype=np.int64)).reshape(-1)
    if matrix.shape != (free_dofs.size, free_dofs.size):
        raise ValueError("offline operator dimension disagrees with regional candidate free-node map")
    if int(arrays["factor_valid"][selected_env]) != 1:
        raise ValueError("selected-Newton regional RAP factor_valid is false")
    pivots = arrays["pivots"][selected_env].astype(np.int64)
    if sorted(pivots.tolist()) != list(range(modes)):
        raise ValueError("selected-Newton regional RAP pivot permutation is invalid")
    node_blocks = arrays["node_block_inverse"][selected_env]
    block_rows = node_blocks[..., :3]
    block_valid = node_blocks[:, 0, 3] > 0.5
    if not np.all(block_valid[free_nodes]):
        raise ValueError("selected-Newton sidecar lacks exact nodal inverse for every free node")
    inv_block = block_rows[free_nodes].astype(np.float64)
    effective = arrays["effective_P"][selected_env]
    ap = arrays["AP"][selected_env]
    P = arrays["P"]
    expected_effective = P.copy()
    expected_effective[:, np.isin(tags, [1.0, 3.0]), :] = 0.0
    if not np.array_equal(effective.view("<u4"), expected_effective.view("<u4")):
        raise ValueError("selected-Newton effective RAP basis differs from source P under constraints")
    fixed = np.ones(nodes, dtype=bool)
    fixed[free_nodes] = False
    if np.any(effective[:, fixed, :3] != 0.0):
        raise ValueError("selected-Newton effective RAP basis is nonzero on constrained nodes")
    Pfree = effective[:, free_nodes, :3].transpose(1, 2, 0).reshape(free_dofs.size, modes).astype(np.float64)
    APfree = ap[:, free_nodes, :3].transpose(1, 2, 0).reshape(free_dofs.size, modes).astype(np.float64)
    coarse = Pfree.T @ APfree
    lu = arrays["pivoted_RAP"][selected_env].astype(np.float64)
    lower = np.tril(lu, -1) + np.eye(modes)
    upper = np.triu(lu)
    lu_reconstruction_rel = float(np.linalg.norm(lower @ upper - coarse[pivots]) /
                                   max(np.linalg.norm(coarse[pivots]), 1e-300))
    if lu_reconstruction_rel > 2.0e-3:
        raise ValueError(f"selected-Newton RAP LU does not reconstruct captured P^TAP ({lu_reconstruction_rel:.3g})")
    ap_offline = np.column_stack([matrix @ Pfree[:, mode] for mode in range(modes)])
    ap_rel = float(np.linalg.norm(ap_offline - APfree) / max(np.linalg.norm(APfree), 1e-300))
    if ap_rel > 1.0e-3:
        raise ValueError(f"offline A*P differs from selected-Newton AP by {ap_rel:.6g}")
    rap = {"P_free": Pfree, "AP_free": APfree, "pivoted_lu": lu, "pivots": pivots,
           "inverse_blocks": inv_block, "coarse_matrix_from_native_AP": coarse,
           "candidate_positions_f32": state_raw[:, :3], "native_ap_relative_error": ap_rel,
           "native_lu_reconstruction_relative_error": lu_reconstruction_rel,
           "native_factor_valid": int(arrays["factor_valid"][selected_env]), "mode_count": modes}
    rap_binding = {"root": int(manifest["root"]), "iteration": int(manifest["iteration"]),
       "environment": selected_env, "linearization_generation": int(manifest["linearization_generation"]),
       "micro_generation": int(manifest["micro_generation"]), "capture_point": manifest["capture_point"],
       "candidate_linearization_sha256": next(x["sha256"] for x in manifest["candidate_linearization_sha256"] if int(x["environment"]) == selected_env),
       "capture_sha256": capture_binding["capture_sha256"], "tangent_cache_used": manifest.get("tangent_cache_used") is True,
       "operator_generation": manifest.get("operator_generation"), "capture_path": capture_binding["capture_path"]}
    return rap, hashes, rap_binding


def load_native_rap(folder: Path, report: dict, matrix: sparse.csr_matrix,
                    vectors: dict) -> tuple[dict, dict]:
    manifest, files = _manifest_files(folder)
    selected_env = int(report["capture_selection"]["environment"])
    nodes = int(manifest["fem_node_count"])
    modes = int(manifest["mode_count"])
    envs = int(manifest["environment_count"])
    def read_f32(name: str, shape: tuple[int, ...]) -> np.ndarray:
        raw = np.fromfile(files[name], dtype="<f4")
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"RAP {name} entry count differs from its declared layout")
        value = raw.reshape(shape)
        require_finite(f"RAP {name}", value)
        return value
    P = read_f32("P", (modes, nodes, 4))
    effective = read_f32("effective_P", (envs, modes, nodes, 4))[selected_env]
    AP = read_f32("AP", (envs, modes, nodes, 4))[selected_env]
    lu = read_f32("pivoted_RAP", (envs, modes, modes))[selected_env]
    pivots = np.fromfile(files["pivots"], dtype="<u4").reshape(envs, modes)[selected_env]
    valid = np.fromfile(files["factor_valid"], dtype="<u4").reshape(envs)[selected_env]
    node_blocks = read_f32("node_block_inverse", (envs, nodes, 3, 4))[selected_env]
    state_raw = np.fromfile(files["candidate_nodes"], dtype="<f4").reshape(envs, nodes, 20)[selected_env]
    expected_state = np.asarray(vectors["captured_node_state_f32"], dtype=np.float32)
    if not np.array_equal(state_raw.view(np.uint32), expected_state.view(np.uint32)):
        raise ValueError("native RAP candidate node state differs from operator capture")
    free_nodes = np.asarray(vectors["free_node_ids"], dtype=np.int64)
    tags = state_raw[:, 11]  # restAndFixed.w in NMFEMNodeStateGPU (5 float4s)
    expected_free = np.flatnonzero(~np.isin(tags, [1.0, 3.0]))
    if not np.array_equal(free_nodes, expected_free):
        raise ValueError("RAP candidate constraints differ from offline free-node mapping")
    free_dofs = (3 * free_nodes[:, None] + np.arange(3, dtype=np.int64)).reshape(-1)
    if matrix.shape != (free_dofs.size, free_dofs.size):
        raise ValueError("offline operator dimension disagrees with native free-node map")
    if int(valid) != 1:
        raise ValueError("native RAP factor_valid is false for selected environment")
    if sorted(pivots.tolist()) != list(range(modes)):
        raise ValueError("native RAP pivot permutation is invalid")
    block_rows = node_blocks[:, :, :3]
    block_valid = node_blocks[:, 0, 3] > 0.5
    if not np.all(block_valid[free_nodes]):
        raise ValueError("native export lacks exact nodal 3x3 inverse for one or more free nodes")
    inv_block = block_rows[free_nodes].astype(np.float64)
    require_finite("native nodal inverse blocks", inv_block)
    eff = effective[:, :, :3].astype(np.float64)
    ap = AP[:, :, :3].astype(np.float64)
    expected_effective = P.copy()
    expected_effective[:, np.isin(tags, [1.0, 3.0]), :] = 0.0
    if not np.array_equal(effective.view(np.uint32),
                          expected_effective.view(np.uint32)):
        raise ValueError("native effective RAP basis differs from source P under captured constraints")
    # Native projected basis must be zero on all constraints.
    fixed = np.ones(nodes, dtype=bool)
    fixed[free_nodes] = False
    if np.any(effective[:, fixed, :3] != 0.0):
        raise ValueError("native effective RAP basis is nonzero on constrained nodes")
    Pfree = eff[:, free_nodes, :].transpose(1, 2, 0).reshape(free_dofs.size, modes)
    APfree = ap[:, free_nodes, :].transpose(1, 2, 0).reshape(free_dofs.size, modes)
    coarse = Pfree.T @ APfree
    # Recover and verify the native final-row-permutation LU, not a replacement
    # coarse factorization. The binary stores the in-place pivoted LU.
    lower = np.tril(lu.astype(np.float64), -1) + np.eye(modes)
    upper = np.triu(lu.astype(np.float64))
    permuted_coarse = coarse[pivots]
    lu_reconstruction_rel = float(np.linalg.norm(lower @ upper - permuted_coarse) /
                                   max(np.linalg.norm(permuted_coarse), 1e-300))
    if lu_reconstruction_rel > 2.0e-3:
        raise ValueError(f"native RAP LU does not reconstruct captured P^TAP ({lu_reconstruction_rel:.3g})")
    # Independent exact-candidate action on P must match the AP export before
    # this is used as a preconditioner comparison.
    ap_offline = np.column_stack([matrix @ Pfree[:, mode] for mode in range(modes)])
    ap_rel = float(np.linalg.norm(ap_offline - APfree) /
                   max(np.linalg.norm(APfree), 1e-300))
    if ap_rel > 1.0e-3:
        raise ValueError(f"offline A*P differs from native AP export by {ap_rel:.6g}")
    return {
        "P_free": Pfree,
        "AP_free": APfree,
        "pivoted_lu": lu.astype(np.float64),
        "pivots": pivots.astype(np.int64),
        "inverse_blocks": inv_block,
        "coarse_matrix_from_native_AP": coarse,
        "candidate_positions_f32": state_raw[:, 0:3],
        "native_ap_relative_error": ap_rel,
        "native_lu_reconstruction_relative_error": lu_reconstruction_rel,
        "native_factor_valid": int(valid),
        "mode_count": modes,
    }, {path.name: sha256(path) for path in files.values()} | {
        "manifest.json": sha256(folder / "manifest.json")}


def native_balanced_rap_preconditioner(matrix: sparse.csr_matrix, rap: dict) -> Callable[[np.ndarray], np.ndarray]:
    P = rap["P_free"]
    AP = rap["AP_free"]
    lu = rap["pivoted_lu"]
    piv = rap["pivots"]
    inv_blocks = rap["inverse_blocks"]
    nodes = inv_blocks.shape[0]
    modes = P.shape[1]
    def solve_native_lu(rhs: np.ndarray) -> np.ndarray:
        y = np.asarray(rhs, dtype=np.float64)[piv].copy()
        for i in range(modes):
            if i:
                y[i] -= np.dot(lu[i, :i], y[:i])
        x = np.empty_like(y)
        for i in range(modes - 1, -1, -1):
            if i + 1 < modes:
                y[i] -= np.dot(lu[i, i + 1:], x[i + 1:])
            diagonal = lu[i, i]
            if not math.isfinite(float(diagonal)) or diagonal == 0.0:
                raise ValueError("native RAP triangular solve encountered invalid pivot")
            x[i] = y[i] / diagonal
        return x
    def apply(rhs: np.ndarray) -> np.ndarray:
        r = np.asarray(rhs, dtype=np.float64)
        coefficients = solve_native_lu(P.T @ r)
        fine = r - AP @ coefficients
        fine3 = fine.reshape(nodes, 3)
        correction = np.einsum("nij,nj->ni", inv_blocks, fine3).reshape(-1)
        return correction + P @ coefficients
    return apply


def build_graph_patches(topology: dict, free_nodes: np.ndarray,
                        max_patch_nodes: int, anchor_stride: int) -> tuple[list[np.ndarray], np.ndarray]:
    if max_patch_nodes < 1 or max_patch_nodes > 32:
        raise ValueError("max patch node count must be in [1,32]")
    if anchor_stride < 1 or anchor_stride > max_patch_nodes:
        raise ValueError("anchor stride must be in [1,max_patch_nodes]")
    conn = np.asarray(topology.get("nodes"), dtype=np.int64)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError("topology sidecar does not contain tetrahedron connectivity")
    node_count_value = topology.get("node_count")
    node_count = int(np.asarray(node_count_value).reshape(-1)[0]) if node_count_value is not None \
        else int(conn.max(initial=-1) + 1)
    if np.any(conn < 0) or np.any(conn >= node_count):
        raise ValueError("tetrahedron connectivity is out of bounds")
    free = np.asarray(free_nodes, dtype=np.int64)
    free_set = set(free.tolist())
    adjacency = [set() for _ in range(node_count)]
    for tet in conn:
        ids = [int(x) for x in tet if int(x) in free_set]
        for n in ids:
            adjacency[n].update(x for x in ids if x != n)
    anchors = free[::anchor_stride].tolist()
    if not anchors or anchors[-1] != int(free[-1]):
        anchors.append(int(free[-1]))
    patches: list[np.ndarray] = []
    for anchor in anchors:
        selected = [int(anchor)]
        visited = {int(anchor)}
        frontier = [int(anchor)]
        while frontier and len(selected) < max_patch_nodes:
            current = frontier.pop(0)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                selected.append(neighbor)
                frontier.append(neighbor)
                if len(selected) == max_patch_nodes:
                    break
        patches.append(np.asarray(sorted(selected), dtype=np.int64))
    covered = set(int(n) for patch in patches for n in patch)
    for node in free:
        if int(node) not in covered:
            patches.append(np.asarray([int(node)], dtype=np.int64))
    membership = np.zeros(node_count, dtype=np.int32)
    for patch in patches:
        membership[patch] += 1
    if np.any(membership[free] == 0) or np.any(membership[np.setdiff1d(np.arange(node_count), free)] != 0):
        raise ValueError("patch collection does not exactly cover free nodes only")
    return patches, membership


def schwarz_preconditioner(matrix: sparse.csr_matrix, free_nodes: np.ndarray,
                           patches: list[np.ndarray], membership: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    dofs_by_node = {int(node): np.arange(3 * i, 3 * i + 3, dtype=np.int64)
                    for i, node in enumerate(free_nodes)}
    factors = []
    estimated = 0
    factor_ops = 0
    for patch_index, nodes in enumerate(patches):
        dofs = np.concatenate([dofs_by_node[int(node)] for node in nodes])
        dense = matrix[dofs][:, dofs].toarray()
        require_finite(f"Schwarz patch {patch_index} matrix", dense)
        n = dense.shape[0]
        # Keep the materialized patch, LU result, and one factorization-work
        # copy in the budget; include pivots/work vectors as well.
        estimated += 3 * dense.nbytes + 4 * n * 8
        if estimated > MAX_HOST_BYTES:
            raise ValueError("bounded Schwarz factors exceed the 1 GiB analysis budget")
        try:
            lu, piv = linalg.lu_factor(dense, check_finite=False)
        except Exception as error:
            raise ValueError(f"Schwarz patch {patch_index} factorization failed") from error
        scale = float(np.max(np.abs(np.diag(lu)))) if n else 0.0
        if not math.isfinite(scale) or scale == 0.0 or np.any(np.abs(np.diag(lu)) <= scale * 1.0e-12):
            raise ValueError(f"Schwarz patch {patch_index} has an unresolved pivot")
        factors.append((nodes.copy(), dofs, lu, piv))
        factor_ops += int((2.0 / 3.0) * n ** 3)
    def apply(rhs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(rhs, dtype=np.float64))
        for nodes, dofs, lu, piv in factors:
            local = linalg.lu_solve((lu, piv), rhs[dofs], check_finite=False)
            weights = np.repeat(1.0 / membership[nodes].astype(np.float64), 3)
            np.add.at(out, dofs, local * weights)
        return out
    details = {
        "patch_count": len(factors),
        "max_patch_nodes": max(int(len(p)) for p in patches),
        "min_patch_nodes": min(int(len(p)) for p in patches),
        "patch_node_membership": [p.astype(int).tolist() for p in patches],
        "per_node_patch_coverage": membership.astype(int).tolist(),
        "dense_factor_bytes_estimate": int(estimated),
        "estimated_factorization_flops": int(factor_ops),
        "application_solve_count_per_gmres_iteration": len(factors),
        "weighting": "partition of unity by 1 / number of patches containing each node; applied equally to its xyz DOFs",
    }
    return apply, details


def restarted_right_gmres(matrix: sparse.csr_matrix, rhs: np.ndarray,
                          precondition: Callable[[np.ndarray], np.ndarray],
                          restart: int = 16, max_iterations: int = 128,
                          relative_tolerance: float = 1.0e-8) -> dict:
    b = np.asarray(rhs, dtype=np.float64)
    bnorm = float(np.linalg.norm(b))
    if not math.isfinite(bnorm) or bnorm == 0.0:
        raise ValueError("GMRES RHS norm must be finite and positive")
    x = np.zeros_like(b)
    history = []
    matvecs = 0
    preconditioner_apps = 0
    for _cycle in range((max_iterations + restart - 1) // restart):
        r = b - matrix @ x
        matvecs += 1
        true_rel = float(np.linalg.norm(r) / bnorm)
        if not math.isfinite(true_rel):
            raise ValueError("true residual became nonfinite")
        if not history:
            history.append({"iteration": 0, "true_relative_residual": true_rel})
        if true_rel <= relative_tolerance:
            break
        beta = float(np.linalg.norm(r))
        if not math.isfinite(beta) or beta == 0.0:
            raise ValueError("initial residual is zero/nonfinite")
        m = min(restart, max_iterations - len(history) + 1)
        V = np.zeros((b.size, m + 1), dtype=np.float64)
        Z = np.zeros((b.size, m), dtype=np.float64)
        H = np.zeros((m + 1, m), dtype=np.float64)
        V[:, 0] = r / beta
        best_x = x
        for j in range(m):
            Z[:, j] = np.asarray(precondition(V[:, j]), dtype=np.float64)
            preconditioner_apps += 1
            w = np.asarray(matrix @ Z[:, j], dtype=np.float64)
            matvecs += 1
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0.0:
                V[:, j + 1] = w / H[j + 1, j]
            target = np.zeros(j + 2, dtype=np.float64)
            target[0] = beta
            y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], target, rcond=None)
            best_x = x + rowwise_fsum_matvec(Z[:, :j + 1], y)
            actual_r = b - matrix @ best_x
            matvecs += 1
            actual_rel = float(np.linalg.norm(actual_r) / bnorm)
            if not math.isfinite(actual_rel):
                raise ValueError("true residual became nonfinite")
            history.append({"iteration": len(history),
                            "true_relative_residual": actual_rel})
            if actual_rel <= relative_tolerance or H[j + 1, j] == 0.0:
                x = best_x
                break
        else:
            x = best_x
        if history[-1]["true_relative_residual"] <= relative_tolerance:
            break
    final = float(np.linalg.norm(b - matrix @ x) / bnorm)
    return {"status": "diagnostic_converged" if final <= relative_tolerance else "diagnostic_not_converged",
            "restart": restart, "max_iterations": max_iterations,
            "relative_tolerance": relative_tolerance,
            "rhs_l2": bnorm, "final_true_relative_residual": final,
            "true_residual_history": history,
            "true_operator_matvec_count": matvecs,
            "preconditioner_application_count": preconditioner_apps}


def run(args: argparse.Namespace) -> dict:
    report_path = Path(args.operator_report)
    report, matrix, vectors, topology, operator_hashes = load_hashed_operator(report_path)
    regional_path = Path(args.regional_sidecar) if getattr(args, "regional_sidecar", None) else None
    if regional_path is not None:
        rap, rap_hashes, regional_binding = load_selected_newton_rap(
            regional_path, report, matrix, vectors)
        regional_manifest = json.loads(regional_path.read_text())
        normalized_rap = {
            "control_step": regional_manifest["root"],
            "environment_count": regional_manifest["environment_count"],
            "fem_node_count": regional_manifest["node_count"],
            "source_physics_fingerprint": str(int(regional_manifest["source_physics_fingerprint"], 16)),
            "device_program_fingerprint": str(int(regional_manifest["device_program_fingerprint"], 16)),
        }
        binding = validate_pairing(report, matrix, vectors, topology, normalized_rap)
        rap_source = str(regional_path)
        rap_source_kind = "selected_newton_factor_barrier_sidecar"
    else:
        rap_dir = Path(args.rap_export)
        manifest, _ = _manifest_files(rap_dir)
        binding = validate_pairing(report, matrix, vectors, topology, manifest)
        rap, rap_hashes = load_native_rap(rap_dir, report, matrix, vectors)
        regional_binding = None
        rap_source = str(rap_dir)
        rap_source_kind = "retrospective_native_export"
    free_nodes = np.asarray(vectors["free_node_ids"], dtype=np.int64)
    rhs = np.asarray(vectors["native_rhs_free_impulse_Ns"], dtype=np.float64)
    native_apply = native_balanced_rap_preconditioner(matrix, rap)
    native_result = restarted_right_gmres(matrix, rhs, native_apply,
                                           args.restart, args.max_iterations,
                                           args.relative_tolerance)
    patches, membership = build_graph_patches(
        topology, free_nodes, args.schwarz_patch_nodes, args.schwarz_anchor_stride)
    patch_factor_bound = sum(3 * (3 * len(patch)) ** 2 * 8 +
                             4 * (3 * len(patch)) * 8 for patch in patches)
    matrix_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    sidecar_bytes = sum(int(value.nbytes) for value in vectors.values()) + \
        sum(int(value.nbytes) for value in topology.values()) + \
        sum(int(value.nbytes) for value in rap.values() if isinstance(value, np.ndarray))
    gmres_workspace_bound = int((3 * 17 + 12) * matrix.shape[0] * 8 + 17 * 17 * 8)
    total_memory_bound = matrix_bytes + sidecar_bytes + gmres_workspace_bound + patch_factor_bound
    if total_memory_bound > MAX_HOST_BYTES:
        raise ValueError(f"offline matrix/GMRES/Schwarz envelope exceeds 1 GiB ({total_memory_bound} bytes)")
    schwarz_apply, schwarz_details = schwarz_preconditioner(
        matrix, free_nodes, patches, membership)
    schwarz_result = restarted_right_gmres(matrix, rhs, schwarz_apply,
                                            args.restart, args.max_iterations,
                                            args.relative_tolerance)
    # RHS must be the same exact pre-FGMRES residual vector in both trials.
    return {
        "schema": SCHEMA,
        "verdict": "offline same-matrix preconditioner diagnostic; not a native solver run or qualification",
        "binding": binding,
        "operator_report": str(report_path),
        "operator_report_sha256": sha256(report_path),
        "operator_sidecar_sha256": operator_hashes,
        "native_rap_export": rap_source,
        "native_rap_source_kind": rap_source_kind,
        "native_rap_manifest_sha256": rap_hashes.get("manifest.json", rap_hashes.get(regional_path.name) if regional_path else None),
        "native_rap_file_sha256": rap_hashes,
        "native_rap_selected_generation": regional_binding,
        "native_rap_validation": {
            "native_ap_relative_error": rap["native_ap_relative_error"],
            "native_lu_reconstruction_relative_error": rap["native_lu_reconstruction_relative_error"],
            "factor_valid": rap["native_factor_valid"],
            "mode_count": rap["mode_count"],
            "factor_scope": "captured selected environment; native pivoted LU and 3x3 nodal inverse",
        },
        "fixed_physical_operator": {
            "shape": list(matrix.shape), "nnz": int(matrix.nnz),
            "coordinate_order": "ascending free node ID, xyz",
            "matrix_hash": operator_hashes[report_path.stem + ".operator.npz"],
            "rhs_l2": float(np.linalg.norm(rhs)),
            "rhs_source": "native captured center residual passed to FGMRES before the selected Newton solve",
            "rhs_hash": hashlib.sha256(np.asarray(rhs, dtype="<f8").tobytes()).hexdigest(),
            "same_A_and_b_for_both_methods": True,
        },
        "gmres_restart16_balanced_native_rap": native_result,
        "gmres_restart16_bounded_graph_schwarz": {
            **schwarz_result,
            "construction": schwarz_details,
            "parameters": {"maximum_patch_nodes": args.schwarz_patch_nodes,
                           "anchor_stride_free_nodes": args.schwarz_anchor_stride,
                           "graph": "tetrahedron adjacency induced by the captured explicit FEM topology; no assumed flute pitch"},
        },
        "analysis_budget": {"host_memory_limit_bytes": MAX_HOST_BYTES,
                            "matrix_csr_bytes": matrix_bytes,
                            "captured_sidecar_arrays_bytes": sidecar_bytes,
                            "gmres_workspace_bound_bytes": gmres_workspace_bound,
                            "schwarz_factor_workspace_bound_bytes": patch_factor_bound,
                            "total_estimated_analysis_bytes": total_memory_bound,
                            "limit_status": "within_limit"},
        "limits": [
            "The same native captured center residual is used as b; it is not a manufactured RHS.",
            "The offline A must pass the native operator-action parity gate and native AP parity check before either preconditioner is run.",
            "The native RAP is required to match the exact candidate node state, world/device fingerprints, control step, topology, constraints, and hashes.",
            "The failed/rejected physical transaction that produced a diagnostic matrix is not relabeled as an accepted material state.",
            "This CPU GMRES experiment does not reproduce Metal arithmetic, restart scheduling, or certify a physical box result.",
            "Graph Schwarz patches are computational comparisons only; they are not inferred flute cells or physical substructures.",
            "No conclusion is made if RAP/AP validation, nodal inverse coverage, or matrix-action parity fails.",
            "A selected-Newton regional sidecar is admitted only after exact root/Newton/environment, source/device fingerprint, candidate-state, local F high/low, accepted material history, optional cached tangent, generation digest, payload hashes, and A*P checks pass.",
            "The retrospective --rap-export path remains a separate compatibility path and cannot establish same-Newton candidate generation without its own selected-iteration binding.",
            "The offline GMRES update accumulation uses scalar math.fsum because the installed NumPy/Accelerate matrix multiply emits spurious FP-status warnings on finite products; this does not change A, b, preconditioner, or the convergence criterion.",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-report", required=True)
    rap_source = p.add_mutually_exclusive_group(required=True)
    rap_source.add_argument("--rap-export", help="legacy retrospective native RAP folder")
    rap_source.add_argument("--regional-sidecar", help="selected-Newton .regional.json manifest")
    p.add_argument("--restart", type=int, default=16)
    p.add_argument("--max-iterations", type=int, default=128)
    p.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    p.add_argument("--schwarz-patch-nodes", type=int, default=16)
    p.add_argument("--schwarz-anchor-stride", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 2
    if args.restart != 16 or args.max_iterations < 16 or args.max_iterations > 512:
        p.error("restart is fixed at 16; max iterations must be in [16,512]")
    if not math.isfinite(args.relative_tolerance) or not 0.0 < args.relative_tolerance < 1.0:
        p.error("relative tolerance must be finite and in (0,1)")
    try:
        result = run(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        print(f"offline conditioning study rejected: {type(error).__name__}: {error}")
        return 2
    print(json.dumps({"status": "diagnostic_complete",
                      "native_rap_final_true_residual": result["gmres_restart16_balanced_native_rap"]["final_true_relative_residual"],
                      "graph_schwarz_final_true_residual": result["gmres_restart16_bounded_graph_schwarz"]["final_true_relative_residual"],
                      "report": str(out)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
