#!/usr/bin/env python3
"""Prepare the exact source101/360 read-only native operator replay payload.

This only repacks already captured center inputs. It does not run a solver or
write into the native run directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

BASE = Path("/Users/home/cardboard-implementation-20261014")
SOURCE = BASE / "snapshots/source-boundary-reset-order-101"
RUN = BASE / "solver/reset-augmentation-first-failure-capture-360"
CAPTURE = RUN / "operator-capture"
PACKAGE = RUN / "output/compiled.nmatterpack"
PREDICTION = RUN / "prediction.json"
RECEIPT = RUN / "receipt.json"
METALLIB = RUN / "inputs/runtime/matter/shaders/NumiMatter.metallib"
OUTPUT = Path(__file__).resolve().parent / "prepared"

EXPECTED_SOURCE_FILES = (
    "matter/include/numi/matter/shared.h",
    "matter/src/runtime.mm",
    "matter/src/metal/common.metalinc",
    "matter/src/metal/fem.metalinc",
    "matter/src/metal/fgmres.metalinc",
    "matter/tools/cardboard_probe.mm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def f32(value: object, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label} is nonfinite")
    result = struct.unpack("<f", struct.pack("<f", result))[0]
    require(math.isfinite(result), f"{label} overflows binary32")
    return result


def append_floats(out: bytearray, values: object, count: int, label: str) -> None:
    require(isinstance(values, list) and len(values) == count,
            f"{label} must contain {count} values")
    out.extend(struct.pack("<" + "f" * count,
                           *(f32(v, label) for v in values)))


def main() -> int:
    for path in (CAPTURE, PACKAGE, PREDICTION, RECEIPT, METALLIB):
        require(path.is_file() and path.stat().st_size > 0,
                f"source-bound input missing or empty: {path}")
    prediction = json.loads(PREDICTION.read_text())
    receipt = json.loads(RECEIPT.read_text())
    require(prediction.get("schema") == "numi.cardboard.native-prediction.v1",
            "unexpected prediction schema")
    require(Path(prediction.get("source_snapshot", "")).resolve() == SOURCE,
            "run is not bound to source101")
    require(prediction.get("environment", {}).get("NM_FEM_OPERATOR_FD_SELECTOR") == "2:23",
            "run does not bind the selected 2:23 operator capture")
    require(prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_ROOT") == "2" and
            prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR") == "23",
            "run does not bind the selected 2:23 true-residual observation")
    require(receipt.get("timed_out") is False and "exit_code" in receipt,
            "native run does not have a terminal receipt")

    source_hashes = {}
    inputs = prediction.get("inputs_sha256", {})
    for relative in EXPECTED_SOURCE_FILES:
        keys = [key for key in (relative, "source/" + relative) if key in inputs]
        require(len(keys) >= 1, f"prediction is missing source hash for {relative}")
        expected = {inputs[key] for key in keys}
        require(len(expected) == 1, f"prediction source hash ambiguous for {relative}")
        actual = sha256(SOURCE / relative)
        require(actual == next(iter(expected)), f"source101 binding mismatch for {relative}")
        source_hashes[relative] = actual

    runtime_metallib_key = "runtime/matter/shaders/NumiMatter.metallib"
    require(inputs.get(runtime_metallib_key) == sha256(METALLIB),
            "selected runtime metallib differs from the native run's build input")

    records = []
    with CAPTURE.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"capture line {line_number} is invalid JSON") from error
    require(records and records[0].get("record") == "header",
            "capture lacks leading header")
    header = records[0]
    require(header.get("schema") == "numi.cardboard.fem-newton-capture.v1" and
            (header.get("root"), header.get("newton_iteration")) == (2, 23),
            "capture is not source101 root 2/Newton 23")
    require(header.get("production_inputs_bitwise_unchanged") is True,
            "capture did not certify unchanged production inputs")
    cache_meta = header.get("cached_algorithmic_tangent_capture", {})
    require(cache_meta.get("policy_enabled") is True and cache_meta.get("available") is True and
            cache_meta.get("snapshot_bitwise_guard_included") is True,
            "production tangent cache capture is absent or unguarded")
    environments = int(header["environment_count"])
    nodes_per_environment = int(header["node_count_per_environment"])
    tets_per_environment = int(header["tetrahedron_count_per_environment"])
    require((environments, nodes_per_environment, tets_per_environment) == (2, 882, 1920),
            "unexpected source101 capture dimensions")
    state_stride = 15
    params_by_environment = header["parameters_by_environment"]
    parameter_count = len(params_by_environment[0])
    require(parameter_count == 59 and len(params_by_environment) == environments,
            "unexpected parameter dimensions")
    require(all(len(row) == parameter_count for row in params_by_environment),
            "parameter rows differ in length")
    dt_values = header.get("dt_seconds_by_environment")
    require(isinstance(dt_values, list) and len(dt_values) == environments,
            "capture timestep rows missing")
    dt = f32(dt_values[0], "capture timestep")
    require(dt > 0.0 and all(f32(value, "capture timestep") == dt for value in dt_values),
            "capture environments do not share one timestep")

    nodes = {}
    tets = {}
    integrity = None
    for record in records[1:]:
        kind = record.get("record")
        if kind == "node":
            key = (int(record["environment"]), int(record["node_id"]))
            require(key not in nodes, f"duplicate captured node {key}")
            nodes[key] = record
        elif kind == "tetrahedron":
            key = (int(record["environment"]), int(record["tet_id"]))
            require(key not in tets, f"duplicate captured tetrahedron {key}")
            tets[key] = record
        elif kind == "integrity":
            require(integrity is None, "capture has multiple integrity records")
            integrity = record
        else:
            raise ValueError(f"unknown capture record kind {kind!r}")
    require(integrity is not None and integrity.get("production_inputs_bitwise_unchanged") is True,
            "capture lacks terminal bitwise-integrity record")
    require(len(nodes) == environments * nodes_per_environment,
            "capture node rows incomplete")
    require(len(tets) == environments * tets_per_environment,
            "capture tetrahedron rows incomplete")

    payload = bytearray()
    fingerprint = int(header["world_fingerprint_hex"], 16)
    # Little-endian ABI: 8-byte magic; version,E,N,T,stateStride,paramCount;
    # captured world fingerprint; timestep; reserved.
    payload.extend(struct.pack("<8s6IQfI", b"NMCACHE1", 1, environments,
        nodes_per_environment, tets_per_environment, state_stride,
        parameter_count, fingerprint, dt, 0))

    for environment in range(environments):
        for node_id in range(nodes_per_environment):
            record = nodes[(environment, node_id)]
            center = record.get("center", {})
            append_floats(payload, [component for slot in center["raw_state_float4s"]
                                    for component in slot], 20,
                          f"node {environment}:{node_id} state")
    for environment in range(environments):
        for node_id in range(nodes_per_environment):
            record = nodes[(environment, node_id)]
            append_floats(payload, [*record["correction_delta_v_m_s"], 0.0], 4,
                          f"node {environment}:{node_id} correction")
    for environment in range(environments):
        for node_id in range(nodes_per_environment):
            record = nodes[(environment, node_id)]
            append_floats(payload, record["center_residual_impulse_Ns"], 4,
                          f"node {environment}:{node_id} center RHS")
    for environment in range(environments):
        for tet_id in range(tets_per_environment):
            record = tets[(environment, tet_id)]
            accepted = record["center"]["accepted_state"]
            require(len(accepted) == int(record["material_state_count"]) <= state_stride,
                    f"tet {environment}:{tet_id} material-history shape mismatch")
            values = [f32(v, f"tet {environment}:{tet_id} accepted material state")
                      for v in accepted]
            values.extend([0.0] * (state_stride - len(values)))
            payload.extend(struct.pack("<" + "f" * state_stride, *values))
    for environment in range(environments):
        for tet_id in range(tets_per_environment):
            record = tets[(environment, tet_id)]
            cache = record.get("cached_algorithmic_tangent_local", {})
            require(cache.get("valid") is True,
                    f"tet {environment}:{tet_id} lacks a valid production cache entry")
            append_floats(payload,
                cache["matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"], 81,
                f"tet {environment}:{tet_id} cached tangent")
    payload.extend(struct.pack("<" + "I" * (environments * tets_per_environment),
                               *([1] * (environments * tets_per_environment))))
    for environment in range(environments):
        payload.extend(struct.pack("<" + "f" * parameter_count,
                                   *(f32(v, f"environment {environment} parameter")
                                     for v in params_by_environment[environment])))
    for environment in range(environments):
        for node_id in range(nodes_per_environment):
            record = nodes[(environment, node_id)]
            append_floats(payload, record["production_A_delta_v_impulse_Ns"], 4,
                          f"node {environment}:{node_id} captured production action")

    # Static compiled tetrahedron identity is independently compared by the
    # native checker against readPackage(); this defeats a capture/package mix.
    for environment in range(environments):
        for tet_id in range(tets_per_environment):
            record = tets[(environment, tet_id)]
            nodes4 = [int(v) for v in record["nodes"]]
            require(len(set(nodes4)) == 4 and all(0 <= v < nodes_per_environment for v in nodes4),
                    f"tet {environment}:{tet_id} connectivity invalid")
            inv = [f32(v, "inverse rest") for row in record["inverse_rest_rows_per_m"] for v in row]
            volume = f32(record["rest_volume_m3"], "rest volume")
            identity = [int(record["material_index"]), int(record["object_index"]),
                        int(record["topology_generation"]), int(record["tet_flags"])]
            frame = [f32(v, "material frame") for v in record["material_frame_quaternion_xyzw"]]
            payload.extend(struct.pack("<4I9ff4I4f", *nodes4, *inv, volume,
                                       *identity, *frame))

    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload_path = OUTPUT / "replay-input.bin"
    payload_path.write_bytes(payload)
    manifest = {
        "schema": "numi.cardboard.native-cached-operator-replay-input.v1",
        "claim_scope": "captured source101 FEM element action and nodal gather only; no solver step or contact/rigid unknowns",
        "source_snapshot": str(SOURCE),
        "source_hashes": source_hashes,
        "run_directory": str(RUN),
        "run_receipt_sha256": sha256(RECEIPT),
        "prediction_sha256": sha256(PREDICTION),
        "capture_sha256": sha256(CAPTURE),
        "compiled_package_sha256": sha256(PACKAGE),
        "metallib_sha256": sha256(METALLIB),
        "payload_sha256": sha256(payload_path),
        "payload_bytes": len(payload),
        "native_capture": {
            "root": 2, "newton_iteration": 23,
            "world_fingerprint_hex": header["world_fingerprint_hex"],
            "execution_fingerprint_hex": header["execution_fingerprint_hex"],
            "environment_count": environments,
            "node_count_per_environment": nodes_per_environment,
            "tetrahedron_count_per_environment": tets_per_environment,
            "material_state_stride": state_stride,
            "parameter_count": parameter_count,
            "dt_seconds": dt,
            "tangent_entries_valid": len(tets),
            "correction_max_abs_m_per_s": max(
                abs(float(v)) for record in nodes.values()
                for v in record["correction_delta_v_m_s"]),
        },
        "integrity_record": integrity,
        "payload_layout": [
            "header: <8s6IQfI>",
            "nodes: E*N*NMFEMNodeStateGPU (5 float4)",
            "direction: E*N*float4 (captured fem correction delta-v)",
            "rhs: E*N*float4 (captured pre-FGMRES FEM RHS)",
            "accepted_material_state: E*T*materialStateStride floats",
            "tangent_cache: E*T*81 floats row-major local dP/dF",
            "tangent_cache_valid: E*T uint32",
            "environment_parameters: E*parameterCount floats",
            "captured_production_action: E*N*float4 (independent comparison output)",
            "static_tet_audit: E*T records <4I9ff4I4f> compared bitwise to compiled package",
        ],
        "limits": [
            "The replay uses production element/gather kernels with cache enabled and disabled on identical captured inputs.",
            "The selected Newton was rejected; this is a diagnostic replay of that exact captured linearization, not an accepted physical state.",
            "The run has zero rigid proxies and zero contact pairs; FEM operator/gather is the complete mechanical linear slice for these captured dimensions.",
            "No physical transaction, material update, or solver convergence claim is made by the replay.",
        ],
    }
    manifest_path = OUTPUT / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["input_manifest_sha256"] = sha256(manifest_path)
    (OUTPUT / "input-manifest.seal.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"payload": str(payload_path), "payload_bytes": len(payload),
                      "payload_sha256": sha256(payload_path),
                      "manifest": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path),
                      "capture_sha256": sha256(CAPTURE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prepare replay rejected input: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
