#!/usr/bin/env python3
"""Source-bound CPU replay of the rejected source105 coupon tet 1820.

This is deliberately an offline equation replay. It does not launch Metal and
does not claim that the Python reconstruction reproduces native instruction
rounding. All retained FP32 inputs are widened from the failed run's package,
candidate buffer, and rolled-back accepted-history export.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys
from typing import Iterable

import numpy as np

BASE = pathlib.Path("/Users/home/cardboard-implementation-20261014")
RUN = BASE / "solver/cached-true-residual-certification-coupon-370"
OUT = pathlib.Path(__file__).resolve().parent
TET_ID = 1820
ENV = 0
SOURCE = BASE / "snapshots/source-true-residual-certification-105"
REF_PATH = BASE / "analysis/bilinear15-fp64-replay-267/bilinear15_reference.py"
POLICY_PATH = BASE / "analysis/local-projection-scaled-correction-082/replay_scaled_policy.py"

spec = importlib.util.spec_from_file_location("bilinear15_reference", REF_PATH)
ref = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ref)
spec = importlib.util.spec_from_file_location("replay_scaled_policy", POLICY_PATH)
pol = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pol)

F32 = np.float32


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32bits(values: Iterable[float]) -> list[str]:
    return [f"0x{struct.unpack('<I', struct.pack('<f', float(F32(x))))[0]:08x}" for x in values]


def hash_bytes(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def package_sections(path: pathlib.Path) -> tuple[dict, dict[int, dict]]:
    raw = path.read_bytes()
    if len(raw) < 56:
        raise ValueError("truncated package header")
    magic, version, endian, abi, section_count, phys, fingerprint, header_hash = struct.unpack_from(
        "<16sIIIIQQQ", raw, 0
    )
    if magic != b"NUMIMATTERPKG\0\0\0" or version != 19 or endian != 0x01020304:
        raise ValueError("unexpected compiled package identity")
    header = {
        "magic": magic.decode("ascii", errors="replace").rstrip("\0"),
        "version": version,
        "endian_marker": f"0x{endian:08x}",
        "matter_abi": abi,
        "section_count": section_count,
        "physics_fingerprint": phys,
        "fingerprint": fingerprint,
        "header_hash": header_hash,
    }
    if hash_bytes(raw[:48] + b"\0" * 8) != header_hash:
        raise ValueError("compiled package header hash mismatch")
    sections = {}
    at = 56
    for _ in range(section_count):
        sid, stride, count, byte_count, content_hash = struct.unpack_from("<IIQQQ", raw, at)
        at += 32
        data = raw[at : at + byte_count]
        if len(data) != byte_count or byte_count != count * stride:
            raise ValueError(f"bad section {sid} byte/count/stride")
        if hash_bytes(data) != content_hash:
            raise ValueError(f"bad section {sid} content hash")
        sections[sid] = {
            "element_size": stride,
            "element_count": count,
            "data": data,
            "content_hash": content_hash,
        }
        at += byte_count
    if at != len(raw):
        raise ValueError("trailing package bytes")
    return header, sections


def fadd(a: float, b: float) -> np.float32:
    return F32(F32(a) + F32(b))


def fsub(a: float, b: float) -> np.float32:
    return F32(F32(a) - F32(b))


def fmul(a: float, b: float) -> np.float32:
    return F32(F32(a) * F32(b))


def round_odd(high: np.float32, low: np.float32) -> np.float32:
    if low == 0 or (struct.unpack("<I", struct.pack("<f", float(high)))[0] & 1):
        return F32(high)
    direction = F32(math.inf if low > 0 else -math.inf)
    return F32(np.nextafter(F32(high), direction, dtype=np.float32))


def compensated_sum(a: float, b: float) -> tuple[np.float32, np.float32]:
    a, b = F32(a), F32(b)
    high = fadd(a, b)
    virtual_b = fsub(high, a)
    low = fadd(fsub(a, fsub(high, virtual_b)), fsub(b, virtual_b))
    return high, low


def compensated_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[np.float32, np.float32]:
    high = compensated_sum(a[0], b[0])
    low = compensated_sum(a[1], b[1])
    middle = compensated_sum(high[1], low[0])
    merged = compensated_sum(high[0], middle[0])
    tail0 = compensated_sum(merged[1], middle[1])
    tail1 = compensated_sum(tail0[0], low[1])
    sticky = round_odd(*compensated_sum(tail1[1], tail0[1]))
    tail = round_odd(*compensated_sum(tail1[0], sticky))
    return compensated_sum(merged[0], tail)


def compensated_product(a: float, b: float) -> tuple[np.float32, np.float32]:
    a, b = F32(a), F32(b)
    high = fmul(a, b)
    # The source calls IEEE single-precision fma for the exact product tail.
    # Python's math.fma is not available on all supported macOS interpreters;
    # use libm's fmaf through ctypes in the helper below.
    import ctypes
    libm = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fmaf = libm.fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float
    low = F32(fmaf(float(a), float(b), -float(high)))
    return high, low


def compensated_multiply(a: tuple[float, float], b: tuple[float, float]) -> tuple[np.float32, np.float32]:
    p0 = compensated_product(a[0], b[0])
    p1 = compensated_product(a[0], b[1])
    p2 = compensated_product(a[1], b[0])
    p3 = compensated_product(a[1], b[1])
    return compensated_add(p0, compensated_add(p1, compensated_add(p2, p3)))


def frame_from_quaternion(qraw: np.ndarray) -> np.ndarray:
    q = np.asarray(qraw, dtype=np.float32).copy()
    dot = F32(F32(q[0] * q[0]) + F32(F32(q[1] * q[1]) + F32(q[2] * q[2])))
    dot = F32(dot + F32(q[3] * q[3]))
    # Source is `raw * rsqrt(dot(raw,raw))`; CPU reciprocal sqrt is a
    # transcription, not a claimed Metal rsqrt bit-match.
    inv = F32(F32(1.0) / F32(np.sqrt(dot, dtype=np.float32)))
    q = np.asarray([F32(x * inv) for x in q], dtype=np.float32)
    x, y, z, w = q
    tw = F32(2.0)
    r = np.zeros((3, 3), dtype=np.float32)
    r[0, 0] = fsub(1.0, fmul(tw, fadd(fmul(y, y), fmul(z, z))))
    r[0, 1] = fmul(tw, fsub(fmul(x, y), fmul(z, w)))
    r[0, 2] = fmul(tw, fadd(fmul(x, z), fmul(y, w)))
    r[1, 0] = fmul(tw, fadd(fmul(x, y), fmul(z, w)))
    r[1, 1] = fsub(1.0, fmul(tw, fadd(fmul(x, x), fmul(z, z))))
    r[1, 2] = fmul(tw, fsub(fmul(y, z), fmul(x, w)))
    r[2, 0] = fmul(tw, fsub(fmul(x, z), fmul(y, w)))
    r[2, 1] = fmul(tw, fadd(fmul(y, z), fmul(x, w)))
    r[2, 2] = fsub(1.0, fmul(tw, fadd(fmul(x, x), fmul(y, y))))
    return r


def source_deformation_words(u: np.ndarray, inverse: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world = [[(F32(0.0), F32(0.0)) for _ in range(3)] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            value = (F32(1.0 if row == col else 0.0), F32(0.0))
            for k in range(3):
                difference = compensated_add((u[k + 1, row], F32(0.0)), (-u[0, row], F32(0.0)))
                product = compensated_multiply(difference, (inverse[k, col], F32(0.0)))
                value = compensated_add(value, product)
            world[row][col] = value
    high = np.zeros((3, 3), dtype=np.float32)
    low = np.zeros((3, 3), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            value = (F32(0.0), F32(0.0))
            for k in range(3):
                product = compensated_multiply(world[row][k], (frame[k, col], F32(0.0)))
                value = compensated_add(value, product)
            high[row, col], low[row, col] = value
    return high, low


def source_initial_seed(p: np.ndarray, f: np.ndarray, old: np.ndarray) -> np.ndarray:
    # The source's 15 state-update programs leave six ep/accumulation histories
    # at accepted values and initialize six stress states to C(E-ep)/sigma_ref.
    # This is the authored generic update guess, not the converged root.
    z = np.asarray(old, dtype=np.float64).copy()
    z[6:12] = ref.stiffness(p) @ (ref.strain(f) - z[:6])
    return z


def main() -> int:
    outdir = RUN / "output"
    pkg = outdir / "compiled.nmatterpack"
    cand = outdir / "rejected-conditioning-000002/candidate_nodes.bin"
    cond_manifest = outdir / "rejected-conditioning-000002/manifest.json"
    mesh_path = outdir / "mesh.json"
    state_path = outdir / "material_state_000002_bent.json"
    obs_path = outdir / "observations.csv"
    pred_path = RUN / "prediction.json"
    run_manifest = outdir / "manifest.json"
    result_path = outdir / "result.json"
    source_paths = [
        SOURCE / "matter/src/metal/common.metalinc",
        SOURCE / "matter/src/metal/fem.metalinc",
        SOURCE / "matter/src/compiler_impl.cpp",
        SOURCE / "matter/src/validation_impl.cpp",
        SOURCE / "matter/include/numi/matter/fem_compensated_deformation.h",
        SOURCE / "include/metalrobo/compensated_geometry_gpu.h",
        SOURCE / "include/metalrobo/compensated_translation_gpu.h",
        SOURCE / "matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter",
        SOURCE / "matter/tools/cardboard_probe.mm",
    ]
    package_header, sections = package_sections(pkg)
    mesh = json.loads(mesh_path.read_text())
    accepted = json.loads(state_path.read_text())
    runmeta = json.loads(run_manifest.read_text())
    prediction = json.loads(pred_path.read_text())
    result = json.loads(result_path.read_text())
    cmanifest = json.loads(cond_manifest.read_text())
    if accepted["status_code"] != 5 or accepted["environment"] != ENV:
        raise ValueError("accepted history export is not the expected rolled-back failure row")
    if cmanifest["accepted_physical_state"] is not False or cmanifest["control_step"] != 1:
        raise ValueError("conditioning candidate is not the rejected control-step-1 state")

    tet_section = sections[20]
    if tet_section["element_size"] != 96 or TET_ID >= tet_section["element_count"]:
        raise ValueError("compiled tet layout/count mismatch")
    traw = tet_section["data"][TET_ID * 96 : (TET_ID + 1) * 96]
    node_ids = list(struct.unpack_from("<4I", traw, 0))
    invrow0 = struct.unpack_from("<4f", traw, 16)
    invrow1 = struct.unpack_from("<4f", traw, 32)
    invrow2 = struct.unpack_from("<4f", traw, 48)
    identity = struct.unpack_from("<4I", traw, 64)
    quaternion = struct.unpack_from("<4f", traw, 80)
    inverse = np.asarray([invrow0[:3], invrow1[:3], invrow2[:3]], dtype=np.float32)
    materials = sections[3]
    if materials["element_size"] != 160 or identity[0] >= materials["element_count"]:
        raise ValueError("compiled material descriptor layout/index mismatch")
    material_raw = materials["data"][identity[0] * materials["element_size"] : (identity[0] + 1) * materials["element_size"]]
    material_words = struct.unpack_from("<18I", material_raw, 0)
    # NMMaterialGPU parameterOffset, parameterCount, stateCount are words 1-3;
    # localNewtonIterations follows the aligned uint64 stateTransferMask.
    parameter_offset, parameter_count, state_count = material_words[1:4]
    local_iterations = struct.unpack_from("<I", material_raw, 80)[0]
    if parameter_count != 28 or state_count != 15:
        raise ValueError("tet 1820 is not the expected 15-state/28-parameter material")
    parameter_section = sections[6]
    if parameter_section["element_size"] != 16:
        raise ValueError("compiled parameter records are not nm_float4")
    all_parameter_vectors = np.frombuffer(
        parameter_section["data"], dtype="<f4"
    ).reshape(-1, 4)
    parameter_vectors = all_parameter_vectors[
        parameter_offset : parameter_offset + parameter_count
    ]
    # NMMaterialGPU offsets count float4 records. The scalar VM reads each
    # record's x lane; slicing the section as a flat float array is incorrect.
    p = np.asarray(parameter_vectors[:, 0], dtype=np.float32)

    candidate_nodes = cand.read_bytes()
    if len(candidate_nodes) != int(cmanifest["fem_node_count"]) * 2 * 80:
        raise ValueError("candidate node buffer length does not match E*N*80")
    u = np.asarray([struct.unpack_from("<3f", candidate_nodes, (ENV * 882 + n) * 80 + 64) for n in node_ids], dtype=np.float32)
    v = np.asarray([struct.unpack_from("<3f", candidate_nodes, (ENV * 882 + n) * 80 + 16) for n in node_ids], dtype=np.float32)
    high, low = source_deformation_words(u, inverse, frame_from_quaternion(np.asarray(quaternion, dtype=np.float32)))
    f = high.astype(np.float64) + low.astype(np.float64)
    old_list = accepted["tetrahedra"][TET_ID]["state"]
    if len(old_list) != 15:
        raise ValueError("accepted material history is not 15 entries")
    old = np.asarray(old_list, dtype=np.float32).astype(np.float64)
    seed = source_initial_seed(p.astype(np.float64), f, old)

    diag_det = float(np.linalg.det(np.eye(3) + np.column_stack((u[1] - u[0], u[2] - u[0], u[3] - u[0])).astype(np.float64) @ inverse.astype(np.float64)))
    # The initial guess is precisely the production update-program value rounded
    # to FP32 (up to bytecode operation order); the independent root equations
    # are evaluated in FP64 and are not a Metal execution.
    old_max = pol.MAXITER
    replay = {}
    for budget in (16, 17, 32):
        pol.MAXITER = budget
        z, info = pol.project_scaled(p.astype(np.float64), f, old, seed)
        replay[str(budget)] = {
            **info,
            "iterations_are_newton_updates": info.get("iterations"),
            "returned_state": [float(x) for x in z],
        }
    pol.MAXITER = old_max

    cli = prediction.get("argv", [])
    try:
        requested = int(cli[cli.index("--material-iterations") + 1])
    except (ValueError, IndexError):
        requested = None
    compiled_requested = runmeta.get("solver", {}).get("local_material_newton_iterations")
    material_policy = runmeta.get("compensated_source_paper_projection", {})
    report = {
        "schema": "numi.cardboard.source105-rejected-row-replay.v1",
        "classification": "CPU-only independent equation replay from exact retained candidate-node, compiled package, material-parameter, and rolled-back accepted-history inputs. Python transcription of compensated F construction is source-derived but not native-bit-confirmed. No production physics or tolerances changed.",
        "run": str(RUN),
        "input_sha256": {
            str(path.relative_to(BASE)): sha(path)
            for path in [pkg, cand, cond_manifest, mesh_path, state_path, obs_path, pred_path, run_manifest, result_path]
        },
        "source_sha256": {str(path.relative_to(SOURCE)): sha(path) for path in source_paths},
        "package_header": package_header,
        "compiled_world_fingerprint": package_header["physics_fingerprint"],
        "failed_record": {
            "control_step": 1,
            "environment": ENV,
            "status_code": 5,
            "status_name": "NM_STATUS_INVALID_DEFORMATION",
            "object_index": 0,
            "tetrahedron_index": TET_ID,
            "native_status_diagnostic_x": 0.99774569272994995,
            "candidate_geometry_det_fp64_from_node_positions": diag_det,
            "diagnostic_det_difference": diag_det - 0.99774569272994995,
            "candidate_was_rejected_not_accepted_history": True,
        },
        "compiled_tet": {
            "nodes": node_ids,
            "inverse_rest_rows_fp32": inverse.tolist(),
            "inverse_rest_row0_w_rest_volume": float(invrow0[3]),
            "identity": list(identity),
            "material_frame_quaternion_xyzw_fp32": list(quaternion),
            "material_frame_quaternion_norm_squared_fp32": float(np.dot(np.asarray(quaternion, np.float32), np.asarray(quaternion, np.float32))),
            "material_index": identity[0],
        },
        "candidate_nodes": {
            "layout": "environment-major NMFEMNodeStateGPU, stride 80; reference displacement offset 64; velocity offset 16",
            "reference_displacement_fp32": u.tolist(),
            "reference_displacement_fp32_bits": [f32bits(row) for row in u],
            "velocity_fp32_m_per_s": v.tolist(),
            "velocity_fp32_bits": [f32bits(row) for row in v],
        },
        "compiled_material": {
            "index": identity[0],
            "parameter_offset": parameter_offset,
            "parameter_count": parameter_count,
            "state_count": state_count,
            "local_newton_iterations_compiled": local_iterations,
            "flags": material_words[7],
            "parameters_fp32": [float(x) for x in p],
            "parameter_records_float4_fp32": parameter_vectors.tolist(),
            "parameter_fp32_bits": f32bits(p),
            "accepted_state_rolled_back_fp32": [float(x) for x in old.astype(np.float32)],
            "accepted_state_fp32_bits": f32bits(old),
        },
        "deformation_words_source_transcription": {
            "F_high_fp32": high.tolist(),
            "F_high_bits": [f32bits(row) for row in high],
            "F_low_fp32": low.tolist(),
            "F_low_bits": [f32bits(row) for row in low],
            "F_widened_sum": f.tolist(),
            "native_words_confirmed": False,
            "limitation": "Frame normalization uses CPU float32 reciprocal-sqrt; Metal rsqrt/FMA contraction may change last bits. Capture374 is the pending native confirmation.",
        },
        "budget_binding": {
            "argv_material_iterations_requested": requested,
            "manifest_requested_local_material_iterations": compiled_requested,
            "compiled_material_local_newton_iterations": local_iterations,
            "accurate_projection_enabled": material_policy.get("accurate_bilinear15_projection"),
            "compiler_source_rule": "genericBilinear15AccurateProjection ? 16u : options.localMaterialNewtonIterations",
            "compiler_validation_rule": "accurate bilinear15 requires localNewtonIterations == 16",
            "observed_contract_mismatch": requested != local_iterations,
            "interpretation": "Explicit CLI request 32 is recorded but silently ignored for this opt-in; compiler and validator deliberately force 16. The run manifest therefore needs both requested and effective values to avoid ambiguity.",
        },
        "fp64_independent_scaled_projection": {
            "method": "bilinear15_reference.py equations and row/variable-scaled local solve; same residual tolerance 2e-8, correction tolerance 2e-8, authored residual infinity-norm Armijo with 16 backtracks; no tolerance changes",
            "initialization": "accepted ep and histories; six stress-state seed components initialized from the authored generic state-update equations C(E-ep)/sigma_ref",
            "f32_Fpair_widened_sum_used": True,
            "Fdot_and_timestep_are_constitutively_unused": True,
            "budgets": replay,
            "interpretation": "If the source-transcribed F words are confirmed, this CPU reference needs 17 updates to satisfy the unchanged scaled-correction gate, while compiled budget16 ends after 16 updates with a correction above 2e-8. This is consistent with, but does not prove the cause of, native projection failure.",
        },
        "evidence_limits": [
            "The candidate deformation determinant matches the native failure diagnostic; it is not the accepted checkpoint geometry.",
            "The accepted history is exported after rollback and is used as the pre-failure accepted state.",
            "The Python source transcription is not bitwise Metal output and its F high/low pair awaits the selected native capture.",
            "The CPU FP64 replay cannot establish production FP32 iteration behavior or a physical constitutive qualification.",
        ],
    }
    (OUT / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(OUT / "replay.json"),
        "failed_det": diag_det,
        "requested_iterations": requested,
        "compiled_iterations": local_iterations,
        "replays": {k: {a: v.get(a) for a in ("status", "iterations", "residual_inf_scaled", "scaled_correction_inf", "correction_threshold")} for k, v in replay.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
