#!/usr/bin/env python3
"""Independent FP64 check of a captured Haj-Ali medium elastic projection.

This is a diagnostic of state representability, not a solver replay. It decodes
the exact binary32 values represented by the capture JSON, adds the captured
high/low deformation words in binary64, and evaluates the authored normalized
StVK stress equations from the retained .nmatter source.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT / "capture.jsonl"
MATERIAL = ROOT / "inputs/matter/materials/hajali2009_medium_hill_ideal.nmatter"
COMPILED = ROOT / "output/compiled.nmatterpack"
SHADER = ROOT / "inputs/build-cardboard-abi45-009/shaders/NumiMatter.metallib"
PROBE = ROOT / "inputs/build-cardboard-abi45-009/numi-matter-cardboard-probe"
COMMON = ROOT / "inputs/matter/src/metal/common.metalinc"
BYTECODE = ROOT / "inputs/matter/include/numi/matter/paper_compensated_bytecode.h"
OUT = ROOT / "fp32-state-floor.json"
RESULT = ROOT / "output/result.json"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def next_f32(value: float, upward: bool) -> float:
    value = f32(value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if value == 0.0:
        return struct.unpack("<f", struct.pack("<I", 1 if upward else 0x80000001))[0]
    if (value > 0.0) == upward:
        bits += 1
    else:
        bits -= 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def nearest_f32(value: float) -> float:
    rounded = f32(value)
    options = (rounded, next_f32(rounded, False), next_f32(rounded, True))
    return min(options, key=lambda candidate: abs(candidate - value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_capture() -> tuple[dict, dict]:
    header = None
    selected = None
    with CAPTURE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record") == "header":
                header = row
            elif row.get("record") == "tetrahedron" and row.get("environment") == 0 and row.get("tet_id") == 1434:
                selected = row
                break
    if header is None or selected is None:
        raise RuntimeError("capture header or requested tetrahedron was not found")
    return header, selected


def analyze() -> dict:
    header, tet = read_capture()
    run_result = json.loads(RESULT.read_text())
    if run_result.get("schema") != "numi.cardboard.probe-result.v1":
        raise RuntimeError("unexpected bound run result schema")
    parameters = [f32(x) for x in header["parameters_by_environment"][0][20:40]]
    captured_state = [f32(x) for x in tet["center"]["projected_state"]]

    # The capture explicitly labels F_local as binary32 high plus binary32 low.
    # Promoting both decoded words to Python binary64 preserves their sum.
    high = tet["center"]["F_local"]
    low = tet["center"]["F_local_low"]
    F = [[f32(high[i][j]) + f32(low[i][j]) for j in range(3)] for i in range(3)]

    ep = captured_state[:6]
    stored_stress = captured_state[6:12]
    delta_lambda = captured_state[12] - f32(tet["center"]["accepted_state"][12])

    # Green normal strains and engineering shears, in the .nmatter state order.
    green = [
        0.5 * (sum(F[k][i] * F[k][i] for k in range(3)) - 1.0) - ep[i]
        for i in range(3)
    ]
    shear = [
        sum(F[k][i] * F[k][j] for k in range(3)) - 2.0 * ep[3 + index]
        for index, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)))
    ]

    # Parameter order is recorded in the capture header and the retained
    # material source: density, C11,C22,C33,C12,C13,C23,G12,G13,G23,sigma_ref.
    c11, c22, c33, c12, c13, c23, g12, g13, g23 = parameters[1:10]
    sigma_ref = parameters[10]
    exact_stress = [
        (c11 * green[0] + c12 * green[1] + c13 * green[2]) / sigma_ref,
        (c12 * green[0] + c22 * green[1] + c23 * green[2]) / sigma_ref,
        (c13 * green[0] + c23 * green[1] + c33 * green[2]) / sigma_ref,
        g23 * shear[0] / sigma_ref,
        g13 * shear[1] / sigma_ref,
        g12 * shear[2] / sigma_ref,
    ]

    residual_scale = parameters[19]
    stress_rows = []
    stress_residual = []
    stress_floor = []
    for index, (candidate, exact) in enumerate(zip(stored_stress, exact_stress)):
        closest = nearest_f32(exact)
        row = {
            "state": ("sbar11", "sbar22", "sbar33", "sbar23", "sbar13", "sbar12")[index],
            "exact_elastic_root_sbar": exact,
            "captured_binary32_sbar": candidate,
            "nearest_binary32_sbar": closest,
            "captured_equals_nearest": candidate == closest,
            "captured_unscaled_residual_sbar": candidate - exact,
            "captured_physical_stress_error_Pa": (candidate - exact) * sigma_ref,
            "captured_scaled_residual": abs(candidate - exact) * residual_scale,
            "nearest_scaled_residual_floor": abs(closest - exact) * residual_scale,
            "nearest_physical_stress_error_floor_Pa": abs(closest - exact) * sigma_ref,
        }
        stress_rows.append(row)
        stress_residual.append((candidate - exact) * residual_scale)
        stress_floor.append(abs(closest - exact) * residual_scale)

    # In this captured state Ep=0 and delta-lambda=0, so all six flow rows are
    # zero in the exact reference. Compute the smooth FB row independently.
    Fh, Gh, Hh, Lh, Mh, Nh = parameters[11:17]
    s11, s22, s33, s23, s13, s12 = stored_stress
    q2 = (
        Fh * (s22 - s33) ** 2
        + Gh * (s33 - s11) ** 2
        + Hh * (s11 - s22) ** 2
        + 2.0 * Lh * s23**2
        + 2.0 * Mh * s13**2
        + 2.0 * Nh * s12**2
        + parameters[17] ** 2
    )
    q = math.sqrt(q2)
    gap = 1.0 - q
    fb_epsilon = parameters[18]
    fb_residual = residual_scale * (math.sqrt(delta_lambda**2 + gap**2 + fb_epsilon**2) - delta_lambda - gap)
    full_reference_norm = max([abs(x) for x in stress_residual] + [abs(fb_residual)])

    captured_diagnostics = tet["center"]["projection"]
    capture_status = tet["center"]["flags_terminal_status_parameter_count"]
    native_residual = captured_diagnostics[0]
    state_norm = captured_diagnostics[1]
    configured_tolerance = 2.0e-8 * (1.0 + state_norm)
    floor_index = max(range(6), key=lambda index: stress_floor[index])
    floor_row = stress_rows[floor_index]

    return {
        "schema": "numi.cardboard.fp32-state-floor-diagnostic.v1",
        "classification": "independent_source_equation_quantization_diagnostic; not_native_replay_or_physical_validation",
        "capture": {
            "path": str(CAPTURE),
            "sha256": sha256(CAPTURE),
            "schema": header["schema"],
            "world_fingerprint_hex": header["world_fingerprint_hex"],
            "execution_fingerprint_hex": header["execution_fingerprint_hex"],
            "environment": tet["environment"],
            "tet_id": tet["tet_id"],
            "material_name": tet["material_name"],
            "F_encoding": header["F_local_word_encoding"],
            "F_local_high_plus_low": F,
            "accepted_steps": run_result["accepted_steps"],
            "run_status": run_result["status"],
            "native_recorded_projection_residual": native_residual,
            "native_recorded_state_norm": state_norm,
            "native_projection_succeeded": bool(captured_diagnostics[3]),
            "native_terminal_failure_code": capture_status[1],
            "native_terminal_failure_name": "ArmijoNoAcceptedTrial" if capture_status[1] == 6 else "unmapped",
        },
        "bindings": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (MATERIAL, COMPILED, SHADER, PROBE, COMMON, BYTECODE, RESULT)
        },
        "method": {
            "reference": "binary64 evaluation of retained Haj-Ali .nmatter normalized StVK stress equations using exact decoded binary32 parameters/state and decoded F high+low words",
            "stress_components": "normalized sbar = second Piola stress / sigma_ref; physical stress errors multiply sigma_ref in Pa",
            "local_residual_scaling": "authored residual_scale multiplies each implicit row; it is numerical conditioning and does not alter the zero set",
            "candidate_states_are_binary32": True,
            "no_tolerance_or_material_parameter_changes": True,
        },
        "material_scales": {
            "sigma_ref_Pa": sigma_ref,
            "residual_scale": residual_scale,
            "q_floor": parameters[17],
            "fb_epsilon": fb_epsilon,
            "configured_scaled_residual_threshold": configured_tolerance,
        },
        "stress_rows": stress_rows,
        "reference": {
            "exact_equation_scaled_residual_inf": full_reference_norm,
            "exact_equation_FB_row_scaled_residual": fb_residual,
            "captured_flow_rows_exactly_zero": all(x == 0.0 for x in ep) and delta_lambda == 0.0,
            "max_binary32_stress_state_floor": stress_floor[floor_index],
            "floor_dominant_row": floor_row["state"],
            "floor_dominant_row_physical_error_Pa": floor_row["nearest_physical_stress_error_floor_Pa"],
            "floor_over_configured_threshold": stress_floor[floor_index] / configured_tolerance,
            "native_residual_over_reference_quantization_floor": native_residual / stress_floor[floor_index],
            "native_residual_minus_reference_norm": native_residual - full_reference_norm,
        },
        "interpretation": {
            "single_precision_stress_state_can_meet_configured_threshold": stress_floor[floor_index] <= configured_tolerance,
            "finding": "The nearest representable binary32 sbar11 state has a source-equation residual floor above the captured solver threshold. This proves that exact local convergence at that threshold is impossible for this row with a single binary32 stress-state word. The equivalent physical error is small in absolute units, but no application-level error allowance is specified here.",
            "unresolved_native_residual_gap": "The captured native local residual is larger than the independently computed representability floor. The floor does not explain the full native residual; this diagnostic does not identify whether the remainder comes from compiled evaluator arithmetic, another implementation detail, or capture formatting.",
            "precision_repair_options": [
                "Store candidate and accepted normalized stress state as two-word (high+low) values, and carry both words through snapshots, rollback, and local projection.",
                "Alternatively, add an independently evaluated backward-error certificate that reports residual floors and physical stress error in Pa; admission still needs an explicit application error bound, not a blanket tolerance relaxation."
            ],
            "prohibited_inference": "This local material-point diagnostic is not evidence of board response, global FEM convergence, or physical validation.",
        },
    }


if __name__ == "__main__":
    result = analyze()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "floor_row": result["reference"]["floor_dominant_row"],
        "scaled_floor": result["reference"]["max_binary32_stress_state_floor"],
        "configured_threshold": result["material_scales"]["configured_scaled_residual_threshold"],
        "native_residual": result["capture"]["native_recorded_projection_residual"],
        "native_residual_gap_unresolved": result["reference"]["native_residual_minus_reference_norm"],
    }, indent=2))
