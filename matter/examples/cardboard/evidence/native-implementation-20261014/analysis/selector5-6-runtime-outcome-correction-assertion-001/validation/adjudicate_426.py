#!/usr/bin/env python3
"""Audit the retained selector5/6 native outcome without changing its gates."""
from __future__ import annotations
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
EXPECTED = {
    "input/source131/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm": "1b91d8888c178cd5719be48eff8c1b0ee993d8078598ef6179e2dc45c44b1819",
    "input/source131/frozen-source.json": "8c076e2876df023b387f248b851d1224720e8096171f560ae72983c51d8e8b50",
    "input/source131/matter/include/numi/matter/shared.h": "bd13b1baf8790956404cd564685919c5a18e339d79e7c5c5165956f94d2f010c",
    "input/source-build.json": "367bd05b371e40f9bc3a1ebf21d0710efa0d65b70ebe4c781e179471795b2bdb",
    "input/native-426/receipt.json": "1bbdf3ba4c078c01c70e7c595bde0d083cc56a7b79feea535aafa95302fcb099",
    "input/native-426/stdout.log": "f280e06af35aea2b8be37a551d36439f942f554ab317a829ab031be0b9a7b3fd",
    "input/native-426/stderr.log": "b114f4efb00d91028653179e342174e3f13fa90f186fa09cd7f5eb341ceb19e1",
    "input/native-426/prediction.json": "9c8a1fef69d3b2b87a3b95206d2c8968084069ab220feac49efe48f1720419c8",
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]*)", line))

def one_line(lines: list[str], prefix: str) -> str:
    rows = [line for line in lines if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError(f"expected one {prefix!r} row, found {len(rows)}")
    return rows[0]

def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]

def main() -> int:
    for rel, expected in EXPECTED.items():
        actual = sha(PKG / rel)
        if actual != expected:
            raise ValueError(f"input hash mismatch {rel}: {actual}")

    source = (PKG / "input/source131/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm").read_text()
    candidate_path = PKG / "source/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm"
    candidate = candidate_path.read_text()
    if sha(candidate_path) != "882dbe998fea3511f76da39492c38c2e9f715f2487f4f37d8267010e17c70da7":
        raise ValueError("checker candidate hash mismatch")
    if "state->statuses[0].fgmresIterations > 0u" not in candidate:
        raise ValueError("candidate does not check recorded Krylov work")
    if "loadedNodeMotion > 1.0e-8" not in candidate:
        raise ValueError("candidate changed the independent loaded-motion gate")
    build = json.loads((PKG / "input/source-build.json").read_text())
    if build.get("success") is not True or build.get("source_sha256", {}).get(
        "matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm"
    ) != EXPECTED["input/source131/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm"]:
        raise ValueError("source build does not bind the frozen base checker")
    if "1.000020, 0.000014, -0.000006" not in source or \
       "Point{0.03131, -0.01630, 0.027008}" not in source:
        raise ValueError("authored baseline geometry/deformation literals changed")
    if "relativeCorrection > 0.0f" not in source:
        raise ValueError("base source no longer contains the audited correction assertion")

    receipt = json.loads((PKG / "input/native-426/receipt.json").read_text())
    if receipt.get("execution_started") is not True or receipt.get("source_inputs_unchanged") is not True:
        raise ValueError("native receipt does not bind an executed unchanged-input run")
    if receipt.get("exit_code") != 1 or receipt.get("timed_out") is not False:
        raise ValueError("expected the retained checker failure, not a timeout or pass")

    text = (PKG / "input/native-426/stdout.log").read_text()
    lines = text.splitlines()
    stderr = (PKG / "input/native-426/stderr.log").read_text()
    if "FAIL selector runtime fixture accepted without a nonzero mechanical correction" not in stderr:
        raise ValueError("retained first checker failure text changed")

    per_mode = {}
    for mode in (5, 6):
        label = f"selector{mode}"
        status = fields(one_line(lines, f"selector_runtime_status label={label} "))
        cert = fields(one_line(lines, f"selector_runtime_certificate label={label} "))
        outcome = fields(one_line(lines, f"selector_runtime_outcome label={label} "))
        if int(status["code"]) != 0 or int(status["completed_microsteps"]) != 1:
            raise ValueError(f"selector{mode} did not report accepted one-step status")
        if int(status["fgmres_iterations"]) != 5:
            raise ValueError(f"selector{mode} FGMRES evidence changed")
        if int(outcome["snapshot_available"]) != 1 or int(outcome["fem_node_count"]) != 4:
            raise ValueError(f"selector{mode} outcome snapshot missing")
        nonlinear = tuple(float(x) for x in cert["nonlinear"].split(","))
        validity = tuple(float(x) for x in cert["validity"].split(","))
        if len(nonlinear) != 4 or len(validity) != 4 or not all(map(math.isfinite, nonlinear + validity)):
            raise ValueError(f"selector{mode} certificate is malformed")
        if validity[3] <= 0.5:
            raise ValueError(f"selector{mode} certificate is not accepted")
        if nonlinear[1] != 0.0:
            raise ValueError(f"selector{mode} certificate correction unexpectedly changed")
        nodes_hex = outcome["fem_nodes_raw_hex"]
        raw = bytes.fromhex(nodes_hex)
        if len(raw) != 4 * 80:
            raise ValueError(f"selector{mode} expected four 80-byte FEM node records")
        final = struct.unpack_from("<3f", raw, 2 * 80)
        per_mode[mode] = {
            "status_code": int(status["code"]),
            "completed_microsteps": int(status["completed_microsteps"]),
            "fgmres_iterations": int(status["fgmres_iterations"]),
            "certificate_nonlinear": list(nonlinear),
            "certificate_validity": list(validity),
            "final_loaded_node2_position_m": list(final),
            "fem_nodes_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "source_fingerprint": int(outcome["source_fingerprint"]),
            "device_program_fingerprint": int(outcome["device_program_fingerprint"]),
        }

    # Authored baseline from source131 cook(): object.femNodes = F * referenceNodes,
    # where F is baselineDeformation. The compiler stores those positions as FP32.
    f = (1.000020, 0.000014, -0.000006,
         -0.000008, 0.999985, 0.000011,
         0.000005, -0.000009, 1.000012)
    points = ((0.031, -0.018, 0.027),
              (0.0333, -0.01783, 0.027004),
              (0.03131, -0.01630, 0.027008),
              (0.03106, -0.01773, 0.027035))
    point = points[2]
    initial = tuple(f32(f[3*r] * point[0] + f[3*r+1] * point[1] + f[3*r+2] * point[2]) for r in range(3))
    motions = {}
    for mode, result in per_mode.items():
        delta = tuple(result["final_loaded_node2_position_m"][i] - initial[i] for i in range(3))
        motions[str(mode)] = {
            "initial_loaded_node2_position_m": list(initial),
            "delta_m": list(delta),
            "norm_m": math.sqrt(sum(x*x for x in delta)),
            "unchanged_source_gate_m": 1.0e-8,
            "gate_passes": math.sqrt(sum(x*x for x in delta)) > 1.0e-8,
        }

    if per_mode[5]["fgmres_iterations"] == 0 or per_mode[6]["fgmres_iterations"] == 0:
        raise ValueError("cannot adjudicate the telemetry assertion without recorded Krylov work")
    if per_mode[5]["certificate_nonlinear"] != per_mode[6]["certificate_nonlinear"] or \
       per_mode[5]["certificate_validity"] != per_mode[6]["certificate_validity"] or \
       per_mode[5]["fem_nodes_raw_sha256"] != per_mode[6]["fem_nodes_raw_sha256"]:
        raise ValueError("selector5 and selector6 accepted outputs differ")
    if any(motions[str(mode)]["gate_passes"] for mode in (5, 6)):
        raise ValueError("motion gate unexpectedly passes; this retained failure needs re-adjudication")
    if motions["5"]["norm_m"] != motions["6"]["norm_m"]:
        raise ValueError("loaded-node motion differs between selectors")

    report = {
        "schema": "numi.cardboard.selector5-6-outcome-audit.v1",
        "native_case": "hybrid-runtime-armijo-outcomes-426",
        "native_receipt_exit_code": receipt["exit_code"],
        "first_checker_failure": "relativeCorrection > 0 was used as a proxy for Krylov work; both modes actually report five FGMRES iterations",
        "selector_outcomes": {str(k): v for k, v in per_mode.items()},
        "loaded_node_motion": motions,
        "conclusion": "The correction telemetry assertion is invalid as a Krylov-work assertion. Replacing it with the recorded FGMRES count exposes the separate unchanged >1e-8 m loaded-node-motion gate, which fails: the measured motion is one FP32 ULP (~1.86e-9 m). Keep that motion gate and do not claim selector5/6 runtime qualification.",
        "qualification": {"selector5_qualified": False, "selector6_qualified": False, "motion_threshold_relaxed": False},
        "inputs_sha256": EXPECTED,
    }
    out = PKG / "validation/fixture-audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "report": str(out), "loaded_node_motion_m": motions["6"]["norm_m"], "gate_m": 1.0e-8, "selector6_qualified": False}, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
