#!/usr/bin/env python3
"""Static, source-bound checks for the opt-in coupled Armijo trial capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


BASE = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/"
    "source-direct-projection-diagnostic-089/matter/src/runtime.mm"
)
OVERLAY = Path(
    "/Users/home/cardboard-implementation-20261014/analysis/"
    "coupled-trial-capture-overlay-001/overlay/matter/src/runtime.mm"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    base = BASE.read_text()
    overlay = OVERLAY.read_text()

    required = [
        ("NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR", 
         "selector environment variable"),
        ("NM_FEM_COUPLED_TRIAL_CAPTURE_PATH",
         "path environment variable"),
        ('phase != "pre" && phase != "post" && phase != "both"',
         "phase allowlist"),
        ("femCoupledCaptureCapBytes - bytes", "checked aggregate byte bound"),
        ("femCoupledTrialCaptureArrays.size() + present > 192u",
         "array count bound"),
        ("newBufferWithLength:static_cast<NSUInteger>(\n                            femCoupledCaptureCapBytes)",
         "fixed-size bounded readback allocation"),
        ("O_WRONLY | O_CREAT | O_EXCL", "exclusive output creation"),
        ('numi.matter.coupled_armijo_trial_capture.v1', "versioned schema"),
        ("sha256Hex(data + array.offset", "per-array hash metadata"),
        ("sha256Hex(data,", "payload hash metadata"),
        ("material_projected_trial_stress_state",
         "explicit projected-state limitation"),
        ("nm_contact_validate_armijo_deformable_sweep binds state.femCandidate at buffer 5",
         "exact pre/post sweep producer binding"),
        ("nm_contact_narrowphase_deformable receives trialNodes at buffer 4",
         "exact endpoint narrowphase binding"),
        ("no standalone TOI scalar is exported",
         "explicit TOI limitation"),
        ("physical_state_changed\\\":false",
         "failure remains observational"),
    ]
    for needle, label in required:
        require(needle in overlay, f"missing {label}: {needle}")

    # The overlay is intentionally a narrow runtime-only patch against 089.
    require(base != overlay, "overlay unexpectedly equals frozen base")
    require(overlay.count("NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR") >= 1,
            "capture selector is absent")
    require("dispatchThreads(\n                        \"nm_fem_armijo_decide\"" in overlay,
            "production Armijo decision path missing")

    def position(text: str, needle: str, start: int = 0) -> int:
        found = text.find(needle, start)
        require(found >= 0, f"missing callsite marker: {needle}")
        return found

    callsites = [
        position(overlay, '"candidate_before_private_contact_rebuild"'),
        position(overlay, '"residual_after_bulk_mechanical"'),
        position(overlay, '"residual_after_cohesive"'),
        position(overlay, '"residual_after_rigid_contact"'),
        position(overlay, '"private_contact_rebuild_before_free_body_terms"'),
        position(overlay, '"residual_after_deformable_contact"'),
        position(overlay, '"full_residual_before_armijo_measure"'),
        position(overlay, 'if (!captureTrialMeasurement())'),
    ]
    decision = position(overlay, '"nm_fem_armijo_decide", environments',
                        callsites[-1])
    callsites.append(decision)
    require(callsites == sorted(callsites),
            "capture producer/decision ordering changed")

    # Validate simple schema literals independently with a strict JSON parser.
    sample = {
        "schema": "numi.matter.coupled_armijo_trial_capture.v1",
        "byte_order": "little_endian",
        "capture_kind": "observational_private_trial_not_checkpoint",
        "arrays": [{"name": "sample", "phase": "pre", "dtype": "uint32",
                    "count": 1, "element_bytes": 4, "offset": 0,
                    "byte_count": 4, "sha256": "0" * 64}],
    }
    reparsed = json.loads(json.dumps(sample, allow_nan=False))
    require(reparsed["schema"] == sample["schema"],
            "schema sample did not survive strict JSON roundtrip")

    report = {
        "schema": "numi.matter.coupled_armijo_trial_capture_static_check.v1",
        "base": str(BASE),
        "base_sha256": sha256(BASE),
        "overlay": str(OVERLAY),
        "overlay_sha256": sha256(OVERLAY),
        "checks": len(required) + 4,
        "result": "PASS_STATIC_ONLY",
        "native_or_physical_qualification": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
