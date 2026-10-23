#!/usr/bin/env python3
"""Check unchanged rows of an existing, row-preserving NHSKIN layer candidate.

This is a preparation check, not geometry or physical admission. Refined meshes
must use their explicit parent ancestry and are intentionally not accepted here.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np


class BaselineError(ValueError):
    pass


def _section(text, name):
    try:
        return text.split(name + "=", 1)[1].split(";", 1)[0]
    except IndexError as exc:
        raise BaselineError("missing layer array: " + name) from exc


def _numbers(text):
    return [float(v.strip().removesuffix("f")) for v in text.split(",") if v.strip()]


def _vectors(text, name):
    values = re.findall(r"(?:std::array<double,3>)?\{([^{}]+)\}", _section(text, name))
    out = np.asarray([_numbers(v) for v in values], dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3 or not np.isfinite(out).all():
        raise BaselineError("invalid layer vectors: " + name)
    return out


def read_baseline(path, expected_sha256):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or actual != expected_sha256:
        raise BaselineError("native baseline header SHA-256 mismatch")
    text = raw.decode("utf-8")
    frames = re.findall(r"WholeBodySkinDonorFrameV1\{(\d+)u,\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\},\{([^{}]+)\}\}", text)
    if not frames:
        raise BaselineError("missing native donor frames")
    # Donor-frame fields are float in the native struct; honor literal rounding.
    outer = np.asarray([_numbers(f[3]) for f in frames], dtype=np.float32).astype(np.float64)
    local = np.asarray([_numbers(f[1]) for f in frames], dtype=np.float32).astype(np.float64)
    out = {
        "outer_world_m": outer + _vectors(text, "kWholeBodySkinMechanicalWorldDeltas"),
        "outer_donor_local_m": local + _vectors(text, "kWholeBodySkinMechanicalDonorLocalDeltas"),
        "skin_inner_world_m": _vectors(text, "kWholeBodySkinInnerWorld"),
        "skin_inner_donor_local_m": _vectors(text, "kWholeBodySkinInnerLocal"),
        "fat_deep_world_m": _vectors(text, "kWholeBodyFatDeepWorld"),
        "fat_deep_donor_local_m": _vectors(text, "kWholeBodyFatDeepLocal"),
        "donor_body_indices": np.asarray([int(f[0]) for f in frames], dtype=np.int64),
    }
    for key, name in (("skin_thickness_m", "kWholeBodySkinThicknessM"), ("fat_thickness_m", "kWholeBodyFatThicknessM")):
        out[key] = np.asarray(_numbers(_section(text, name).strip().strip("{}")))
    for key, name in (("source_vertex_ids", "kWholeBodySkinSourceVertices"), ("source_face_parents", "kWholeBodySkinSourceFaces")):
        out[key] = np.asarray([int(v) for v in re.findall(r"\d+", _section(text, name))], dtype=np.int64)
    out["faces"] = np.asarray([[int(v) for v in f] for f in re.findall(r"Face\s*\{\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(\d+)u?\s*\}", _section(text, "kWholeBodySkinFaces"))], dtype=np.int64)
    return out


def check_arrays(baseline, candidate, edited_rows, maximum_displacement_mm=None):
    if maximum_displacement_mm is not None and (
        not np.isfinite(maximum_displacement_mm) or maximum_displacement_mm <= 0
    ):
        raise BaselineError("maximum displacement must be finite and positive in millimeters")
    n = len(baseline["outer_world_m"])
    edited = list(edited_rows)
    if not edited or len(set(edited)) != len(edited) or any(type(i) is not int or i < 0 or i >= n for i in edited):
        raise BaselineError("edited rows must be unique, explicit native row indices")
    outside = np.ones(n, dtype=bool)
    outside[edited] = False
    immutable = {"faces", "source_vertex_ids", "source_face_parents", "donor_body_indices"}
    checks = {}
    for key, reference in baseline.items():
        if key not in candidate:
            raise BaselineError("missing candidate array: " + key)
        value = np.asarray(candidate[key])
        if value.shape != reference.shape or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise BaselineError("invalid candidate shape or finite coordinates: " + key)
        if key in immutable:
            equal = np.all(value == reference, axis=tuple(range(1, value.ndim))) if value.ndim > 1 else value == reference
        else:
            # A double snapshot preserves exact emitted header values, including signed zero.
            v = np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)
            r = np.ascontiguousarray(reference, dtype=np.float64).view(np.uint64)
            equal = np.all(v == r, axis=tuple(range(1, v.ndim))) if v.ndim > 1 else v == r
            equal = equal | ~outside
        bad = np.flatnonzero(~equal)
        checks[key] = {"passed": not bool(len(bad)), "changed_forbidden_rows": int(len(bad)), "first_rows": bad[:16].tolist()}
    displacement = {}
    for key in ("outer_world_m", "skin_inner_world_m", "fat_deep_world_m"):
        if key not in baseline:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            distance_mm = 1000.0 * np.linalg.norm(
                np.asarray(candidate[key], dtype=np.float64) - baseline[key], axis=1
            )
        if not np.isfinite(distance_mm).all():
            raise BaselineError("candidate displacement overflows millimeter units: " + key)
        row = int(np.argmax(distance_mm))
        displacement[key] = {
            "maximum_mm": float(distance_mm[row]), "witness_row": row,
            "passed": bool(maximum_displacement_mm is None or distance_mm[row] <= maximum_displacement_mm),
        }
    return {"passed": all(c["passed"] for c in checks.values()) and all(c["passed"] for c in displacement.values()),
            "maximum_permitted_displacement_mm": maximum_displacement_mm, "displacement": displacement,
            "native_rows": n, "permitted_edited_rows": sorted(edited), "arrays": checks,
            "scope": "Exact native rest-baseline and unchanged-row preparation only; no geometry, mass, contact, motion or physical admission."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--edited-rows", required=True, help="comma-separated native row indices")
    p.add_argument("--maximum-displacement-mm", type=float, help="declared local edit budget; catches unit errors before geometry audits")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    report = {"passed": False}
    try:
        baseline = read_baseline(args.baseline, args.expected_sha256)
        candidate_bytes = args.candidate.read_bytes()
        with np.load(io.BytesIO(candidate_bytes), allow_pickle=False) as candidate:
            report = check_arrays(baseline, candidate, [int(v) for v in args.edited_rows.split(",")], args.maximum_displacement_mm)
        report.update(baseline_header_sha256=args.expected_sha256, candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(), tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except (BaselineError, ValueError, OSError) as exc:
        report["error"] = str(exc)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
