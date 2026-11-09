"""Validated compressed reference input and predeclared refinement metrics.

No temporal or spatial qualification is implied by an input passing validation.
"""
import gzip
import json
import math
from pathlib import Path


def read_reference(path, divisions, h):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        x = json.load(stream)
    if (x.get("schema") not in ("numi.independent-fem-rk4.v1", "numi.independent-fem-rk4-compact.v1") or
        x.get("mesh_divisions") != divisions or x.get("h") != h or x.get("duration") != .01 or
        x.get("nodes") != (divisions + 1) ** 3 or x.get("tetrahedra") != 6 * divisions ** 3):
        raise ValueError("incorrect reference configuration")
    weights = x.pop("contact_area_weights_m2")
    if len(weights) != (divisions + 1) ** 3:
        raise ValueError("incorrect contact weight count")
    expected = [0.0] * (divisions + 1) ** 2
    share = .02 * .02 / (divisions * divisions * 6.)
    for y in range(divisions):
        for u in range(divisions):
            a = y * (divisions + 1) + u
            b, c, d = a + 1, a + divisions + 1, a + divisions + 2
            for index in (a, b, d, a, d, c):
                expected[index] += share
    tolerance = .0004 * 1e-12
    for i, value in enumerate(weights):
        target = expected[i] if i < len(expected) else 0.0
        if not math.isfinite(value) or value < 0 or abs(value - target) > tolerance:
            raise ValueError("invalid area weight at " + str(i))
    total = math.fsum(weights)
    if (not all(math.isfinite(x[k]) for k in ("area_m2", "mass_kg", "min_gap_m")) or
        abs(total - .0004) >= 1e-14 or abs(x["area_m2"] - .0004) >= 1e-14 or
        abs(x["mass_kg"] - .0088) >= 1e-8 or x["min_gap_m"] <= 1e-7):
        raise ValueError("invalid area, mass, or clearance")
    del weights
    trace, bins = x["trace"], x["force_bins_N"]
    if len(trace) != 5001 or len(bins) != 10:
        raise ValueError("incomplete reference trace")
    previous = -1.0
    for i, row in enumerate(trace):
        if (len(row) != 3 or not all(math.isfinite(v) for v in row) or
            abs(row[0] - i * .000002) > 1e-12 or row[1] < 0 or row[2] < previous - 1e-15):
            raise ValueError("invalid trace sample " + str(i))
        previous = row[2]
    if (not all(math.isfinite(v) and v >= 0 for v in bins) or
        not math.isfinite(x["peak_force_N"]) or x["peak_force_N"] <= 0 or
        not math.isfinite(x["impulse_Ns"]) or x["impulse_Ns"] <= 0 or
        not math.isfinite(x["onset_time_s"]) or not 0 <= x["onset_time_s"] <= .01 or
        x["peak_force_N"] + 1e-14 * x["peak_force_N"] < max(row[1] for row in trace) or
        abs(trace[-1][2] - x["impulse_Ns"]) >= 1e-12 or
        abs(sum(bins) * .001 - x["impulse_Ns"]) >= 1e-12):
        raise ValueError("invalid force or impulse")
    x["verified_area_m2"] = total
    return x


def compare(a, b):
    if len(a["trace"]) != len(b["trace"]) or any(abs(u[0] - v[0]) > 1e-12 for u, v in zip(a["trace"], b["trace"])):
        raise ValueError("inconsistent reference sample times")
    peak, impulse = b["peak_force_N"], b["impulse_Ns"]
    result = {
        "binned_force": max(abs(u - v) for u, v in zip(a["force_bins_N"], b["force_bins_N"])) / max(b["force_bins_N"]),
        "sampled_peak_force": abs(a["peak_force_N"] - peak) / peak,
        "total_impulse": abs(a["impulse_Ns"] - impulse) / impulse,
        "cumulative_impulse": max(abs(u[2] - v[2]) for u, v in zip(a["trace"], b["trace"])) / impulse,
        "sampled_force_trace": max(abs(u[1] - v[1]) for u, v in zip(a["trace"], b["trace"])) / peak,
    }
    result["passes_refinement"] = (result["binned_force"] <= .0025 and result["total_impulse"] <= .0025 and
                                   result["sampled_peak_force"] <= .01 and result["cumulative_impulse"] <= .01)
    return result
