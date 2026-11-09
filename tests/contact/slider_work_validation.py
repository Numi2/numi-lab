"""Independently check the retained per-node FEM work reconstruction."""
import math
import sys


def validate_fem_work_rows(step, fixture):
    areas = fixture["reference_area_weights_m2"]
    h = fixture["h"]
    width = fixture["activation_width_m"]
    coefficient = fixture["contact_coefficient"]
    active = {i for i, area in enumerate(areas) if area > 0}
    rows = step["fem_work_nodes"]
    if (len(rows) != len(active) or any(len(row) != 5 for row in rows) or
            {row[0] for row in rows} != active or
            not all(all(math.isfinite(value) for value in row) for row in rows)):
        raise ValueError("missing or invalid per-node FEM contact work")
    if not all(math.isfinite(x) and x > 0 for x in (h, width, coefficient)):
        raise ValueError("invalid contact law parameters")
    for node, _, recorded, _, gap in rows:
        if gap <= 0:
            raise ValueError("nonpositive contact midpoint gap")
        pressure = 0.0
        if gap < width:
            squared, thickness = gap * gap, width * width
            shift = squared - thickness
            pressure = 2 * gap * coefficient * (2 * shift * math.log(squared / thickness)
                                                + shift * shift / squared)
        expected = h * areas[int(node)] * pressure
        if not math.isclose(recorded, expected, rel_tol=1e-12, abs_tol=1e-28):
            raise ValueError("recorded nodal impulse disagrees with independently evaluated law")
    observed = sum(row[1] * row[3] for row in rows)
    reconstructed = sum(row[2] * row[3] for row in rows)
    magnitude = sum(abs(row[1] * row[3]) + abs(row[2] * row[3]) for row in rows)
    roundoff = 64 * sys.float_info.epsilon * magnitude + 1e-30
    if (abs(observed - step["fem_contact_work"]) > roundoff or
            abs(reconstructed - step["fem_law_work"]) > roundoff):
        raise ValueError("per-node FEM work does not reproduce published totals")
    ratio = max(abs((row[1] - row[2]) * row[3]) /
                (max(1e-12, 1e-5 * abs(row[2])) * abs(row[3]) + 1e-24)
                for row in rows)
    if not math.isclose(ratio, step["fem_work_reconstruction_gate_ratio"],
                        rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("FEM work gate does not reproduce from per-node data")
    return ratio
