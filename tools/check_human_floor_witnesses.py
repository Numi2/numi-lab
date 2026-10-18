#!/usr/bin/env python3
"""Check retained pre-dynamics floor witnesses; sampled viewer audits alone are insufficient."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

def check(run, boundary_rows, tolerance, expected_steps, expected_regions=32):
    run = Path(run)
    worst = None
    failures = []
    seen = {}
    first_step_source_geometry = {}
    below = 0
    path = run / "resting-support-selected-skin-witness.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            step = int(row["accepted_step"])
            region = int(row["region"])
            regions = seen.setdefault(step, set())
            if region in regions:
                failures.append(f"duplicate step/region {step}/{region}")
            regions.add(region)
            if step == 1:
                first_step_source_geometry[region] = int(row["source_geometry_index"])
            if any(int(row[k]) != 0 for k in
                   ("region_error", "global_error", "nonfinite_jacobian_count")):
                failures.append(f"invalid witness step/region {step}/{region}")
            if region in boundary_rows:
                continue
            xyz = [float(row["selected_point_world_" + axis + "_m"]) for axis in "xyz"]
            if not all(math.isfinite(v) for v in xyz):
                failures.append(f"nonfinite witness step/region {step}/{region}")
                continue
            if worst is None or xyz[2] < worst["gap_m"]:
                worst = {"accepted_step_slot": step,
                         "pose_step": int(row.get("pre_dynamics_step_index", step - 1)),
                         "pose_time_s": float(row.get("pre_dynamics_time_s", row["accepted_time_s"])),
                         "region": region, "gap_m": xyz[2],
                         "source_geometry_index": int(row["source_geometry_index"]),
                         "skin_vertex_local": int(row["selected_skin_vertex_local"]),
                         "vertex_map_index": int(row["selected_vertex_map_index"])}
            below += xyz[2] < -tolerance
    if not seen or worst is None:
        failures.append("no floor witnesses")
    else:
        all_regions = set.union(*seen.values())
        if all_regions != set(range(expected_regions)):
            failures.append("region IDs do not match admitted region count")
        if not boundary_rows.issubset(all_regions):
            failures.append("declared boundary-only row missing")
        if not all_regions - boundary_rows:
            failures.append("no floor regions remain")
        if any(regions != all_regions for regions in seen.values()):
            failures.append("incomplete region coverage in retained steps")
        if set(seen) != set(range(1, max(seen) + 1)):
            failures.append("retained step sequence is incomplete")
    if expected_regions == 38:
        log_path = run / "launcher.log"
        if not log_path.is_file():
            failures.append("38-region foot manifold run lacks launcher.log allocation receipt")
        else:
            log_text = log_path.read_text(errors="replace")
            allocation_rows = re.findall(
                r"resting_support_geometry=full_registered_skin\s+vertices=(\d+)\s+regions=(\d+).*?region_sizes=\[([0-9,]+)\]",
                log_text,
            )
            if len(allocation_rows) != 1:
                failures.append("expected exactly one full-skin region allocation receipt")
            else:
                vertex_count, region_count, encoded_sizes = allocation_rows[0]
                sizes = [int(value) for value in encoded_sizes.split(",")]
                if int(region_count) != 38 or len(sizes) != 38:
                    failures.append("allocation receipt does not describe exactly 38 runtime regions")
                if sum(sizes) != int(vertex_count):
                    failures.append("per-region allocation counts do not cover every registered skin vertex exactly once")
                for group in ((20, 32, 33, 34), (30, 35, 36, 37)):
                    if len(sizes) == 38 and any(sizes[index] <= 0 for index in group):
                        failures.append(f"foot manifold source/derived regions are empty: {group}")
        for parent, children in ((20, (32, 33, 34)), (30, (35, 36, 37))):
            source_id = first_step_source_geometry.get(parent)
            if source_id is None or any(first_step_source_geometry.get(child) != source_id
                                        for child in children):
                failures.append(f"four contact regions do not preserve source identity for parent row {parent}")
    if expected_steps is not None and len(seen) != expected_steps:
        failures.append(f"expected {expected_steps} accepted steps; got {len(seen)}")
    if below:
        failures.append(f"{below} floor witnesses exceed penetration tolerance")
    surface_count = 0
    with (run / "resting-surface-audit.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            surface_count += 1
            if not math.isfinite(float(row["min_skin_bed_gap_m"])):
                failures.append("nonfinite surface gap")
            if float(row["min_skin_bed_gap_m"]) < -tolerance:
                failures.append("sampled full surface exceeds penetration tolerance")
            if any(int(row[k]) for k in ("nonfinite_skin_vertices",
                                        "mesh_zero_area_triangles",
                                        "mesh_nonfinite_area_triangles",
                                        "functional_geometry_status")):
                failures.append("invalid sampled anatomical surface")
    if not surface_count:
        failures.append("missing full-surface audit")
    return {"run": str(run.resolve()), "passed": not failures,
            "scope": "retained pre-dynamics floor-witness and sampled surface consistency only; final endpoint and full physics qualification require separate evidence",
            "boundary_only_rows": sorted(boundary_rows),
            "penetration_tolerance_m": tolerance, "accepted_steps_checked": len(seen),
            "floor_witnesses_below_tolerance": below, "worst_floor_witness": worst,
            "failures": sorted(set(failures))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--boundary-only-rows", required=True,
                        help="Explicit comma-separated non-floor region IDs from scene admission")
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.001)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-regions", type=int, choices=(32, 38), default=32,
                        help="Exact admitted region count: 32 authored rows or 38 rows with four-point foot manifolds")
    args = parser.parse_args()
    if not math.isfinite(args.penetration_tolerance_m) or args.penetration_tolerance_m < 0:
        parser.error("penetration tolerance must be finite and nonnegative")
    rows = {int(v) for v in args.boundary_only_rows.split(",") if v}
    result = check(args.run, rows, args.penetration_tolerance_m, args.expected_steps, args.expected_regions)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
