#!/usr/bin/env python3
"""Select source-bound skin/fat faces from matched accepted FEM endpoint captures.

This is an offline compiler-input selector. It does not estimate solver error or
claim that a selected face's discrepancy disappears after refinement. It ranks
absolute volume-weighted Green-strain and free-node force differences between
one matched full and candidate endpoint, then emits source tet representatives.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

DEFAULT_OBJECT = "BodyParts3D whole-body skin-fat bonded shell"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_fields(line: str) -> tuple[str, dict[str, str]]:
    items = line.rstrip("\n").split(",")
    record = items[0]
    fields: dict[str, str] = {}
    for item in items[1:]:
        key, sep, value = item.partition("=")
        if not sep or key in fields:
            raise ValueError(f"malformed or duplicate field in {record}: {item!r}")
        fields[key] = value
    return record, fields


def number(fields: dict[str, str], key: str) -> float:
    value = float(fields[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def integer(fields: dict[str, str], key: str) -> int:
    return int(fields[key], 10)


def endpoint(log: Path, object_name: str):
    process_path = log.parent / "process.json"
    if not process_path.is_file():
        raise ValueError(f"missing process receipt: {process_path}")
    process = json.loads(process_path.read_text())
    if process.get("exit_code") != 0:
        raise ValueError(f"endpoint process did not succeed: {process_path}")
    invocation_path = log.parent / "invocation.json"
    if not invocation_path.is_file():
        raise ValueError(f"missing invocation provenance: {invocation_path}")
    invocation = json.loads(invocation_path.read_text())
    begin = None
    nodes: dict[int, dict] = {}
    tets: dict[int, dict] = {}
    for line in log.read_text(errors="strict").splitlines():
        if line.startswith("resting_FEM_endpoint_capture=begin "):
            if begin is not None:
                raise ValueError(f"multiple endpoint captures in {log}")
            _, tail = line.split(" ", 1)
            begin = dict(item.split("=", 1) for item in tail.split(" "))
        elif line.startswith("resting_FEM_node,"):
            _, f = parse_fields(line)
            if f.get("object") != object_name:
                continue
            local = integer(f, "local")
            if local in nodes:
                raise ValueError(f"duplicate local FEM node {local} in {log}")
            nodes[local] = {
                "global": integer(f, "global"),
                "rest": tuple(number(f, f"rest_{a}") for a in "xyz"),
                "accepted": tuple(number(f, f"accepted_{a}") for a in "xyz"),
                "mass": number(f, "mass_kg"),
                "initial_mass": number(f, "initial_mass_kg"),
                "inverse_mass": number(f, "inverse_mass"),
                "constraint": integer(f, "constraint_tag"),
                "force": tuple(number(f, f"free_internal_force_{a}_N") for a in "xyz"),
            }
        elif line.startswith("resting_FEM_tet,"):
            _, f = parse_fields(line)
            if f.get("object") != object_name:
                continue
            local = integer(f, "local")
            if local in tets:
                raise ValueError(f"duplicate local FEM tet {local} in {log}")
            tets[local] = {
                "global": integer(f, "global"),
                "nodes_global": tuple(integer(f, f"n{i}") for i in range(4)),
                "j": number(f, "J"),
                "donor": integer(f, "anatomical_region_id"),
            }
    if begin is None or not nodes or not tets:
        raise ValueError(f"incomplete endpoint capture for {object_name!r} in {log}")
    if begin.get("version") != "1" or begin.get("steps") != "1":
        raise ValueError(f"expected a single-step version-1 capture in {log}")
    if sorted(nodes) != list(range(len(nodes))) or sorted(tets) != list(range(len(tets))):
        raise ValueError(f"source local IDs are not contiguous in {log}")
    if len(nodes) % 3 != 0:
        raise ValueError(f"registered skin/fat node count is not three layers in {log}")
    width = len(nodes) // 3
    global_to_local = {v["global"]: local for local, v in nodes.items()}
    if len(global_to_local) != len(nodes):
        raise ValueError(f"duplicate global node IDs in {log}")
    for tet in tets.values():
        try:
            tet["nodes"] = tuple(global_to_local[g] for g in tet.pop("nodes_global"))
        except KeyError as exc:
            raise ValueError(f"tet references a node outside the captured object in {log}") from exc
        if tet["donor"] < 0 or tet["donor"] == 0xFFFFFFFF:
            raise ValueError(f"tet has no original donor label in {log}")
    if (int(begin["nodes"]) < len(nodes) or
            int(begin["tetrahedra"]) < len(tets) or
            int(begin.get("source_region_labeled_nodes", "-1")) != len(nodes) or
            int(begin.get("source_region_labeled_tetrahedra", "-1")) != len(tets)):
        raise ValueError(f"object capture does not exactly match the labeled source region in {log}")
    return {"log": log, "process": process, "invocation": invocation,
            "begin": begin, "nodes": nodes, "tets": tets, "width": width}


def determinant(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def inverse3(a):
    d = determinant(a)
    scale = max(abs(x) for row in a for x in row)
    if not math.isfinite(d) or scale == 0.0 or abs(d) <= 1e-14 * scale**3:
        raise ValueError("degenerate source tetrahedron in strain calculation")
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rows = [r for r in range(3) if r != j]
            cols = [c for c in range(3) if c != i]
            minor = a[rows[0]][cols[0]] * a[rows[1]][cols[1]] - a[rows[0]][cols[1]] * a[rows[1]][cols[0]]
            result[i][j] = ((-1.0) ** (i + j)) * minor / d
    return result


def columns_matrix(points):
    return [[points[c + 1][r] - points[0][r] for c in range(3)] for r in range(3)]


def green_strain(coords, node_records, local_nodes):
    rest_points = [node_records[n]["rest"] for n in local_nodes]
    current_points = [coords[n] for n in local_nodes]
    dm = columns_matrix(rest_points)
    ds = columns_matrix(current_points)
    inv = inverse3(dm)
    f = [[sum(ds[r][k] * inv[k][c] for k in range(3)) for c in range(3)]
         for r in range(3)]
    e = [[0.5 * (sum(f[k][r] * f[k][c] for k in range(3)) - (1.0 if r == c else 0.0))
          for c in range(3)] for r in range(3)]
    return (e[0][0], e[1][1], e[2][2], e[0][1], e[0][2], e[1][2]), abs(determinant(dm)) / 6.0


def norm_strain_delta(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 +
                     2.0*((a[3]-b[3])**2 + (a[4]-b[4])**2 + (a[5]-b[5])**2))


def compatible(full, candidate):
    for key in ("version", "steps", "dt_s", "runtime_dt_s", "timestep_multiplier",
                "timestep_divisor", "endpoint_time_s", "completed_microsteps",
                "source_region_total_nodes", "source_region_total_tetrahedra"):
        if full["begin"].get(key) != candidate["begin"].get(key):
            raise ValueError(f"matched endpoints differ in {key}")
    if len(full["nodes"]) != len(candidate["nodes"]) or len(full["tets"]) != len(candidate["tets"]):
        raise ValueError("matched endpoints have different object topology sizes")
    if (full["begin"].get("runtime_coordinate_projection") != "disabled" or
            candidate["begin"].get("runtime_coordinate_projection") != "enabled"):
        raise ValueError("expected full-coordinate control and hybrid-coordinate candidate endpoint captures")
    for local in full["nodes"]:
        a, b = full["nodes"][local], candidate["nodes"][local]
        if (a["global"] != b["global"] or a["rest"] != b["rest"] or
                a["mass"] != b["mass"] or a["initial_mass"] != b["initial_mass"] or
                a["inverse_mass"] != b["inverse_mass"] or
                a["constraint"] != b["constraint"]):
            raise ValueError(
                f"matched endpoints changed source node identity, geometry, mass, inverse mass, or constraint at {local}")
    for local in full["tets"]:
        a, b = full["tets"][local], candidate["tets"][local]
        if a["global"] != b["global"] or a["nodes"] != b["nodes"] or a["donor"] != b["donor"]:
            raise ValueError(f"matched endpoints changed source tet identity/connectivity/donor at {local}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--pose-label", required=True,
                        help="human-readable pose identifier; not a qualification claim")
    parser.add_argument("--pose-marker", default="",
                        help="optional exact line substring required in both logs")
    parser.add_argument("--target-fraction", type=float, default=0.95)
    parser.add_argument("--patch-count", type=int, default=60)
    parser.add_argument("--max-faces", type=int, default=64)
    parser.add_argument("--optic-regions", type=int, default=20)
    args = parser.parse_args()
    if not (0.0 < args.target_fraction <= 1.0 and 1 <= args.patch_count <= 170 and
            36 <= args.max_faces <= 256 and 0 <= args.optic_regions <= 170):
        raise SystemExit("invalid target fraction or bounded region settings")
    full = endpoint(args.full_log, args.object)
    candidate = endpoint(args.candidate_log, args.object)
    compatible(full, candidate)
    full_environment = full["invocation"].get("simulation_environment", {})
    candidate_environment = candidate["invocation"].get("simulation_environment", {})
    environment_differences = [
        {"name": key, "full": full_environment.get(key),
         "candidate": candidate_environment.get(key)}
        for key in sorted(set(full_environment) | set(candidate_environment))
        if full_environment.get(key) != candidate_environment.get(key)]
    if args.pose_marker:
        for log in (args.full_log, args.candidate_log):
            if args.pose_marker not in log.read_text(errors="strict"):
                raise SystemExit(f"required pose marker absent from {log}")
    width = full["width"]
    face_data = {}
    tet_strain = {}
    full_coords = {i: n["accepted"] for i, n in full["nodes"].items()}
    candidate_coords = {i: n["accepted"] for i, n in candidate["nodes"].items()}
    for tet_id, tet in full["tets"].items():
        vertices = tuple(sorted({node % width for node in tet["nodes"]}))
        if len(vertices) != 3:
            raise ValueError(f"skin/fat prism tet {tet_id} does not map to three base vertices")
        row = face_data.setdefault(vertices, {"tets": [], "donor": tet["donor"]})
        if row["donor"] != tet["donor"]:
            raise ValueError(f"source face {vertices} maps to more than one registered donor")
        row["tets"].append(tet_id)
        e0, vol0 = green_strain(full_coords, full["nodes"], tet["nodes"])
        e1, vol1 = green_strain(candidate_coords, full["nodes"], tet["nodes"])
        if vol0 <= 0 or vol0 != vol1:
            raise ValueError(f"source tet {tet_id} has invalid or changed reference volume")
        tet_strain[tet_id] = vol0 * norm_strain_delta(e0, e1)
    for face, row in face_data.items():
        if len(row["tets"]) != 6:
            raise ValueError(f"source face {face} does not own exactly six skin/fat prism tets")
        row["tets"].sort()
        row["strain"] = math.fsum(tet_strain[t] for t in row["tets"])
        row["force_nodes"] = {vertex + layer * width for vertex in face for layer in range(3)}
    force_error = {}
    for local, a in full["nodes"].items():
        b = candidate["nodes"][local]
        if a["constraint"] == 0 and a["inverse_mass"] > 0.0:
            force_error[local] = math.sqrt(sum((a["force"][i]-b["force"][i])**2 for i in range(3)))
    total_strain = math.fsum(row["strain"] for row in face_data.values())
    total_force = math.fsum(force_error.values())
    if not math.isfinite(total_strain + total_force) or (total_strain <= 0 and total_force <= 0):
        raise SystemExit("matched endpoint discrepancy is zero or non-finite; no targets can be ranked")
    selected = set()
    selected_sequence = []
    covered_nodes = set()
    selected_strain = 0.0
    selected_force = 0.0
    ordered = sorted(face_data)
    strain_fraction = lambda: selected_strain / total_strain if total_strain > 0 else 1.0
    force_fraction = lambda: selected_force / total_force if total_force > 0 else 1.0
    while (strain_fraction() < args.target_fraction or
           force_fraction() < args.target_fraction):
        best_face = None
        best_score = -1.0
        best_delta = None
        for face in ordered:
            if face in selected:
                continue
            row = face_data[face]
            ds = row["strain"]
            new_nodes = row["force_nodes"] - covered_nodes
            df = math.fsum(force_error[n] for n in new_nodes if n in force_error)
            score = (ds / total_strain if total_strain > 0 else 0.0) + \
                    (df / total_force if total_force > 0 else 0.0)
            if score > best_score or (score == best_score and (best_face is None or face < best_face)):
                best_face, best_score, best_delta = face, score, (ds, df, new_nodes)
        if best_face is None or best_score <= 0 or best_delta is None:
            raise SystemExit("greedy source-face selection cannot reach both requested discrepancy fractions")
        selected.add(best_face)
        ds, df, new_nodes = best_delta
        selected_sequence.append((best_face, ds, df))
        selected_strain += ds
        selected_force += df
        covered_nodes.update(new_nodes)
        if len(selected) > 4096:
            raise SystemExit("candidate selection exceeds the compiler's 4096-tet bounded input")
    selected_rows = []
    for face, selected_ds, selected_df in selected_sequence:
        row = face_data[face]
        selected_rows.append({"face_base_vertices": face,
                              "representative_source_tet": row["tets"][0],
                              "original_donor": row["donor"],
                              "prism_tet_count": len(row["tets"]),
                              "volume_weighted_strain_difference": selected_ds,
                              "free_force_difference_newly_covered_N": selected_df})
    donors = {row["donor"] for row in face_data.values()}
    target_donors = {face_data[face]["donor"] for face in selected}
    base_regions = sum((sum(1 for row in face_data.values() if row["donor"] == donor) - 1) // args.max_faces + 1
                       for donor in donors)
    if args.patch_count < len(target_donors) or args.patch_count > len(selected):
        raise SystemExit("patch count cannot provide nonempty donor-local groups for this selected set")
    skin_regions_upper = base_regions + args.patch_count
    combined_regions_upper = skin_regions_upper + args.optic_regions
    if combined_regions_upper > 170:
        raise SystemExit(f"source refinement exceeds combined basis region budget: {combined_regions_upper}/170")
    output = {
        "schema": "numi.human.skin_fat.greedy_refinement_targets.v1",
        "pose_label": args.pose_label,
        "pose_marker_required": args.pose_marker or None,
        "matched_endpoint_capture": {
            "steps": 1, "dt_s": full["begin"]["dt_s"],
            "endpoint_time_s": full["begin"]["endpoint_time_s"],
            "completed_microsteps": int(full["begin"]["completed_microsteps"]),
            "object": args.object, "node_count": len(full["nodes"]),
            "tet_count": len(full["tets"]),
            "full_log_sha256": sha256(args.full_log),
            "candidate_log_sha256": sha256(args.candidate_log),
            "full_process": full["process"], "candidate_process": candidate["process"],
            "full_binary_sha256": full["invocation"].get("binary_sha256"),
            "candidate_binary_sha256": candidate["invocation"].get("binary_sha256"),
            "full_source_head": full["invocation"].get("source_head"),
            "candidate_source_head": candidate["invocation"].get("source_head"),
            "simulation_environment_differences": environment_differences},
        "metric": {
            "strain": "sum over source prism tets of rest-volume times Green-Lagrange strain-difference Frobenius norm",
            "force": "sum over free tag-0 shell nodes of Euclidean free-internal-force difference in N",
            "target_fraction_each": args.target_fraction,
            "total_strain_difference_volume_weighted": total_strain,
            "selected_strain_fraction": strain_fraction(),
            "total_free_force_difference_N": total_force,
            "selected_free_force_fraction": force_fraction(),
            "selection_is_attribution_not_predicted_solver_error_reduction": True},
        "capacity": {"face_count": len(face_data), "selected_face_count": len(selected),
                     "distinct_target_donors": len(target_donors),
                     "base_skin_regions_upper_bound": base_regions,
                     "refinement_patches": args.patch_count,
                     "skin_regions_upper_bound": skin_regions_upper,
                     "optic_regions_reserved": args.optic_regions,
                     "combined_regions_upper_bound": combined_regions_upper,
                     "maximum_faces_per_base_region": args.max_faces},
        "compiler_environment": {
            "NM_FEM_HUMAN_PARTITIONED_SPLIT_SKIN_TETS": ",".join(
                str(row["representative_source_tet"]) for row in selected_rows),
            "NM_FEM_HUMAN_PARTITIONED_SPLIT_SKIN_PATCHES": str(args.patch_count),
            "NM_FEM_HUMAN_PARTITIONED_MAX_SKIN_FACES_PER_REGION": str(args.max_faces)},
        "selected_faces": selected_rows,
        "limitations": [
            "Only the named skin/fat source object is ranked; optic, vessel, contact, attachment, and other FEM rows remain under existing owners.",
            "The score measures disagreement in this endpoint pair only and does not predict accuracy or speed after refinement.",
            "The supplied full and candidate invocations may differ in solver controls; all differences are retained above and prevent interpreting this as a strict solver A/B.",
            "Runtime quality gates and full-coordinate certification remain mandatory."
        ]
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "selected_faces": len(selected),
                      "strain_fraction": strain_fraction(),
                      "force_fraction": force_fraction(),
                      "skin_regions_upper_bound": skin_regions_upper,
                      "combined_regions_upper_bound": combined_regions_upper}, sort_keys=True))


if __name__ == "__main__":
    main()
