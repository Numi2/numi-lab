#!/usr/bin/env python3
"""Reproduce Hill-q and plastic-state summaries from the retained FD rollback state.

This audits accepted history only. It does not contain the Newton-3 trial state,
the paired FD-projection states, or an independent solver implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_STATE = Path(
    "/Users/home/cardboard-followup-20261013/solver/operator-fd-001/output/"
    "material_state_000020_bent.json"
)
DEFAULT_MANIFEST = DEFAULT_STATE.with_name("manifest.json")
DEFAULT_RESULT = DEFAULT_STATE.with_name("result.json")
DEFAULT_OBSERVATIONS = DEFAULT_STATE.with_name("observations.csv")
DEFAULT_MESH = DEFAULT_STATE.with_name("mesh.json")
DEFAULT_OUTCOME = Path(
    "/Users/home/cardboard-followup-20261013/operator-fd/outcome-comparison.json"
)
DEFAULT_OUTPUT = Path(
    "/Users/home/cardboard-followup-20261013/operator-fd/"
    "accepted-history-yield-audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_parameter(path: Path, name: str) -> float:
    source = path.read_text()
    pattern = rf"^\s*parameter\s+{re.escape(name)}\s*:\s*[^=]+?=\s*([-+0-9.eE]+)\s*;"
    match = re.search(pattern, source, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"parameter {name!r} not found in {path}")
    return float(match.group(1))


def material_parameters(path: Path) -> dict[str, float]:
    names = ("hill_F", "hill_G", "hill_H", "hill_L", "hill_M", "hill_N", "q_floor")
    return {name: read_parameter(path, name) for name in names}


def hill_q(state: list[float], p: dict[str, float]) -> float:
    # State layout is read from the retained material-state JSON and asserted below.
    s11, s22, s33 = state[6], state[7], state[8]
    s23, s13, s12 = state[9], state[10], state[11]
    q_squared = (
        p["hill_F"] * (s22 - s33) ** 2
        + p["hill_G"] * (s33 - s11) ** 2
        + p["hill_H"] * (s11 - s22) ** 2
        + 2.0 * p["hill_L"] * s23**2
        + 2.0 * p["hill_M"] * s13**2
        + 2.0 * p["hill_N"] * s12**2
        + p["q_floor"] ** 2
    )
    return math.sqrt(max(q_squared, 0.0))


def edge_statistics(mesh: dict[str, Any]) -> dict[str, float | int]:
    nodes = mesh["nodes_m"]
    element_min_edges = []
    for tetrahedron in mesh["tetrahedra"]:
        edges = [
            math.dist(nodes[tetrahedron[i]], nodes[tetrahedron[j]])
            for i in range(4)
            for j in range(i)
        ]
        element_min_edges.append(min(edges))
    ordered = sorted(element_min_edges)
    median = ordered[len(ordered) // 2]
    return {
        "tetrahedron_count": len(element_min_edges),
        "minimum_edge_m": min(ordered),
        "median_per_tetrahedron_minimum_edge_m": median,
        "maximum_per_tetrahedron_minimum_edge_m": max(ordered),
        "tetrahedra_with_minimum_edge_below_0_1mm": sum(x < 1.0e-4 for x in ordered),
    }


def accepted_row_check(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(newline="")))
    env0 = [row for row in rows if int(row["environment"]) == 0]
    row18 = next(row for row in env0 if int(row["step"]) == 18)
    row19 = next(row for row in env0 if int(row["step"]) == 19)
    if int(row18["step_accepted"]) != 1 or int(row18["status_code"]) != 0:
        raise ValueError("expected step 18 to be the last accepted environment-0 row")
    if int(row19["step_accepted"]) != 0 or int(row19["status_code"]) != 10:
        raise ValueError("expected step 19 to be the rejected environment-0 row")
    if int(result.get("accepted_steps", -1)) != 19 or result.get("status") != "failed":
        raise ValueError("result.json does not match the expected failed 19-accepted-step run")
    return {
        "last_accepted_row": {key: row18[key] for key in ("step", "time_s", "status_code", "step_accepted")},
        "rejected_row": {key: row19[key] for key in ("step", "time_s", "status_code", "step_accepted")},
        "result_accepted_steps": result["accepted_steps"],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.state.resolve()
    manifest_path = args.manifest.resolve()
    result_path = args.result.resolve()
    observations_path = args.observations.resolve()
    mesh_path = args.mesh.resolve()
    manifest = read_json(manifest_path)
    result = read_json(result_path)
    state = read_json(state_path)
    mesh = read_json(mesh_path)

    if state.get("status_code") != 10:
        raise ValueError("retained state does not carry expected rollback status 10")
    if "rolled-back" not in state.get("note", ""):
        raise ValueError("state JSON does not explicitly describe the failed-step rollback")
    if state.get("environment") != 0:
        raise ValueError("expected the bent specimen in environment 0")

    expected_layout = [
        "ep11", "ep22", "ep33", "ep23", "ep13", "ep12",
        "sbar11", "sbar22", "sbar33", "sbar23", "sbar13", "sbar12",
        "accumulated_multiplier",
    ]
    materials = state["materials"]
    material_paths = {
        0: Path(manifest["materials"]["liner_path"]),
        1: Path(manifest["materials"]["medium_path"]),
    }
    expected_asset_hashes = {
        0: manifest["materials"]["liner_sha256"],
        1: manifest["materials"]["medium_sha256"],
    }
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tet in state["tetrahedra"]:
        if tet["material_index"] in (0, 1):
            groups[tet["material_index"]].append(tet)

    material_summaries = []
    material_sha = {}
    for index in (0, 1):
        asset = material_paths[index].resolve()
        actual_hash = sha256(asset)
        if actual_hash != expected_asset_hashes[index]:
            raise ValueError(f"material asset hash differs from run manifest: {asset}")
        material_sha[str(index)] = actual_hash
        material_record = next(item for item in materials if item["index"] == index)
        if material_record["state_names"] != expected_layout:
            raise ValueError(f"unexpected state layout for material {index}")
        params = material_parameters(asset)
        samples = []
        for tet in groups[index]:
            values = [float(value) for value in tet["state"]]
            if len(values) != len(expected_layout):
                raise ValueError(f"unexpected state length in tetrahedron {tet['index']}")
            samples.append({
                "tetrahedron_index": int(tet["index"]),
                "q": hill_q(values, params),
                "lambda": values[12],
                "max_abs_Ep": max(abs(value) for value in values[:6]),
            })
        near = sorted(samples, key=lambda sample: abs(sample["q"] - 1.0))
        max_q = max(samples, key=lambda sample: sample["q"])
        max_lambda = max(samples, key=lambda sample: sample["lambda"])
        max_ep = max(samples, key=lambda sample: sample["max_abs_Ep"])
        material_summaries.append({
            "material_index": index,
            "material_name": material_record["name"],
            "tetrahedron_count": len(samples),
            "hill_parameters_from_asset": params,
            "q_min": min(sample["q"] for sample in samples),
            "q_max": max_q["q"],
            "q_max_tetrahedron": max_q["tetrahedron_index"],
            "q_gt_0_999_count": sum(sample["q"] > 0.999 for sample in samples),
            "abs_q_minus_1_le_1e_6_count": sum(abs(sample["q"] - 1.0) <= 1.0e-6 for sample in samples),
            "lambda_gt_1e_8_count": sum(sample["lambda"] > 1.0e-8 for sample in samples),
            "lambda_max": max_lambda["lambda"],
            "lambda_max_tetrahedron": max_lambda["tetrahedron_index"],
            "max_abs_Ep": max_ep["max_abs_Ep"],
            "max_abs_Ep_tetrahedron": max_ep["tetrahedron_index"],
            "eight_tetrahedra_nearest_q_1": near[:8],
            "diagnostic_cutoffs": {
                "q_gt": 0.999,
                "abs_q_minus_1_lte": 1.0e-6,
                "lambda_gt": 1.0e-8,
                "meaning": "descriptive thresholds only; not solver active-set flags or constitutive acceptance criteria",
            },
        })

    fd_context = None
    if args.outcome.exists():
        outcome = read_json(args.outcome)
        trace = outcome["fd_traces"][0]
        pair = trace["environments"][0]["epsilon_pairs"][0]
        edge_info = edge_statistics(mesh)
        max_perturbation = pair["maximum_position_perturbation"]
        fd_context = {
            "root": trace["root"],
            "newton_iteration": trace["newton_iteration"],
            "epsilon": pair["epsilon"],
            "maximum_position_perturbation_m": max_perturbation,
            "mesh_edge_statistics": edge_info,
            "max_position_perturbation_divided_by_minimum_edge": max_perturbation / edge_info["minimum_edge_m"],
            "max_position_perturbation_divided_by_median_element_minimum_edge": max_perturbation / edge_info["median_per_tetrahedron_minimum_edge_m"],
            "interpretation_limit": "coordinate-displacement to edge ratios are not deformation-gradient increments; nodal displacement differences are required to estimate local delta-F",
        }

    inputs = {
        "state_json": {"path": str(state_path), "sha256": sha256(state_path)},
        "run_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "result_json": {"path": str(result_path), "sha256": sha256(result_path)},
        "observations_csv": {"path": str(observations_path), "sha256": sha256(observations_path)},
        "mesh_json": {"path": str(mesh_path), "sha256": sha256(mesh_path)},
        "material_assets": material_sha,
        "manifest_compiled_world_fingerprint": manifest.get("compiled_world_fingerprint"),
        "result_compiled_world_fingerprint": result.get("compiled_world_fingerprint"),
    }
    return {
        "schema": "numi.cardboard.accepted-history-yield-audit.v1",
        "audited_at_utc": "2026-10-13",
        "method": "CPU recomputation of Hill equivalent stress q from retained accepted-state sbar components and the F..N, q_floor values parsed from hash-verified .nmatter source assets",
        "state_classification": {
            "source_status_code": state["status_code"],
            "source_note": state["note"],
            "accepted_row_check": accepted_row_check(observations_path, result),
            "label": "rolled-back accepted history after the rejected physical row; corresponds to pre-failure accepted history (last accepted CSV step 18)",
        },
        "materials": material_summaries,
        "fd_context": fd_context,
        "input_bindings": inputs,
        "conclusion_boundary": [
            "These q/Ep/lambda summaries describe the saved accepted history only.",
            "They do not reveal the actual Newton-3 candidate state or either +/- finite-difference projected state.",
            "Near-yield accepted history makes yield-transition crossing a testable hypothesis, but the audit does not prove that the compared candidate/perturbations cross the transition or explain the global operator mismatch.",
            "No physical cardboard qualification is implied.",
        ],
        "audit_script_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = audit(args)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "input_bindings": value["input_bindings"]}, indent=2))


if __name__ == "__main__":
    main()
