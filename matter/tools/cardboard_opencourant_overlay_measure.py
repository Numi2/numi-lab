#!/usr/bin/env python3
"""Collect measured TYPE2/TYPE43 overlay fixture histories.

This reader is intentionally limited to the retained synthetic overlay runs.
It verifies native/converter hashes, reconstructs the nine interface gaps and
slips, reports per-element solid tractions and energy, and keeps every result
labelled as a solver instrument rather than cardboard qualification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from .cardboard_opencourant_bond_measure import _stencil_indices, _time_average
except ImportError:  # direct execution from matter/tools
    from cardboard_opencourant_bond_measure import _stencil_indices, _time_average


SCHEMA = "cardboard.opencourant.overlay-measurement.v2"
ACTIVE_OFF = 1.0
INACTIVE_OFF = 0.0
THTOCSV_SHA256 = "f8176302da59caba3e373504804beb77715137b5e9e9ef79eaf3231f5e1a1502"
THTOCSV_SOURCE_COMMIT = "1b6bb9200ee7dca899e61c18bd982c3fb1c72fd1"
GAUSS_2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


class MeasurementError(ValueError):
    """Raised when a retained overlay history cannot be verified/read."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MeasurementError(f"expected a JSON object in {path}")
    return value


def _float(value: str, where: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise MeasurementError(f"invalid float {value!r} in {where}") from exc
    if not math.isfinite(result):
        raise MeasurementError(f"nonfinite float in {where}")
    return result


def _nodes_from_deck(deck: str) -> dict[int, tuple[float, float, float]]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().upper() == "/NODE"]
    if len(starts) != 1:
        raise MeasurementError("starter must contain one /NODE block")
    nodes: dict[int, tuple[float, float, float]] = {}
    for line in lines[starts[0] + 1:]:
        if line.lstrip().startswith("/"):
            break
        fields = line.split()
        if not fields:
            continue
        try:
            node_id = int(fields[0])
            xyz = tuple(float(v) for v in fields[1:4])
        except (ValueError, IndexError) as exc:
            raise MeasurementError(f"cannot parse node record {line!r}") from exc
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise MeasurementError(f"invalid node coordinate record {line!r}")
        nodes[node_id] = xyz
    return nodes


def _bricks_from_deck(deck: str, part_id: int) -> dict[int, tuple[int, ...]]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines)
              if re.fullmatch(rf"\s*/BRICK/{part_id}\s*", line, re.I)]
    if len(starts) != 1:
        raise MeasurementError(f"starter must contain one /BRICK/{part_id} block")
    result: dict[int, tuple[int, ...]] = {}
    for line in lines[starts[0] + 1:]:
        if line.lstrip().startswith("/"):
            break
        fields = line.split()
        if not fields:
            continue
        try:
            element_id = int(fields[0])
            nodes = tuple(int(v) for v in fields[1:9])
        except (ValueError, IndexError) as exc:
            raise MeasurementError(f"cannot parse /BRICK/{part_id} record {line!r}") from exc
        if len(nodes) != 8:
            raise MeasurementError(f"/BRICK/{part_id} element {element_id} is not HEX8")
        result[element_id] = nodes
    return result


def _group_nodes_from_deck(deck: str, group_id: int) -> tuple[int, ...]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines)
              if re.fullmatch(rf"\s*/GRNOD/NODE/{group_id}\s*", line, re.I)]
    if len(starts) != 1:
        raise MeasurementError(f"starter must contain one /GRNOD/NODE/{group_id} group")
    nodes: list[int] = []
    for line in lines[starts[0] + 2:]:
        if line.lstrip().startswith("/"):
            break
        for field in line.split():
            try:
                nodes.append(int(field))
            except ValueError:
                continue
    if not nodes:
        raise MeasurementError(f"/GRNOD/NODE/{group_id} is empty")
    return tuple(nodes)


def _engine_solver_cards(run_dir: Path, engines: list[str]) -> list[dict[str, Any]]:
    result = []
    for index, name in enumerate(engines, start=1):
        path = run_dir / name
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^\s*(/IMPL/SOLV/\d+)\s*\n([^\n]+)", text)
        result.append({
            "stage": f"T{index:02d}",
            "engine_file": name,
            "engine_sha256": _sha256(path),
            "solver_card": None if match is None else match.group(1).strip(),
            "solver_parameters": None if match is None else match.group(2).strip(),
        })
    return result


def _area(nodes: dict[int, tuple[float, float, float]], ids: tuple[int, ...]) -> float:
    pts = [nodes[n] for n in ids]

    def tri(a: tuple[float, ...], b: tuple[float, ...], c: tuple[float, ...]) -> float:
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        cross = (u[1] * v[2] - u[2] * v[1],
                 u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0])
        return 0.5 * math.sqrt(sum(x * x for x in cross))

    return tri(pts[0], pts[1], pts[2]) + tri(pts[0], pts[2], pts[3])


def _quad_shape(xi: float, eta: float) -> tuple[float, ...]:
    return ((1.0 - xi) * (1.0 - eta) / 4.0,
            (1.0 + xi) * (1.0 - eta) / 4.0,
            (1.0 + xi) * (1.0 + eta) / 4.0,
            (1.0 - xi) * (1.0 + eta) / 4.0)


def _quad_area_jacobian(points: list[tuple[float, float, float]],
                        xi: float, eta: float) -> float:
    """Surface Jacobian for a bilinear quad in its current/birth geometry."""
    dxi = (-(1.0 - eta) / 4.0, (1.0 - eta) / 4.0,
           (1.0 + eta) / 4.0, -(1.0 + eta) / 4.0)
    deta = (-(1.0 - xi) / 4.0, -(1.0 + xi) / 4.0,
            (1.0 + xi) / 4.0, (1.0 - xi) / 4.0)
    tangent_xi = tuple(sum(dxi[i] * points[i][axis] for i in range(4))
                       for axis in range(3))
    tangent_eta = tuple(sum(deta[i] * points[i][axis] for i in range(4))
                        for axis in range(3))
    cross = (tangent_xi[1] * tangent_eta[2] - tangent_xi[2] * tangent_eta[1],
             tangent_xi[2] * tangent_eta[0] - tangent_xi[0] * tangent_eta[2],
             tangent_xi[0] * tangent_eta[1] - tangent_xi[1] * tangent_eta[0])
    jacobian = math.sqrt(sum(value * value for value in cross))
    if not math.isfinite(jacobian) or jacobian <= 0.0:
        raise MeasurementError("cohesive quad has a nonpositive surface Jacobian")
    return jacobian


def _integrate_quad_elastic(
        reference_points: list[tuple[float, float, float]],
        corner_separation_local: list[tuple[float, float, float]],
        kn_mpa_per_mm: float, kt_mpa_per_mm: float,
        tangent_x_global: tuple[float, float, float],
        tangent_y_global: tuple[float, float, float],
        normal_global: tuple[float, float, float]) -> dict[str, Any]:
    """Integrate a linear elastic cohesive quad and consistent nodal forces.

    Separation components are ordered (local shear x, local shear y, normal).
    Returned upper-face forces use the restoring-force convention: the upper
    substrate receives -K*separation and the lower receives its opposite.
    Two-by-two Gauss integration is exact for the square of a bilinear field.
    """
    energy = 0.0
    reference_area = 0.0
    upper_corner_forces = [[0.0, 0.0, 0.0] for _ in range(4)]
    lower_corner_forces = [[0.0, 0.0, 0.0] for _ in range(4)]
    quadrature: list[dict[str, Any]] = []
    for xi in GAUSS_2:
        for eta in GAUSS_2:
            shape = _quad_shape(xi, eta)
            jacobian = _quad_area_jacobian(reference_points, xi, eta)
            separation = tuple(sum(shape[i] * corner_separation_local[i][axis]
                                   for i in range(4)) for axis in range(3))
            area_weight = jacobian  # Gauss weights are 1 at both points.
            density = 0.5 * (kn_mpa_per_mm * separation[2] ** 2
                             + kt_mpa_per_mm * (separation[0] ** 2
                                                + separation[1] ** 2))
            energy += density * area_weight
            reference_area += area_weight

            traction_local = (kt_mpa_per_mm * separation[0],
                              kt_mpa_per_mm * separation[1],
                              kn_mpa_per_mm * separation[2])
            traction_global = tuple(
                traction_local[0] * tangent_x_global[axis]
                + traction_local[1] * tangent_y_global[axis]
                + traction_local[2] * normal_global[axis]
                for axis in range(3))
            for corner in range(4):
                for axis in range(3):
                    contribution = -shape[corner] * traction_global[axis] * area_weight
                    upper_corner_forces[corner][axis] += contribution
                    lower_corner_forces[corner][axis] -= contribution
            quadrature.append({
                "xi": xi, "eta": eta,
                "separation_local_mm": list(separation),
                "area_weight_mm2": area_weight,
            })
    return {
        "elastic_energy_N_mm": energy,
        "reference_area_mm2": reference_area,
        "upper_corner_forces_N": upper_corner_forces,
        "lower_corner_forces_N": lower_corner_forces,
        "quadrature": quadrature,
    }


def _law117_elastic_parameters(deck: str) -> dict[str, float]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines)
              if re.fullmatch(r"\s*/MAT/LAW117/\d+\s*", line, re.I)]
    if len(starts) != 1:
        raise MeasurementError("expected one /MAT/LAW117 card")
    start = starts[0]
    try:
        elastic = [float(v) for v in lines[start + 3].split()[:2]]
        peak = [float(v) for v in lines[start + 4].split()[2:4]]
    except (IndexError, ValueError) as exc:
        raise MeasurementError("cannot parse LAW117 stiffness/peak inputs") from exc
    if len(elastic) != 2 or len(peak) != 2 or min(elastic) <= 0.0 or min(peak) <= 0.0:
        raise MeasurementError("LAW117 stiffness and peak inputs must be positive")
    return {"normal_stiffness_MPa_per_mm": elastic[0],
            "tangential_stiffness_MPa_per_mm": elastic[1],
            "normal_peak_stress_MPa": peak[0],
            "tangential_peak_stress_MPa": peak[1]}


def _birth_position(nodes: dict[int, tuple[float, float, float]],
                    node_motion: dict[str, Any], pair_node: int, node: int,
                    side: str) -> tuple[float, float, float]:
    displacement = node_motion[str(pair_node)][f"{side}_displacement_global_mm"]
    return tuple(nodes[node][axis] + displacement[axis] for axis in range(3))


def _separation_at_node(point: dict[str, Any], birth_point: dict[str, Any],
                        node: int) -> tuple[float, float, float]:
    current = point["interface_node_motion"][str(node)]
    birth = birth_point["interface_node_motion"][str(node)]
    return (current["slip_x_local_mm"] - birth["slip_x_local_mm"],
            current["slip_y_local_mm"] - birth["slip_y_local_mm"],
            current["gap_mm"] - birth["gap_mm"])


def _reference_points_at_birth(nodes: dict[int, tuple[float, float, float]],
                               birth_point: dict[str, Any],
                               conn: tuple[int, ...]) -> dict[str, list[tuple[float, float, float]]]:
    lower, upper = [], []
    motion = birth_point["interface_node_motion"]
    for index in range(4):
        lower_id, upper_id = conn[index], conn[index + 4]
        lower.append(_birth_position(nodes, motion, lower_id, lower_id, "lower"))
        upper.append(_birth_position(nodes, motion, lower_id, upper_id, "upper"))
    midsurface = [tuple((lower[i][axis] + upper[i][axis]) / 2.0 for axis in range(3))
                  for i in range(4)]
    return {"lower": lower, "upper": upper, "midsurface": midsurface}


def _quad_area(points: list[tuple[float, float, float]]) -> float:
    return sum(_quad_area_jacobian(points, xi, eta)
               for xi in GAUSS_2 for eta in GAUSS_2)


def _apply_quadrature_diagnostics(
        stages: dict[str, dict[str, Any]],
        cohesive: dict[int, tuple[int, ...]],
        nodes: dict[int, tuple[float, float, float]],
        pair_map: dict[int, int],
        tangent_x: tuple[float, float, float],
        tangent_y: tuple[float, float, float],
        normal: tuple[float, float, float],
        stiffness: dict[str, float],
        first_active: tuple[str, int, dict[str, Any]] | None,
        birth_time_s: float) -> dict[str, Any]:
    """Attach exact 2x2 quadrature energies and consistent nodal force checks."""
    if first_active is None:
        raise MeasurementError("cannot establish cohesive reference geometry without an active sample")
    birth_point = first_active[2]
    kn = stiffness["normal_stiffness_MPa_per_mm"]
    kt = stiffness["tangential_stiffness_MPa_per_mm"]
    upper_ids = sorted(pair_map.values())
    lower_ids = sorted(pair_map)
    node_ids = lower_ids + upper_ids
    reference = {eid: _reference_points_at_birth(nodes, birth_point, conn)
                 for eid, conn in cohesive.items()}
    reference_areas = {eid: {
        "lower_face_mm2": _quad_area(points["lower"]),
        "upper_face_mm2": _quad_area(points["upper"]),
        "midsurface_mm2": _quad_area(points["midsurface"]),
    } for eid, points in reference.items()}

    for _stage, data in stages.items():
        for point in data["observations"]:
            nodal_force = {node: [0.0, 0.0, 0.0] for node in node_ids}
            element_energy: dict[str, float] = {}
            element_area: dict[str, float] = {}
            native_stress_resultant = [0.0, 0.0, 0.0]
            active_count = 0
            for eid, conn in cohesive.items():
                key = str(eid)
                state = point["cohesive_elements"][key]
                if state["OFF"] != ACTIVE_OFF or point["time_s"] < birth_time_s:
                    state["quadrature_elastic_energy_N_mm"] = 0.0
                    state["quadrature_energy_minus_native_IE_N_mm"] = -state["internal_energy_N_mm"]
                    element_energy[key] = 0.0
                    element_area[key] = reference_areas[eid]["midsurface_mm2"]
                    continue
                active_count += 1
                corners = [_separation_at_node(point, birth_point, node)
                           for node in conn[:4]]
                integrated = _integrate_quad_elastic(
                    reference[eid]["midsurface"], corners, kn, kt,
                    tangent_x, tangent_y, normal)
                value = integrated["elastic_energy_N_mm"]
                element_energy[key] = value
                element_area[key] = integrated["reference_area_mm2"]
                state["birth_reference_area_mm2"] = integrated["reference_area_mm2"]
                state["quadrature_elastic_energy_N_mm"] = value
                state["quadrature_energy_minus_native_IE_N_mm"] = (
                    value - state["internal_energy_N_mm"])
                state["quadrature_mean_energy_density_N_per_mm"] = (
                    value / integrated["reference_area_mm2"])
                for index in range(4):
                    low_id, high_id = conn[index], conn[index + 4]
                    if pair_map.get(low_id) != high_id:
                        raise MeasurementError(f"TYPE43 {eid} node pair {low_id}/{high_id} is inconsistent")
                    for axis in range(3):
                        nodal_force[high_id][axis] += integrated["upper_corner_forces_N"][index][axis]
                        nodal_force[low_id][axis] += integrated["lower_corner_forces_N"][index][axis]
                stress = state["stress_MPa"]
                native_stress_resultant[0] += stress["SXZ"] * integrated["reference_area_mm2"]
                native_stress_resultant[1] += stress["SYZ"] * integrated["reference_area_mm2"]
                native_stress_resultant[2] += stress["SZ"] * integrated["reference_area_mm2"]

            total_energy = sum(element_energy.values())
            native_energy = point["cohesive_internal_energy_sum_N_mm"]
            upper_resultant = [sum(nodal_force[node][axis] for node in upper_ids)
                               for axis in range(3)]
            lower_resultant = [sum(nodal_force[node][axis] for node in lower_ids)
                               for axis in range(3)]
            force_balance = [upper_resultant[axis] + lower_resultant[axis]
                             for axis in range(3)]
            native_upper_force = [-value for value in native_stress_resultant]
            force_residual = [upper_resultant[axis] - native_upper_force[axis]
                              for axis in range(3)]
            point["quadrature_elastic_energy_N_mm"] = total_energy
            point["quadrature_energy_minus_native_IE_N_mm"] = total_energy - native_energy
            point["quadrature_energy_relative_error"] = (
                None if abs(native_energy) <= 1e-18
                else (total_energy - native_energy) / native_energy)
            point["quadrature_reference_area_by_element_mm2"] = element_area
            point["quadrature_element_energy_N_mm"] = element_energy
            point["quadrature_force_check"] = {
                "sign_convention": "restoring interface force on upper substrate is -K*separation; lower force is equal and opposite",
                "upper_surface_nodal_force_N": {
                    str(node): nodal_force[node] for node in upper_ids},
                "lower_surface_nodal_force_N": {
                    str(node): nodal_force[node] for node in lower_ids},
                "upper_surface_resultant_N": upper_resultant,
                "lower_surface_resultant_N": lower_resultant,
                "action_reaction_balance_N": force_balance,
                "native_stress_area_upper_face_resultant_N": native_upper_force,
                "quadrature_minus_native_resultant_N": force_residual,
                "active_TYPE43_count": active_count,
            }

    # A linear traction-separation law has exact trapezoidal work at each
    # Gauss point: midpoint(K*delta) dot delta_increment equals Delta(1/2 K delta^2).
    chronological = sorted((p for data in stages.values() for p in data["observations"]),
                           key=lambda p: p["time_s"])
    active_points = [p for p in chronological
                     if all(elem["OFF"] == ACTIVE_OFF
                            for elem in p["cohesive_elements"].values())]
    path_work = 0.0
    path_work_components = {"normal_N_mm": 0.0,
                            "shear_x_N_mm": 0.0,
                            "shear_y_N_mm": 0.0}
    for before, after in zip(active_points, active_points[1:]):
        for eid, conn in cohesive.items():
            refpoints = reference[eid]["midsurface"]
            old_corners = [_separation_at_node(before, birth_point, node) for node in conn[:4]]
            new_corners = [_separation_at_node(after, birth_point, node) for node in conn[:4]]
            for xi in GAUSS_2:
                for eta in GAUSS_2:
                    shape = _quad_shape(xi, eta)
                    jacobian = _quad_area_jacobian(refpoints, xi, eta)
                    old = tuple(sum(shape[i] * old_corners[i][axis] for i in range(4))
                                for axis in range(3))
                    new = tuple(sum(shape[i] * new_corners[i][axis] for i in range(4))
                                for axis in range(3))
                    increments = [new[i] - old[i] for i in range(3)]
                    stiffness_components = (kt, kt, kn)
                    for name, axis in (("shear_x_N_mm", 0),
                                       ("shear_y_N_mm", 1),
                                       ("normal_N_mm", 2)):
                        average_traction = (stiffness_components[axis]
                                            * (old[axis] + new[axis]) / 2.0)
                        path_work_components[name] += average_traction * increments[axis] * jacobian
    path_work = sum(path_work_components.values())
    final_energy = active_points[-1]["quadrature_elastic_energy_N_mm"] if active_points else 0.0
    first_energy = active_points[0]["quadrature_elastic_energy_N_mm"] if active_points else 0.0
    final_force = active_points[-1]["quadrature_force_check"] if active_points else None
    return {
        "law_inputs": stiffness,
        "elastic_branch_check": {
            "maximum_reported_absolute_stress_MPa": max(
                abs(value) for data in stages.values() for point in data["observations"]
                for element in point["cohesive_elements"].values()
                for value in element["stress_MPa"].values()),
            "maximum_input_peak_stress_MPa": min(stiffness["normal_peak_stress_MPa"],
                                                  stiffness["tangential_peak_stress_MPa"]),
            "all_reported_stress_below_both_peak_limits": all(
                max(abs(value) for value in element["stress_MPa"].values())
                < min(stiffness["normal_peak_stress_MPa"],
                      stiffness["tangential_peak_stress_MPa"])
                for data in stages.values() for point in data["observations"]
                for element in point["cohesive_elements"].values()),
            "limit": "This supports using the undamaged linear branch for this synthetic fixture; it is not material calibration.",
        },
        "reference_area_semantics": {
            "geometry_sample": "paired TYPE43 midsurface at first sampled OFF=1 accepted birth geometry",
            "per_element_area_mm2": {
                str(eid): {key: value for key, value in areas.items()}
                for eid, areas in reference_areas.items()},
            "area_source_limit": "Pinned upstream SUSER43 uses per-Gauss AREAP values saved in VOL for ISMSTR=1; the staged-birth repair initializes a dormant element's missing reference area/frame at first active geometry. The current calculation uses that sampled birth midsurface and does not infer unsampled geometry.",
        },
        "sample_comparison": {
            stage: [{
                "sample_index": index,
                "time_s": point["time_s"],
                "active_TYPE43_count": point["quadrature_force_check"]["active_TYPE43_count"],
                "quadrature_elastic_energy_N_mm": point["quadrature_elastic_energy_N_mm"],
                "native_cohesive_IE_N_mm": point["cohesive_internal_energy_sum_N_mm"],
                "quadrature_minus_native_N_mm": point["quadrature_energy_minus_native_IE_N_mm"],
                "relative_error": point["quadrature_energy_relative_error"],
            } for index, point in enumerate(data["observations"])]
            for stage, data in stages.items()},
        "maximum_absolute_relative_energy_error": max(
            (abs(point["quadrature_energy_relative_error"])
             for data in stages.values() for point in data["observations"]
             if point["quadrature_energy_relative_error"] is not None), default=0.0),
        "maximum_absolute_energy_error_N_mm": max(
            abs(point["quadrature_energy_minus_native_IE_N_mm"])
            for data in stages.values() for point in data["observations"]),
        "correct_elastic_path_work_N_mm": {
            **path_work_components,
            "total_N_mm": path_work,
            "first_active_energy_N_mm": first_energy,
            "last_active_energy_N_mm": final_energy,
            "work_minus_energy_change_N_mm": path_work - (final_energy - first_energy),
            "method": "For each active adjacent sample pair, integrate midpoint(K*delta) dot delta_increment at each of four TYPE43 Gauss points using birth reference area; exact for this linear undamaged law.",
        },
        "final_sample_force_check": final_force,
    }


def _native_reaction_force_diagnostics(
        stages: dict[str, dict[str, Any]],
        lower_support_nodes: tuple[int, ...],
        upper_support_nodes: tuple[int, ...]) -> dict[str, Any]:
    """Compare converted support REAC with window-matched quad resultants."""
    per_stage: dict[str, Any] = {}
    residuals: dict[str, list[float]] = {
        f"{face}_{axis}": [] for face in ("lower", "upper") for axis in "XYZ"}
    support_balance: dict[str, list[float]] = {axis: [] for axis in "XYZ"}
    for stage, data in stages.items():
        points = data["observations"]
        times = [point["time_s"] for point in points]
        stage_examples = []
        stage_residuals = {key: [] for key in residuals}
        stage_balance = {axis: [] for axis in "XYZ"}
        for index, point in enumerate(points):
            left, right, window = _stencil_indices(index, len(points))
            values: dict[str, Any] = {"sample_index": index, "time_s": times[index],
                                      "window": window, "window_start_s": times[left],
                                      "window_end_s": times[right], "axes": {}}
            for axis_index, axis in enumerate("XYZ"):
                lower_reaction = point["support_reaction_N"]["lower"][axis]
                upper_reaction = point["support_reaction_N"]["upper"][axis]
                lower_internal = _time_average(
                    times,
                    [p["quadrature_force_check"]["lower_surface_resultant_N"][axis_index]
                     for p in points], left, right)
                upper_internal = _time_average(
                    times,
                    [p["quadrature_force_check"]["upper_surface_resultant_N"][axis_index]
                     for p in points], left, right)
                if lower_reaction is None or upper_reaction is None or \
                        lower_internal is None or upper_internal is None:
                    continue
                # REAC is the support reaction (opposite the internal cohesive
                # force acting on each substrate). Convert it to internal-force
                # sign before comparing with the integrated cohesive field.
                lower_internal_from_reaction = -lower_reaction
                upper_internal_from_reaction = -upper_reaction
                lower_residual = lower_internal_from_reaction - lower_internal
                upper_residual = upper_internal_from_reaction - upper_internal
                balance = lower_reaction + upper_reaction
                residuals[f"lower_{axis}"].append(lower_residual)
                residuals[f"upper_{axis}"].append(upper_residual)
                stage_residuals[f"lower_{axis}"].append(lower_residual)
                stage_residuals[f"upper_{axis}"].append(upper_residual)
                support_balance[axis].append(balance)
                stage_balance[axis].append(balance)
                values["axes"][axis] = {
                    "lower_support_REAC_window_average_N": lower_reaction,
                    "upper_support_REAC_window_average_N": upper_reaction,
                    "lower_internal_force_from_REAC_N": lower_internal_from_reaction,
                    "upper_internal_force_from_REAC_N": upper_internal_from_reaction,
                    "lower_quadrature_internal_force_window_average_N": lower_internal,
                    "upper_quadrature_internal_force_window_average_N": upper_internal,
                    "lower_force_residual_N": lower_residual,
                    "upper_force_residual_N": upper_residual,
                    "native_support_balance_N": balance,
                }
            if values["axes"]:
                stage_examples.append(values)

        per_stage[stage] = {
            "sample_count": len(stage_examples),
            "converter_window_policy": "same grid as bond_measure: forward first, centered interior, backward last; window-average quadrature force with the pinned th_to_csv source stencil before comparison",
            "lower_force_residual_rms_N": {
                axis: (math.sqrt(sum(v * v for v in stage_residuals[f"lower_{axis}"])
                                / len(stage_residuals[f"lower_{axis}"]))
                       if stage_residuals[f"lower_{axis}"] else None)
                for axis in "XYZ"},
            "upper_force_residual_rms_N": {
                axis: (math.sqrt(sum(v * v for v in stage_residuals[f"upper_{axis}"])
                                / len(stage_residuals[f"upper_{axis}"]))
                       if stage_residuals[f"upper_{axis}"] else None)
                for axis in "XYZ"},
            "support_balance_max_abs_N": {
                axis: max((abs(v) for v in stage_balance[axis]), default=0.0)
                for axis in "XYZ"},
            "sample_examples": stage_examples[:2] + stage_examples[-1:],
        }
    combined_residuals = {
        key: {"max_abs_N": max((abs(v) for v in values), default=0.0),
              "rms_N": math.sqrt(sum(v * v for v in values) / len(values)) if values else None}
        for key, values in residuals.items()}
    combined_balance = {
        axis: {"max_abs_N": max((abs(v) for v in values), default=0.0),
               "rms_N": math.sqrt(sum(v * v for v in values) / len(values)) if values else None}
        for axis, values in support_balance.items()}
    return {
        "status": "measured_windowed_native_reaction_diagnostic",
        "lower_support_nodes": list(lower_support_nodes),
        "upper_support_nodes": list(upper_support_nodes),
        "reaction_sign_convention": "negated native support REAC is compared to the cohesive internal force on the same substrate face",
        "window_semantics": "Converted /TH/NODE REAC is a source-stencil impulse-derived time-window average; quadrature traction is averaged over the matching sample window using the same source stencil.",
        "converter_binary_sha256": THTOCSV_SHA256,
        "converter_source_commit": THTOCSV_SOURCE_COMMIT,
        "per_face_axis_force_residuals": combined_residuals,
        "upper_plus_lower_native_support_balance": combined_balance,
        "stages": per_stage,
        "interpretation_limit": "Support reactions include the complete deformable substrate and imposed-grip response; residuals measure this synthetic instrument's force transfer and are not paper-interface calibration or a quasi-static gate.",
    }


def _entity_column(headers: list[str], entity: int, variable: str) -> int:
    matches = [i for i, header in enumerate(headers)
               if len(header.split()) >= 2
               and header.split()[-2] == str(entity)
               and header.split()[-1].upper() == variable.upper()]
    if len(matches) != 1:
        raise MeasurementError(
            f"expected one {variable} history column for entity {entity}, found {len(matches)}")
    return matches[0]


def _read_stage(post_dir: Path, stage: str) -> tuple[list[str], list[list[float]], dict[str, Any]]:
    directory = post_dir / stage
    csv_path = directory / "probeT01.csv"
    titles_path = directory / "probeT01_TITLES"
    alias_path = directory / "probeT01"
    conversion_path = directory / "conversion.json"
    if not all(p.is_file() for p in (csv_path, titles_path, alias_path, conversion_path)):
        raise MeasurementError(f"missing converted history artifacts in {directory}")
    conversion = _json(conversion_path)
    checks = {
        "input_sha256": _sha256(alias_path),
        "output_sha256": _sha256(csv_path),
        "titles_sha256": _sha256(titles_path),
    }
    for key, actual in checks.items():
        if conversion.get(key) != actual:
            raise MeasurementError(f"{directory}: conversion receipt {key} mismatch")
    if conversion.get("converter_sha256") != THTOCSV_SHA256:
        raise MeasurementError(f"{directory}: unrecognized th_to_csv binary")
    if conversion.get("exit_code") != 0:
        raise MeasurementError(f"{directory}: converter did not complete successfully")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        headers = next(reader, [])
        if not headers or headers[0].strip().casefold() != "time":
            raise MeasurementError(f"{csv_path}: missing leading time column")
        rows: list[list[float]] = []
        for row_no, values in enumerate(reader, start=2):
            if len(values) != len(headers):
                raise MeasurementError(f"{csv_path}:{row_no}: column count mismatch")
            rows.append([_float(v, f"{csv_path}:{row_no}") for v in values])
    if not rows:
        raise MeasurementError(f"{csv_path}: no history samples")
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise MeasurementError(f"{csv_path}: non-increasing timebase")
    metadata = {
        "csv_sha256": checks["output_sha256"],
        "native_history_sha256": checks["input_sha256"],
        "native_history_relative_path": conversion.get("input_relative_path"),
        "titles_sha256": checks["titles_sha256"],
        "conversion_sha256": _sha256(conversion_path),
        "converter_sha256": conversion.get("converter_sha256"),
        "converter_source_commit": THTOCSV_SOURCE_COMMIT,
    }
    return headers, rows, metadata


def _read_run(run_dir: Path, post_dir: Path) -> dict[str, Any]:
    case_path, receipt_path = run_dir / "case.json", run_dir / "run.json"
    case, receipt = _json(case_path), _json(receipt_path)
    if case.get("id") != "type2_cohesive_overlay_03":
        raise MeasurementError(f"unexpected case id in {case_path}")
    if receipt.get("schema") != "cardboard.opencourant.run.v1":
        raise MeasurementError(f"unexpected run receipt schema in {receipt_path}")
    if receipt.get("case_sha256") != _sha256(case_path):
        raise MeasurementError(f"{run_dir}: receipt case hash mismatch")
    for rel, meta in receipt.get("inputs", {}).items():
        path = run_dir / rel
        if not path.is_file() or _sha256(path) != meta.get("sha256"):
            raise MeasurementError(f"{run_dir}: native input hash mismatch at {rel}")
    deck_path = run_dir / case["starter"]
    deck = deck_path.read_text(encoding="utf-8", errors="replace")
    nodes = _nodes_from_deck(deck)
    cohesive = _bricks_from_deck(deck, 1)
    law117 = _law117_elastic_parameters(deck)
    lower_support_nodes = _group_nodes_from_deck(deck, 1)
    driven_nodes = _group_nodes_from_deck(deck, 2)
    if len(cohesive) != 4:
        raise MeasurementError(f"expected four cohesive HEX8s, found {len(cohesive)}")
    lower_ids = sorted({n for conn in cohesive.values() for n in conn[:4]})
    upper_ids = sorted({n for conn in cohesive.values() for n in conn[4:]})
    pair_map: dict[int, int] = {}
    for lower in lower_ids:
        lx, ly, _ = nodes[lower]
        candidates = sorted(upper_ids, key=lambda upper: (
            (nodes[upper][0] - lx) ** 2 + (nodes[upper][1] - ly) ** 2,
            abs(nodes[upper][2] - nodes[lower][2])))
        upper = candidates[0]
        if math.hypot(nodes[upper][0] - lx, nodes[upper][1] - ly) > 1e-8:
            raise MeasurementError(f"no coincident XY overlay pair for lower node {lower}")
        if upper in pair_map.values():
            raise MeasurementError("overlay node pairing is not one-to-one")
        pair_map[lower] = upper
    vector = [sum(nodes[pair_map[n]][axis] - nodes[n][axis] for n in lower_ids)
              / len(lower_ids) for axis in range(3)]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        raise MeasurementError("cannot derive initial interface normal")
    normal = [v / norm for v in vector]
    angle = math.radians(float(case.get("initial_rotation_deg", 0.0)))
    local_x = [math.cos(angle), math.sin(angle), 0.0]
    local_y = [-math.sin(angle), math.cos(angle), 0.0]
    elem_areas = {eid: _area(nodes, conn[:4]) for eid, conn in cohesive.items()}

    stages: dict[str, dict[str, Any]] = {}
    for stage_index, _engine in enumerate(case["engines"], start=1):
        stage = f"T{stage_index:02d}"
        headers, rows, meta = _read_stage(post_dir, stage)
        outputs = receipt.get("outputs", {})
        for relative, hash_key in (
                (meta["native_history_relative_path"], "native_history_sha256"),
                (str(meta["native_history_relative_path"]) + "_TITLES", "titles_sha256")):
            if not isinstance(relative, str) or outputs.get(relative, {}).get("sha256") != meta[hash_key]:
                raise MeasurementError(
                    f"{run_dir}: converted {stage} input does not match retained native output {relative!r}")
        stage_columns: dict[str, Any] = {
            "global": {key: headers.index(key) for key in (
                "INTERNAL ENERGY", "KINETIC ENERGY", "EXTERNAL WORK", "MASS", "ADDED MASS")},
            "nodes": {node: {axis: _entity_column(headers, node, "D" + axis)
                             for axis in ("X", "Y", "Z")}
                      for node in lower_ids + upper_ids + list(driven_nodes)},
            "reactions": {node: {axis: _entity_column(headers, node, "REAC" + axis)
                                  for axis in ("X", "Y", "Z")}
                          for node in tuple(lower_support_nodes) + tuple(driven_nodes)},
            "cohesive": {eid: {var: _entity_column(headers, eid, var)
                                for var in ("OFF", "SX", "SY", "SZ", "SXZ", "SYZ", "IE")}
                         for eid in cohesive},
        }
        observations: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            gaps: list[float] = []
            slips_x: list[float] = []
            slips_y: list[float] = []
            node_motion: dict[int, dict[str, float]] = {}
            for lower, upper in pair_map.items():
                relative = [row[stage_columns["nodes"][upper][axis]]
                            - row[stage_columns["nodes"][lower][axis]]
                            for axis in ("X", "Y", "Z")]
                base = [nodes[upper][axis] - nodes[lower][axis] for axis in range(3)]
                gap = sum((base[i] + relative[i]) * normal[i] for i in range(3))
                slip_x = sum(relative[i] * local_x[i] for i in range(3))
                slip_y = sum(relative[i] * local_y[i] for i in range(3))
                lower_displacement = [row[stage_columns["nodes"][lower][axis]]
                                      for axis in ("X", "Y", "Z")]
                upper_displacement = [row[stage_columns["nodes"][upper][axis]]
                                      for axis in ("X", "Y", "Z")]
                gaps.append(gap)
                slips_x.append(slip_x)
                slips_y.append(slip_y)
                node_motion[lower] = {
                    "gap_mm": gap,
                    "slip_x_local_mm": slip_x,
                    "slip_y_local_mm": slip_y,
                    "lower_displacement_global_mm": lower_displacement,
                    "upper_displacement_global_mm": upper_displacement,
                }
            element_states: dict[str, Any] = {}
            force = {"normal_local_N": 0.0, "shear_x_local_N": 0.0,
                     "shear_y_local_N": 0.0}
            for eid in sorted(cohesive):
                cols = stage_columns["cohesive"][eid]
                values = {name: row[col] for name, col in cols.items()}
                area = elem_areas[eid]
                element_states[str(eid)] = {
                    "OFF": values["OFF"],
                    "stress_MPa": {axis: values[axis] for axis in ("SX", "SY", "SZ", "SXZ", "SYZ")},
                    "internal_energy_N_mm": values["IE"],
                    "nominal_interface_area_mm2": area,
                    "mean_interface_separation_mm": {
                        "gap": sum(node_motion[n]["gap_mm"] for n in cohesive[eid][:4]) / 4.0,
                        "slip_x_local": sum(node_motion[n]["slip_x_local_mm"]
                                            for n in cohesive[eid][:4]) / 4.0,
                        "slip_y_local": sum(node_motion[n]["slip_y_local_mm"]
                                            for n in cohesive[eid][:4]) / 4.0,
                    },
                }
                force["normal_local_N"] += values["SZ"] * area
                force["shear_x_local_N"] += values["SXZ"] * area
                force["shear_y_local_N"] += values["SYZ"] * area
            g = stage_columns["global"]
            observations.append({
                "time_s": row[0],
                "global_internal_energy_N_mm": row[g["INTERNAL ENERGY"]],
                "global_kinetic_energy_N_mm": row[g["KINETIC ENERGY"]],
                "global_external_work_N_mm": row[g["EXTERNAL WORK"]],
                "total_mass_Mg": row[g["MASS"]],
                "added_mass_Mg": row[g["ADDED MASS"]],
                "interface_gap_mm": {"min": min(gaps), "max": max(gaps),
                                      "mean": sum(gaps) / len(gaps)},
                "interface_node_motion": {str(node): value
                                          for node, value in node_motion.items()},
                "interface_slip_local_mm": {
                    "x_min": min(slips_x), "x_max": max(slips_x),
                    "y_min": min(slips_y), "y_max": max(slips_y),
                    "max_magnitude": max(math.hypot(x, y)
                                         for x, y in zip(slips_x, slips_y)),
                },
                "cohesive_elements": element_states,
                "cohesive_internal_energy_sum_N_mm": sum(
                    row[stage_columns["cohesive"][eid]["IE"]] for eid in cohesive),
                "area_integrated_stress_resultant_N": force,
                "support_reaction_N": {
                    "lower": {axis: sum(row[stage_columns["reactions"][node][axis]]
                                         for node in lower_support_nodes)
                              for axis in ("X", "Y", "Z")},
                    "upper": {axis: sum(row[stage_columns["reactions"][node][axis]]
                                         for node in driven_nodes)
                              for axis in ("X", "Y", "Z")},
                },
            })
        stages[stage] = {"observations": observations, "history_integrity": meta}

    stage_summaries: dict[str, Any] = {}
    for stage, data in stages.items():
        points = data["observations"]
        stage_summaries[stage] = {
            "sample_count": len(points),
            "time_start_s": points[0]["time_s"],
            "time_end_s": points[-1]["time_s"],
            "first_global_state": {key: points[0][key] for key in (
                "global_internal_energy_N_mm", "global_kinetic_energy_N_mm",
                "global_external_work_N_mm", "total_mass_Mg", "added_mass_Mg")},
            "last_global_state": {key: points[-1][key] for key in (
                "global_internal_energy_N_mm", "global_kinetic_energy_N_mm",
                "global_external_work_N_mm", "total_mass_Mg", "added_mass_Mg")},
            "gap_range_over_stage_mm": {
                "min": min(p["interface_gap_mm"]["min"] for p in points),
                "max": max(p["interface_gap_mm"]["max"] for p in points),
            },
            "max_abs_kinetic_energy_N_mm": max(
                abs(p["global_kinetic_energy_N_mm"]) for p in points),
            "observed_OFF_values": sorted({element["OFF"]
                for point in points for element in point["cohesive_elements"].values()}),
            "max_abs_cohesive_stress_MPa": max(
                abs(value) for p in points for element in p["cohesive_elements"].values()
                for value in element["stress_MPa"].values()),
        }

    def transition(left_stage: str, right_stage: str) -> dict[str, Any]:
        before = stages[left_stage]["observations"][-1]
        after = stages[right_stage]["observations"][0]
        return {
            "from_stage": left_stage,
            "to_stage": right_stage,
            "time_before_s": before["time_s"],
            "time_after_s": after["time_s"],
            "sample_time_gap_s": after["time_s"] - before["time_s"],
            "delta_global_internal_energy_N_mm": (
                after["global_internal_energy_N_mm"] - before["global_internal_energy_N_mm"]),
            "delta_global_kinetic_energy_N_mm": (
                after["global_kinetic_energy_N_mm"] - before["global_kinetic_energy_N_mm"]),
            "delta_global_external_work_N_mm": (
                after["global_external_work_N_mm"] - before["global_external_work_N_mm"]),
            "delta_mass_Mg": after["total_mass_Mg"] - before["total_mass_Mg"],
            "after_interface_gap_mm": after["interface_gap_mm"],
            "after_area_integrated_stress_resultant_N": after["area_integrated_stress_resultant_N"],
            "after_cohesive_internal_energy_N_mm": after["cohesive_internal_energy_sum_N_mm"],
        }

    # Evaluate the final prescribed approach segment using actual output
    # displacements. This is a finite-difference rate, not a solver velocity
    # channel; exact endpoint velocity continuity is therefore not asserted.
    approach = stages["T01"]["observations"]
    approach_headers, approach_rows, _ = _read_stage(post_dir, "T01")
    top_column = {
        node: {axis: _entity_column(approach_headers, node, "D" + axis)
               for axis in ("X", "Y", "Z")}
        for node in driven_nodes}
    left, right = approach_rows[-2], approach_rows[-1]
    dt = right[0] - left[0]
    last_rates = {axis: {
        "per_node_mm_per_s": {
            str(node): (right[top_column[node][axis]] - left[top_column[node][axis]]) / dt
            for node in driven_nodes},
        "range_mm_per_s": max(
            (right[top_column[node][axis]] - left[top_column[node][axis]]) / dt
            for node in driven_nodes) - min(
            (right[top_column[node][axis]] - left[top_column[node][axis]]) / dt
            for node in driven_nodes),
    } for axis in ("X", "Y", "Z")}
    endpoint_spread = {axis: max(approach_rows[-1][top_column[node][axis]] for node in driven_nodes)
                       - min(approach_rows[-1][top_column[node][axis]] for node in driven_nodes)
                       for axis in ("X", "Y", "Z")}

    # Activation is read from native /TH/BRIC OFF, where 0 is dormant/deleted
    # and 1 is active; first sampled active row is after the exact /ACTIV time.
    all_rows = [(stage, i, point) for stage, data in stages.items()
                for i, point in enumerate(data["observations"])]
    birth_time = float(case["birth_time_s"])
    active_rows = [(s, i, p) for s, i, p in all_rows
                   if any(e["OFF"] == ACTIVE_OFF for e in p["cohesive_elements"].values())]
    dormant_before = [(s, i, p) for s, i, p in all_rows
                      if p["time_s"] < birth_time and
                      all(e["OFF"] == INACTIVE_OFF for e in p["cohesive_elements"].values())]
    first_active = min(active_rows, key=lambda item: item[2]["time_s"]) if active_rows else None
    last_dormant = max(dormant_before, key=lambda item: item[2]["time_s"]) if dormant_before else None

    quadrature_diagnostic = _apply_quadrature_diagnostics(
        stages, cohesive, nodes, pair_map,
        tuple(local_x), tuple(local_y), tuple(normal), law117,
        first_active, birth_time)
    reaction_diagnostic = _native_reaction_force_diagnostics(
        stages, lower_support_nodes, driven_nodes)
    for stage, data in stages.items():
        active_points = [p for p in data["observations"]
                         if p["quadrature_force_check"]["active_TYPE43_count"] == len(cohesive)]
        if active_points:
            stage_summaries[stage]["last_quadrature_energy_N_mm"] = (
                active_points[-1]["quadrature_elastic_energy_N_mm"])
            stage_summaries[stage]["last_native_cohesive_IE_N_mm"] = (
                active_points[-1]["cohesive_internal_energy_sum_N_mm"])
            stage_summaries[stage]["last_quadrature_energy_relative_error"] = (
                active_points[-1]["quadrature_energy_relative_error"])
            relative_errors = [abs(p["quadrature_energy_relative_error"])
                               for p in active_points
                               if p["quadrature_energy_relative_error"] is not None]
            stage_summaries[stage]["maximum_active_quadrature_energy_relative_error"] = (
                max(relative_errors) if relative_errors else None)

    quadrature_diagnostic["native_source_basis"] = {
        "upstream_source_commit": "cb760c206e346eff12161a2268b134ef9d48a4ef",
        "source_files_and_lines": {
            "engine/source/elements/solid/sconnect/scoor43.F":
                "lines 348-360 initialize the frame at TIME=0; lines 474-476 call SDET43",
            "engine/source/elements/solid/sconnect/sdet43.F":
                "lines 67-127 derive four AREAP Gauss weights from the paired-face midsurface",
            "engine/source/elements/solid/sconnect/suser43.F":
                "lines 312-315 save AREAP in VOL at TIME=0 for ISMSTR=1; lines 357-363 select the saved VOL or current AREAP; lines 601-603 accumulate stress work times the selected AREA",
            "engine/source/materials/mat/mat117/sigeps117.F90":
                "lines 40-70 identify normal/tangential elastic stiffness; lines 381-392 give undamaged elastic stress as stiffness times separation strain",
        },
        "delayed_birth_repair": "The retained repair note records that first active use initializes an unset dormant TYPE43 reference area and frame at the accepted current geometry; the run source commit and binary are recorded beside each case.",
        "area_instrumentation": "For the pinned source's ISMSTR=1 path, after the delayed-birth repair initializes the unset state at first active use, its four birth AREAP weights are held through VOL. The calculation reconstructs the paired midsurface at first sampled OFF=1 and reports the four-Gauss summed area per element.",
    }
    quadrature_diagnostic["native_source_basis"]["runtime_source_commit"] = receipt.get("source_commit")

    cohesive_work = {"normal_N_mm": 0.0, "shear_x_N_mm": 0.0,
                     "shear_y_N_mm": 0.0, "active_time_intervals_integrated": 0}
    chronological = sorted((p for _, _, p in all_rows), key=lambda p: p["time_s"])
    for before, after in zip(chronological, chronological[1:]):
        if not all(before["cohesive_elements"][str(eid)]["OFF"] == ACTIVE_OFF
                   and after["cohesive_elements"][str(eid)]["OFF"] == ACTIVE_OFF
                   for eid in cohesive):
            continue
        for eid in sorted(cohesive):
            key = str(eid)
            old = before["cohesive_elements"][key]
            new = after["cohesive_elements"][key]
            area = elem_areas[eid]
            old_sep, new_sep = (old["mean_interface_separation_mm"],
                                new["mean_interface_separation_mm"])
            for component, stress, displacement in (
                    ("normal_N_mm", "SZ", "gap"),
                    ("shear_x_N_mm", "SXZ", "slip_x_local"),
                    ("shear_y_N_mm", "SYZ", "slip_y_local")):
                traction_mean = 0.5 * (old["stress_MPa"][stress] + new["stress_MPa"][stress])
                displacement_change = new_sep[displacement] - old_sep[displacement]
                cohesive_work[component] += traction_mean * area * displacement_change
        cohesive_work["active_time_intervals_integrated"] += 1

    direct = stages["T03"]["observations"][-1]
    cohesive_work["total_N_mm"] = sum(cohesive_work[k] for k in (
        "normal_N_mm", "shear_x_N_mm", "shear_y_N_mm"))
    cohesive_work["final_cohesive_internal_energy_N_mm"] = direct[
        "cohesive_internal_energy_sum_N_mm"]
    cohesive_work["integral_minus_final_cohesive_internal_energy_N_mm"] = (
        cohesive_work["total_N_mm"] - direct["cohesive_internal_energy_sum_N_mm"])
    cohesive_work["relative_error_vs_final_cohesive_IE"] = (
        None if abs(direct["cohesive_internal_energy_sum_N_mm"]) <= 1e-18
        else cohesive_work["integral_minus_final_cohesive_internal_energy_N_mm"]
        / direct["cohesive_internal_energy_sum_N_mm"])
    cohesive_work["relative_error_percent_vs_final_cohesive_IE"] = (
        None if cohesive_work["relative_error_vs_final_cohesive_IE"] is None
        else 100.0 * cohesive_work["relative_error_vs_final_cohesive_IE"])

    # The exact input and measured mass ledger are independent checks.
    starter_out = run_dir / f"{Path(case['starter']).stem}.out"
    out_text = starter_out.read_text(encoding="utf-8", errors="replace")
    mass_match = re.search(
        r"TOTAL MASS AND MASS CENTER\s*-+\s*\n\s*MASS\s+X\s+Y\s+Z\s*\n\s*([+\-0-9.Ee]+)",
        out_text, re.I)
    added_match = re.search(r"TOTAL ADDED MASS\s*=\s*([+\-0-9.Ee]+)", out_text, re.I)
    measured_mass = float(mass_match.group(1)) if mass_match else None
    measured_added_mass = float(added_match.group(1)) if added_match else None
    expected_mass = float(case.get("mass_ledger", {}).get("expected_starter_total_mass_Mg", math.nan))
    expected_substrate_mass = float(case["mass_ledger"]["substrate_only_total_mass_Mg"])
    observed_overlay_mass = None if measured_mass is None else measured_mass - expected_substrate_mass

    # Per-stage receipt status preserves the PCG failure as a failed run rather
    # than treating the partial converted T03 output as a completed history.
    commands = receipt.get("commands", [])
    command_status = [{"index": index, "argv": command.get("argv"),
                       "elapsed_s": command.get("elapsed_s"),
                       "exit_code": command.get("exit_code"),
                       "normal_termination": command.get("normal_termination"),
                       "timed_out": command.get("timed_out")}
                      for index, command in enumerate(commands)]
    starter_stem = Path(case["starter"]).stem.removesuffix("_0000")
    final_out = run_dir / f"{starter_stem}_0003.out"
    failure = None
    if final_out.is_file():
        errors = [line.strip() for line in final_out.read_text(
            encoding="utf-8", errors="replace").splitlines()
                  if "ERROR:" in line and "ERROR TERMINATION" not in line]
        if errors:
            failure = errors[-1]
    return {
        "run_dir": str(run_dir),
        "post_dir": str(post_dir),
        "case_sha256": _sha256(case_path),
        "run_receipt_sha256": _sha256(receipt_path),
        "starter_sha256": _sha256(deck_path),
        "native_source_commit": receipt.get("source_commit"),
        "runtime_identity_sha256": receipt.get("runtime_identity_sha256"),
        "binary_sha256": receipt.get("binary_sha256"),
        "engine_solver_cards": _engine_solver_cards(run_dir, case["engines"]),
        "native_input_hashes": receipt.get("inputs", {}),
        "execution_complete": receipt.get("execution_complete"),
        "case_sequence_complete": receipt.get("case_sequence_complete"),
        "native_command_elapsed_total_s": sum(float(c.get("elapsed_s", 0.0)) for c in commands),
        "stage_summaries": stage_summaries,
        "stage_transition_diagnostics": [transition("T01", "T02"),
                                         transition("T02", "T03")],
        "approach_endpoint_diagnostics": {
            "last_finite_difference_interval_s": dt,
            "last_interval_drive_rates_mm_per_s": last_rates,
            "driven_grip_node_displacement_spread_at_approach_end_mm": endpoint_spread,
            "measurement_limit": "Displacement finite differences are not a native velocity channel; endpoint velocity continuity cannot be established from these histories alone.",
        },
        "commands": command_status,
        "stage_failure_text": failure,
        "mass_ledger": {
            "expected_substrate_only_mass_Mg": float(case["mass_ledger"]["substrate_only_total_mass_Mg"]),
            "expected_overlay_mass_Mg": float(case["mass_ledger"]["cohesive_overlay_mass_Mg"]),
            "expected_overlay_mass_mg": float(case["type2_mass_audit"]["expected_total_model_mass_change_mg"]),
            "expected_total_mass_Mg": expected_mass,
            "starter_reported_total_mass_Mg": measured_mass,
            "starter_reported_overlay_mass_Mg": observed_overlay_mass,
            "starter_reported_overlay_mass_mg": None if observed_overlay_mass is None else observed_overlay_mass * 1e9,
            "starter_mass_error_Mg": None if measured_mass is None else measured_mass - expected_mass,
            "starter_mass_error_mg": None if measured_mass is None else (measured_mass - expected_mass) * 1e9,
            "starter_reported_total_added_nodal_mass_Mg": measured_added_mass,
            "final_history_total_mass_Mg": stages["T03"]["observations"][-1]["total_mass_Mg"],
            "film_thickness_assumption_mm": float(case["adhesive_mass_analogue"]["declared_film_thickness_mm"]),
            "density_analogue_kg_per_litre": float(case["adhesive_mass_analogue"]["analogue_density_kg_per_litre"]),
            "areal_mass_Mg_per_mm2": float(case["adhesive_mass_analogue"]["computed_areal_mass_Mg_per_mm2"]),
            "adhesive_source_sha256": case["source_document"]["sha256"],
            "adhesive_source_url": case["source_document"]["url"],
            "qualification_limit": "Sika density is only an engineering density analogue; 0.05 mm film is an assumption and not measured starch/adhesive geometry.",
        },
        "geometry": {
            "interface_node_pair_count": len(pair_map),
            "cohesive_hex8_count": len(cohesive),
            "element_face_area_mm2": elem_areas,
            "total_cohesive_face_area_mm2": sum(elem_areas.values()),
            "normal_unit_global": normal,
            "local_tangent_x_unit_global": local_x,
            "local_tangent_y_unit_global": local_y,
            "initial_rotation_deg": angle * 180.0 / math.pi,
        },
        "activation": {
            "card_birth_time_s": birth_time,
            "OFF_semantics": {"dormant": INACTIVE_OFF, "active": ACTIVE_OFF},
            "last_dormant_sample": None if last_dormant is None else {
                "stage": last_dormant[0], "index": last_dormant[1], **last_dormant[2]},
            "first_active_sample": None if first_active is None else {
                "stage": first_active[0], "index": first_active[1], **first_active[2]},
            "dormant_before_birth_observed": bool(dormant_before),
            "active_after_birth_observed": bool(active_rows),
            "activation_sample_time_is_after_card_time_due_output_sampling": True,
        },
        "invalid_coarse_average_work_diagnostic": {
            **cohesive_work,
            "method": "Across adjacent sampled pairs with both TYPE43 OFF=1, trapezoidal native solid stress components SZ/SXZ/SYZ times nominal face area times per-element mean normal gap/local shear increments.",
            "validity": "invalid_for_nonuniform_quad_fields",
            "reason": "Multiplying separately averaged stress and separation omits the within-element variance term. It underestimates the work for this nonuniform field and must not be compared as a conservative energy check.",
            "preserved_historical_value": True,
        },
        "cohesive_quadrature_diagnostic": quadrature_diagnostic,
        "native_reaction_force_diagnostic": reaction_diagnostic,
        "stages": stages,
    }


def collect(integration_root: Path, output: Path) -> dict[str, Any]:
    runs = integration_root / "runs"
    post = integration_root / "post"
    source_case_path = (integration_root /
                        "candidate-overlay-03-hot-melt-areal-mass-20261023" /
                        "case" / "case.json")
    source_case = _json(source_case_path)
    relative_source = Path(source_case["source_document"]["relative_path"])
    source_path = (source_case_path.parent / relative_source).resolve()
    source_hash = _sha256(source_path)
    if source_hash != source_case["source_document"]["sha256"]:
        raise MeasurementError("analogue source PDF does not match the declared SHA-256")
    results = {
        "schema": SCHEMA,
        "classification": "synthetic TYPE2/TYPE43 solver instrument; not cardboard qualification",
        "energy_units": "N mm",
        "stress_units": "MPa (N/mm2)",
        "mass_units": "Mg unless the key explicitly says mg",
        "traction_method": "The legacy element resultant uses reported solid stresses times nominal face area. The added cohesive diagnostic reconstructs local separation at each TYPE43 quad corner, evaluates 2x2 Gauss elastic energy and consistent per-node forces over the repaired birth-reference area, then checks the force resultant against element stress times that area.",
        "reaction_semantics": "bond_measure establishes that converted T01 nodal reaction output is source-stencil/window averaged; this collector does not treat it as instantaneous endpoint force",
        "source_document_verification": {
            "path": str(source_path),
            "sha256": source_hash,
            "url": source_case["source_document"]["url"],
            "status": "verified_bytes_match_case_sidecar",
        },
        "runs": {
            "overlay-03-pcg": _read_run(runs / "overlay-03", post / "overlay-03"),
            "overlay-04-direct": _read_run(runs / "overlay-04-direct", post / "overlay-04-direct"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integration_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = collect(args.integration_root, args.output)
    print(f"Wrote {args.output}")
    for name, result in results["runs"].items():
        final = result["stages"]["T03"]["observations"][-1]
        print(f"{name}: complete={result['execution_complete']} "
              f"t={final['time_s']:.9g}s mass={final['total_mass_Mg']:.12g}Mg "
              f"IE={final['global_internal_energy_N_mm']:.9g}Nmm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
