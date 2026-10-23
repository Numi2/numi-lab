#!/usr/bin/env python3
"""Export a source-labelled, contacting score-and-fold strip fixture.

This deck is an integration instrument, not a calibrated carton prediction.
It consumes the all-HEX8 geometry from ``cardboard_opencourant_hex`` and adds
only boundary grippers and contacting tools. The corrugated liner/medium and
finite elastic starch volumes remain deformable. Duplicate material faces
are joined with native TYPE2 kinematic ties, representing the recipe's
perfect-bond idealization; the deck has no adhesive failure/debonding law.
TYPE2 Spotflag=27 is selected because the supported native formulation is
explicitly mass-conserving; rigid gripper inertia is computed from the board
nodes and adds no artificial mass or inertia. Native TYPE7 self-contact uses
all exposed liner/medium boundary faces except exact bonded interface facets;
gripper-adjacent faces remain active through boundary release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

try:  # importable as matter.tools.cardboard_opencourant_strip
    from . import cardboard_opencourant_hex as hex_mesh
    from . import cardboard_opencourant_paper as paper
except ImportError:  # directly executable from this tools directory
    import cardboard_opencourant_hex as hex_mesh
    import cardboard_opencourant_paper as paper


SCHEMA = "numi.cardboard.opencourant.strip-score-fold.v1"
RELEASE = paper.RELEASE
SOURCE_COMMIT = paper.SOURCE_COMMIT
ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "matter/examples/cardboard/manufacturing-source-manifest.json"
SOURCE_NOTES = ROOT / "matter/examples/cardboard/manufacturing-sources.md"
FIT_NODES = ROOT / "matter/examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv"
YIELD_ANCHORS = ROOT / "matter/examples/cardboard/source-data/haj_ali_2009_yield_anchors.csv"
SOURCE_CATALOG = ROOT / "matter/examples/cardboard/source-data/catalog.json"
ADHESION_SOURCE = ROOT / "matter/tools/cardboard_opencourant_bond.py"

# All schedules are fixture choices. Their values are recorded in case.json
# and do not assert a manufacturing press speed or rate-independent response.
SCORE_STROKE_DURATION_S = 2.0e-3
SCORE_HOLD_DURATION_S = 5.0e-4
FOLD_START_S = 5.0e-3
FOLD_DURATION_S = 3.0e-3
FOLD_ANGLE_DEG = -90.0  # right wing moves upward about the global Y score axis
FOLD_PATH_INTERVAL_DEG = 5.0
ANIMATION_INTERVAL_S = 8.0e-5
HISTORY_INTERVAL_S = 8.0e-6
DT_SCALE = 0.9
DT_MIN_S = 1.0e-12
ENGINE_MODES = ("explicit", "implicit-slow")
IMPLICIT_FORCE_TOLERANCE = 1.0e-6
IMPLICIT_INITIAL_DT_S = 1.0e-1
IMPLICIT_MIN_DT_S = 1.0e-8
IMPLICIT_MAX_DT_S = 1.0e-1
IMPLICIT_HISTORY_INTERVAL_S = 2.5e-2
IMPLICIT_ANIMATION_INTERVAL_S = 1.0e-1
IMPLICIT_RELEASE_DT_S = 1.0e-3
IMPLICIT_SCORE_SPEED_MM_S = 12.5 / 60.0
IMPLICIT_SCORE_HOLD_S = 1.0
IMPLICIT_PRE_FOLD_DWELL_S = 1.0
IMPLICIT_FOLD_RATE_DEG_S = 10.0
EXPLICIT_PRE_FOLD_DWELL_S = 5.0e-4
PUNCH_PARK_SPEED_MM_S = 5.0
PUNCH_PARK_CLEARANCE_MM = 2.0
FOLD_END_DWELL_S = 1.0
POST_RELEASE_OBSERVATION_S = 1.0
SELF_CONTACT_GAP_MM = 1.0e-3
ENGINE_MASS_BALANCE_REL_TOLERANCE = 5.0e-4
_FLOAT_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _starter_mass_values(text: str) -> tuple[float, float, float]:
    marker = "TOTAL MASS AND MASS CENTER"
    if text.count(marker) != 1:
        raise ValueError("Starter output must contain exactly one total-mass section")
    mass_rows = []
    after_mass = text.split(marker, 1)[1]
    for line in after_mass.splitlines():
        values = _FLOAT_TOKEN.findall(line)
        if len(values) >= 4:
            mass_rows.append(_parse_float(values[0]))
            break
    if len(mass_rows) != 1:
        raise ValueError("could not parse Starter total structural mass")

    added_mass_rows = re.findall(
        r"^\s*TOTAL ADDED MASS\s*=\s*(" + _FLOAT_TOKEN.pattern + r")\s*$",
        text, flags=re.MULTILINE)
    added_inertia_rows = re.findall(
        r"^\s*TOTAL ADDED INERTIA\s*=\s*(" + _FLOAT_TOKEN.pattern + r")\s*$",
        text, flags=re.MULTILINE)
    if len(added_mass_rows) != 1 or len(added_inertia_rows) != 1:
        raise ValueError("Starter output must report exactly one added-mass and added-inertia total")
    return mass_rows[0], _parse_float(added_mass_rows[0]), _parse_float(added_inertia_rows[0])


def _engine_initial_mass_values(text: str) -> tuple[float, float, float]:
    headers = [line_index for line_index, line in enumerate(text.splitlines())
               if "CYCLE" in line and "MAS.ERR" in line and "TOTAL MASS" in line and "MASS ADDED" in line]
    if len(headers) != 1:
        raise ValueError("Engine output must contain exactly one mass-history header")
    lines = text.splitlines()
    for line in lines[headers[0] + 1:]:
        columns = line.split()
        if not columns or columns[0] != "0":
            continue
        numeric = [_parse_float(value) for value in _FLOAT_TOKEN.findall(line)]
        # Cycle zero is emitted with the three mass-history columns last:
        # relative mass error, total mass, and added mass.
        if len(numeric) >= 3:
            mass_error, total_mass, added_mass = numeric[-3:]
            return mass_error, total_mass, added_mass
    raise ValueError("could not parse Engine cycle-zero mass, added mass, and mass error")


def collect_mass_accounting(starter_output: Path | str, engine_output: Path | str, *,
                            known_rigid_tool_mass_Mg: float,
                            tool_mass_in_starter: bool,
                            tool_mass_identity: str) -> dict[str, Any]:
    """Read Starter and Engine mass summaries without running or changing them.

    Callers must account for rigid tooling explicitly. tool_mass_in_starter
    states whether the supplied tool mass is already included in Starter's
    total. This separates structural board mass from tooling and keeps the
    Engine's added-mass report visible. The function is diagnostic only; it
    does not qualify any solver result.
    """
    tool_mass = float(known_rigid_tool_mass_Mg)
    if not math.isfinite(tool_mass) or tool_mass < 0.0:
        raise ValueError("known_rigid_tool_mass_Mg must be finite and nonnegative")
    if not isinstance(tool_mass_in_starter, bool):
        raise ValueError("tool_mass_in_starter must be explicitly true or false")
    if not isinstance(tool_mass_identity, str) or not tool_mass_identity.strip():
        raise ValueError("tool_mass_identity must identify the mass source")

    starter_path, engine_path = Path(starter_output), Path(engine_output)
    starter_text = starter_path.read_text(encoding="utf-8", errors="replace")
    engine_text = engine_path.read_text(encoding="utf-8", errors="replace")
    starter_mass, starter_added_mass, starter_added_inertia = _starter_mass_values(starter_text)
    engine_mass_error, engine_total_mass, engine_added_mass = _engine_initial_mass_values(engine_text)
    if min(starter_mass, starter_added_mass, starter_added_inertia,
           engine_total_mass, engine_added_mass) < 0.0:
        raise ValueError("mass outputs must be nonnegative")

    board_mass = starter_mass - (tool_mass if tool_mass_in_starter else 0.0)
    if board_mass < 0.0:
        raise ValueError("known rigid-tool mass exceeds Starter's reported total mass")
    engine_model_mass_expected = starter_mass + engine_added_mass
    balance_error = engine_total_mass - engine_model_mass_expected
    balance_scale = max(abs(engine_total_mass), abs(engine_model_mass_expected), 1.0e-300)
    balance_error_relative = balance_error / balance_scale
    external_tool_mass = 0.0 if tool_mass_in_starter else tool_mass
    return {
        "schema": "numi.cardboard.opencourant.strip-mass-accounting.v1",
        "read_only": True,
        "mass_unit": "Mg",
        "source_files": {"starter_output": str(starter_path), "engine_output": str(engine_path)},
        "tooling": {
            "identity": tool_mass_identity.strip(),
            "known_mass_Mg": tool_mass,
            "included_in_starter_total": tool_mass_in_starter,
            "external_to_solver_model_mass_Mg": external_tool_mass,
        },
        "starter": {
            "total_mass_Mg": starter_mass,
            "added_mass_Mg": starter_added_mass,
            "added_inertia_reported": starter_added_inertia,
            "board_mass_Mg": board_mass,
        },
        "engine_cycle_zero": {
            "total_mass_Mg": engine_total_mass,
            "added_mass_Mg": engine_added_mass,
            "mass_error_ratio_reported": engine_mass_error,
            "model_mass_expected_from_starter_Mg": engine_model_mass_expected,
            "mass_balance_error_Mg": balance_error,
            "mass_balance_error_relative": balance_error_relative,
            "mass_balance_within_output_rounding": abs(balance_error_relative) <= ENGINE_MASS_BALANCE_REL_TOLERANCE,
            "added_mass_detected": engine_added_mass > 0.0,
        },
        "physical_assembly_mass_Mg": engine_total_mass + external_tool_mass,
        "mass_conservation_gate": {
            "passed": (engine_added_mass == 0.0 and
                       abs(balance_error_relative) <= ENGINE_MASS_BALANCE_REL_TOLERANCE),
            "basis": "cycle-zero Engine added mass must be exactly zero at reported precision and total mass must balance Starter plus Engine-added mass",
            "output_rounding_relative_tolerance": ENGINE_MASS_BALANCE_REL_TOLERANCE,
        },
        "qualification": "mass-accounting diagnostic only; not mechanics or physical validation",
    }


def _material_transfer_module() -> Any:
    """Load the cycle-free source material module in package and script use."""
    module_name = "cardboard_opencourant_material"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    tools_path = str(Path(__file__).resolve().parent)
    added_tools_path = tools_path not in sys.path
    added_paper_alias = module_name != "cardboard_opencourant_paper" and \
        "cardboard_opencourant_paper" not in sys.modules
    if added_tools_path:
        sys.path.insert(0, tools_path)
    if added_paper_alias:
        # The material module uses this stable standalone import name; alias
        # the already-loaded paper module to avoid a second parameter reader.
        sys.modules["cardboard_opencourant_paper"] = paper
    try:
        return importlib.import_module(module_name)
    finally:
        if added_paper_alias:
            sys.modules.pop("cardboard_opencourant_paper", None)
        if added_tools_path:
            sys.path.remove(tools_path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _ids(values: Iterable[int], columns: int = 10) -> str:
    values = list(values)
    return "\n".join("".join(f"{value:10d}" for value in values[i:i + columns])
                     for i in range(0, len(values), columns))


def _func(fid: int, title: str, points: list[tuple[float, float]]) -> str:
    rows = "\n".join(paper._float_fields(time, value) for time, value in points)
    return f"/FUNCT/{fid}\n{title}\n{rows}"


def _type6(prop_id: int, title: str) -> str:
    return "\n".join((
        f"/PROP/TYPE6/{prop_id}", title,
        paper._mixed_fields(17, 2, None, 0, 0, 0, 0, -1, 0.0),
        paper._float_fields(0.0, 0.0, 0.0),
        paper._float_fields(0.0, 0.0, 0.0) + paper._int_fields(0, 20, 0),
        paper._float_fields(0.0, 0.0, 0.0, 0.0),
        paper._float_fields(0.0, 0.0, 0.0, 0.0, 0.0),
    ))


def _type14(prop_id: int) -> str:
    # Copied from the accepted elastic-adherend input format in
    # cardboard_opencourant_bond.py; isotropic starch is a separate volume.
    return "\n".join((
        f"/PROP/TYPE14/{prop_id}",
        "Fully integrated isotropic eight-node starch bridge",
        f"{14:10d}{0:10d}{'':10}{0:10d}{'':10}{222:10d}{'':10}{0:10d}{0.0:20g}",
        f"{0.0:20g}" * 5,
        f"{0.0:20g}" * 5,
    ))


def _part(part_id: int, title: str, material_id: int, property_id: int) -> str:
    # Native Radioss order is property ID, material ID, and option flag.
    return f"/PART/{part_id}\n{title}\n{paper._int_fields(property_id, material_id, 0)}"


def _group(group_id: int, title: str, node_ids: Iterable[int]) -> str:
    ids = sorted(set(int(node) for node in node_ids))
    if not ids:
        raise ValueError(f"empty node group: {title}")
    return f"/GRNOD/NODE/{group_id}\n{title}\n{_ids(ids)}"


def _bcs(bcs_id: int, title: str, mask: str, group_id: int) -> str:
    if len(mask.replace(" ", "")) != 6:
        raise ValueError("BCS mask must encode six translational/rotational flags")
    return "\n".join((f"/BCS/{bcs_id}", title,
                      f"{mask:>10}{0:10d}{group_id:10d}"))


def _imposed(fid: int, direction: str, group_id: int, start: float,
             stop: float, *, scale_y: float = 1.0) -> str:
    if direction not in {"X", "Y", "Z", "XX", "YY", "ZZ"}:
        raise ValueError(f"unsupported imposed-motion direction: {direction}")
    return "\n".join((
        f"/IMPDISP/{fid}/0", f"Boundary-gripper {direction} trajectory",
        f"{fid:10d}{direction:>10}{0:10d}{0:10d}{group_id:10d}{0:10d}{0:10d}",
        paper._float_fields(1.0, scale_y, start, stop),
    ))


def _imposed_velocity(fid: int, direction: str, group_id: int,
                      peak_velocity: float, stop: float, *, start: float = 0.0) -> str:
    return "\n".join((
        f"/IMPVEL/{fid}/0", "Score, withdraw, and park boundary punch",
        f"{fid:10d}{direction:>10}{0:10d}{0:10d}{group_id:10d}{0:10d}{0:10d}",
        paper._float_fields(1.0, peak_velocity, start, stop),
    ))


def _type7_self_contact(mesh: hex_mesh.HexMesh, *, end_time_s: float,
                        gap_mm: float = SELF_CONTACT_GAP_MM) -> tuple[str, str, str, dict[str, Any]]:
    """Build a self-contact surface from actual free paper boundary faces.

    Only exterior liner/medium faces are included; exact bonded TYPE2 faces
    are omitted. Exposed faces adjacent to bonded faces or current grippers
    remain in the surface so self-contact is available after release. Type7
    uses its documented constant-gap mode;
    the positive Gapmin and Gapmax records match, and INACTIV=1000 requests no
    initial-coordinate adjustment.
    """
    if not math.isfinite(gap_mm) or gap_mm <= 0.0:
        raise ValueError("self-contact gap must be finite and positive")
    if not math.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("self-contact end time must be finite and positive")
    minimum_jacobian = float(mesh.quality.get("min_gauss_jacobian_mm3", 0.0))
    if not math.isfinite(minimum_jacobian) or minimum_jacobian <= 0.0:
        raise ValueError("self-contact requires initially positive HEX8 Gauss Jacobians")

    elements = {int(element.id): element for element in mesh.elements}
    roles = {element_id: element.material_role for element_id, element in elements.items()}
    node_xyz = {int(node["id"]): tuple(float(v) for v in node["xyz_mm"])
                for node in mesh.nodes}
    interface_face_refs: set[tuple[int, str]] = set()
    interface_nodes: set[int] = set()
    type2_secondary_nodes: set[int] = set()
    for pair in mesh.interface_pairs:
        for side_name in ("side_a", "side_b"):
            side = pair[side_name]
            interface_face_refs.add((int(side["element_id"]), str(side["face"])))
            interface_nodes.update(int(node) for node in side["nodes"])
        secondary_side = ("side_a" if pair["side_a_role"] == "starch" else "side_b")
        type2_secondary_nodes.update(int(node) for node in pair[secondary_side]["nodes"])
    grip_nodes = {int(node) for key in ("grip_left", "grip_right")
                  for node in mesh.node_sets[key]}
    faces = mesh.surface_sets.get("all_free_surfaces")
    if not faces:
        raise ValueError("self-contact requires builder-provided free boundary surfaces")
    eligible = [face for face in faces if roles.get(int(face.element_id)) in {"liner", "medium"}]
    direct_tie_faces = [face for face in eligible
                        if (int(face.element_id), str(face.face)) in interface_face_refs]
    # Keep exposed facets next to bonds and rigid-grip boundary nodes. Exact
    # paired interface faces are excluded from the builder surface already,
    # and are explicitly filtered again here if a future builder includes one.
    excluded_refs = {(int(face.element_id), str(face.face)) for face in direct_tie_faces}
    contact_faces = [face for face in eligible
                     if (int(face.element_id), str(face.face)) not in excluded_refs]
    tied_adjacent_retained = [face for face in contact_faces
                              if set(int(node) for node in face.nodes) & interface_nodes]
    grip_adjacent_retained = [face for face in contact_faces
                              if set(int(node) for node in face.nodes) & grip_nodes]
    if len(contact_faces) < 100 or not contact_faces:
        raise ValueError("free paper self-contact surface is unexpectedly small")

    # FaceRef uses the builder's outward-oriented HEX8 local face ordering.
    # Verify that every emitted segment still matches that element topology.
    face_geometry: list[tuple[int, tuple[int, int, int, int]]] = []
    segment_keys: set[tuple[int, ...]] = set()
    for face in contact_faces:
        element = elements[int(face.element_id)]
        local = hex_mesh._FACE_LOCAL_NODES.get(str(face.face))
        if local is None:
            raise ValueError(f"unsupported HEX8 boundary face {face.face!r}")
        expected = tuple(int(element.nodes[index]) for index in local)
        nodes = tuple(int(node) for node in face.nodes)
        if nodes != expected:
            raise ValueError("free contact face node order differs from outward HEX8 face order")
        if len(set(nodes)) != 4 or any(node not in node_xyz for node in nodes):
            raise ValueError("free contact face has invalid or missing node references")
        key = tuple(sorted(nodes))
        if key in segment_keys:
            raise ValueError("duplicate free boundary segment in self-contact surface")
        segment_keys.add(key)
        # A nonzero oriented face area rejects collapsed or reversed geometry.
        p0, p1, p2, p3 = (node_xyz[node] for node in nodes)
        u = tuple(p1[i] - p0[i] for i in range(3))
        v = tuple(p2[i] - p0[i] for i in range(3))
        w = tuple(p3[i] - p0[i] for i in range(3))
        cross_a = (u[1] * v[2] - u[2] * v[1],
                   u[2] * v[0] - u[0] * v[2],
                   u[0] * v[1] - u[1] * v[0])
        cross_b = (v[1] * w[2] - v[2] * w[1],
                   v[2] * w[0] - v[0] * w[2],
                   v[0] * w[1] - v[1] * w[0])
        area2 = math.sqrt(sum((cross_a[i] + cross_b[i]) ** 2 for i in range(3)))
        if not math.isfinite(area2) or area2 <= 1.0e-12:
            raise ValueError("self-contact surface contains a degenerate boundary face")
        face_geometry.append((int(face.element_id), nodes))

    secondary_nodes = {node for _, nodes in face_geometry for node in nodes}
    if secondary_nodes & type2_secondary_nodes:
        raise ValueError("TYPE7 secondary nodes overlap a TYPE2 starch-tie secondary group")
    if secondary_nodes != {node for _, nodes in face_geometry for node in nodes}:
        raise AssertionError("self-contact secondary-node inventory is inconsistent")

    contact_id, group_id, surface_id = 3, 9, 33
    group = _group(group_id, "free liner and flute self-contact secondary nodes", secondary_nodes)
    segment_rows = "\n".join(
        f"{segment_id:10d}" + "".join(f"{node:10d}" for node in nodes)
        for segment_id, (_, nodes) in enumerate(face_geometry, start=1))
    surface = f"/SURF/SEG/{surface_id}\nfree exposed liner and flute segments for self-contact\n{segment_rows}"

    # Radioss 2026 TYPE7 card. Igap=1000 is constant-gap mode; explicitly set
    # Gapmin=Gapmax=1 micron. INACTIV=1000 performs no nodal coordinate
    # correction; Ibag=0 avoids requesting monitored-volume behavior when no
    # /MONVOL is defined; ICOG=1 also removes TYPE2-tied slave nodes.
    card = "\n".join((
        f"/INTER/TYPE7/{contact_id}",
        "Free liner and flute self-contact; numerical clearance only",
        f"{group_id:10d}{surface_id:10d}{2:10d}{0:10d}{1000:10d}{'':10}"
        f"{0:10d}{1000:10d}{0:10d}{0:10d}",
        f"{1.0:20g}{gap_mm:20g}{0.0:20g}{'':20}{0:10d}{0:10d}",
        f"{0.0:20g}{0.0:20g}{0.0:20g}{0.0:20g}{1:10d}{1:10d}",
        f"{1.0:20g}{0.0:20g}{gap_mm:20g}{0.0:20g}{end_time_s:20g}",
        f"{'':7}{0:1d}{0:1d}{0:1d}{'':20}{1000:10d}"
        f"{0.0:20g}{0.0:20g}{0.0:20g}",
        f"{0:10d}{0:10d}{0.0:20g}{0:10d}{0:10d}{0:10d}{1.0:20g}{0:10d}",
    ))
    metadata = {
        "type": "/INTER/TYPE7",
        "contact_id": contact_id,
        "main_surface": {"card": f"/SURF/SEG/{surface_id}",
                          "segment_count": len(face_geometry),
                          "material_roles": ["liner", "medium"],
                          "source": "builder all_free_surfaces; boundary faces only"},
        "secondary_group": {"card": f"/GRNOD/NODE/{group_id}",
                            "node_count": len(secondary_nodes)},
        "excluded_faces": {
            "exact_interface_faces_present_in_builder_free_surface": len(direct_tie_faces),
            "policy": "exclude exact TYPE2 interface faces only; retain all other exterior liner/medium faces, including facets adjacent to bonds and grippers"},
        "retained_adjacent_faces": {
            "sharing_interface_nodes": len(tied_adjacent_retained),
            "sharing_gripper_nodes": len(grip_adjacent_retained),
        },
        "type2_secondary_overlap_count": len(secondary_nodes & type2_secondary_nodes),
        "gripper_slave_overlap_count": len(secondary_nodes & grip_nodes),
        "gripper_contact_policy": "gripper boundary nodes remain in TYPE7 secondary set; native rigid-body/contact compatibility and post-release self-contact must be verified in Starter/Engine",
        "same_surface_contact_filtering": "export all exposed boundary segments; verify native TYPE7 handles shared-node adjacent segments without exporter one-ring removal",
        "gap_mm": gap_mm,
        "gap_mode": 1000,
        "gap_max_mm": gap_mm,
        "ibag": 0,
        "ibag_semantics": "no monitored-volume flag; no /MONVOL is defined",
        "gap_identity": "declared numerical contact clearance; not an added material layer or calibrated physical gap",
        "initial_coordinate_adjustment": {"inactiv": 1000, "semantics": "no action"},
        "monitored_volume_flag": {"ibag": 0, "reason": "no /MONVOL is defined; avoid Starter ID614 warning"},
        "stiffness_mode": 2,
        "friction": 0.0,
        "friction_identity": "instrument choice; source friction is unidentified",
        "penetration_gate": f"measured maximum penetration must remain < {gap_mm:.6g} mm",
        "sensitivity_case": {"gap_mm": gap_mm / 2.0,
                              "status": "required for qualification; not run by exporter"},
        "initial_mesh_min_gauss_jacobian_mm3": minimum_jacobian,
        "initial_geometry_gate": "positive HEX8 Gauss Jacobians; Starter must confirm no initial penetration and no coordinate adjustment",
        "starter_only_export": True,
    }
    return group, surface, card, metadata


def _rbody(body_id: int, master_id: int, slave_group: int, title: str) -> str:
    # Ispher=2 derives rigid-body inertia from the actual secondary-node masses.
    # The main node itself receives no synthetic mass or inertia.
    return "\n".join((
        f"/RBODY/{body_id}", title,
        f"{master_id:10d}{0:10d}{0:10d}{2:10d}{0.0:20g}{slave_group:10d}{0:10d}{0:10d}{0:10d}",
        f"{0.0:20g}{0.0:20g}{0.0:20g}",
        f"{0.0:20g}{0.0:20g}{0.0:20g}",
        f"{0:10d}{'':10}{0:10d}",
    ))


def _rwall_plane(wall_id: int, node_group: int, title: str,
                 z: float, *, normal_sign: float = 1.0) -> str:
    # Static 2026-format plane; contact is restricted to a boundary node set.
    return "\n".join((
        f"/RWALL/PLANE/{wall_id}", title,
        f"{0:10d}{0:10d}{node_group:10d}{0:10d}",
        paper._float_fields(0.0, 0.0, 0.0, 0.0) + f"{0:10d}",
        paper._float_fields(0.0, 0.0, z),
        paper._float_fields(0.0, 0.0, z + normal_sign),
    ))


def _rwall_cylinder(wall_id: int, node_group: int, master_id: int,
                    title: str, center: tuple[float, float, float],
                    radius: float) -> str:
    x, y, z = center
    # The line from its moving master to the second point fixes its axis along
    # global +Y. The node's /IMPVEL history translates it only along -Z/+Z.
    return "\n".join((
        f"/RWALL/CYL/{wall_id}", title,
        f"{master_id:10d}{0:10d}{node_group:10d}{0:10d}",
        paper._float_fields(0.0, 0.0, 2.0 * radius, 0.0) + f"{0:10d}",
        paper._float_fields(0.0, 0.0, 0.0, 0.0),
        paper._float_fields(x, y + 1.0, z),
    ))


def _type2_ties(mesh: hex_mesh.HexMesh,
                part_ids_by_role: dict[str, list[int]]) -> tuple[list[str], list[str], dict[str, Any]]:
    pairs_by_name: dict[str, list[dict[str, Any]]] = {}
    node_xyz = {int(node["id"]): tuple(float(v) for v in node["xyz_mm"])
                for node in mesh.nodes}
    for pair in mesh.interface_pairs:
        if pair.get("law") != "TYPE43" or pair.get("law_material_id") is not None:
            raise ValueError("unexpected interface binding: strip exporter expects unbound face pairs")
        a, b = pair["side_a"], pair["side_b"]
        xyz_a = sorted(tuple(round(v, 10) for v in node_xyz[node]) for node in a["nodes"])
        xyz_b = sorted(tuple(round(v, 10) for v in node_xyz[node]) for node in b["nodes"])
        if xyz_a != xyz_b:
            raise ValueError(f"noncoincident tie pair at interface {pair['interface']}:{pair['id']}")
        pairs_by_name.setdefault(str(pair["interface"]), []).append(pair)

    sections: list[str] = []
    tied_metadata: list[dict[str, Any]] = []
    secondary_nodes_by_interface: dict[str, set[int]] = {}
    ids_used: set[int] = set()
    for ordinal, interface_name in enumerate(sorted(pairs_by_name), start=1):
        pairs = pairs_by_name[interface_name]
        role_pairs = {(str(pair["side_a_role"]), str(pair["side_b_role"]))
                      for pair in pairs}
        if len(role_pairs) != 1:
            raise ValueError(f"interface {interface_name} has inconsistent material roles")
        side_a_role, side_b_role = next(iter(role_pairs))
        if (side_a_role == "starch") == (side_b_role == "starch"):
            raise ValueError(f"interface {interface_name} must pair starch with exactly one paper role")
        if side_a_role == "starch":
            secondary_side, master_role = "side_a", side_b_role
        else:
            secondary_side, master_role = "side_b", side_a_role
        if master_role not in {"liner", "medium"}:
            raise ValueError(f"interface {interface_name} has unsupported paper master role {master_role!r}")
        # The finite starch bridge is the TYPE2 secondary on both faces. This
        # keeps paper solids as mains and prevents one interface's main solid
        # nodes from becoming the other interface's secondary nodes.
        slave_nodes_all = {
            int(node) for pair in pairs for node in pair[secondary_side]["nodes"]
        }
        # Every finite-starch node stays on the TYPE2 secondary side. The
        # grippers are filtered to paper-only nodes in _deck, so no starch
        # secondary node also receives a rigid-body constraint.
        slave_nodes = sorted(slave_nodes_all)
        master_parts = part_ids_by_role.get(master_role, [])
        if not slave_nodes or not master_parts:
            raise ValueError(f"interface {interface_name} lacks slave nodes or master parts")
        slave_group = 20 + ordinal
        surface_id = 30 + ordinal
        tie_id = ordinal
        if tie_id in ids_used:
            raise AssertionError("duplicate TYPE2 identifier")
        ids_used.add(tie_id)
        secondary_nodes_by_interface[interface_name] = set(slave_nodes)
        # Spotflag=27 is kinematic like standard flag 5, with automatic
        # penalty fallback only for incompatible constraints. Altair documents
        # that this formulation adds no mass. The following defaults are used
        # only if that fallback is triggered: Stfac=1, Visc=0.05, Istf=2.
        # Pinned radioss2025 TYPE2 schema formats this row as
        # ``%20lg%20lg                    %10d`` (70 columns total): Istf
        # begins at column 61. Keep the native 20-space gap exactly; an extra
        # blank 20-column field puts Istf past the supported record width.
        # dsearch=0 requests the engine's native search distance.
        card = "\n".join((
            f"/INTER/TYPE2/{tie_id}", f"Perfectly tied source-recipe interface: {interface_name}",
            f"{slave_group:10d}{surface_id:10d}{0:10d}{27:10d}{0:10d}{0:10d}{0:10d}{'':10}{0.0:20g}",
            f"{1.0:20g}{0.05:20g}{'':20}{2:10d}",
            _group(slave_group, f"secondary nodes for {interface_name}", slave_nodes),
            f"/SURF/PART/EXT/{surface_id}\nmaster solid-part external faces for {interface_name}\n{_ids(master_parts)}",
        ))
        sections.append(card)
        tied_metadata.append({
            "interface": interface_name,
            "type": "/INTER/TYPE2",
            "behavior": "kinematic perfect tie; no relative slip or opening; no debonding",
            "pair_count": len(pairs),
            "slave_material_role": "starch",
            "master_material_role": master_role,
            "secondary_side": secondary_side,
            "candidate_secondary_node_count": len(slave_nodes_all),
            "excluded_gripper_slave_node_count": 0,
            "secondary_nodes_omitted": 0,
            "all_candidate_secondary_nodes_tied": True,
            "slave_node_count": len(slave_nodes),
            "master_part_ids": master_parts,
            "spotflag": 27,
            "mass_addition": "native formulation documented to add no mass",
            "incompatible_kinematics": "native automatic penalty fallback; Starter warnings are a run rejection gate",
            "fallback_parameters": {
                "stfac": 1.0, "visc": 0.05, "istf": 2,
                "status": "native documented defaults; relevant only after fallback",
            },
            "search_distance_mm": 0.0,
        })
    names = sorted(secondary_nodes_by_interface)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = secondary_nodes_by_interface[left_name] & secondary_nodes_by_interface[right_name]
            if overlap:
                raise ValueError(
                    f"TYPE2 secondary groups chain across {left_name} and {right_name}; "
                    f"shared node IDs include {sorted(overlap)[:8]}"
                )
    if len(tied_metadata) != 2 or {item["interface"] for item in tied_metadata} != {
            "medium_starch", "starch_liner"}:
        raise ValueError("expected medium-starch and starch-liner interfaces")
    return sections, [], {"interfaces": tied_metadata}


def _paper_gripper_memberships(mesh: hex_mesh.HexMesh) -> tuple[dict[str, set[int]], dict[str, Any]]:
    """Return paper-only rigid-gripper members and explicit starch exclusions."""
    roles_by_node: dict[int, set[str]] = {}
    for element in mesh.elements:
        role = str(element.material_role)
        for node_id in element.nodes:
            roles_by_node.setdefault(int(node_id), set()).add(role)
    if any(len(roles) != 1 for roles in roles_by_node.values()):
        raise ValueError("gripper membership requires material-separated coincident interface nodes")

    memberships: dict[str, set[int]] = {}
    side_metadata: dict[str, Any] = {}
    for side in ("left", "right"):
        raw = {int(node) for node in mesh.node_sets[f"grip_{side}"]}
        if not raw or raw - roles_by_node.keys():
            raise ValueError(f"{side} gripper contains empty or unmeshed node membership")
        by_role = {role: {node for node in raw if roles_by_node[node] == {role}}
                   for role in ("liner", "medium", "starch")}
        unexpected = raw - set().union(*by_role.values())
        if unexpected:
            raise ValueError(f"{side} gripper contains unsupported material roles")
        paper_nodes = by_role["liner"] | by_role["medium"]
        if not paper_nodes:
            raise ValueError(f"{side} gripper has no paper nodes")
        memberships[side] = paper_nodes
        side_metadata[side] = {
            "raw_geometry_node_count": len(raw),
            "rigidbody_paper_node_count": len(paper_nodes),
            "rigidbody_material_roles": ["liner", "medium"],
            "starch_nodes_excluded_from_rigidbody": len(by_role["starch"]),
        }
    if memberships["left"] & memberships["right"]:
        raise ValueError("left and right paper-only gripper groups overlap")
    return memberships, {
        "membership_policy": "RBODY secondary groups contain paper nodes only; all starch nodes remain TYPE2 secondary nodes",
        "sides": side_metadata,
        "type2_rigidbody_main_compatibility": {
            "finding": "paper solids are TYPE2 main surfaces and may also provide rigid-body main nodes; starch is the TYPE2 secondary and is not in either RBODY slave group",
            "official_type2_reference": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/inter_type2_starter_r.htm",
            "pinned_engine_source": "engine/source/interfaces/interf/i2_imp1.F, NDM=6 main-segment branch calls UPDKB_RB",
            "qualification": "source/card semantics checked; Starter-only compatibility for this exported case remains pending",
        },
    }


def _unique_node_center(mesh: hex_mesh.HexMesh, node_ids: Iterable[int]) -> tuple[float, float, float]:
    lookup = {int(node["id"]): tuple(float(v) for v in node["xyz_mm"])
              for node in mesh.nodes}
    unique = {tuple(round(v, 10) for v in lookup[int(node_id)]) for node_id in node_ids}
    if not unique:
        raise ValueError("cannot find gripper center for an empty node set")
    return tuple(sum(point[axis] for point in unique) / len(unique) for axis in range(3))


def _face_nodes(faces: Iterable[Any]) -> set[int]:
    return {int(node) for face in faces for node in face.nodes}


def _rigid_wing_sweep_envelope(mesh: hex_mesh.HexMesh, pivot_x_mm: float,
                               pivot_z_mm: float) -> dict[str, float | int]:
    """Bound every rigid rotation of the right-hand board wing about Y."""
    center_x = float(mesh.score["centerline_x_mm"])
    right_nodes = [node for node in mesh.nodes
                   if float(node["xyz_mm"][0]) >= center_x - 1.0e-9]
    if not right_nodes:
        raise ValueError("cannot bound the right-wing fold sweep without nodes")
    radii = [math.hypot(float(node["xyz_mm"][0]) - pivot_x_mm,
                        float(node["xyz_mm"][2]) - pivot_z_mm)
             for node in right_nodes]
    radius_max = max(radii)
    return {
        "right_wing_node_count": len(right_nodes),
        "max_radius_from_fold_pivot_mm": radius_max,
        "max_z_under_any_rigid_rotation_mm": pivot_z_mm + radius_max,
    }


def _make_motion(mesh: hex_mesh.HexMesh, *, punch_master: int,
                 left_master: int, right_master: int,
                 t_score: float, t_hold: float,
                 fold_start: float, fold_duration: float,
                 fold_angle_deg: float, pivot_z_mm: float,
                 park_center_z_mm: float, park_speed_mm_s: float,
                 fold_end_dwell_s: float, post_release_observation_s: float,
                 sweep_envelope: dict[str, float | int]) -> tuple[list[str], list[str], dict[str, Any]]:
    score = mesh.score
    top_nodes = _face_nodes(mesh.surface_sets["score_punch_contact"])
    if not top_nodes:
        raise ValueError("score punch contact surface is empty")
    node_xyz = {int(node["id"]): tuple(float(v) for v in node["xyz_mm"])
                for node in mesh.nodes}
    center_x = float(score["centerline_x_mm"])
    center_y = 0.5 * float(mesh.parameters["width_mm"])
    top_z = max(node_xyz[node][2] for node in top_nodes)
    radius = float(mesh.tool_targets["punch_radius_mm"])
    indentation = float(mesh.tool_targets["indentation_mm"])
    if radius <= 0.0 or indentation <= 0.0 or park_speed_mm_s <= 0.0:
        raise ValueError("punch radius and indentation must be positive")
    start_punch = (center_x, center_y, top_z + radius)
    punch_end = 2.0 * t_score + t_hold
    peak_speed = indentation / (0.9 * t_score)
    park_distance = park_center_z_mm - start_punch[2]
    if park_distance <= 0.0:
        raise ValueError("punch park target must be above its initial position")
    # The velocity history ramps over 10%, holds over 80%, then ramps down;
    # its integrated area is exactly 0.9 * peak * duration.
    park_duration = park_distance / (0.9 * park_speed_mm_s)
    park_end = punch_end + park_duration
    points = [
        (0.0, 0.0), (0.1 * t_score, -1.0), (0.9 * t_score, -1.0), (t_score, 0.0),
        (t_score + t_hold, 0.0), (t_score + t_hold + 0.1 * t_score, 1.0),
        (t_score + t_hold + 0.9 * t_score, 1.0), (punch_end, 0.0),
        (punch_end + 0.1 * park_duration, park_speed_mm_s / peak_speed),
        (punch_end + 0.9 * park_duration, park_speed_mm_s / peak_speed),
        (park_end, 0.0),
    ]
    fold_end = fold_start + fold_duration
    release_start = fold_end + fold_end_dwell_s
    total_time = release_start
    final_time = release_start + post_release_observation_s
    if park_end > fold_start:
        raise ValueError("fold must start after the punch reaches its park target")
    points.extend(((fold_start, 0.0), (release_start, 0.0)))
    fold_angle = math.radians(fold_angle_deg)
    right_center = _unique_node_center(mesh, mesh.node_sets["grip_right"])
    pivot = (center_x, center_y, pivot_z_mm)
    x0, y0, z0 = right_center
    rx, rz = x0 - pivot[0], z0 - pivot[2]
    sample_count = int(round(abs(fold_angle_deg) / FOLD_PATH_INTERVAL_DEG))
    if sample_count < 1 or not math.isclose(sample_count * FOLD_PATH_INTERVAL_DEG,
                                             abs(fold_angle_deg), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("fold angle must be divisible by the 5 degree path interval")
    x_points: list[tuple[float, float]] = [(0.0, 0.0), (fold_start, 0.0)]
    z_points: list[tuple[float, float]] = [(0.0, 0.0), (fold_start, 0.0)]
    angle_points: list[tuple[float, float]] = [(0.0, 0.0), (fold_start, 0.0)]
    for index in range(sample_count + 1):
        fraction = index / sample_count
        theta = fold_angle * fraction
        time = fold_start + fold_duration * fraction
        x_new = pivot[0] + math.cos(theta) * rx + math.sin(theta) * rz
        z_new = pivot[2] - math.sin(theta) * rx + math.cos(theta) * rz
        x_points.append((time, x_new - x0))
        z_points.append((time, z_new - z0))
        angle_points.append((time, theta))
    if total_time < punch_end or post_release_observation_s <= 0.0 or fold_end_dwell_s <= 0.0:
        raise ValueError("release requires punch withdrawal, fold-end dwell, and observation time")
    for function_points in (x_points, z_points, angle_points):
        function_points.append((release_start, function_points[-1][1]))
    if len({point[0] for point in x_points}) != len(x_points):
        # Remove the two zero-angle samples at t=fold_start while preserving
        # the same exactly prescribed geometric transform.
        x_points = [(t, x) for i, (t, x) in enumerate(x_points)
                    if not (i > 1 and t == fold_start)]
        z_points = [(t, z) for i, (t, z) in enumerate(z_points)
                    if not (i > 1 and t == fold_start)]
        angle_points = [(t, a) for i, (t, a) in enumerate(angle_points)
                        if not (i > 1 and t == fold_start)]
    functions = [
        _func(1, "score, withdraw, park, and hold punch velocity", points),
        _func(2, "right boundary gripper X displacement", x_points),
        _func(3, "right boundary gripper Z displacement", z_points),
        _func(4, "right boundary gripper rotation about global Y", angle_points),
    ]
    drives = [
        _imposed_velocity(1, "Z", 4, peak_speed, total_time),
        _imposed(2, "X", 8, fold_start, total_time),
        _imposed(3, "Z", 8, fold_start, total_time),
        _imposed(4, "YY", 8, fold_start, total_time),
    ]
    return functions, drives, {
        "score": {
            "tool": "analytic /RWALL/CYL", "radius_mm": radius,
            "axis": "+global Y", "initial_center_mm": list(start_punch),
            "stroke_mm": indentation, "direction": "-global Z then +global Z",
            "stroke_duration_s": t_score, "stationary_hold_s": t_hold,
            "withdraw_complete_s": punch_end, "peak_speed_mm_per_s": peak_speed,
            "speed_identity": "fixture choice; not a measured press speed",
            "park": {
                "trajectory": "vertical +Z trapezoidal-velocity clearance path, then zero-velocity dwell",
                "start_s": punch_end, "end_s": park_end,
                "initial_center_mm": [center_x, center_y, start_punch[2]],
                "target_center_mm": [center_x, center_y, park_center_z_mm],
                "distance_mm": park_distance,
                "speed_mm_per_s": park_speed_mm_s,
                "speed_identity": "declared boundary-tool clearance choice; not a measured manufacturing rate",
                "swept_envelope": sweep_envelope,
                "clearance_above_rigid_wing_envelope_mm": park_center_z_mm - radius - float(sweep_envelope["max_z_under_any_rigid_rotation_mm"]),
                "clearance_margin_mm": PUNCH_PARK_CLEARANCE_MM,
            },
            "support": "two analytic /RWALL/PLANE walls with a 6 mm center channel",
        },
        "fold": {
            "actuator": "right boundary RBody gripper only",
            "axis": "+global Y", "target_angle_deg": fold_angle_deg,
            "start_s": fold_start, "duration_s": fold_duration,
            "pivot_mm": list(pivot),
            "pivot_z_identity": "geometric mid-caliper fixture assumption; not a measured crease neutral axis",
            "path_interpolation": "piecewise-linear /FUNCT points at 5 degree rigid-grip positions",
            "end_s": fold_end,
            "endpoint_dwell_s": fold_end_dwell_s,
            "release_start_s": release_start,
            "release_observation_end_s": final_time,
            "prescribed_domains": ["right grip boundary node set", "analytic punch master node"],
            "interior_kinematic_angles": False,
            "endpoint_velocity_at_release": "zero; final displacement is held constant through the explicit dwell",
        },
        "engine_1_end_time_s": total_time,
        "release_observation_end_time_s": final_time,
        "end_time_s": final_time,
        "right_grip_master_initial_mm": list(right_center),
        "left_grip_master_initial_mm": list(_unique_node_center(mesh, mesh.node_sets["grip_left"])),
        "punch_master_id": punch_master,
        "left_grip_master_id": left_master,
        "right_grip_master_id": right_master,
    }


def _integration_schedule(mesh: hex_mesh.HexMesh, engine_mode: str,
                         fold_angle_deg: float) -> dict[str, Any]:
    if engine_mode == "explicit":
        return {
            "mode": "explicit",
            "t_score_s": SCORE_STROKE_DURATION_S,
            "t_hold_s": SCORE_HOLD_DURATION_S,
            "pre_fold_dwell_s": EXPLICIT_PRE_FOLD_DWELL_S,
            "fold_duration_s": FOLD_DURATION_S,
            "fold_end_dwell_s": FOLD_END_DWELL_S,
            "post_release_observation_s": POST_RELEASE_OBSERVATION_S,
            "punch_park_speed_mm_per_s": None,
            "history_interval_s": HISTORY_INTERVAL_S,
            "animation_interval_s": ANIMATION_INTERVAL_S,
            "rate_identity": "short parser/integration fixture schedule; not a manufacturing rate",
        }
    if engine_mode == "implicit-slow":
        indentation = float(mesh.tool_targets["indentation_mm"])
        t_score = indentation / (0.9 * IMPLICIT_SCORE_SPEED_MM_S)
        t_hold = IMPLICIT_SCORE_HOLD_S
        return {
            "mode": "implicit-slow",
            "t_score_s": t_score,
            "t_hold_s": t_hold,
            "pre_fold_dwell_s": IMPLICIT_PRE_FOLD_DWELL_S,
            "fold_duration_s": abs(fold_angle_deg) / IMPLICIT_FOLD_RATE_DEG_S,
            "fold_end_dwell_s": FOLD_END_DWELL_S,
            "post_release_observation_s": POST_RELEASE_OBSERVATION_S,
            "punch_park_speed_mm_per_s": PUNCH_PARK_SPEED_MM_S,
            "history_interval_s": IMPLICIT_HISTORY_INTERVAL_S,
            "animation_interval_s": IMPLICIT_ANIMATION_INTERVAL_S,
            "score_peak_speed_mm_per_s": IMPLICIT_SCORE_SPEED_MM_S,
            "fold_rate_deg_per_s": IMPLICIT_FOLD_RATE_DEG_S,
            "release_initial_dt_s": IMPLICIT_RELEASE_DT_S,
            "rate_identity": "declared engineering stage rates; not measured manufacturing conditions or rate acceleration",
        }
    raise ValueError(f"unsupported engine_mode {engine_mode!r}; expected one of {ENGINE_MODES}")


def _deck(mesh: hex_mesh.HexMesh, parameters: dict[str, Any],
          part_ids: dict[str, int], *, stem: str, engine_mode: str,
          fold_pivot_z_mm: float | None = None,
          fold_angle_deg: float = FOLD_ANGLE_DEG) -> tuple[str, str, str, dict[str, Any]]:
    pivot_z = (0.5 * float(mesh.parameters["caliper_mm"])
               if fold_pivot_z_mm is None else float(fold_pivot_z_mm))
    if not math.isfinite(pivot_z):
        raise ValueError("fold pivot z must be finite")
    schedule = _integration_schedule(mesh, engine_mode, fold_angle_deg)
    center_x = float(mesh.score["centerline_x_mm"])
    center_y = 0.5 * float(mesh.parameters["width_mm"])
    top_nodes_for_park = _face_nodes(mesh.surface_sets["score_punch_contact"])
    node_xyz = {int(node["id"]): tuple(float(v) for v in node["xyz_mm"])
                for node in mesh.nodes}
    top_z_for_park = max(node_xyz[node][2] for node in top_nodes_for_park)
    punch_radius = float(mesh.tool_targets["punch_radius_mm"])
    start_punch_center_z = top_z_for_park + punch_radius
    sweep_envelope = _rigid_wing_sweep_envelope(mesh, center_x, pivot_z)
    fold_envelope_z = float(sweep_envelope["max_z_under_any_rigid_rotation_mm"])
    park_center_z = fold_envelope_z + punch_radius + PUNCH_PARK_CLEARANCE_MM
    park_distance = park_center_z - start_punch_center_z
    if park_distance <= 0.0:
        raise ValueError("rigid-wing fold envelope does not permit a positive punch park path")
    t_score = float(schedule["t_score_s"])
    t_hold = float(schedule["t_hold_s"])
    score_peak_speed = float(mesh.tool_targets["indentation_mm"]) / (0.9 * t_score)
    park_speed = (float(schedule["punch_park_speed_mm_per_s"])
                  if schedule["punch_park_speed_mm_per_s"] is not None
                  else score_peak_speed)
    if park_speed <= 0.0:
        raise ValueError("punch park speed must be positive")
    park_duration = park_distance / (0.9 * park_speed)
    punch_return_time = 2.0 * t_score + t_hold
    park_end = punch_return_time + park_duration
    fold_start = park_end + float(schedule["pre_fold_dwell_s"])
    fold_duration = float(schedule["fold_duration_s"])
    fold_end = fold_start + fold_duration
    release_start = fold_end + float(schedule["fold_end_dwell_s"])
    final_time = release_start + float(schedule["post_release_observation_s"])
    schedule.update({
        "punch_return_complete_s": punch_return_time,
        "punch_park_start_s": punch_return_time,
        "punch_park_end_s": park_end,
        "punch_park_distance_mm": park_distance,
        "punch_park_speed_mm_per_s": park_speed,
        "punch_park_duration_s": park_duration,
        "punch_park_target_center_z_mm": park_center_z,
        "fold_start_s": fold_start,
        "fold_end_s": fold_end,
        "release_start_s": release_start,
        "engine_1_end_time_s": release_start,
        "engine_2_end_time_s": final_time,
        "end_time_s": final_time,
        "rate_identity": f"{schedule['rate_identity']}; punch park speed is a declared boundary-tool clearance choice",
    })
    material_transfer = _material_transfer_module()
    role_mat = {"liner": 1, "medium": 2, "starch": 3}
    role_prop = {"liner": 1, "medium": 1, "starch": 2}
    part_ids_by_role: dict[str, list[int]] = {role: [] for role in role_mat}
    for part_name, part_id in part_ids.items():
        role = next(element.material_role for element in mesh.elements if element.part == part_name)
        part_ids_by_role[role].append(part_id)

    tie_cards, _, ties_meta = _type2_ties(mesh, part_ids_by_role)
    liner_material_card = material_transfer.law_card(role_mat["liner"], "liner")
    medium_material_card = material_transfer.law_card(role_mat["medium"], "medium")
    liner_source_record = material_transfer.source_record("liner")
    medium_source_record = material_transfer.source_record("medium")
    max_node = max(node_xyz)
    left_master, right_master, punch_master = max_node + 1, max_node + 2, max_node + 3
    paper_grippers, gripper_membership_meta = _paper_gripper_memberships(mesh)
    left_nodes = paper_grippers["left"]
    right_nodes = paper_grippers["right"]
    top_contact = _face_nodes(mesh.surface_sets["score_punch_contact"])
    bottom_faces = mesh.surface_sets["lower_liner_outer"]
    center = float(mesh.score["centerline_x_mm"])
    channel = float(mesh.tool_targets["channel_width_mm"])
    half_channel = 0.5 * channel
    x_eps = 1.0e-8
    lower_left_candidates = {node for face in bottom_faces for node in face.nodes
                             if node_xyz[node][0] <= center - half_channel + x_eps}
    lower_right_candidates = {node for face in bottom_faces for node in face.nodes
                              if node_xyz[node][0] >= center + half_channel - x_eps}
    gripped_nodes = left_nodes | right_nodes
    lower_left_excluded = lower_left_candidates & gripped_nodes
    lower_right_excluded = lower_right_candidates & gripped_nodes
    lower_left = lower_left_candidates - gripped_nodes
    lower_right = lower_right_candidates - gripped_nodes
    if lower_left & lower_right:
        raise ValueError("left and right scoring anvil node groups overlap")
    if not lower_left or not lower_right:
        raise ValueError("failed to form two lower support rails outside the score channel")
    groups = [
        _group(1, "top liner nodes contacted by score punch", top_contact),
        _group(2, "lower liner nodes supported left of score channel", lower_left),
        _group(3, "lower liner nodes supported right of score channel", lower_right),
        _group(4, "moving score punch master node", (punch_master,)),
        _group(5, "left boundary gripper rigid-body slaves", left_nodes),
        _group(6, "right boundary gripper rigid-body slaves", right_nodes),
        _group(7, "fixed left boundary gripper master", (left_master,)),
        _group(8, "moving right boundary gripper master", (right_master,)),
    ]
    bcs = [_bcs(1, "fixed left boundary gripper", "111 111", 7)]
    bcs.extend((
        _bcs(2, "restrain right gripper spanwise translation and out-of-plane rotation",
             "010 101", 8),
        _bcs(3, "restrain punch axial and spanwise translation", "110 000", 4),
    ))

    motions_f, motions_i, motion_meta = _make_motion(
        mesh, punch_master=punch_master, left_master=left_master,
        right_master=right_master, t_score=schedule["t_score_s"],
        t_hold=schedule["t_hold_s"], fold_start=fold_start,
        fold_duration=fold_duration, fold_angle_deg=fold_angle_deg,
        pivot_z_mm=pivot_z, park_center_z_mm=park_center_z,
        park_speed_mm_s=park_speed,
        fold_end_dwell_s=float(schedule["fold_end_dwell_s"]),
        post_release_observation_s=float(schedule["post_release_observation_s"]),
        sweep_envelope=sweep_envelope,
    )
    motion_meta["gripper_membership"] = gripper_membership_meta
    contact_group, contact_surface, type7_contact_card, self_contact_meta = _type7_self_contact(
        mesh, end_time_s=final_time)
    groups.append(contact_group)
    motion_meta["support_rails"] = {
        "left": {
            "surface": "lower_liner_outer left of score channel",
            "candidate_node_count": len(lower_left_candidates),
            "excluded_gripped_node_count": len(lower_left_excluded),
            "contact_node_count": len(lower_left),
        },
        "right": {
            "surface": "lower_liner_outer right of score channel",
            "candidate_node_count": len(lower_right_candidates),
            "excluded_gripped_node_count": len(lower_right_excluded),
            "contact_node_count": len(lower_right),
        },
        "overlap_policy": "gripped boundary nodes are excluded from rigid-wall support contact",
    }
    funcs = "\n".join(motions_f)
    drives = "\n".join(motions_i)
    punch_xyz = motion_meta["score"]["initial_center_mm"]
    lower_z = min(node_xyz[node][2] for node in _face_nodes(bottom_faces))
    walls = "\n".join((
        _rwall_plane(1, 2, "left scoring anvil rail", lower_z),
        _rwall_plane(2, 3, "right scoring anvil rail", lower_z),
        _rwall_cylinder(3, 1, punch_master, "1 mm radius scoring punch",
                        tuple(punch_xyz), float(mesh.tool_targets["punch_radius_mm"])),
    ))
    bodies = "\n".join((
        _rbody(1, left_master, 5, "ideal fixed left boundary clamp"),
        _rbody(2, right_master, 6, "ideal driven right boundary gripper"),
    ))

    elements_by_role = {role: [e for e in mesh.elements if e.material_role == role]
                        for role in role_mat}
    if any(not rows for rows in elements_by_role.values()):
        raise ValueError("strip geometry is missing one or more material roles")
    # Pick deterministic element histories around the score; animation remains
    # available for the full mesh through the engine output.
    score_x = float(mesh.score["centerline_x_mm"])
    score_elements = []
    for role in ("liner", "medium", "starch"):
        candidates = []
        for element in elements_by_role[role]:
            xyz = [node_xyz[node] for node in element.nodes]
            centroid = tuple(sum(p[k] for p in xyz) / 8.0 for k in range(3))
            if abs(centroid[0] - score_x) <= 1.0:
                candidates.append((centroid, element.id))
        candidates.sort(key=lambda item: (item[0][1], item[0][2], item[1]))
        if not candidates:
            candidates = [(None, elements_by_role[role][0].id)]
        stride = max(1, len(candidates) // 8)
        score_elements.extend(element_id for _, element_id in candidates[::stride][:8])
    center_top_node = min(top_contact, key=lambda node: (
        abs(node_xyz[node][0] - score_x) +
        abs(node_xyz[node][1] - 0.5 * float(mesh.parameters["width_mm"])), node))
    center_bottom_nodes = _face_nodes(mesh.surface_sets["score_anvil_contact"])
    center_bottom_node = min(center_bottom_nodes, key=lambda node: (
        abs(node_xyz[node][0] - score_x) +
        abs(node_xyz[node][1] - 0.5 * float(mesh.parameters["width_mm"])), node))
    th_nodes = [left_master, right_master, punch_master, center_top_node, center_bottom_node]
    th = "\n".join((
        "/TH/BRIC/1", "selected score-zone solid element histories",
        "".join(f"{name:<10}" for name in ("STRESS", "STRAIN", "IE")),
        "\n".join(f"{element_id:10d}" for element_id in score_elements),
        "/TH/NODE/2", "gripper, punch, and score-center nodal history",
        "".join(f"{name:<10}" for name in ("DX", "DY", "DZ", "REACX", "REACY", "REACZ")),
        "\n".join(f"{node:10d}{0:10d}" for node in th_nodes),
    ))

    # All geometry nodes plus three separate tool/gripper control nodes.
    all_nodes = list(mesh.nodes)
    left_xyz = motion_meta["left_grip_master_initial_mm"]
    right_xyz = motion_meta["right_grip_master_initial_mm"]
    all_nodes.extend((
        {"id": left_master, "xyz_mm": left_xyz},
        {"id": right_master, "xyz_mm": right_xyz},
        {"id": punch_master, "xyz_mm": punch_xyz},
    ))
    node_rows = "\n".join(
        f"{int(node['id']):10d}{float(node['xyz_mm'][0]):20.10g}"
        f"{float(node['xyz_mm'][1]):20.10g}{float(node['xyz_mm'][2]):20.10g}"
        for node in all_nodes
    )
    part_cards: list[str] = []
    for part_name, part_id in sorted(part_ids.items(), key=lambda item: item[1]):
        role = next(element.material_role for element in mesh.elements if element.part == part_name)
        part_cards.append(_part(part_id, f"{part_name} ({role})", role_mat[role], role_prop[role]))
    brick_cards: list[str] = []
    for part_name, part_id in sorted(part_ids.items(), key=lambda item: item[1]):
        part_elements = [element for element in mesh.elements if element.part == part_name]
        if not part_elements:
            raise ValueError(f"empty brick part: {part_name}")
        # The part ID belongs in the /BRICK/<part_ID> keyword; every data row
        # then contains only brick_ID followed by its eight node IDs.
        brick_rows = "\n".join(
            paper._int_fields(element.id, *element.nodes) for element in part_elements
        )
        brick_cards.append(f"/BRICK/{part_id}\n{brick_rows}")
    material_cards = "\n".join((
        liner_material_card,
        medium_material_card,
        "/MAT/LAW1/3\nDried starch apparent-slope elastic volume; nu/rho assumed\n"
        f"{1.5e-9:20.9g}\n{400.0:20.9g}{0.3:20.9g}",
    ))
    properties = "\n".join((_type6(1, "full-integration orthotropic paperboard HEX8"),
                            _type14(2)))
    starter = f"""#RADIOSS STARTER
# OpenCourant {RELEASE}; source {SOURCE_COMMIT}
# Integration fixture only: no target-grade, score, adhesive-failure, or full-box calibration.
/BEGIN/
{stem}
{2021:10d}{0:10d}
{paper._float_fields(1.0, 1.0, 1.0)}
{paper._float_fields(1.0, 1.0, 1.0)}
/TITLE
Four-pitch corrugated strip: score, park, boundary fold, self-contact and release
/ANALY
{0:41d}
/DEF_SOLID
{paper._int_fields(0, 0, 2)}
{material_cards}
{properties}
{chr(10).join(part_cards)}
/NODE
{node_rows}
{chr(10).join(brick_cards)}
{chr(10).join(groups)}
{chr(10).join(bcs)}
{chr(10).join(tie_cards)}
{contact_surface}
{type7_contact_card}
{bodies}
{walls}
{funcs}
{drives}
{th}
/ANIM/VERS
36
/END
"""
    if engine_mode == "explicit":
        integration_controls = f"""/DT/NODA/STOP
{DT_SCALE:.10g} {DT_MIN_S:.10g}"""
        engine_description = "explicit; no mass scaling requested"
        timestep_control = {"keyword": "/DT/NODA/STOP", "scale": DT_SCALE,
                            "minimum_dt_s": DT_MIN_S, "mass_scaling": False}
    else:
        integration_controls = f"""/IMPL/NONL/01
20 2 {IMPLICIT_FORCE_TOLERANCE:.10g} {IMPLICIT_FORCE_TOLERANCE:.10g}
/IMPL/DTINI
{IMPLICIT_INITIAL_DT_S:.10g}
/IMPL/DT/1
20 1.2 100 0.6666667
/IMPL/DT/STOP
{IMPLICIT_MIN_DT_S:.10g} {IMPLICIT_MAX_DT_S:.10g}
/IMPL/SOLV/02
1 0 3 1.0e-10
/IMPL/PRINT/NONLIN/1"""
        engine_description = "implicit nonlinear fold attempt; direct MUMPS; provisional DT controller; not qualified; no mass scaling"
        timestep_control = {
            "keyword": "/IMPL/DTINI plus /IMPL/DT/1 and /IMPL/DT/STOP",
            "initial_dt_s": IMPLICIT_INITIAL_DT_S,
            "minimum_dt_s": IMPLICIT_MIN_DT_S,
            "maximum_dt_s": IMPLICIT_MAX_DT_S,
            "dt_1_record": "20 1.2 100 0.6666667",
            "dt_1_record_semantics": {
                "NL_DTP": 20,
                "SCAL_DTP": 1.2,
                "NL_DTN": 100,
                "SCAL_DTN": 0.6666667,
                "identity": "provisional integration controller; permits at most 100 Newton iterations per attempt before cutback and allows up to 1.2 timestep growth after convergence; does not relax the force tolerance. The increase from 50 is an extrapolation from strip10's observed residual ratio 0.836 and residual 4.117e-5 at iteration 49, which projects about 71 iterations to reach the unchanged 1e-6 force gate; this projection is not a successful run",
            },
            "nonlinear_control_record": f"20 2 {IMPLICIT_FORCE_TOLERANCE:.10g} {IMPLICIT_FORCE_TOLERANCE:.10g}",
            "force_residual_tolerance": IMPLICIT_FORCE_TOLERANCE,
            "solver": "direct MUMPS", "linear_relative_tolerance": 1.0e-10,
            "mass_scaling": False,
        }
    engine_1 = f"""#RADIOSS ENGINE
/VERS/41
{integration_controls}
/ANIM/DT
0.0 {schedule['animation_interval_s']:.10g}
/TFILE/4
{schedule['history_interval_s']:.10g}
/PRINT/-1000
/ANIM/BRICK/TENS/STRESS
/ANIM/BRICK/TENS/STRAIN
/ANIM/BRICK/EPSP
/ANIM/ELEM/ENER
/ANIM/VECT/DISP
/ANIM/VECT/VEL
/MON/ON
/TH/TITLE
/RUN/{stem}/1
{motion_meta['engine_1_end_time_s']:.10g}
"""
    if engine_mode == "implicit-slow":
        release_mode = "implicit-dynamic"
        release_controls = f"""/IMPL/DYNA/2
0.5 0.25
/IMPL/NONL/01
20 2 {IMPLICIT_FORCE_TOLERANCE:.10g} {IMPLICIT_FORCE_TOLERANCE:.10g}
/IMPL/DTINI
{IMPLICIT_RELEASE_DT_S:.10g}
/IMPL/DT/1
20 1.2 100 0.6666667
/IMPL/DT/STOP
{IMPLICIT_MIN_DT_S:.10g} {IMPLICIT_RELEASE_DT_S:.10g}
/IMPL/SOLV/02
1 0 3 1.0e-10
/IMPL/PRINT/NONLIN/1"""
        release_control_metadata = {
            "integrator": "/IMPL/DYNA/2 general Newmark",
            "record": "0.5 0.25",
            "gamma": 0.5,
            "beta": 0.25,
            "interpretation": "average-acceleration Newmark candidate; alpha-HHT is bypassed, so the exporter adds no algorithmic HHT dissipation or physical damping",
            "initial_dt_s": IMPLICIT_RELEASE_DT_S,
            "maximum_dt_s": IMPLICIT_RELEASE_DT_S,
            "minimum_dt_s": IMPLICIT_MIN_DT_S,
            "nonlinear_force_tolerance": IMPLICIT_FORCE_TOLERANCE,
            "solver": "direct MUMPS",
            "mass_scaling": False,
            "status": "keyword and source parsing verified; this staged release path is unrun and unqualified",
        }
        release_description = "implicit dynamic average-acceleration Newmark release candidate; direct MUMPS; unrun; no mass scaling or physical damping"
    else:
        release_mode = "explicit"
        release_controls = f"""/DT/NODA/STOP
{DT_SCALE:.10g} {DT_MIN_S:.10g}"""
        release_control_metadata = {
            "integration": "explicit, no mass scaling requested",
            "scale": DT_SCALE,
            "minimum_dt_s": DT_MIN_S,
            "mass_scaling": False,
            "status": "unexecuted option; Starter11 predicts at least 97 million solid-element cycles for one second at the 0.9-scaled minimum solid timestep, while the minimum nodal estimate is 6.468e-28 s; actual Engine active dt is unknown",
        }
        release_description = "explicit release option; no mass scaling requested; unexecuted due Starter-only timestep forecast"
    release_engine = f"""#RADIOSS ENGINE
/VERS/41
{release_controls}
/RBODY/OFF
{left_master:10d}{right_master:10d}
/ANIM/DT
0.0 {schedule['animation_interval_s']:.10g}
/TFILE/4
{schedule['history_interval_s']:.10g}
/PRINT/-1000
/ANIM/BRICK/TENS/STRESS
/ANIM/BRICK/TENS/STRAIN
/ANIM/BRICK/EPSP
/ANIM/ELEM/ENER
/ANIM/VECT/DISP
/ANIM/VECT/VEL
/MON/ON
/TH/TITLE
/RUN/{stem}/2
{motion_meta['release_observation_end_time_s']:.10g}
"""
    material_identity = {
        "medium": {
            "material_id": role_mat["medium"], "law": "/MAT/LAW112",
            "source_map": "matter/examples/cardboard/opencourant/paper-parameters.json",
            "identity": "126 gsm NSSC medium; published elasticity, tensile yield anchors, and digitized bilinear tension-fit endpoint transferred to nominal LAW112 tensile C values",
            "source_record": medium_source_record,
            "law_card_sha256": hashlib.sha256(medium_material_card.encode("utf-8")).hexdigest(),
            "calibration_scope": parameters.get("calibration_scope"),
        },
        "liner": {
            "material_id": role_mat["liner"], "law": "/MAT/LAW112",
            "source_map": "Haj-Ali 2009 Table 2 + Table 1 + digitized bilinear fit endpoint; density proxy from separate Popil coupon",
            "identity": "205 gsm liner; published elasticity, tensile yield anchors, and digitized bilinear tension-fit endpoint transferred to nominal LAW112 tensile C values",
            "source_record": liner_source_record,
            "law_card_sha256": hashlib.sha256(liner_material_card.encode("utf-8")).hexdigest(),
            "density_Mg_per_mm3": 7.66e-10,
            "density_status": "separate coupon cross-source proxy, not a matched sheet measurement",
        },
        "starch": {
            "law": "/MAT/LAW1 plus /PROP/TYPE14 finite solid volume",
            "elastic_modulus_MPa": 400.0,
            "modulus_status": "source-bound apparent slope for fully dried starch, Popil et al. 2007 / repeated in the 2009 explicit glue study",
            "poisson_ratio": 0.3, "poisson_status": "FE assumption, not source measurement",
            "density_Mg_per_mm3": 1.5e-9, "density_status": "numerical assumption, no matching published starch density",
            "fracture_model": "none",
        },
    }
    fixture = {
        "schema": SCHEMA,
        "id": stem,
        "starter": f"{stem}_0000.rad",
        "engines": [f"{stem}_0001.rad", f"{stem}_0002.rad"],
        "expected_end_time_s": motion_meta["end_time_s"],
        "engine_stages": [
            {"engine": f"{stem}_0001.rad", "mode": engine_mode,
             "end_time_s": motion_meta["engine_1_end_time_s"],
             "purpose": "score, clear the punch, fold the right boundary gripper, and hold the endpoint"},
            {"engine": f"{stem}_0002.rad", "mode": release_mode,
             "end_time_s": motion_meta["release_observation_end_time_s"],
             "purpose": "turn off both gripper RBODY constraints and observe the undamped unforced board response",
             "release_card": "/RBODY/OFF",
             "released_rbody_ids": [1, 2],
             "support_rails_remain_active": True,
             "damping_added": False,
             "integration_controls": release_control_metadata},
        ],
        "runtime": {"release": RELEASE, "source_commit": SOURCE_COMMIT,
                    "intended_profile": "arm64-mumps-paper", "solver_run_requested_by_exporter": False,
                    "engine_mode": engine_mode,
                    "resource_forecast": None,
                    "resource_forecast_status": "not measured; bounded strip run pending"},
        "geometry": {
            "builder": "matter/tools/cardboard_opencourant_hex.py",
            "mesh_schema": hex_mesh.MESH_SCHEMA,
            "units": mesh.units,
            "parameters": mesh.parameters,
            "quality": mesh.quality,
            "score": mesh.score,
        },
        "materials": material_identity,
        "interfaces": {
            **ties_meta,
            "rigidbody_main_surface_compatibility": gripper_membership_meta[
                "type2_rigidbody_main_compatibility"],
            "adhesive_claim": "perfect-bond integration control only; no cohesive separation, debonding, or fracture prediction",
            "source_recipe_identity": "finite elastic starch bridges with ideal perfect bonding",
            "mass_semantics": "TYPE2 Spotflag=27 is documented as adding no mass; check Starter for fallback-to-penalty warnings",
        },
        "self_contact": self_contact_meta,
        "tools_and_boundary_conditions": motion_meta,
        "mass_accounting": {
            "mass_unit": "Mg",
            "known_rigid_tool_mass_Mg": 0.0,
            "tool_mass_identity": "Analytic RWALL punch and supports are massless; gripper bodies are prescribed board boundary nodes, not tooling masses.",
            "tool_mass_in_starter": False,
            "board_mass_definition": "Starter-reported structural mass; board gripper-node mass is already included.",
            "starter_expected_added_mass_Mg": 0.0,
            "engine_expected_added_mass_Mg": 0.0,
            "collector": "matter.tools.cardboard_opencourant_strip.collect_mass_accounting",
            "interpretation": "Starter plus Engine mass report only; not a mechanics or physical qualification gate by itself.",
        },
        "numerics": {
            "engine": engine_description,
            "release_engine": {
                **release_control_metadata,
                "description": release_description,
                "card": "/RBODY/OFF",
                "released_rbody_ids": [1, 2],
                "release_timing": "new Engine stage after fold-end zero-velocity dwell",
                "observation_duration_s": POST_RELEASE_OBSERVATION_S,
                "damping": "none added; duration is an observation window, not a relaxation or springback qualification",
                "explicit_timestep_forecast": {
                    "source": "Starter-only strip11 output, same mesh and TYPE7 surface",
                    "minimum_solid_dt_s": 1.1420892054297e-8,
                    "explicit_dt_scale": DT_SCALE,
                    "scaled_solid_dt_s": 1.02788028488673e-8,
                    "solid_only_cycles_for_one_second": 97287594,
                    "minimum_nodal_dt_estimate_s": 6.468e-28,
                    "active_engine_dt": "unverified; no Engine run was used for this estimate",
                    "meaning": "bounded forecast only; the nodal estimate may further constrain Engine and explicit release is not resource-qualified",
                },
            },
            "time_step_control": timestep_control,
            "nonlinear_tolerance_semantics": (None if engine_mode == "explicit" else {
                "keyword": "/IMPL/NONL/01",
                "record": {
                    "L_A": 20,
                    "Itol": 2,
                    "Toli": IMPLICIT_FORCE_TOLERANCE,
                    "Tolj": IMPLICIT_FORCE_TOLERANCE,
                },
                "mapping": "Itol=2 selects relative force residual; Toli is the selected tolerance; Tolj is unused for a single criterion.",
                "convergence_status_caution": "A Conv.stat marker is not sufficient evidence by itself; inspect |r|/|r0| against Toli for every accepted increment.",
                "reference": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/impl_nonlin_engine_r.htm",
            }),
            "tool_contact": {"law": "kinematic rigid-wall contact", "friction": 0.0,
                              "friction_status": "instrument choice; source friction not identified"},
            "self_contact": {
                "law": "/INTER/TYPE7 self-contact with /SURF/SEG and matching /GRNOD/NODE",
                "gap_mm": SELF_CONTACT_GAP_MM,
                "gap_status": "declared numerical clearance, not a physical material layer or measured gap",
                "initial_coordinate_adjustment": "INACTIV=1000 (no action)",
                "maximum_penetration_gate": f"measure and require < {SELF_CONTACT_GAP_MM:.6g} mm",
                "sensitivity_gap_mm": SELF_CONTACT_GAP_MM / 2.0,
                "sensitivity_status": "required before qualification; not run here",
            "shared_node_filtering": "all exposed paper segments are exported; verify native TYPE7 handling of shared-node adjacent segments",
            "gripper_nodes_in_contact_set": self_contact_meta["gripper_slave_overlap_count"],
            "gripper_contact_verification": "Starter/Engine must accept TYPE7 secondary nodes that are also rigid-body slaves, and Engine 2 must retain self-contact after RBODY/OFF",
                "friction": 0.0,
                "friction_status": "instrument choice; source friction not identified",
            },
            "ideal_grip": "boundary-only rigid gripper groups; no interior imposed rotations",
            "score_path": "1 mm radius analytic punch, 6 mm channel, 0.25 caliper indentation, retract, then park above the rigid-wing sweep envelope",
            "stage_schedule": schedule,
            "material_axis_orientation": {
                "property": "/PROP/TYPE6 Ip=20",
                "native_orientation": "orthotropic axes from brick connectivity N1-N2 and N1-N4; N1-N2 follows flute-profile/MD tangent and N1-N4 follows +CD",
                "mesh_frame_metadata_used_for_card_orientation": False,
                "orthotropic_material_card": "/MAT/LAW112; no /ORTHO card required for this connectivity convention",
            },
            "limitations": [
                "The geometry is a literature-derived mixed-source coupon, not a matched physical specimen.",
                "The tensile LAW112 branches are nominal transfers from published bilinear-fit endpoints; this is not specimen-level calibration and unidentified multiaxial branches remain assumptions.",
                "The perfect ties suppress all starch-interface opening and slip; adhesive failure is not modeled.",
                "The score is produced by geometric contact only; no crease-specific constitutive damage law is included.",
                "The explicit schedule is a short parser fixture; implicit-slow uses declared engineering stage rates and provisional timestep control, not measured manufacturing conditions. Dynamic/quasi-static equivalence is unqualified.",
                "Self-contact has a declared 1 micron numerical clearance; qualification requires a gap/2 sensitivity case, no initial penetration or coordinate correction, and measured maximum penetration below the declared gap.",
                "Gripper release has no added damping; the final observation window does not establish relaxed springback or a calibrated release response.",
                "No force, springback, fold-angle response, assembly, crush, or physical prediction is qualified by deck export.",
            ],
        },
        "provenance": {},
        "gates": [
            "Normal native termination through scoring, punch parking above the rigid-wing sweep envelope, boundary-driven fold, endpoint dwell, and gripper release",
            "No negative volume or unbounded contact/constraint energy; retain complete time and energy histories",
            "Starter must confirm positive initial geometry, no initial self-contact penetration, and no automatic coordinate adjustment",
            "Starter/Engine must accept gripper nodes in the TYPE7 secondary group, filter adjacent shared-node segments correctly, and retain self-contact after RBODY/OFF",
            f"Measure maximum TYPE7 penetration and require it to stay below the declared {SELF_CONTACT_GAP_MM:.6g} mm numerical contact gap",
            f"Repeat qualification at {SELF_CONTACT_GAP_MM / 2.0:.6g} mm contact gap and compare response",
            "Starter and Engine mass accounting must show no added mass; verify the reported tool mass separately from board structural mass",
            "For implicit cases, verify each accepted increment's selected relative-force residual against the declared Toli; a convergence status marker alone is insufficient",
            "Confirm boundary-only prescribed domains and measured low kinetic/internal-energy ratio before any quasi-static interpretation",
            "Treat this run as integration evidence only; physical validation requires matched coupons and identified crease/adhesive response",
        ],
        "physical_qualification": False,
        "mechanics_verdict": None,
    }
    return starter, engine_1, release_engine, fixture


def export_strip_case(output: Path, *, config: hex_mesh.StripConfig | None = None,
                      fold_pivot_z_mm: float | None = None,
                      fold_angle_deg: float = FOLD_ANGLE_DEG,
                      engine_mode: str = "explicit") -> dict[str, Any]:
    """Export one native strip case into an existing empty directory.

    The generator never starts Radioss. It fails closed if the geometry does
    not contain coincident paired interfaces or the two expected interface
    families. ``fold_angle_deg`` and ``fold_pivot_z_mm`` are fixture controls,
    not estimates of a calibrated scoring or folding process.
    """
    output = Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError(f"output must be an existing empty directory: {output}")
    mesh = hex_mesh.build_score_fold_strip(config)
    source_mapping, _ = paper._read_parameters()
    unique_parts = sorted({element.part for element in mesh.elements})
    part_ids = {name: index for index, name in enumerate(unique_parts, start=1)}
    stem = "cardboard_score_fold_selfcontact_release_instrument"
    starter, engine_1, engine_2, spec = _deck(mesh, source_mapping, part_ids,
                                               stem=stem, fold_pivot_z_mm=fold_pivot_z_mm,
                                               fold_angle_deg=fold_angle_deg, engine_mode=engine_mode)
    starter_name, engine_1_name, engine_2_name = spec["starter"], *spec["engines"]
    (output / starter_name).write_text(starter, encoding="utf-8")
    (output / engine_1_name).write_text(engine_1, encoding="utf-8")
    (output / engine_2_name).write_text(engine_2, encoding="utf-8")
    mesh_sha = hex_mesh.export_mesh_json(mesh, output / "mesh.json")
    material_module = Path(_material_transfer_module().__file__).resolve()
    source_paths = [SOURCE_MANIFEST, SOURCE_NOTES, YIELD_ANCHORS, FIT_NODES,
                    SOURCE_CATALOG, paper.PARAMETERS, material_module,
                    Path(hex_mesh.__file__).resolve(), Path(paper.__file__).resolve(),
                    ADHESION_SOURCE]
    spec["provenance"] = {
        "mesh_sha256": mesh_sha,
        "starter_sha256": _sha(output / starter_name),
        "engine_sha256": _sha(output / engine_1_name),
        "release_engine_sha256": _sha(output / engine_2_name),
        "exporter_sha256": _sha(Path(__file__).resolve()),
        "inputs": {str(path.relative_to(ROOT)): _sha(path) for path in source_paths if path.is_file()},
        "material_transfer": {
            "module": str(material_module.relative_to(ROOT)),
            "module_sha256": _sha(material_module),
            "transfer_schema": spec["materials"]["liner"]["source_record"]["schema"],
            "fit_node_csv_sha256": _sha(FIT_NODES),
            "yield_anchor_csv_sha256": _sha(YIELD_ANCHORS),
            "transfer_status": "nominal source-derived uniaxial tensile bilinear transfer; not specimen-level calibration",
        },
        "frozen_paper_parameter_sha256": _sha(paper.PARAMETERS),
        "opencourant": {"release": RELEASE, "source_commit": SOURCE_COMMIT},
        "tie_card_reference": {
            "upstream_qa_deck": "qa-tests/miniqa/INTERF/INT_2/pl_dam_ilev/data/PLREG41B_0000.rad",
            "source_commit": SOURCE_COMMIT,
            "verified_card": "/INTER/TYPE2 group-to-/SURF/PART/EXT tie; Spotflag=27 native mass-conserving kinematic formulation",
            "native_mass_semantics_reference": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/inter_type2_starter_r.htm",
        },
        "rigid_body_reference": {
            "verified_card": "/RBODY with Ispher=2 computed inertia and zero added mass/inertia",
            "native_inertia_semantics_reference": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/rbody_starter_r.htm",
        },
        "tool_card_reference": {
            "upstream_qa_deck": "qa-tests/miniqa/OPTIONS/damp/new_test_v11/data/FFO25SCB_0000.rad",
            "source_commit": SOURCE_COMMIT,
            "verified_cards": ["/RWALL/CYL", "/IMPVEL"],
        },
        "self_contact_reference": {
            "source_commit": SOURCE_COMMIT,
            "verified_cards": ["/INTER/TYPE7", "/SURF/SEG", "/GRNOD/NODE"],
        "type7_fields": "2026 CFG format; Igap=1000; Ibag=0; Gapmin=Gapmax=0.001 mm; INACTIV=1000; ICOG=1",
            "native_references": {
                "starter_type7_schema": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/inter_type7_starter_r.htm",
                "engine_rbody_off": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/rbody_engine_r.htm",
            },
        },
        "release_reference": {
            "source_commit": SOURCE_COMMIT,
            "upstream_engine_example": "qa-tests/miniqa/INTERF/INT_7/idel7=2/data/IDEL8_0002.rad",
            "verified_card": "/RBODY/OFF followed by the rigid-body master node identifiers",
        },
        "implicit_dynamic_release_reference": {
            "verified_card": "/IMPL/DYNA/2 with general Newmark gamma=0.5, beta=0.25",
            "pinned_source": "engine/source/input/freimpl.F lines 168-185; engine/source/implicit/imp_dyna.F lines 125-130",
            "native_references": {
                "newmark": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/impl_dyna_2_engine_r.htm",
                "implicit_release_status": "keyword/source parsing verified only; actual staged RBODY/OFF integration remains unrun",
            },
        },
    }
    _write_json(output / "case.json", spec)
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory for starter, engine, mesh and case.json")
    parser.add_argument("--fold-pivot-z-mm", type=float, default=None,
                        help="fixture pivot; defaults to half the geometry caliper")
    parser.add_argument("--engine-mode", choices=ENGINE_MODES, default="explicit",
                        help="select the short explicit parser fixture or declared-rate implicit quasi-static schedule")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = export_strip_case(args.output, fold_pivot_z_mm=args.fold_pivot_z_mm,
                               engine_mode=args.engine_mode)
    print(json.dumps({"output": str(args.output), "case_id": result["id"],
                      "starter": result["starter"], "engines": result["engines"],
                      "mesh_sha256": result["provenance"]["mesh_sha256"],
                      "physical_qualification": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
