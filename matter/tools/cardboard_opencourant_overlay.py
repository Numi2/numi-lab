#!/usr/bin/env python3
"""Export a compact TYPE2-tied auxiliary TYPE43 interpolation fixture.

This is a synthetic solver instrument, not a cardboard material model. Two
deformable H8 substrates retain their independent reference meshes. A 2x2
cohesive overlay has nine nodes on each side; each side is tied to one coarse
substrate face by standard kinematic /INTER/TYPE2. Overlay mass is either a
historical zero-density solver control or a declared areal-mass analogue.
The starter contains a prescribed outer-face approach, an accepted stationary
bond birth, and nonuniform mixed opening/shear across an
implicit-explicit-implicit stage sequence. Solver execution is external.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CASE_NAME = "type2_cohesive_overlay"
OVERLAY03_CASE_NAME = "type2_cohesive_overlay_03"
SCHEMA = "cardboard.opencourant.type2-cohesive-overlay.v1"
SIKAMELT_230_SOURCE_URL = "https://industry.sika.com/dam/dms/fi01/q/sikamelt-230.pdf"
SIKAMELT_230_SOURCE_SHA256 = "db2f828fb1c18ffdc94c3dd5e0488cd966967cb1cb6c9a8cfc92b431a7f4cc5c"


@dataclass(frozen=True)
class OverlayConfig:
    """Numerical controls for the small synthetic transfer fixture."""

    patch_width_mm: float = 20.0
    substrate_thickness_mm: float = 2.0
    initial_gap_mm: float = 0.05
    initial_rotation_deg: float = 23.0
    approach_end_s: float = 0.00049
    birth_time_s: float = 0.00050
    stationary_end_s: float = 0.00055
    final_time_s: float = 0.001
    opening_base_mm: float = 0.02
    opening_x_gradient_mm: float = 0.04
    opening_y_gradient_mm: float = 0.01
    shear_x_base_mm: float = 0.01
    shear_x_y_gradient_mm: float = 0.02
    shear_y_base_mm: float = -0.005
    shear_y_x_gradient_mm: float = 0.01
    type2_search_distance_mm: float = 2.0
    cohesive_density_mg_mm3: float = 0.0
    cohesive_mass_calculation_flag: int = 2
    cohesive_film_thickness_mm: float = 0.0
    cohesive_density_source_name: str | None = None
    cohesive_density_source_url: str | None = None
    cohesive_density_source_sha256: str | None = None
    substrate_density_mg_mm3: float = 1.0e-6
    cohesive_stiffness_n_per_mm: float = 100.0
    substrate_young_modulus_n_per_mm2: float = 10000.0

    def validate(self) -> None:
        for name in ("patch_width_mm", "substrate_thickness_mm",
                     "initial_gap_mm", "type2_search_distance_mm",
                     "cohesive_stiffness_n_per_mm",
                     "substrate_young_modulus_n_per_mm2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cohesive_mass_calculation_flag not in (1, 2):
            raise ValueError("cohesive_mass_calculation_flag must be 1 (area) or 2 (volume)")
        if (not math.isfinite(self.cohesive_density_mg_mm3)
                or self.cohesive_density_mg_mm3 < 0.0):
            raise ValueError("cohesive_density_mg_mm3 must be finite and nonnegative")
        if not math.isfinite(self.cohesive_film_thickness_mm) or self.cohesive_film_thickness_mm < 0:
            raise ValueError("cohesive_film_thickness_mm must be finite and nonnegative")
        if self.cohesive_mass_calculation_flag == 1 and self.cohesive_density_mg_mm3 > 0.0:
            if self.cohesive_film_thickness_mm <= 0.0:
                raise ValueError("positive IMASS=1 density requires a declared finite film thickness")
            if not all((self.cohesive_density_source_name,
                        self.cohesive_density_source_url,
                        self.cohesive_density_source_sha256)):
                raise ValueError("positive IMASS=1 density requires source name, URL, and SHA-256")
        if (not math.isfinite(self.substrate_density_mg_mm3)
                or self.substrate_density_mg_mm3 <= 0.0):
            raise ValueError("substrate_density_mg_mm3 must be finite and positive")
        times = (self.approach_end_s, self.birth_time_s,
                 self.stationary_end_s, self.final_time_s)
        if not all(math.isfinite(value) and value > 0.0 for value in times):
            raise ValueError("all stage times must be finite and positive")
        if not (self.approach_end_s < self.birth_time_s <
                self.stationary_end_s < self.final_time_s):
            raise ValueError("stage times must satisfy approach < birth < hold-end < final")
        if not math.isfinite(self.initial_rotation_deg):
            raise ValueError("initial_rotation_deg must be finite")
        for name in ("opening_base_mm", "opening_x_gradient_mm",
                     "opening_y_gradient_mm", "shear_x_base_mm",
                     "shear_x_y_gradient_mm", "shear_y_base_mm",
                     "shear_y_x_gradient_mm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def hot_melt_mass_analogue(cls, **overrides: Any) -> "OverlayConfig":
        """Return the declared SikaMelt-230 density / 50 um film mass analogue.

        The product density is an analogue for a solver instrument only. The
        assumed film thickness is an engineering control, not measured starch
        geometry or a cardboard closure calibration.
        """
        values: dict[str, Any] = {
            "cohesive_density_mg_mm3": 1.0e-9,
            "cohesive_mass_calculation_flag": 1,
            "cohesive_film_thickness_mm": 0.05,
            "cohesive_density_source_name": "SikaMelt-230 hot-melt density analogue (1.0 kg/L)",
            "cohesive_density_source_url": SIKAMELT_230_SOURCE_URL,
            "cohesive_density_source_sha256": SIKAMELT_230_SOURCE_SHA256,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def law117_density_input(self) -> float:
        """MAT_RHO value, in Mg/mm2 for IMASS=1 or Mg/mm3 for IMASS=2."""
        if self.cohesive_mass_calculation_flag == 1:
            return self.cohesive_density_mg_mm3 * self.cohesive_film_thickness_mm
        return self.cohesive_density_mg_mm3

    @property
    def overlay_mass_mg(self) -> float:
        """Expected mass for the square TYPE43 patch, in milligrams."""
        area_mm2 = self.patch_width_mm**2
        if self.cohesive_mass_calculation_flag == 1:
            mass_mg = self.law117_density_input * area_mm2 * 1.0e9
        else:
            initial_volume_mm3 = area_mm2 * self.initial_gap_mm
            mass_mg = self.cohesive_density_mg_mm3 * initial_volume_mm3 * 1.0e9
        return mass_mg

    @property
    def substrate_mass_mg(self) -> float:
        """Total synthetic mass of the two rectangular substrate HEX8s."""
        volume_mm3 = self.patch_width_mm**2 * self.substrate_thickness_mm
        return 2.0 * self.substrate_density_mg_mm3 * volume_mm3 * 1.0e9


@dataclass(frozen=True)
class OverlayModel:
    config: OverlayConfig
    nodes: dict[int, tuple[float, float, float]]
    node_roles: dict[int, str]
    substrate_elements: dict[int, tuple[int, ...]]
    cohesive_elements: dict[int, tuple[int, ...]]
    lower_overlay_nodes: tuple[int, ...]
    upper_overlay_nodes: tuple[int, ...]
    lower_master_face: tuple[int, int, int, int]
    upper_master_face: tuple[int, int, int, int]
    lower_remote_face: tuple[int, int, int, int]
    upper_remote_face: tuple[int, int, int, int]
    upper_drive_groups: dict[int, int]
    upper_drive_functions: dict[tuple[int, str], int]
    close_displacements_mm: dict[int, tuple[float, float, float]]
    final_displacements_mm: dict[int, tuple[float, float, float]]
    min_center_jacobian_mm3: float

    def mesh_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "units": {"length": "mm", "mass": "Mg", "time": "s"},
            "frame": {
                "local_x": "interface tangent direction 1",
                "local_y": "interface tangent direction 2",
                "local_z": "interface normal from lower to upper substrate",
                "initial_rotation_about_global_z_deg": self.config.initial_rotation_deg,
                "rotation_is_initial_geometry_only": True,
            },
            "nodes": [
                {"id": node_id, "xyz_mm": list(xyz), "role": self.node_roles[node_id]}
                for node_id, xyz in sorted(self.nodes.items())
            ],
            "substrate_elements": [
                {"id": element_id, "part": "lower_substrate" if element_id == 1 else "upper_substrate",
                 "nodes": list(nodes)}
                for element_id, nodes in sorted(self.substrate_elements.items())
            ],
            "cohesive_elements": [
                {"id": element_id, "part": "cohesive_overlay", "nodes": list(nodes)}
                for element_id, nodes in sorted(self.cohesive_elements.items())
            ],
            "type2_ties": [
                {"secondary_nodes": list(self.lower_overlay_nodes),
                 "main_surface_id": 21, "main_face_nodes": list(self.lower_master_face),
                 "spot_flag": 5, "formulation": "standard kinematic, no penalty"},
                {"secondary_nodes": list(self.upper_overlay_nodes),
                 "main_surface_id": 22, "main_face_nodes": list(self.upper_master_face),
                 "spot_flag": 5, "formulation": "standard kinematic, no penalty"},
            ],
            "remote_grips": {
                "fixed_lower_nodes": list(self.lower_remote_face),
                "driven_upper_nodes": list(self.upper_remote_face),
                "all_other_substrate_nodes_free": True,
                "overlay_nodes_have_no_direct_bcs": True,
            },
            "stage_times_s": {
                "implicit_approach_end": self.config.approach_end_s,
                "accepted_cohesive_birth": self.config.birth_time_s,
                "explicit_stationary_hold_end": self.config.stationary_end_s,
                "implicit_mixed_mode_end": self.config.final_time_s,
            },
            "initial_gap_mm": self.config.initial_gap_mm,
            "displacement_boundary_values_mm": {
                str(node_id): {
                    "at_contact_and_birth": list(self.close_displacements_mm[node_id]),
                    "at_final_time": list(self.final_displacements_mm[node_id]),
                }
                for node_id in self.upper_remote_face
            },
            "synthetic_material_controls": {
                "classification": "solver instrument; adhesive mass may use a declared hot-melt density analogue, not starch calibration",
                "law117_density_input": self.config.law117_density_input,
                "law117_density_input_unit": (
                    "Mg/mm2" if self.config.cohesive_mass_calculation_flag == 1 else "Mg/mm3"),
                "law117_mass_calculation_flag": self.config.cohesive_mass_calculation_flag,
                "auxiliary_mass_intentionally_zero": self.config.law117_density_input == 0.0,
                "substrate_density_mg_mm3": self.config.substrate_density_mg_mm3,
                "law117_normal_and_shear_stiffness_n_per_mm": self.config.cohesive_stiffness_n_per_mm,
                "law1_substrate_young_modulus_n_per_mm2": self.config.substrate_young_modulus_n_per_mm2,
                "mass_scaling": False,
            },
            "adhesive_mass_analogue": {
                "status": ("engineering analogue; not measured cardboard/starch" if
                           self.config.cohesive_density_source_name else
                           "synthetic massless control"),
                "analogue_name": self.config.cohesive_density_source_name,
                "source_url": self.config.cohesive_density_source_url,
                "source_sha256": self.config.cohesive_density_source_sha256,
                "analogue_density_kg_per_litre": (
                    self.config.cohesive_density_mg_mm3 / 1.0e-9
                    if self.config.cohesive_density_source_name else None),
                "analogue_density_Mg_per_mm3": self.config.cohesive_density_mg_mm3,
                "declared_film_thickness_mm": self.config.cohesive_film_thickness_mm,
                "computed_areal_mass_Mg_per_mm2": self.config.law117_density_input,
                "cohesive_patch_area_mm2": self.config.patch_width_mm**2,
                "expected_overlay_mass_mg": self.config.overlay_mass_mg,
                "mass_basis": ("areal density; independent of initial inter-substrate gap" if
                               self.config.cohesive_mass_calculation_flag == 1 else
                               "volumetric density times initial cohesive-element volume"),
                "not_starch_mass_or_fracture_calibration": True,
                "do_not_combine_with_explicit_glue_mass_for_same_interface": True,
            },
            "mass_ledger": {
                "units": {"volume": "mm3", "density": "Mg/mm3", "mass": "Mg"},
                "lower_substrate_H8_volume_mm3": (
                    self.config.patch_width_mm**2 * self.config.substrate_thickness_mm),
                "upper_substrate_H8_volume_mm3": (
                    self.config.patch_width_mm**2 * self.config.substrate_thickness_mm),
                "substrate_density_Mg_per_mm3": self.config.substrate_density_mg_mm3,
                "substrate_only_total_mass_Mg": self.config.substrate_mass_mg * 1.0e-9,
                "cohesive_overlay_mass_Mg": self.config.overlay_mass_mg * 1.0e-9,
                "expected_starter_total_mass_Mg": (
                    self.config.substrate_mass_mg + self.config.overlay_mass_mg) * 1.0e-9,
                "expected_starter_total_mass_mg": (
                    self.config.substrate_mass_mg + self.config.overlay_mass_mg),
                "total_mass_includes_overlay_once": True,
                "TYPE2_projection_can_redistribute_but_must_not_create_mass": True,
            },
            "type2_mass_audit": {
                "spot_flag": 5,
                "formulation": "standard kinematic constraint",
                "projection_flag": 1,
                "projection_flag_source": "documented TYPE2 default; redistributes secondary mass/inertia on the main segment",
                "reference": "https://help.altair.com/hwsolvers/rad/topics/solvers/rad/inter_type2_starter_r.htm",
                "expected_total_model_mass_change_mg": self.config.overlay_mass_mg,
                "expected_total_starter_mass_Mg": (
                    self.config.substrate_mass_mg + self.config.overlay_mass_mg) * 1.0e-9,
                "starter_mass_check": "compare total reported mass with substrate-only mass plus expected overlay mass; reject any unexplained additional overlay mass",
                "no_arbitrary_mass_scale": True,
            },
            "outputs": {
                "outer_face_reaction_and_motion": "/TH/NODE/101 and /TH/NODE/102",
                "auxiliary_interface_node_motion_for_gap_reconstruction": "/TH/NODE/103",
                "cohesive_stress_strain_internal_energy_and_active_flag": "/TH/BRIC/104",
                "substrate_state": "/TH/BRIC/105",
                "global_energy_work": "automatic global time history through /TFILE/4",
                "interpretation": "Reaction balance, gap, and work transfer must be evaluated from native output; deck generation is not evidence of conservation.",
            },
            "quality": {
                "substrate_hex8_count": len(self.substrate_elements),
                "cohesive_hex8_count": len(self.cohesive_elements),
                "overlay_nodes_per_side": 9,
                "overlay_connectivity_is_independent": True,
                "min_center_jacobian_mm3": self.min_center_jacobian_mm3,
                "substrate_reference_coordinates_preserved": True,
            },
            "qualification": "unrun synthetic TYPE2-to-TYPE43 transfer fixture; not shipping closure qualification",
        }


def _rotated_xyz(local: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = local
    return (c * x - s * y, s * x + c * y, z)


def _hex_center_jacobian(points: list[tuple[float, float, float]]) -> float:
    """Determinant of the trilinear HEX8 Jacobian at its center."""
    if len(points) != 8:
        raise ValueError("HEX8 requires eight coordinates")
    sx = (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
    sy = (-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0)
    sz = (-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
    deriv = [[0.0, 0.0, 0.0] for _ in range(3)]
    for p, x, y, z in zip(points, sx, sy, sz):
        for coord in range(3):
            deriv[coord][0] += x * p[coord] / 8.0
            deriv[coord][1] += y * p[coord] / 8.0
            deriv[coord][2] += z * p[coord] / 8.0
    a, b, c = deriv
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def build_overlay_fixture(config: OverlayConfig | None = None) -> OverlayModel:
    """Build two fixed-reference H8 blocks plus a 2x2 TYPE43 overlay."""
    cfg = config or OverlayConfig()
    cfg.validate()
    half = 0.5 * cfg.patch_width_mm
    angle = math.radians(cfg.initial_rotation_deg)
    nodes: dict[int, tuple[float, float, float]] = {}
    roles: dict[int, str] = {}

    def add_node(node_id: int, role: str, local: tuple[float, float, float]) -> None:
        nodes[node_id] = _rotated_xyz(local, angle)
        roles[node_id] = role

    corners = ((-half, -half), (half, -half), (half, half), (-half, half))
    # Lower substrate: outer face at -thickness, tie face at z=0.
    for i, (x, y) in enumerate(corners, start=1):
        add_node(i, "lower_substrate_remote", (x, y, -cfg.substrate_thickness_mm))
    for i, (x, y) in enumerate(corners, start=5):
        add_node(i, "lower_substrate_interface", (x, y, 0.0))
    # Upper substrate: tie face at the initial gap, remote grip above it.
    for i, (x, y) in enumerate(corners, start=9):
        add_node(i, "upper_substrate_interface", (x, y, cfg.initial_gap_mm))
    for i, (x, y) in enumerate(corners, start=13):
        add_node(i, "upper_substrate_remote",
                 (x, y, cfg.initial_gap_mm + cfg.substrate_thickness_mm))

    # The two 3x3 overlay surfaces are separate from both substrate meshes.
    lower_grid: dict[tuple[int, int], int] = {}
    upper_grid: dict[tuple[int, int], int] = {}
    for j in range(3):
        for i in range(3):
            x = -half + cfg.patch_width_mm * i / 2.0
            y = -half + cfg.patch_width_mm * j / 2.0
            lower_id = 17 + 3 * j + i
            upper_id = 26 + 3 * j + i
            add_node(lower_id, "cohesive_lower_auxiliary", (x, y, 0.0))
            add_node(upper_id, "cohesive_upper_auxiliary",
                     (x, y, cfg.initial_gap_mm))
            lower_grid[i, j] = lower_id
            upper_grid[i, j] = upper_id

    substrate = {
        1: (1, 2, 3, 4, 5, 6, 7, 8),
        2: (9, 10, 11, 12, 13, 14, 15, 16),
    }
    cohesive: dict[int, tuple[int, ...]] = {}
    element_id = 101
    for j in range(2):
        for i in range(2):
            a = (lower_grid[i, j], lower_grid[i + 1, j],
                 lower_grid[i + 1, j + 1], lower_grid[i, j + 1])
            b = (upper_grid[i, j], upper_grid[i + 1, j],
                 upper_grid[i + 1, j + 1], upper_grid[i, j + 1])
            cohesive[element_id] = a + b
            element_id += 1

    # The substrate coordinates are not moved to match one another. TYPE2 ties
    # project the independent overlay nodes onto each original coarse face.
    lower_master = (5, 6, 7, 8)
    upper_master = (9, 12, 11, 10)  # outward normal from upper substrate is -Z
    lower_remote = (1, 2, 3, 4)
    upper_remote = (13, 14, 15, 16)
    drive_groups = {node_id: 1000 + node_id for node_id in upper_remote}
    function_ids: dict[tuple[int, str], int] = {}
    close_displacements: dict[int, tuple[float, float, float]] = {}
    final_displacements: dict[int, tuple[float, float, float]] = {}
    c, s = math.cos(angle), math.sin(angle)
    for node_id in upper_remote:
        # Convert the already-rotated coordinates back to local coordinates so
        # the imposed nonuniform field is objective under initial rotation.
        xg, yg, _ = nodes[node_id]
        x, y = c * xg + s * yg, -s * xg + c * yg
        xi, eta = (x + half) / cfg.patch_width_mm, (y + half) / cfg.patch_width_mm
        opening = (cfg.opening_base_mm + cfg.opening_x_gradient_mm * xi
                   + cfg.opening_y_gradient_mm * eta)
        shear_x = cfg.shear_x_base_mm + cfg.shear_x_y_gradient_mm * eta
        shear_y = cfg.shear_y_base_mm + cfg.shear_y_x_gradient_mm * xi
        global_shear_x = c * shear_x - s * shear_y
        global_shear_y = s * shear_x + c * shear_y
        close_displacements[node_id] = (0.0, 0.0, -cfg.initial_gap_mm)
        final_displacements[node_id] = (
            global_shear_x, global_shear_y, -cfg.initial_gap_mm + opening)
        for offset, axis in enumerate(("X", "Y", "Z")):
            function_ids[node_id, axis] = 100 + 3 * (node_id - upper_remote[0]) + offset

    all_elements = [*substrate.values(), *cohesive.values()]
    jacobians = [_hex_center_jacobian([nodes[n] for n in connectivity])
                 for connectivity in all_elements]
    if any(not math.isfinite(value) or value <= 0.0 for value in jacobians):
        raise ValueError(f"fixture contains nonpositive HEX8 Jacobian: {min(jacobians)}")

    return OverlayModel(
        config=cfg, nodes=nodes, node_roles=roles,
        substrate_elements=substrate, cohesive_elements=cohesive,
        lower_overlay_nodes=tuple(lower_grid[i, j] for j in range(3) for i in range(3)),
        upper_overlay_nodes=tuple(upper_grid[i, j] for j in range(3) for i in range(3)),
        lower_master_face=lower_master, upper_master_face=upper_master,
        lower_remote_face=lower_remote, upper_remote_face=upper_remote,
        upper_drive_groups=drive_groups, upper_drive_functions=function_ids,
        close_displacements_mm=close_displacements,
        final_displacements_mm=final_displacements,
        min_center_jacobian_mm3=min(jacobians),
    )


def _f(value: float) -> str:
    return f"{value:.12g}"


def _group_block(group_id: int, title: str, node_ids: tuple[int, ...] | list[int]) -> list[str]:
    lines = [f"/GRNOD/NODE/{group_id}", title]
    for start in range(0, len(node_ids), 10):
        lines.append("".join(f"{value:10d}" for value in node_ids[start:start + 10]))
    return lines


def _th_variables(names: tuple[str, ...]) -> str:
    """Render accepted 10-column TH variable names, one field per variable."""
    if any(not name or len(name) > 10 for name in names):
        raise ValueError("TH variable names must contain 1 to 10 characters")
    return "".join(f"{name:<10s}" for name in names)


def _th_node_row(node_id: int) -> str:
    """Render accepted /TH/NODE node and skew identifiers (skew 0)."""
    return f"{node_id:10d}{0:10d}"


def render_starter(model: OverlayModel, case_name: str = CASE_NAME) -> str:
    """Render a starter deck using the accepted four-stage synthetic fixture pattern."""
    cfg = model.config
    lines = [
        "#RADIOSS STARTER",
        "# Synthetic TYPE2-to-TYPE43 auxiliary-overlay transfer fixture.",
        "# No physical cardboard calibration; no solver run is included.",
        "/BEGIN", case_name, "2026",
        "                  Mg                  mm                   s",
        "                  Mg                  mm                   s",
        "/TITLE", ("Synthetic TYPE2-tied cohesive overlay; initial rigid geometry rotation"
                    if not cfg.cohesive_density_source_name else
                    "Overlay03 hot-melt mass analogue; engineering film assumption, not starch calibration"),
        "/ANALY", "         0                   0",
        "/MAT/LAW117/1", ("Synthetic only; inherited from accepted cohesive fixture"
                           if not cfg.cohesive_density_source_name else
                           "SikaMelt-230 density analogue; assumed 0.05 mm film"),
        f"{cfg.law117_density_input:20g}",
        f"{cfg.cohesive_stiffness_n_per_mm:20g}{cfg.cohesive_stiffness_n_per_mm:20g}{cfg.cohesive_mass_calculation_flag:10d}{1:10d}{1:10d}{0:10d}",
        "         0         0                  10                  10                   0",
        f"{1.5:20g}{1.5:20g}{2:20d}{2:20d}{1:20d}",
        "         1         1                 0.3                 0.3                   3",
        "/PROP/TYPE43/1", "Synthetic cohesive solid property with finite initial offset", "         1",
        "/MAT/LAW1/2", "Synthetic elastic substrate only",
        f"{cfg.substrate_density_mg_mm3:20g}",
        f"{cfg.substrate_young_modulus_n_per_mm2:20g}{0:20d}",
        "/PROP/TYPE14/2", "Fully integrated H8 substrate",
        "        14         0                   0                 222                   0                   0",
        "                   0                   0                   0                   0                   0",
        "                   0                   0                   0                   0                   0",
        "         0         0         2",
        "/NODE",
    ]
    for node_id, xyz in sorted(model.nodes.items()):
        lines.append(f"{node_id:10d}{_f(xyz[0]):>20s}{_f(xyz[1]):>20s}{_f(xyz[2]):>20s}")

    lines.extend(_group_block(1, "fixed lower outer substrate grip", model.lower_remote_face))
    lines.extend(_group_block(2, "driven upper outer substrate grip", model.upper_remote_face))
    lines.extend(_group_block(31, "lower overlay secondary nodes for TYPE2", model.lower_overlay_nodes))
    lines.extend(_group_block(32, "upper overlay secondary nodes for TYPE2", model.upper_overlay_nodes))
    for node_id, group_id in model.upper_drive_groups.items():
        lines.extend(_group_block(group_id, f"single upper outer drive node {node_id}", [node_id]))

    lines.extend([
        "/SURF/SEG/21/0", "Lower substrate interface face",
        "".join(f"{value:10d}" for value in (1, 5, 6, 7, 8)),
        "/SURF/SEG/22/0", "Upper substrate interface face, outward -Z",
        "".join(f"{value:10d}" for value in (2, 9, 12, 11, 10)),
        "/INTER/TYPE2/1", "Lower 3x3 auxiliary nodes tied to coarse substrate face",
        f"{31:10d}{21:10d}{1000:10d}{5:10d}{0:10d}{2:10d}{0:10d}{'':10s}{cfg.type2_search_distance_mm:20g}",
        "/INTER/TYPE2/2", "Upper 3x3 auxiliary nodes tied to coarse substrate face",
        f"{32:10d}{22:10d}{1000:10d}{5:10d}{0:10d}{2:10d}{0:10d}{'':10s}{cfg.type2_search_distance_mm:20g}",
        "/PART/1", "Synthetic cohesive overlay", "         1         1         0",
        "/PART/2", "Lower deformable H8 substrate", "         2         2         0",
        "/PART/3", "Upper deformable H8 substrate", "         2         2         0",
        "/BRICK/1",
    ])
    for element_id, connectivity in sorted(model.cohesive_elements.items()):
        lines.append(f"{element_id:10d}" + "".join(f"{node_id:10d}" for node_id in connectivity))
    lines.extend(["/BRICK/2", "         1         1         2         3         4         5         6         7         8",
                  "/BRICK/3", "         2         9        10        11        12        13        14        15        16",
                  "/BCS/1", "Fix lower remote grip only", "   111 000         0         1",
                  "/GRBRIC/BRIC/7", "dormant 2x2 cohesive overlay", "       101       102       103       104",
                  "/ACTIV/1", "activate overlay at stationary accepted closure geometry",
                  "         0         7         0         0         0         0         0         0                   2",
                  f"{cfg.birth_time_s:>18g}              1e+30"])

    # All top displacement controls are attached only to the remote grip.
    # Local shears are transformed to global XY before writing the functions.
    for node_id in model.upper_remote_face:
        group_id = model.upper_drive_groups[node_id]
        for axis in ("X", "Y", "Z"):
            function_id = model.upper_drive_functions[node_id, axis]
            close = model.close_displacements_mm[node_id][{"X": 0, "Y": 1, "Z": 2}[axis]]
            final = model.final_displacements_mm[node_id][{"X": 0, "Y": 1, "Z": 2}[axis]]
            lines.extend([
                f"/FUNCT/{function_id}", f"synthetic {axis} displacement at grip node {node_id}",
                "# time (s) displacement (mm)",
                f"{0.0:>18g}{0.0:>18g}",
                f"{cfg.approach_end_s:>18g}{close:>18g}",
                f"{cfg.stationary_end_s:>18g}{close:>18g}",
                f"{cfg.final_time_s:>18g}{final:>18g}",
                f"/IMPDISP/{function_id}", f"outer grip {axis} displacement node {node_id}",
                "# Ifunct DIR Iskew Isensor Gnod_id Frame Icoor",
                f"{function_id:10d}{axis:>10s}{0:10d}{0:10d}{group_id:10d}{0:10d}{0:10d}",
                "# Scale_x Scale_y Tstart Tstop",
                f"{1.0:>18g}{1.0:>18g}{0.0:>18g}{cfg.final_time_s:>18g}",
            ])

    lines.extend([
        "/TH/NODE/101", "lower anchored outer-face reactions",
        _th_variables(("REACX", "REACY", "REACZ")),
    ])
    lines.extend(_th_node_row(node_id) for node_id in model.lower_remote_face)
    lines.extend(["/TH/NODE/102", "upper driven outer-face reactions and displacement",
                  _th_variables(("DX", "DY", "DZ", "REACX", "REACY", "REACZ"))])
    lines.extend(_th_node_row(node_id) for node_id in model.upper_remote_face)
    lines.extend(["/TH/NODE/103", "overlay side-node motion for gap reconstruction",
                  _th_variables(("DX", "DY", "DZ"))])
    lines.extend(_th_node_row(node_id)
                 for node_id in (*model.lower_overlay_nodes, *model.upper_overlay_nodes))
    lines.extend(["/TH/BRIC/104", "cohesive state and energy density",
                  _th_variables(("STRESS", "STRAIN", "IE", "OFF"))])
    lines.extend(f"{element_id:10d}" for element_id in sorted(model.cohesive_elements))
    lines.extend(["/TH/BRIC/105", "substrate state and energy density",
                  _th_variables(("STRESS", "STRAIN", "IE", "OFF"))])
    lines.extend(f"{element_id:10d}" for element_id in sorted(model.substrate_elements))
    lines.extend(["/END", ""])
    return "\n".join(lines)


def render_engine_stage(stage: int, final_time_s: float,
                        case_name: str = CASE_NAME) -> str:
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    lines = []
    if stage in (1, 3):
        lines.extend([
            "/IMPL/NONL/01", "20 2 1e-6 1e-6",
            "/IMPL/DTINI", "0.00001",
            "/IMPL/DT/1", "4 1.0 15 0.6666667",
            # Direct MUMPS is the accepted default for this stiff cohesive
            # overlay fixture. Immutable overlay03 evidence retains PCG.
            "/IMPL/SOLV/02", "1 0 3 1e-10",
            "/IMPL/PRINT/NONLIN/1",
        ])
    else:
        lines.extend(["/DT/NODA/STOP", "                0.02               1e-12"])
    lines.extend([
        "#RADIOSS ENGINE", "/ANIM/DT", "                   0               1e-05",
        "/ANIM/BRICK/DAMA", "/ANIM/BRICK/TENS/STRESS", "/ANIM/ELEM/ENER",
        "/ANIM/VECT/DISP", "/MON/ON", "/RFILE", "      5000", "/TFILE/4",
        f"{min(final_time_s, 1.0e-6):>18g}", "/TH/TITLE", f"/RUN/{case_name}/{stage}",
        f"{final_time_s:>18g}",
    ])
    return "\n".join(lines) + "\n"


def export_overlay_case(output_dir: str | Path,
                        config: OverlayConfig | None = None,
                        case_name: str = CASE_NAME) -> dict[str, Any]:
    """Write one small starter plus three staged Engine decks and JSON audit."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    model = build_overlay_fixture(config)
    stem = case_name
    starter = f"{stem}_0000.rad"
    engines = [f"{stem}_000{stage}.rad" for stage in (1, 2, 3)]
    (out / starter).write_text(render_starter(model, case_name=case_name), encoding="utf-8")
    for stage, name in zip((1, 2, 3), engines):
        end = {1: model.config.approach_end_s,
               2: model.config.stationary_end_s,
               3: model.config.final_time_s}[stage]
        (out / name).write_text(render_engine_stage(stage, end, case_name=case_name), encoding="utf-8")
    mesh = model.mesh_dict()
    mesh_path = out / "mesh.json"
    mesh_bytes = json.dumps(mesh, indent=2, sort_keys=True, allow_nan=False) + "\n"
    mesh_path.write_text(mesh_bytes, encoding="utf-8")
    case = {
        "schema": "cardboard.opencourant.bond.v1",
        "id": case_name,
        "starter": starter,
        "engines": engines,
        "expected_end_time_s": model.config.final_time_s,
        "implicit": "implicit approach; explicit stationary accepted birth; implicit opening/shear",
        "activation": True,
        "purpose": ("Synthetic load-transfer test for two independent TYPE2-tied cohesive overlay faces"
                    if not model.config.cohesive_density_source_name else
                    "Synthetic transfer test with a declared hot-melt areal-mass analogue"),
        "stage_end_times_s": [model.config.approach_end_s,
                              model.config.stationary_end_s,
                              model.config.final_time_s],
        "birth_time_s": model.config.birth_time_s,
        "initial_gap_mm": model.config.initial_gap_mm,
        "initial_rotation_deg": model.config.initial_rotation_deg,
        "mass_scaling": False,
        "physical_qualification": False,
        "mesh_sidecar": "mesh.json",
        "mesh_sidecar_sha256": hashlib.sha256(mesh_bytes.encode("utf-8")).hexdigest(),
        "native_solver_run": False,
        "adhesive_mass_analogue": mesh["adhesive_mass_analogue"],
        "type2_mass_audit": mesh["type2_mass_audit"],
        "source_document": ({
            "relative_path": "../sources/sikamelt-230-product-data-sheet.pdf",
            "url": model.config.cohesive_density_source_url,
            "sha256": model.config.cohesive_density_source_sha256,
        } if model.config.cohesive_density_source_name else None),
        "mass_ledger": mesh["mass_ledger"],
        "gates": [
            ("Starter accepts zero-density LAW117 mass with I_mass=2 and massless independent overlay nodes tied through standard TYPE2 on both sides"
             if model.config.law117_density_input == 0.0 else
             f"Starter reports substrate-only mass plus the declared {model.config.overlay_mass_mg:.9g} mg overlay without an initial-gap volume multiplier or unexplained TYPE2 mass"),
            "Dormant TYPE43 force remains zero during approach and stationary pre-birth hold",
            "Accepted birth at zero interface gap preserves finite reactions without force/energy jump",
            "Nonuniform mixed-mode opening/shear converges in implicit stage with interface nodes free of direct BCS",
            "Main-side reactions balance cohesive resultant and external work tracks cohesive/substrate energy change",
            "Initial 23 degree rigid geometry rotation gives equivalent local response after component transformation",
        ],
        "classification": ("synthetic instrument; unrun; no material calibration or box qualification"
                           if not model.config.cohesive_density_source_name else
                           "synthetic instrument with declared engineering hot-melt density/film-mass analogue; unrun; no cardboard/starch calibration or box qualification"),
    }
    (out / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    return case


def export_overlay03_hot_melt_case(output_dir: str | Path) -> dict[str, Any]:
    """Export overlay03 with the explicitly labeled hot-melt mass analogue."""
    return export_overlay_case(
        output_dir,
        OverlayConfig.hot_melt_mass_analogue(),
        case_name=OVERLAY03_CASE_NAME,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory; it must not already exist")
    parser.add_argument("--rotation-deg", type=float, default=23.0,
                        help="initial rigid geometry rotation about global Z")
    parser.add_argument("--gap-mm", type=float, default=0.05,
                        help="initial normal gap closed at the remote grip")
    parser.add_argument("--hot-melt-mass-analogue", action="store_true",
                        help="use the declared SikaMelt-230 density and 0.05 mm film-mass analogue (overlay03)")
    args = parser.parse_args(argv)
    if args.hot_melt_mass_analogue:
        config = OverlayConfig.hot_melt_mass_analogue(
            initial_rotation_deg=args.rotation_deg,
            initial_gap_mm=args.gap_mm,
        )
        result = export_overlay_case(args.output, config, case_name=OVERLAY03_CASE_NAME)
    else:
        result = export_overlay_case(
            args.output,
            OverlayConfig(initial_rotation_deg=args.rotation_deg,
                          initial_gap_mm=args.gap_mm),
        )
    print(json.dumps({"output": str(args.output.resolve()),
                      "starter": result["starter"],
                      "engines": result["engines"],
                      "native_solver_run": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
