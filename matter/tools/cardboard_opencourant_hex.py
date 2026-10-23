#!/usr/bin/env python3
"""Structured all-HEX8 corrugated-cardboard score/fold coupon geometry.

This is a Python port of the existing sinusoidal finite-glue section geometry
in matter/tools/cardboard_glue_mesh.hpp. It sweeps that section along the flute
axis and preserves the lower/upper finite starch bridge dimensions from the
checked-in literature2009 recipe. It produces geometry and paired interface
faces only; it does not assign a cohesive law, material calibration, score
damage, fold trajectory, or solver deck.

Coordinates are millimetres: X runs across the flute profile (and along the
coupon's paper MD convention), Y runs along the flutes / coupon width, and Z is
board caliper. The initial coupon is unscored. The exported top/bottom tool
contact groups and score metadata let a solver physically form the score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable


SCHEMA = "numi.cardboard.opencourant.hex-score-fold.v1"
MESH_SCHEMA = "numi.cardboard.hex8.v1"
MATERIAL_ROLES = ("liner", "medium", "starch")
GEOMETRY_SOURCE_REVISION = "c622d2323261d6ae26e5b860b1ca4098c8b23951"
GEOMETRY_SOURCE_SHA256 = "8db61e54ff4a0e2e4a47d6bcfe02244699b7820bf97474727bb77f1f9f389260"
_FACE_LOCAL_NODES: dict[str, tuple[int, int, int, int]] = {
    # Outward-oriented local faces for the element node ordering documented
    # below. IDs within face references remain one-based.
    "z_minus": (0, 3, 2, 1),
    "z_plus": (4, 5, 6, 7),
    "y_minus": (0, 1, 5, 4),
    "x_plus": (1, 2, 6, 5),
    "y_plus": (3, 7, 6, 2),
    "x_minus": (0, 4, 7, 3),
}
_FACE_ORDER = {name: i for i, name in enumerate(_FACE_LOCAL_NODES)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _recipe() -> tuple[dict[str, Any], dict[str, Any]]:
    path = _repo_root() / "matter/examples/cardboard/manufacturing-source-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"cardboard source manifest not found: {path}")
    raw = path.read_bytes()
    doc = json.loads(raw)
    return doc["implemented_starting_recipe"], {
        "path": "matter/examples/cardboard/manufacturing-source-manifest.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@dataclass(frozen=True)
class StripConfig:
    """Small source-recipe strip with one centered 6 mm scoring band.

    The defaults retain the existing four-pitch x 20 mm coupon dimensions.
    Mesh density and boundary grip width are explicit numerical fixture
    choices, not published test conditions.
    """

    length_mm: float | None = None
    width_mm: float | None = None
    score_x_mm: float | None = None
    score_band_width_mm: float = 6.0
    grip_width_mm: float = 5.0
    flute_subdivisions_per_pitch: int = 16
    width_subdivisions: int = 8
    liner_thickness_slices: int = 2
    medium_thickness_slices: int = 2
    starch_thickness_slices: int = 1
    pitch_mm: float | None = None
    caliper_mm: float | None = None
    liner_thickness_mm: float | None = None
    medium_thickness_mm: float | None = None
    lower_starch_gap_mm: float | None = None
    lower_starch_width_mm: float | None = None
    upper_starch_gap_mm: float | None = None
    upper_starch_width_mm: float | None = None
    # These are explicit fixture/tool inputs from the root task, not source-
    # measured manufacturing values.
    punch_radius_mm: float = 1.0
    score_channel_width_mm: float = 6.0
    score_indentation_fraction_of_caliper: float = 0.25

    def resolved(self) -> "StripConfig":
        recipe, _ = _recipe()
        geometry = recipe["paper_thickness_density_proxy"]
        coupon = recipe["coupon"]
        glue = recipe["glue"]
        lower = glue["double_back_lower_gap"]
        upper = glue["single_face_upper_gap"]
        defaults = {
            "length_mm": coupon["length_mm"],
            "width_mm": coupon["width_mm"],
            "pitch_mm": coupon["flute_pitch_mm"],
            "caliper_mm": recipe["board_caliper_mm"],
            "liner_thickness_mm": geometry["liner_thickness_mm"],
            "medium_thickness_mm": geometry["medium_thickness_mm"],
            "lower_starch_gap_mm": lower["thickness_mm"],
            "lower_starch_width_mm": lower["width_mm"],
            "upper_starch_gap_mm": upper["thickness_mm"],
            "upper_starch_width_mm": upper["width_mm"],
        }
        values = asdict(self)
        for key, value in defaults.items():
            if values[key] is None:
                values[key] = value
        if values["score_x_mm"] is None:
            values["score_x_mm"] = 0.5 * values["length_mm"]
        return StripConfig(**values)


@dataclass(frozen=True)
class BlankConfig:
    """Provisional score-to-score RSC blank dimensions and mesh resolution.

    The 300 x 200 x 200 mm inputs are score-to-score dimensions. Score
    allowance is recorded as an explicit zero placeholder because no scored
    coupon measurement is available; these values therefore do not establish
    the requested internal dimensions.
    """

    panel_length_mm: float = 300.0
    panel_width_mm: float = 200.0
    wall_height_mm: float = 200.0
    manufacturer_tab_width_mm: float = 25.0
    slot_kerf_mm: float = 1.0
    score_allowance_mm: float = 0.0
    score_band_width_mm: float = 6.0
    flute_subdivisions_per_pitch: int = 8
    blank_axis_spacing_mm: float = 10.0
    liner_thickness_slices: int = 1
    medium_thickness_slices: int = 1
    starch_thickness_slices: int = 1
    punch_radius_mm: float = 1.0
    score_channel_width_mm: float = 6.0
    score_indentation_fraction_of_caliper: float = 0.25

    def resolved_strip_config(self) -> tuple[StripConfig, dict[str, Any]]:
        _positive("panel_length_mm", self.panel_length_mm)
        _positive("panel_width_mm", self.panel_width_mm)
        _positive("wall_height_mm", self.wall_height_mm)
        _positive("manufacturer_tab_width_mm", self.manufacturer_tab_width_mm)
        _positive("slot_kerf_mm", self.slot_kerf_mm)
        if self.score_allowance_mm < 0.0 or not math.isfinite(self.score_allowance_mm):
            raise ValueError("score_allowance_mm must be finite and nonnegative")
        if not math.isclose(self.score_allowance_mm, 0.0, abs_tol=1.0e-12):
            raise ValueError("only the declared zero score-allowance placeholder is currently supported")
        if self.blank_axis_spacing_mm <= 0.0 or not math.isfinite(self.blank_axis_spacing_mm):
            raise ValueError("blank_axis_spacing_mm must be finite and positive")
        span = 2.0 * (self.panel_length_mm + self.panel_width_mm) + self.manufacturer_tab_width_mm
        blank_height = self.wall_height_mm + self.panel_width_mm
        # FEFCO 0201 has four panels L-W-L-W, then a body-height-only tab.
        score_x = (self.panel_length_mm,
                   self.panel_length_mm + self.panel_width_mm,
                   2.0 * self.panel_length_mm + self.panel_width_mm,
                   2.0 * (self.panel_length_mm + self.panel_width_mm))
        band = 0.5 * self.panel_width_mm
        score_y = (band, band + self.wall_height_mm)
        panel_spans = (
            (0.0, self.panel_length_mm),
            (self.panel_length_mm, self.panel_length_mm + self.panel_width_mm),
            (self.panel_length_mm + self.panel_width_mm,
             2.0 * self.panel_length_mm + self.panel_width_mm),
            (2.0 * self.panel_length_mm + self.panel_width_mm,
             2.0 * (self.panel_length_mm + self.panel_width_mm)),
        )
        slot_boundaries = score_x[:3]
        x_refinement = []
        for center in score_x:
            x_refinement.extend((center - 0.5 * self.score_band_width_mm,
                                 center, center + 0.5 * self.score_band_width_mm))
        for center in slot_boundaries:
            x_refinement.extend((center - 0.5 * self.slot_kerf_mm,
                                 center + 0.5 * self.slot_kerf_mm))
        y_refinement = [0.0, band, band + self.wall_height_mm, blank_height]
        for center in score_y:
            y_refinement.extend((center - 0.5 * self.score_band_width_mm,
                                 center, center + 0.5 * self.score_band_width_mm))
        return StripConfig(
            length_mm=span,
            width_mm=blank_height,
            score_x_mm=score_x[0],
            score_band_width_mm=self.score_band_width_mm,
            grip_width_mm=5.0,
            flute_subdivisions_per_pitch=self.flute_subdivisions_per_pitch,
            width_subdivisions=max(1, math.ceil(blank_height / self.blank_axis_spacing_mm)),
            liner_thickness_slices=self.liner_thickness_slices,
            medium_thickness_slices=self.medium_thickness_slices,
            starch_thickness_slices=self.starch_thickness_slices,
            punch_radius_mm=self.punch_radius_mm,
            score_channel_width_mm=self.score_channel_width_mm,
            score_indentation_fraction_of_caliper=self.score_indentation_fraction_of_caliper,
        ), {
            "perimeter_span_mm": span,
            "blank_height_mm": blank_height,
            "flap_depth_mm": band,
            "score_to_score_dimensions_mm": {
                "panel_length_L": self.panel_length_mm,
                "panel_width_W": self.panel_width_mm,
                "wall_height_H": self.wall_height_mm,
            },
            "internal_dimensions_mm": None,
            "score_allowance_mm": self.score_allowance_mm,
            "score_allowance_status": "declared zero placeholder; not measured",
            "manufacturer_tab_width_mm": self.manufacturer_tab_width_mm,
            "slot_kerf_mm": self.slot_kerf_mm,
            "panel_spans_x_mm": [list(span) for span in panel_spans],
            "tab_span_x_mm": [span - self.manufacturer_tab_width_mm, span],
            "vertical_score_x_mm": list(score_x),
            "flap_score_y_mm": list(score_y),
            "slot_boundaries_x_mm": list(slot_boundaries),
            "score_band_width_mm": self.score_band_width_mm,
            "blank_axis_spacing_mm": self.blank_axis_spacing_mm,
            "mesh_resolution_status": "fixture input; no spatial or material convergence claim",
            "x_refinement_mm": x_refinement,
            "y_refinement_mm": y_refinement,
        }


@dataclass(frozen=True)
class FaceRef:
    element_id: int
    face: str
    nodes: tuple[int, int, int, int]


@dataclass(frozen=True)
class HexElement:
    id: int
    nodes: tuple[int, int, int, int, int, int, int, int]
    part: str
    material_role: str
    frame_angle_rad: float
    # Local material axes expressed in global coordinates as columns:
    # [MD, CD, thickness].
    frame_axes: tuple[tuple[float, float, float], ...]


@dataclass
class HexMesh:
    schema: str
    units: dict[str, str]
    nodes: list[dict[str, Any]]
    elements: list[HexElement]
    interface_pairs: list[dict[str, Any]]
    surface_sets: dict[str, list[FaceRef]]
    node_sets: dict[str, list[int]]
    parameters: dict[str, Any]
    starch_patches: list[dict[str, Any]]
    score: dict[str, Any]
    tool_targets: dict[str, Any]
    frame_convention: dict[str, Any]
    provenance: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mesh_schema": MESH_SCHEMA,
            "units": self.units,
            "node_order": [
                "x0_y0_z0", "x1_y0_z0", "x1_y1_z0", "x0_y1_z0",
                "x0_y0_z1", "x1_y0_z1", "x1_y1_z1", "x0_y1_z1",
            ],
            "nodes": self.nodes,
            "elements": [asdict(element) for element in self.elements],
            "interface_pairs": self.interface_pairs,
            "surface_sets": {
                name: [asdict(face) for face in faces]
                for name, faces in self.surface_sets.items()
            },
            "node_sets": self.node_sets,
            "parameters": self.parameters,
            "starch_patches": self.starch_patches,
            "score": self.score,
            "tool_targets": self.tool_targets,
            "frame_convention": self.frame_convention,
            "provenance": self.provenance,
            "quality": self.quality,
        }


def export_mesh_json(mesh: HexMesh, path: str | Path) -> str:
    """Write stable JSON geometry; does not create the output directory."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory must already exist: {path.parent}")
    payload = json.dumps(mesh.to_dict(), indent=2, sort_keys=True,
                         allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


_BINARY_MAGIC = b"OCHEX8\x00\x01"
_BINARY_NODE = struct.Struct("<dddH")
_BINARY_ELEMENT = struct.Struct("<8IHBf")
_BINARY_INTERFACE = struct.Struct("<BHI B I B")
_BINARY_FACE = struct.Struct("<IB")


def export_mesh_binary(mesh: HexMesh, path: str | Path) -> str:
    """Stream a compact little-endian mesh archive without building JSON.

    Node coordinates, HEX8 connectivity/material frames, independent
    TYPE43 face references, named surface sets, and node sets are stored as
    fixed-size records. A small JSON header carries units and provenance.
    This avoids the multi-gigabyte expanded JSON representation for a full
    shipping blank. It is a geometry interchange file, not a solver deck.
    """
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory must already exist: {path.parent}")
    part_names = sorted({node["part"] for node in mesh.nodes} |
                        {element.part for element in mesh.elements})
    if len(part_names) >= 65536:
        raise ValueError("binary schema supports fewer than 65536 part names")
    part_ids = {name: i for i, name in enumerate(part_names)}
    material_ids = {"liner": 0, "medium": 1, "starch": 2}
    face_ids = {name: i for i, name in enumerate(_FACE_LOCAL_NODES)}
    interface_ids = {"medium_starch": 1, "starch_liner": 2}
    header = {
        "schema": "numi.cardboard.opencourant.hex8-binary.v1",
        "source_mesh_schema": mesh.schema,
        "units": mesh.units,
        "node_count": len(mesh.nodes),
        "hex8_count": len(mesh.elements),
        "interface_pair_count": len(mesh.interface_pairs),
        "surface_sets": {name: len(faces) for name, faces in mesh.surface_sets.items()},
        "node_sets": {name: len(nodes) for name, nodes in mesh.node_sets.items()},
        "part_names_by_id": part_names,
        "material_role_by_id": ["liner", "medium", "starch"],
        "face_name_by_id": list(_FACE_LOCAL_NODES),
        "interface_name_by_id": {str(v): k for k, v in interface_ids.items()},
        "interface_law": "TYPE43",
        "interface_law_material_id": None,
        "interface_property_binding_required": True,
        "parameters": mesh.parameters,
        "starch_patches": mesh.starch_patches,
        "score": mesh.score,
        "tool_targets": mesh.tool_targets,
        "frame_convention": mesh.frame_convention,
        "provenance": mesh.provenance,
        "quality": mesh.quality,
        "record_order": ["nodes", "hex8_elements", "interface_pairs",
                         "surface_sets", "node_sets"],
        "byte_order": "little-endian; packed with no alignment padding",
        "record_layout": {
            "file_prefix": "8-byte magic OCHEX8\\0\\1, then uint32 header byte length, then compact UTF-8 JSON header",
            "node": "<dddH: x_mm, y_mm, z_mm, part_id",
            "hex8_element": "<8IHBf: eight 1-based node IDs, part_id, material_role_id, frame_angle_rad",
            "interface_pair": "<BHIBIB: interface_id, patch_id, side_a_element_id, side_a_face_id, side_b_element_id, side_b_face_id",
            "surface_set": "uint16 name byte length, uint64 face count, UTF-8 name, then repeated <IB element_id/face_id",
            "node_set": "uint16 name byte length, uint64 node count, UTF-8 name, then repeated uint32 1-based node_id",
        },
    }
    header_bytes = json.dumps(header, sort_keys=True, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    with path.open("wb") as output:
        def write(data: bytes) -> None:
            output.write(data)
            digest.update(data)

        write(_BINARY_MAGIC)
        write(struct.pack("<I", len(header_bytes)))
        write(header_bytes)
        for node in mesh.nodes:
            xyz = node["xyz_mm"]
            write(_BINARY_NODE.pack(float(xyz[0]), float(xyz[1]),
                                    float(xyz[2]), part_ids[node["part"]]))
        for element in mesh.elements:
            if element.material_role not in material_ids:
                raise ValueError(f"unknown material role: {element.material_role}")
            write(_BINARY_ELEMENT.pack(
                *element.nodes, part_ids[element.part],
                material_ids[element.material_role],
                float(element.frame_angle_rad)))
        for pair in mesh.interface_pairs:
            kind = pair["interface"]
            if kind not in interface_ids:
                raise ValueError(f"unsupported interface kind: {kind}")
            side_a, side_b = pair["side_a"], pair["side_b"]
            write(_BINARY_INTERFACE.pack(
                interface_ids[kind], int(pair["patch_id"]),
                int(side_a["element_id"]), face_ids[side_a["face"]],
                int(side_b["element_id"]), face_ids[side_b["face"]]))
        for name, faces in mesh.surface_sets.items():
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > 65535:
                raise ValueError("surface-set name exceeds binary schema limit")
            write(struct.pack("<HQ", len(name_bytes), len(faces)))
            write(name_bytes)
            for face in faces:
                write(_BINARY_FACE.pack(face.element_id, face_ids[face.face]))
        for name, node_ids in mesh.node_sets.items():
            name_bytes = name.encode("utf-8")
            if len(name_bytes) > 65535:
                raise ValueError("node-set name exceeds binary schema limit")
            write(struct.pack("<HQ", len(name_bytes), len(node_ids)))
            write(name_bytes)
            for node_id in node_ids:
                write(struct.pack("<I", node_id))
    return digest.hexdigest()


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _round_unique(values: Iterable[float], tol: float = 1.0e-10) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tol:
            result.append(value)
    return result


class _Profile:
    def __init__(self, cfg: StripConfig) -> None:
        assert cfg.pitch_mm is not None
        assert cfg.caliper_mm is not None
        assert cfg.liner_thickness_mm is not None
        assert cfg.medium_thickness_mm is not None
        assert cfg.lower_starch_gap_mm is not None
        assert cfg.upper_starch_gap_mm is not None
        self.pitch = cfg.pitch_mm
        self.caliper = cfg.caliper_mm
        self.liner = cfg.liner_thickness_mm
        self.medium = cfg.medium_thickness_mm
        self.lower_gap = cfg.lower_starch_gap_mm
        self.upper_gap = cfg.upper_starch_gap_mm
        self.rise = (self.caliper - 2.0 * self.liner - self.medium -
                     self.lower_gap - self.upper_gap)
        self.amplitude = 0.5 * self.rise
        self.midline = 0.5 * ((self.liner + self.lower_gap) +
                              (self.caliper - self.liner - self.upper_gap))
        self.wave_number = 2.0 * math.pi / self.pitch

    def slope(self, u: float) -> float:
        return self.amplitude * self.wave_number * math.sin(self.wave_number * u)

    def angle(self, u: float) -> float:
        return math.atan(self.slope(u))

    def offset(self, u: float, signed_distance: float) -> tuple[float, float]:
        angle = self.angle(u)
        return (u - signed_distance * math.sin(angle),
                self.midline - self.amplitude * math.cos(self.wave_number * u) +
                signed_distance * math.cos(angle))

    def offset_x(self, u: float, signed_distance: float) -> float:
        return self.offset(u, signed_distance)[0]

    def inverse_offset_x(self, target: float, center: float,
                         signed_distance: float) -> float:
        if abs(target - center) <= 2.0e-14:
            return center
        if target < center:
            lo, hi = center - 0.5 * self.pitch, center
        else:
            lo, hi = center, center + 0.5 * self.pitch
        vlo, vhi = self.offset_x(lo, signed_distance), self.offset_x(hi, signed_distance)
        if target < vlo - 1.0e-10 or target > vhi + 1.0e-10:
            raise ValueError("starch footprint boundary is outside invertible half-pitch")
        for _ in range(72):
            mid = 0.5 * (lo + hi)
            if self.offset_x(mid, signed_distance) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _inverse_offset_global(profile: _Profile, target_x: float,
                           signed_distance: float) -> float:
    """Invert a monotone offset curve at a global X cut coordinate."""
    if abs(signed_distance) <= 1.0e-15:
        return target_x
    lo = target_x - abs(signed_distance) - 1.0e-12
    hi = target_x + abs(signed_distance) + 1.0e-12
    if not (profile.offset_x(lo, signed_distance) <= target_x <=
            profile.offset_x(hi, signed_distance)):
        raise ValueError("offset curve does not bracket the requested global X")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if profile.offset_x(mid, signed_distance) < target_x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _offset_surface_at_global_x(profile: _Profile, target_x: float,
                                signed_distance: float) -> tuple[float, float]:
    u = _inverse_offset_global(profile, target_x, signed_distance)
    _, z = profile.offset(u, signed_distance)
    return target_x, z


def _frame_axes(theta: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(theta), math.sin(theta)
    # Stored as columns: local MD tangent, CD along flute, caliper normal.
    # The flute profile is z = midline - A*cos(k*u), so its positive-u
    # tangent points toward +z on the rising side. Keep the local frame
    # right-handed: MD x CD = thickness.
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _det3(a: tuple[float, float, float],
          b: tuple[float, float, float],
          c: tuple[float, float, float]) -> float:
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) -
            a[1] * (b[0] * c[2] - b[2] * c[0]) +
            a[2] * (b[0] * c[1] - b[1] * c[0]))


def _jacobian_determinants(points: list[tuple[float, float, float]]) -> list[float]:
    # Eight Gauss points for trilinear HEX8, in the node order in the schema.
    g = 1.0 / math.sqrt(3.0)
    signs = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
    determinants = []
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                dxi = [0.0, 0.0, 0.0]
                deta = [0.0, 0.0, 0.0]
                dzeta = [0.0, 0.0, 0.0]
                for p, (sx, sy, sz) in zip(points, signs):
                    dxi_w = sx * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0
                    deta_w = sy * (1.0 + sx * xi) * (1.0 + sz * zeta) / 8.0
                    dzeta_w = sz * (1.0 + sx * xi) * (1.0 + sy * eta) / 8.0
                    for axis in range(3):
                        dxi[axis] += dxi_w * p[axis]
                        deta[axis] += deta_w * p[axis]
                        dzeta[axis] += dzeta_w * p[axis]
                determinants.append(_det3(tuple(dxi), tuple(deta), tuple(dzeta)))
    return determinants


class _HexBuilder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.node_ids: dict[tuple[str, tuple[int, int, int]], int] = {}
        self.elements: list[HexElement] = []
        self.part_faces: dict[str, dict[tuple[int, ...], list[FaceRef]]] = {}
        self.part_element_ids: dict[str, list[int]] = {}
        self.min_jacobian = math.inf
        self.volume_by_material: dict[str, float] = {}
        self.elements_by_material: dict[str, int] = {}

    @staticmethod
    def _key(xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        # 1e-9 mm key spacing is well below the smallest 0.030 mm bridge.
        return tuple(round(v * 1.0e9) for v in xyz)  # type: ignore[return-value]

    def _node(self, part: str, xyz: tuple[float, float, float]) -> int:
        if any(not math.isfinite(v) for v in xyz):
            raise ValueError("nonfinite mesh coordinate")
        key = (part, self._key(xyz))
        found = self.node_ids.get(key)
        if found is not None:
            return found
        node_id = len(self.nodes) + 1
        self.node_ids[key] = node_id
        self.nodes.append({"id": node_id, "xyz_mm": [float(v) for v in xyz],
                           "part": part})
        return node_id

    def add_hex(self, part: str, material_role: str,
                points: list[tuple[float, float, float]],
                frame_angle_rad: float) -> int:
        if len(points) != 8:
            raise ValueError("HEX8 must have exactly eight nodes")
        dets = _jacobian_determinants(points)
        local_min = min(dets)
        if not math.isfinite(local_min) or local_min <= 1.0e-14:
            raise ValueError(f"nonpositive or singular HEX8 Jacobian in {part}: {local_min}")
        self.min_jacobian = min(self.min_jacobian, local_min)
        self.volume_by_material[material_role] = (
            self.volume_by_material.get(material_role, 0.0) + sum(dets))
        self.elements_by_material[material_role] = (
            self.elements_by_material.get(material_role, 0) + 1)
        node_ids = tuple(self._node(part, p) for p in points)
        element_id = len(self.elements) + 1
        element = HexElement(element_id, node_ids, part, material_role,
                             float(frame_angle_rad), _frame_axes(frame_angle_rad))
        self.elements.append(element)
        self.part_element_ids.setdefault(part, []).append(element_id)
        face_map = self.part_faces.setdefault(part, {})
        for face_name, local_ids in _FACE_LOCAL_NODES.items():
            face_nodes = tuple(node_ids[i] for i in local_ids)
            key = tuple(sorted(face_nodes))
            face_map.setdefault(key, []).append(
                FaceRef(element_id, face_name, face_nodes))
        return element_id


def _face_centroid(face: FaceRef, nodes_by_id: dict[int, tuple[float, float, float]]) -> tuple[float, float, float]:
    xyz = [nodes_by_id[node_id] for node_id in face.nodes]
    return tuple(sum(p[k] for p in xyz) / 4.0 for k in range(3))  # type: ignore[return-value]


def _validate_config(cfg: StripConfig) -> None:
    for name in ("length_mm", "width_mm", "score_band_width_mm", "grip_width_mm",
                 "punch_radius_mm", "score_channel_width_mm"):
        _positive(name, float(getattr(cfg, name)))
    for name in ("pitch_mm", "caliper_mm", "liner_thickness_mm",
                 "medium_thickness_mm", "lower_starch_gap_mm",
                 "lower_starch_width_mm", "upper_starch_gap_mm",
                 "upper_starch_width_mm"):
        value = getattr(cfg, name)
        if value is None:
            raise ValueError(f"{name} did not resolve from the recipe")
        _positive(name, float(value))
    if cfg.flute_subdivisions_per_pitch < 4 or cfg.flute_subdivisions_per_pitch > 256:
        raise ValueError("flute_subdivisions_per_pitch must be in [4,256]")
    if cfg.width_subdivisions < 1 or cfg.width_subdivisions > 512:
        raise ValueError("width_subdivisions must be in [1,512]")
    if min(cfg.liner_thickness_slices, cfg.medium_thickness_slices,
           cfg.starch_thickness_slices) < 1:
        raise ValueError("all material thickness slice counts must be positive")
    assert cfg.pitch_mm is not None and cfg.caliper_mm is not None
    assert cfg.lower_starch_width_mm is not None and cfg.upper_starch_width_mm is not None
    if cfg.grip_width_mm * 2.0 >= cfg.length_mm:
        raise ValueError("left and right grip bands must not overlap")
    if not (0.0 < cfg.score_x_mm < cfg.length_mm):
        raise ValueError("score_x_mm must be strictly inside the coupon")
    if cfg.score_band_width_mm >= min(cfg.score_x_mm, cfg.length_mm - cfg.score_x_mm):
        raise ValueError("score band must fit inside coupon with positive margins")
    if not (cfg.lower_starch_width_mm < 0.5 * cfg.pitch_mm and
            cfg.upper_starch_width_mm < 0.5 * cfg.pitch_mm):
        raise ValueError("starch footprints must be narrower than half a flute pitch")
    if cfg.score_indentation_fraction_of_caliper <= 0.0:
        raise ValueError("score indentation fraction must be positive")
    rise = (cfg.caliper_mm - 2.0 * cfg.liner_thickness_mm -
            cfg.medium_thickness_mm - cfg.lower_starch_gap_mm -
            cfg.upper_starch_gap_mm)  # type: ignore[operator]
    if rise <= 0.0:
        raise ValueError("recipe leaves no positive flute centerline rise")
    max_curvature = 0.5 * rise * (2.0 * math.pi / cfg.pitch_mm) ** 2
    if 0.5 * cfg.medium_thickness_mm * max_curvature >= 1.0:  # type: ignore[operator]
        raise ValueError("normal-thickness medium offset would fold or become singular")


def _build_laminate_mesh(
    config: StripConfig,
    *,
    occupancy: Any = None,
    x_refinement_mm: Iterable[float] = (),
    y_refinement_mm: Iterable[float] = (),
    score_regions: list[dict[str, Any]] | None = None,
    require_whole_pitch: bool = False,
    section_cut_planes: bool = False,
) -> HexMesh:
    """Build a swept all-solid explicit-flute board over an optional 2-D mask.

    The mesh uses independently numbered material parts. Every starch-to-paper
    interface is represented by coincident, duplicated quadrilateral face
    references in ``interface_pairs``; the finite starch itself is also
    volumetrically meshed. The caller must attach the appropriate material law
    and execute score/fold/tool stages.
    """
    cfg = config.resolved()
    _validate_config(cfg)
    if require_whole_pitch and abs(cfg.length_mm / cfg.pitch_mm -
                                    round(cfg.length_mm / cfg.pitch_mm)) > 1.0e-10:
        raise ValueError("coupon length must span a whole number of source flute pitches")
    assert cfg.pitch_mm is not None and cfg.caliper_mm is not None
    assert cfg.liner_thickness_mm is not None and cfg.medium_thickness_mm is not None
    assert cfg.lower_starch_gap_mm is not None and cfg.lower_starch_width_mm is not None
    assert cfg.upper_starch_gap_mm is not None and cfg.upper_starch_width_mm is not None
    recipe, manifest_hash = _recipe()
    profile = _Profile(cfg)

    if cfg.length_mm < cfg.pitch_mm:
        raise ValueError("coupon must include at least one complete flute pitch")
    pitch_count = int(math.ceil(cfg.length_mm / cfg.pitch_mm))
    regular = [i * cfg.pitch_mm / cfg.flute_subdivisions_per_pitch
               for i in range(pitch_count * cfg.flute_subdivisions_per_pitch + 1)
               if i * cfg.pitch_mm / cfg.flute_subdivisions_per_pitch < cfg.length_mm]
    regular.append(cfg.length_mm)
    aux_x = [cfg.score_x_mm,
             cfg.score_x_mm - 0.5 * cfg.score_band_width_mm,
             cfg.score_x_mm + 0.5 * cfg.score_band_width_mm,
             cfg.grip_width_mm, cfg.length_mm - cfg.grip_width_mm,
             *x_refinement_mm]
    aux_x = _round_unique(aux_x)
    if any(x <= 0.0 or x >= cfg.length_mm for x in aux_x):
        raise ValueError("score/grip refinement lines must lie inside coupon")

    # Finite bridge patches exactly follow the source sinusoid's normal-offset
    # profile and source-recipe horizontal widths/gaps.
    patches: list[dict[str, Any]] = []
    for period in range(pitch_count + 1):
        for crest, center, width, offset in (
            (False, period * cfg.pitch_mm, cfg.lower_starch_width_mm,
             -0.5 * cfg.medium_thickness_mm),
            (True, (period + 0.5) * cfg.pitch_mm, cfg.upper_starch_width_mm,
             0.5 * cfg.medium_thickness_mm),
        ):
            if center > cfg.length_mm + 1.0e-12:
                continue
            x_left = max(0.0, center - 0.5 * width)
            x_right = min(cfg.length_mm, center + 0.5 * width)
            if not x_right > x_left:
                continue
            u_left = profile.inverse_offset_x(x_left, center, offset)
            u_right = profile.inverse_offset_x(x_right, center, offset)
            patches.append({
                "index": len(patches), "kind": "upper_crest" if crest else "lower_valley",
                "side": "upper" if crest else "lower", "period": period,
                "center_x_mm": center, "x_left_mm": x_left, "x_right_mm": x_right,
                "u_left_mm": u_left, "u_right_mm": u_right,
                "normal_offset_mm": offset,
                "requested_width_mm": width,
            })

    # Refine the medium grid at exact score/grip coordinates. If such a line
    # crosses a bond patch, invert the normal-offset map so the paper/starch
    # surface faces retain one-to-one segmentation.
    medium_u = list(regular) + list(aux_x)
    if section_cut_planes:
        for patch in patches:
            medium_u.extend((patch["x_left_mm"], patch["x_right_mm"]))
    else:
        for patch in patches:
            medium_u.extend((patch["u_left_mm"], patch["u_right_mm"]))
            offset = patch["normal_offset_mm"]
            for x in aux_x:
                if patch["x_left_mm"] + 1.0e-10 < x < patch["x_right_mm"] - 1.0e-10:
                    medium_u.append(profile.inverse_offset_x(
                        x, patch["center_x_mm"], offset))
    medium_u = _round_unique(medium_u)

    # Preserve the legacy section's liner segmentation policy: regular liner
    # nodes inside each bridge are replaced by the exact curved-face nodes,
    # preventing T-junctions at the finite starch interface.
    liner_x = [x for x in regular if not any(
        p["x_left_mm"] + 1.0e-10 < x < p["x_right_mm"] - 1.0e-10
        for p in patches)]
    liner_x.extend(aux_x)
    for patch in patches:
        if section_cut_planes:
            lo, hi = patch["x_left_mm"], patch["x_right_mm"]
        else:
            lo, hi = patch["u_left_mm"], patch["u_right_mm"]
        us = [u for u in medium_u if lo - 1.0e-10 <= u <= hi + 1.0e-10]
        if not us or abs(us[0] - lo) > 1.0e-9:
            us.append(lo)
        if not us or abs(us[-1] - hi) > 1.0e-9:
            us.append(hi)
        us = _round_unique(us)
        patch["u_nodes_mm"] = us
        patch["surface_nodes_mm"] = [
            (_offset_surface_at_global_x(profile, u, patch["normal_offset_mm"])
             if section_cut_planes else profile.offset(u, patch["normal_offset_mm"]))
            for u in us
        ]
        surface_x = [point[0] for point in patch["surface_nodes_mm"]]
        if any(b <= a for a, b in zip(surface_x, surface_x[1:])):
            raise ValueError("finite starch interface surface is not x-monotone")
        if abs(surface_x[0] - patch["x_left_mm"]) > 1.0e-8 or \
                abs(surface_x[-1] - patch["x_right_mm"]) > 1.0e-8:
            raise ValueError("finite starch footprint endpoints do not close")
        liner_x.extend(surface_x)
        liner_x.extend((patch["x_left_mm"], patch["x_right_mm"]))
    liner_x = _round_unique(liner_x)
    y_grid = _round_unique(
        [cfg.width_mm * j / cfg.width_subdivisions
         for j in range(cfg.width_subdivisions + 1)] + list(y_refinement_mm))
    if y_grid[0] < -1.0e-10 or y_grid[-1] > cfg.width_mm + 1.0e-10:
        raise ValueError("y refinement lies outside the swept board span")

    def occupied(x_mm: float, y_mm: float) -> bool:
        return True if occupancy is None else bool(occupancy(x_mm, y_mm))

    b = _HexBuilder()
    liner_cells: dict[str, dict[tuple[int, int, int], int]] = {
        "lower_liner": {}, "upper_liner": {}}
    medium_cells: dict[tuple[int, int, int], int] = {}
    starch_cells: dict[int, dict[tuple[int, int, int], int]] = {}

    # Two flat kraft liner bodies, each with its own material-node domain.
    # Element order: x/profile, y/flute axis, z/caliper.
    for part, z_inner, z_outer, outer_face in (
        ("lower_liner", 0.0, cfg.liner_thickness_mm, "z_minus"),
        ("upper_liner", cfg.caliper_mm - cfg.liner_thickness_mm,
         cfg.caliper_mm, "z_plus"),
    ):
        for ix, (x0, x1) in enumerate(zip(liner_x, liner_x[1:])):
            for iy, (y0, y1) in enumerate(zip(y_grid, y_grid[1:])):
                if not occupied(0.5 * (x0 + x1), 0.5 * (y0 + y1)):
                    continue
                for iz in range(cfg.liner_thickness_slices):
                    a0 = iz / cfg.liner_thickness_slices
                    a1 = (iz + 1) / cfg.liner_thickness_slices
                    z0 = z_inner + a0 * (z_outer - z_inner)
                    z1 = z_inner + a1 * (z_outer - z_inner)
                    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
                    eid = b.add_hex(part, "liner", pts, 0.0)
                    liner_cells[part][(ix, iy, iz)] = eid

    # The medium is a true normal-thickness sinusoidal solid, swept without
    # flattening its measured-source-parameterized centerline.
    medium_depths = [
        (iz / cfg.medium_thickness_slices - 0.5) * cfg.medium_thickness_mm
        for iz in range(cfg.medium_thickness_slices + 1)
    ]
    medium_section_nodes: list[list[tuple[float, float]]] | None = None
    medium_section_parameters: list[list[float]] | None = None
    if section_cut_planes:
        medium_section_nodes = [
            [_offset_surface_at_global_x(profile, x, depth) for x in medium_u]
            for depth in medium_depths
        ]
        medium_section_parameters = [
            [_inverse_offset_global(profile, x, depth) for x in medium_u]
            for depth in medium_depths
        ]
    for ix, (u0, u1) in enumerate(zip(medium_u, medium_u[1:])):
        if section_cut_planes:
            assert medium_section_parameters is not None
            theta = profile.angle(0.25 * (
                medium_section_parameters[0][ix] +
                medium_section_parameters[0][ix + 1] +
                medium_section_parameters[-1][ix] +
                medium_section_parameters[-1][ix + 1]))
        else:
            theta = profile.angle(0.5 * (u0 + u1))
        for iy, (y0, y1) in enumerate(zip(y_grid, y_grid[1:])):
            if not occupied(0.5 * (u0 + u1), 0.5 * (y0 + y1)):
                continue
            for iz in range(cfg.medium_thickness_slices):
                if section_cut_planes:
                    assert medium_section_nodes is not None
                    p00 = medium_section_nodes[iz][ix]
                    p10 = medium_section_nodes[iz][ix + 1]
                    p01 = medium_section_nodes[iz + 1][ix]
                    p11 = medium_section_nodes[iz + 1][ix + 1]
                else:
                    off0 = medium_depths[iz]
                    off1 = medium_depths[iz + 1]
                    p00 = profile.offset(u0, off0)
                    p10 = profile.offset(u1, off0)
                    p01 = profile.offset(u0, off1)
                    p11 = profile.offset(u1, off1)
                pts = [(p00[0], y0, p00[1]), (p10[0], y0, p10[1]),
                       (p10[0], y1, p10[1]), (p00[0], y1, p00[1]),
                       (p01[0], y0, p01[1]), (p11[0], y0, p11[1]),
                       (p11[0], y1, p11[1]), (p01[0], y1, p01[1])]
                eid = b.add_hex("nssc_medium", "medium", pts, theta)
                medium_cells[(ix, iy, iz)] = eid

    # Finite starch volumes follow the source's straight-sided interpolation
    # between each curved medium face and a flat liner face. Each pad is its
    # own body so interface identity remains independent.
    for patch in patches:
        patch_index = patch["index"]
        part = f"starch_{patch_index:03d}"
        starch_cells[patch_index] = {}
        surface_nodes = patch["surface_nodes_mm"]
        u_nodes = patch["u_nodes_mm"]
        if len(u_nodes) != len(surface_nodes):
            raise ValueError("starch patch surface grid is inconsistent")
        liner_face = (cfg.caliper_mm - cfg.liner_thickness_mm
                      if patch["side"] == "upper" else cfg.liner_thickness_mm)
        for iseg, ((x0, z0), (x1, z1)) in enumerate(zip(surface_nodes, surface_nodes[1:])):
            u0, u1 = u_nodes[iseg], u_nodes[iseg + 1]
            theta = profile.angle(0.5 * (u0 + u1))
            for iy, (y0, y1) in enumerate(zip(y_grid, y_grid[1:])):
                if not occupied(0.5 * (x0 + x1), 0.5 * (y0 + y1)):
                    continue
                for iz in range(cfg.starch_thickness_slices):
                    a0 = iz / cfg.starch_thickness_slices
                    a1 = (iz + 1) / cfg.starch_thickness_slices
                    if patch["side"] == "lower":
                        za0 = (1.0 - a0) * liner_face + a0 * z0
                        zb0 = (1.0 - a0) * liner_face + a0 * z1
                        za1 = (1.0 - a1) * liner_face + a1 * z0
                        zb1 = (1.0 - a1) * liner_face + a1 * z1
                    else:
                        # Low-z row is the medium crest; high-z row reaches
                        # the inner surface of the upper liner.
                        za0 = (1.0 - a0) * z0 + a0 * liner_face
                        zb0 = (1.0 - a0) * z1 + a0 * liner_face
                        za1 = (1.0 - a1) * z0 + a1 * liner_face
                        zb1 = (1.0 - a1) * z1 + a1 * liner_face
                    # These rows must be increasing in global Z, preserving
                    # positive orientation for the standard HEX8 ordering.
                    if not (za0 < za1 and zb0 < zb1):
                        raise ValueError("finite starch element has inverted thickness")
                    pts = [(x0, y0, za0), (x1, y0, zb0), (x1, y1, zb0), (x0, y1, za0),
                           (x0, y0, za1), (x1, y0, zb1), (x1, y1, zb1), (x0, y1, za1)]
                    eid = b.add_hex(part, "starch", pts, theta)
                    starch_cells[patch_index][(iseg, iy, iz)] = eid

    # Build independently paired TYPE43 face topology. Both sides have their
    # own nodes even though coordinates coincide; root supplies the law card.
    node_by_id = {node["id"]: tuple(node["xyz_mm"]) for node in b.nodes}
    elements_by_id = {e.id: e for e in b.elements}

    def face_ref(element_id: int, face: str) -> FaceRef:
        element = elements_by_id[element_id]
        return FaceRef(element_id, face,
                       tuple(element.nodes[i] for i in _FACE_LOCAL_NODES[face]))

    def nearest_index(values: list[float], value: float) -> int:
        best = min(range(len(values) - 1), key=lambda i: abs(values[i] - value))
        if abs(values[best] - value) > 2.0e-8:
            raise ValueError(f"cohesive surface x={value} mm has no matching liner segment")
        return best

    interface_pairs: list[dict[str, Any]] = []
    for patch in patches:
        patch_index = patch["index"]
        side = patch["side"]
        part = f"starch_{patch_index:03d}"
        liner_part = "upper_liner" if side == "upper" else "lower_liner"
        surf = patch["surface_nodes_mm"]
        us = patch["u_nodes_mm"]
        # The source section uses the medium's profile-grid intervals to split
        # each adhesive bridge. Resolve each one to matching structured cells.
        for iseg, ((x0, _), (x1, _)) in enumerate(zip(surf, surf[1:])):
            i0 = min(range(len(medium_u)), key=lambda i: abs(medium_u[i] - us[iseg]))
            i1 = min(range(len(medium_u)), key=lambda i: abs(medium_u[i] - us[iseg + 1]))
            if abs(medium_u[i0] - us[iseg]) > 2.0e-8 or \
                    abs(medium_u[i1] - us[iseg + 1]) > 2.0e-8:
                raise ValueError("starch interface node is absent from medium grid")
            if i1 != i0 + 1:
                raise ValueError("starch segment is not one medium grid interval")
            ix_liner = nearest_index(liner_x, x0)
            if abs(liner_x[ix_liner + 1] - x1) > 2.0e-8:
                raise ValueError("starch and liner interface segmentations differ")
            for iy in range(len(y_grid) - 1):
                y0, y1 = y_grid[iy], y_grid[iy + 1]
                if not occupied(0.5 * (x0 + x1), 0.5 * (y0 + y1)):
                    continue
                if side == "lower":
                    med_layer = 0
                    med_face = "z_minus"
                    glue_med_layer = cfg.starch_thickness_slices - 1
                    glue_med_face = "z_plus"
                    glue_liner_layer = 0
                    glue_liner_face = "z_minus"
                    liner_layer = cfg.liner_thickness_slices - 1
                    liner_face_name = "z_plus"
                else:
                    med_layer = cfg.medium_thickness_slices - 1
                    med_face = "z_plus"
                    glue_med_layer = 0
                    glue_med_face = "z_minus"
                    glue_liner_layer = cfg.starch_thickness_slices - 1
                    glue_liner_face = "z_plus"
                    liner_layer = 0
                    liner_face_name = "z_minus"
                med_id = medium_cells[(i0, iy, med_layer)]
                glue_med_id = starch_cells[patch_index][
                    (iseg, iy, glue_med_layer)]
                glue_liner_id = starch_cells[patch_index][
                    (iseg, iy, glue_liner_layer)]
                liner_id = liner_cells[liner_part][(ix_liner, iy, liner_layer)]
                pair_specs = (
                    ("medium_starch", face_ref(med_id, med_face),
                     face_ref(glue_med_id, glue_med_face), "medium", "starch"),
                    ("starch_liner", face_ref(glue_liner_id, glue_liner_face),
                     face_ref(liner_id, liner_face_name), "starch", "liner"),
                )
                for kind, face_a, face_b, role_a, role_b in pair_specs:
                    interface_pairs.append({
                        "id": len(interface_pairs) + 1,
                        "law": "TYPE43",
                        "law_material_id": None,
                        "interface": kind,
                        "patch_id": patch_index,
                        "patch_side": side,
                        "side_a_role": role_a,
                        "side_b_role": role_b,
                        "side_a": asdict(face_a),
                        "side_b": asdict(face_b),
                    })

    # Exposed faces are discovered from actual H8 topology, then cohesive face
    # references are removed so a contact deck cannot double-count bonds.
    paired_refs: set[tuple[int, str]] = set()
    for pair in interface_pairs:
        for side_key in ("side_a", "side_b"):
            face = pair[side_key]
            paired_refs.add((face["element_id"], face["face"]))
    boundary_faces: list[tuple[str, FaceRef]] = []
    for part, by_face in b.part_faces.items():
        for use in by_face.values():
            if len(use) == 1:
                boundary_faces.append((part, use[0]))
            elif len(use) != 2:
                raise ValueError(f"nonmanifold HEX8 face in part {part}")
    boundary_faces.sort(key=lambda item: (item[1].element_id,
                                          _FACE_ORDER[item[1].face]))

    def is_paired(face: FaceRef) -> bool:
        return (face.element_id, face.face) in paired_refs

    top_liner = [face for part, face in boundary_faces
                 if part == "upper_liner" and face.face == "z_plus"]
    bottom_liner = [face for part, face in boundary_faces
                    if part == "lower_liner" and face.face == "z_minus"]
    score_left = cfg.score_x_mm - 0.5 * cfg.score_band_width_mm
    score_right = cfg.score_x_mm + 0.5 * cfg.score_band_width_mm

    def centroid_x(face: FaceRef) -> float:
        return _face_centroid(face, node_by_id)[0]

    def in_band(face: FaceRef, lo: float, hi: float) -> bool:
        cx = centroid_x(face)
        return lo - 1.0e-9 <= cx <= hi + 1.0e-9

    score_top = [f for f in top_liner if in_band(f, score_left, score_right)]
    score_bottom = [f for f in bottom_liner if in_band(f, score_left, score_right)]
    grip_left_faces = [f for f in top_liner + bottom_liner
                       if in_band(f, 0.0, cfg.grip_width_mm)]
    grip_right_faces = [f for f in top_liner + bottom_liner
                        if in_band(f, cfg.length_mm - cfg.grip_width_mm,
                                   cfg.length_mm)]
    external = [(part, face) for part, face in boundary_faces if not is_paired(face)]
    left_cut_faces = [face for _, face in external if face.face == "x_minus"]
    right_cut_faces = [face for _, face in external if face.face == "x_plus"]
    exposed_medium = [face for part, face in external if part == "nssc_medium"]
    exposed_starch = [face for part, face in external if part.startswith("starch_")]
    cut_edges = [face for part, face in external
                 if face.face in ("x_minus", "x_plus", "y_minus", "y_plus")]
    surface_sets = {
        # Includes exposed flute void surfaces as well as the two outer faces;
        # the name avoids implying that every free surface is on the box exterior.
        "all_free_surfaces": [face for _, face in external],
        "upper_liner_outer": top_liner,
        "lower_liner_outer": bottom_liner,
        "score_punch_contact": score_top,
        "score_anvil_contact": score_bottom,
        "grip_left_contact": grip_left_faces,
        "grip_right_contact": grip_right_faces,
        "grip_left_cut_edge": left_cut_faces,
        "grip_right_cut_edge": right_cut_faces,
        "exposed_medium": exposed_medium,
        "exposed_starch_edges": exposed_starch,
        "coupon_cut_edges": cut_edges,
    }

    def node_ids_from_faces(faces: Iterable[FaceRef]) -> list[int]:
        return sorted({node for face in faces for node in face.nodes})

    # Grip node sets are boundary contact/cut surfaces only. No internal volume
    # node set is supplied for prescribed folding.
    grip_left_nodes = node_ids_from_faces(grip_left_faces + left_cut_faces)
    grip_right_nodes = node_ids_from_faces(grip_right_faces + right_cut_faces)
    node_sets = {
        "grip_left": grip_left_nodes,
        "grip_right": grip_right_nodes,
        "score_band": node_ids_from_faces(score_top + score_bottom),
    }

    min_determinant = b.min_jacobian
    if not math.isfinite(min_determinant):
        raise ValueError("empty H8 mesh")
    pair_counts: dict[str, int] = {}
    for pair in interface_pairs:
        pair_counts[pair["interface"]] = pair_counts.get(pair["interface"], 0) + 1
    mesh = HexMesh(
        schema=SCHEMA,
        units={"length": "mm", "coordinate_system": "global right-handed XYZ"},
        nodes=b.nodes,
        elements=b.elements,
        interface_pairs=interface_pairs,
        surface_sets=surface_sets,
        node_sets=node_sets,
        parameters={
            key: value for key, value in asdict(cfg).items()
        },
        starch_patches=[{
            key: value for key, value in patch.items()
            if key not in ("u_nodes_mm", "surface_nodes_mm")
        } for patch in patches],
        score={
            "state_at_export": "unscored; score must be physically formed by tool contact",
            "axis": "global Y",
            "centerline_x_mm": cfg.score_x_mm,
            "band_x_mm": [score_left, score_right],
            "band_width_mm": cfg.score_band_width_mm,
            "score_reference_plane_z_mm": cfg.caliper_mm,
            "geometry_type": "flat full-thickness corrugated laminate; no pre-imposed crease or flattened flute",
        },
        tool_targets={
            "punch_radius_mm": cfg.punch_radius_mm,
            "channel_width_mm": cfg.score_channel_width_mm,
            "indentation_mm": cfg.score_indentation_fraction_of_caliper * cfg.caliper_mm,
            "indentation_fraction_of_caliper": cfg.score_indentation_fraction_of_caliper,
            "status": "explicit test-fixture input; not a calibrated manufacturing recipe",
            "punch_surface_set": "score_punch_contact",
            "anvil_surface_set": "score_anvil_contact",
        },
        frame_convention={
            "global_x": "across flute profile / paper MD convention",
            "global_y": "along flute axes / coupon width",
            "global_z": "board caliper",
            "element_frame_axes": "columns are local MD, CD, thickness axes in global coordinates",
            "liner": "local MD is +global X",
            "medium": "local MD follows the sinusoidal centerline tangent in X-Z; CD is +global Y",
            "starch": "isotropic geometry frame; no anisotropic material claim",
        },
        provenance={
            "geometry_source": "matter/tools/cardboard_glue_mesh.hpp sinusoidal normal-offset finite-glue section, swept as structured HEX8",
            "geometry_source_git_revision": GEOMETRY_SOURCE_REVISION,
            "geometry_source_sha256": GEOMETRY_SOURCE_SHA256,
            "recipe_id": recipe.get("name", "literature2009"),
            "source_manifest": manifest_hash,
            "geometry_status": "derived sinusoidal fallback geometry; not a source-arc flute and not a same-roll reconstruction",
            "starch_geometry": "finite bridge widths/gaps preserved from recipe; material parameters and cohesive law are not assigned here",
            "physical_score_status": "tool-contact regions are mesh sets only; scoring has not been executed",
            "mesh_resolution_status": "fixture choice; spatial/material convergence not established",
        },
        quality={
            "node_count": len(b.nodes),
            "hex8_count": len(b.elements),
            "part_count": len(b.part_element_ids),
            "element_count_by_material": b.elements_by_material,
            "material_volume_mm3": b.volume_by_material,
            "interface_pair_count": len(interface_pairs),
            "interface_pair_counts": pair_counts,
            "min_gauss_jacobian_mm3": min_determinant,
            "flute_centerline_rise_mm": profile.rise,
            "flute_centerline_amplitude_mm": profile.amplitude,
            "flute_centerline_peak_to_trough_mm": profile.rise,
            "flute_samples_per_pitch": cfg.flute_subdivisions_per_pitch,
            "width_subdivisions": cfg.width_subdivisions,
            "liner_thickness_slices": cfg.liner_thickness_slices,
            "medium_thickness_slices": cfg.medium_thickness_slices,
            "starch_thickness_slices": cfg.starch_thickness_slices,
            "surface_sets_face_counts": {name: len(faces) for name, faces in surface_sets.items()},
            "node_set_counts": {name: len(nodes) for name, nodes in node_sets.items()},
            "whole_pitch_count": pitch_count,
        },
    )
    if score_regions is not None:
        top_faces: list[FaceRef] = []
        bottom_faces: list[FaceRef] = []
        line_node_ids: set[int] = set()
        line_metadata: list[dict[str, Any]] = []
        for region in score_regions:
            axis = region["axis"]
            center = float(region["center_mm"])
            span = tuple(float(v) for v in region["span_mm"])
            name = str(region["name"])
            lo = center - 0.5 * cfg.score_band_width_mm
            hi = center + 0.5 * cfg.score_band_width_mm

            def matches(face: FaceRef) -> bool:
                cx, cy, _ = _face_centroid(face, node_by_id)
                if axis == "x":
                    return lo - 1.0e-9 <= cx <= hi + 1.0e-9 and \
                        span[0] - 1.0e-9 <= cy <= span[1] + 1.0e-9
                if axis == "y":
                    return lo - 1.0e-9 <= cy <= hi + 1.0e-9 and \
                        span[0] - 1.0e-9 <= cx <= span[1] + 1.0e-9
                raise ValueError(f"unsupported score region axis: {axis}")

            top_region = [face for face in top_liner if matches(face)]
            bottom_region = [face for face in bottom_liner if matches(face)]
            mesh.surface_sets[f"{name}_punch_contact"] = top_region
            mesh.surface_sets[f"{name}_anvil_contact"] = bottom_region
            top_faces.extend(top_region)
            bottom_faces.extend(bottom_region)
            line_node_ids.update(node_ids_from_faces(top_region + bottom_region))
            line_metadata.append({
                "name": name,
                "coordinate_axis": "global_x" if axis == "x" else "global_y",
                "line_direction": "global_y" if axis == "x" else "global_x",
                "center_mm": center,
                "span_mm": list(span),
                "band_width_mm": cfg.score_band_width_mm,
                "punch_surface_set": f"{name}_punch_contact",
                "anvil_surface_set": f"{name}_anvil_contact",
            })
        mesh.surface_sets["score_punch_contact"] = top_faces
        mesh.surface_sets["score_anvil_contact"] = bottom_faces
        mesh.node_sets["score_band"] = sorted(line_node_ids)
        mesh.score["state_at_export"] = "unscored; all score lines require physical tool contact"
        mesh.score["score_lines"] = line_metadata
        mesh.quality["score_line_surface_face_counts"] = {
            row["name"]: {
                "punch": len(mesh.surface_sets[f"{row['name']}_punch_contact"]),
                "anvil": len(mesh.surface_sets[f"{row['name']}_anvil_contact"]),
            } for row in line_metadata
        }
    _validate_mesh(mesh)
    return mesh


def build_score_fold_strip(config: StripConfig | None = None) -> HexMesh:
    """Build the source-recipe all-solid score/fold strip fixture."""
    cfg = (config or StripConfig()).resolved()
    return _build_laminate_mesh(cfg, require_whole_pitch=True)


def build_fefco0201_blank(config: BlankConfig | None = None) -> HexMesh:
    """Build an unscored, all-solid FEFCO 0201 blank with explicit slots.

    The flat occupancy consists of four L-W-L-W body panels, an explicit
    25 mm body-height-only manufacturer tab, four half-W flaps at both ends,
    and three through-board 1 mm slot strips at each end. No score damage or
    folded deformation is prescribed.
    """
    blank = config or BlankConfig()
    cfg, layout = blank.resolved_strip_config()
    panel_spans = tuple(tuple(float(v) for v in row)
                        for row in layout["panel_spans_x_mm"])
    tab_x0, tab_x1 = (float(v) for v in layout["tab_span_x_mm"])
    score_xs = [float(v) for v in layout["vertical_score_x_mm"]]
    slot_xs = [float(v) for v in layout["slot_boundaries_x_mm"]]
    y0, y1 = layout["flap_score_y_mm"]
    y0, y1 = float(y0), float(y1)
    half_kerf = 0.5 * blank.slot_kerf_mm

    def occupied(x_mm: float, y_mm: float) -> bool:
        if y0 <= y_mm <= y1:
            return (panel_spans[0][0] <= x_mm <= panel_spans[-1][1] or
                    tab_x0 <= x_mm <= tab_x1)
        if not any(lo <= x_mm <= hi for lo, hi in panel_spans):
            return False
        return not any(center - half_kerf <= x_mm <= center + half_kerf
                       for center in slot_xs)

    score_regions: list[dict[str, Any]] = []
    body_y = (y0, y1)
    for i, x in enumerate(score_xs):
        score_regions.append({"name": f"score_panel_fold_{i + 1}", "axis": "x",
                              "center_mm": x, "span_mm": body_y})
    for i, (xlo, xhi) in enumerate(panel_spans):
        score_regions.append({"name": f"score_top_flap_panel_{i + 1}", "axis": "y",
                              "center_mm": y0, "span_mm": (xlo, xhi)})
        score_regions.append({"name": f"score_bottom_flap_panel_{i + 1}", "axis": "y",
                              "center_mm": y1, "span_mm": (xlo, xhi)})

    mesh = _build_laminate_mesh(
        cfg,
        occupancy=occupied,
        x_refinement_mm=layout["x_refinement_mm"],
        y_refinement_mm=layout["y_refinement_mm"],
        score_regions=score_regions,
        section_cut_planes=True,
    )
    for name in ("grip_left_contact", "grip_right_contact",
                 "grip_left_cut_edge", "grip_right_cut_edge"):
        mesh.surface_sets.pop(name, None)
    mesh.node_sets.pop("grip_left", None)
    mesh.node_sets.pop("grip_right", None)
    mesh.score.update({
        "axis": "global X and Y; listed score centers are unscored tool paths",
        "layout": layout,
        "slot_cut_geometry": "1 mm material-absent strips through the complete board stack",
        "physical_score_status": "score bands are contact sets only; no score process executed",
    })
    mesh.parameters["fefco0201_layout"] = layout
    mesh.provenance["blank_geometry"] = (
        "four-panel L-W-L-W RSC occupancy with body-height-only manufacturer tab; "
        "alternating top/bottom flap sectors and paired slot kerfs are absent cells"
    )
    mesh.quality["blank_occupied_cell_area_mm2"] = _blank_occupied_area(blank)
    mesh.quality["blank_score_allowance_status"] = layout["score_allowance_status"]
    mesh.quality["blank_internal_dimensions_mm"] = None
    mesh.quality["surface_sets_face_counts"] = {
        name: len(faces) for name, faces in mesh.surface_sets.items()
    }
    mesh.quality["node_set_counts"] = {
        name: len(nodes) for name, nodes in mesh.node_sets.items()
    }
    return mesh


def _blank_occupied_area(config: BlankConfig) -> float:
    """Exact planar area represented before flute-grid discretization."""
    body = 2.0 * (config.panel_length_mm + config.panel_width_mm) * config.wall_height_mm
    tab = config.manufacturer_tab_width_mm * config.wall_height_mm
    flaps = 2.0 * config.panel_width_mm * (config.panel_length_mm + config.panel_width_mm)
    slots = 6.0 * config.slot_kerf_mm * 0.5 * config.panel_width_mm
    return body + tab + flaps - slots


def _validate_mesh(mesh: HexMesh) -> None:
    if not mesh.nodes or not mesh.elements:
        raise ValueError("mesh has no nodes or HEX8 elements")
    node_ids = {node["id"] for node in mesh.nodes}
    xyz_by_id = {node["id"]: tuple(node["xyz_mm"]) for node in mesh.nodes}
    if node_ids != set(range(1, len(mesh.nodes) + 1)):
        raise ValueError("node IDs are not dense one-based indices")
    element_ids = {element.id for element in mesh.elements}
    if element_ids != set(range(1, len(mesh.elements) + 1)):
        raise ValueError("element IDs are not dense one-based indices")
    for element in mesh.elements:
        if len(set(element.nodes)) != 8 or not set(element.nodes) <= node_ids:
            raise ValueError(f"invalid node connectivity in HEX8 {element.id}")
    for pair in mesh.interface_pairs:
        a = pair["side_a"]["nodes"]
        b = pair["side_b"]["nodes"]
        if len(a) != 4 or len(b) != 4:
            raise ValueError("interface pair must contain two quadrilateral faces")
        pa = sorted(tuple(round(v, 9) for v in xyz_by_id[node_id]) for node_id in a)
        pb = sorted(tuple(round(v, 9) for v in xyz_by_id[node_id]) for node_id in b)
        if pa != pb:
            raise ValueError(f"coincident duplicated interface faces disagree in pair {pair['id']}")
    for name, faces in mesh.surface_sets.items():
        for face in faces:
            if face.element_id not in element_ids or len(face.nodes) != 4:
                raise ValueError(f"invalid face in surface set {name}")


def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="output path; parent directory must already exist")
    parser.add_argument("--blank", action="store_true",
                        help="build the provisional FEFCO 0201 RSC blank")
    parser.add_argument("--format", choices=("json", "binary"),
                        help="output encoding (blank defaults to compact binary)")
    parser.add_argument("--length-mm", type=float, default=31.6)
    parser.add_argument("--width-mm", type=float, default=20.0)
    parser.add_argument("--score-x-mm", type=float)
    parser.add_argument("--flute-subdivisions-per-pitch", type=int)
    parser.add_argument("--width-subdivisions", type=int)
    parser.add_argument("--panel-length-mm", type=float, default=300.0)
    parser.add_argument("--panel-width-mm", type=float, default=200.0)
    parser.add_argument("--wall-height-mm", type=float, default=200.0)
    parser.add_argument("--manufacturer-tab-width-mm", type=float, default=25.0)
    parser.add_argument("--slot-kerf-mm", type=float, default=1.0)
    parser.add_argument("--blank-axis-spacing-mm", type=float, default=10.0)
    args = parser.parse_args()
    if args.blank:
        cfg = BlankConfig(
            panel_length_mm=args.panel_length_mm,
            panel_width_mm=args.panel_width_mm,
            wall_height_mm=args.wall_height_mm,
            manufacturer_tab_width_mm=args.manufacturer_tab_width_mm,
            slot_kerf_mm=args.slot_kerf_mm,
            flute_subdivisions_per_pitch=(args.flute_subdivisions_per_pitch or 8),
            blank_axis_spacing_mm=args.blank_axis_spacing_mm,
        )
        mesh = build_fefco0201_blank(cfg)
    else:
        cfg = StripConfig(length_mm=args.length_mm, width_mm=args.width_mm,
                          score_x_mm=args.score_x_mm,
                          flute_subdivisions_per_pitch=(args.flute_subdivisions_per_pitch or 16),
                          width_subdivisions=(args.width_subdivisions or 8))
        mesh = build_score_fold_strip(cfg)
    output_format = args.format or ("binary" if args.blank else "json")
    digest = (export_mesh_binary(mesh, args.output) if output_format == "binary"
              else export_mesh_json(mesh, args.output))
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      **mesh.quality}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
