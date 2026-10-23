#!/usr/bin/env python3
"""Source-bound nominal tensile transfer and implicit LAW112 coupons.

The post-yield input is a uniaxial conversion of the four Haj-Ali et al.
bilinear *fit lines*, not specimen-level calibration.  This module emits
small verification coupons; it never launches OpenCourant/Radioss.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cardboard_opencourant_paper as paper


ROOT = Path(__file__).resolve().parents[2]
FIT_NODES = ROOT / "matter/examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv"
YIELD_ANCHORS = ROOT / "matter/examples/cardboard/source-data/haj_ali_2009_yield_anchors.csv"
MANIFEST = ROOT / "matter/examples/cardboard/manufacturing-source-manifest.json"
PARAMETERS = paper.PARAMETERS
SOURCE_COMMIT = paper.SOURCE_COMMIT
SCHEMA = "cardboard.opencourant.material-transfer.v1"
COUPON_SCHEMA = "cardboard.opencourant.material-tensile-coupon.v1"

FIT_SHA256 = "83f5778db797f0fb9996416848abb91c18f9102f4c3707aea9ea267b1eeb0d82"
YIELD_SHA256 = "ede6b47229eb02f1d1de278035c995eaa358919c55384398cbe33fe3de3c9f04"
MANIFEST_SHA256 = "6f3c556d6ffc97998d243601a6bfb34a1e8735026f06d6c0e8a8aec625a7d68f"
PARAMETERS_SHA256 = "46d33c7e6e4c796ad3285e269df03b1a0a6442dc70e6cb86de8a38ab5568527b"
ENGINE_LAW112_SHA256 = "c437b2a6a75c71bd4ff9f95aebdf0190e76a1df7c9a53183c904bd3184b0c1b5"
STARTER_LAW112_SHA256 = "6208b17da98e26f3b2aa549f9f5a3d48b1901d51eaa44e721a4f5d48d29119df"

# Haj-Ali does not report tensile conditioning or loading rate in the
# accessible figure/text record.  This is a declared coupon-instrument choice,
# not a recovered source condition; no source-rate acceleration is asserted.
ENGINEERING_STRAIN_RATE_S_1 = 1.0e-2
GAUGE_LENGTH_MM = 10.0
SAMPLE_COUNT = 100

_ROLE_DATA = {
    "liner": {"source_component": "liner", "basis_weight_g_m2": 205},
    "medium": {"source_component": "medium_NSSC", "basis_weight_g_m2": 126},
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_hashes() -> dict[str, str]:
    mapping, _ = paper._read_parameters()
    hashes = {item["path"]: item["sha256"] for item in mapping["source_inputs"]}
    binding = mapping["opencourant_binding"]
    for path_key, hash_key in (
            ("law112_cfg_path", "law112_cfg_sha256"),
            ("starter_law112_reader_path", "starter_law112_reader_sha256"),
            ("engine_law112_path", "engine_law112_sha256"),
            ("one_hex_qa_starter_path", "one_hex_qa_starter_sha256"),
            ("one_hex_qa_engine_path", "one_hex_qa_engine_sha256")):
        hashes[f"OpenCourant/{binding['commit']}/{binding[path_key]}"] = binding[hash_key]
    hashes.update({
        str(FIT_NODES.relative_to(ROOT)): FIT_SHA256,
        str(YIELD_ANCHORS.relative_to(ROOT)): YIELD_SHA256,
        str(MANIFEST.relative_to(ROOT)): MANIFEST_SHA256,
        str(PARAMETERS.relative_to(ROOT)): PARAMETERS_SHA256,
    })
    expected = {
        FIT_NODES: FIT_SHA256,
        YIELD_ANCHORS: YIELD_SHA256,
        MANIFEST: MANIFEST_SHA256,
        PARAMETERS: PARAMETERS_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"pinned source hash mismatch: {path.relative_to(ROOT)}")
    if (binding.get("engine_law112_sha256") != ENGINE_LAW112_SHA256 or
            binding.get("starter_law112_reader_sha256") != STARTER_LAW112_SHA256):
        raise ValueError("LAW112 implementation hashes do not match the pinned source record")
    return dict(sorted(hashes.items()))


def _source_points() -> dict[tuple[str, str], dict[str, Any]]:
    # Source-derived cards must fail closed if either tabular input has drifted.
    _source_hashes()
    with FIT_NODES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with YIELD_ANCHORS.open(newline="", encoding="utf-8") as stream:
        anchors = list(csv.DictReader(stream))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for role, metadata in _ROLE_DATA.items():
        component = metadata["source_component"]
        for axis in ("MD", "CD"):
            key = (role, axis)
            fit_rows = [row for row in rows
                        if row["component"] == component and row["axis"] == axis]
            endpoint_rows = [row for row in fit_rows if row["point_role"] == "digitized_bilinear_fit_endpoint"]
            if len(endpoint_rows) != 1:
                raise ValueError(f"expected one digitized endpoint for {role}/{axis}")
            endpoint = endpoint_rows[0]
            anchor_rows = [row for row in anchors if row["source_id"] == "haj_ali_2009"
                           and row["component"] == component and row["axis"] == axis
                           and row["loading"] == "tension"]
            if len(anchor_rows) != 1:
                raise ValueError(f"expected one tensile yield anchor for {role}/{axis}")
            anchor = anchor_rows[0]
            E = float(endpoint["table2_elastic_modulus_MPa"])
            S0 = float(anchor["stress_MPa"])
            eps_y = float(anchor["strain"])
            eps_endpoint = float(endpoint["strain_mm_per_mm"])
            sigma_endpoint = float(endpoint["stress_MPa"])
            Et = (sigma_endpoint - S0) / (eps_endpoint - eps_y)
            Et_csv = float(endpoint["postyield_tangent_MPa"])
            if not math.isclose(Et, Et_csv, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"fit endpoint does not reproduce stored tangent for {role}/{axis}")
            if not (0.0 < Et < E):
                raise ValueError(f"invalid bilinear tangent for {role}/{axis}")
            H = E * Et / (E - Et)
            H_low = float(endpoint["uniaxial_plastic_hardening_modulus_low_MPa"])
            H_high = float(endpoint["uniaxial_plastic_hardening_modulus_high_MPa"])
            H_csv = float(endpoint["uniaxial_plastic_hardening_modulus_MPa"])
            if not math.isclose(H, H_csv, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"uniaxial H conversion does not match fit record for {role}/{axis}")
            by_key[key] = {
                "role": role,
                "source_component": component,
                "basis_weight_g_m2": int(metadata["basis_weight_g_m2"]),
                "axis": axis,
                "elastic_modulus_MPa": E,
                "yield_anchor": {
                    "stress_MPa": S0,
                    "strain_mm_per_mm": eps_y,
                    "observation_class": anchor["observation_class"],
                    "source_location": anchor["source_location"],
                },
                "digitized_fit_endpoint": {
                    "stress_MPa": sigma_endpoint,
                    "strain_mm_per_mm": eps_endpoint,
                    "source_location": endpoint["source_location"],
                    "observation_class": endpoint["observation_class"],
                    "pixel_readout_halfwidth_px": float(endpoint["pixel_readout_halfwidth_px"]),
                    "axis_line_halfwidth_px": float(endpoint["axis_line_halfwidth_px"]),
                },
                "postyield_total_tangent_MPa": Et,
                "plastic_hardening_modulus_MPa": H,
                "plastic_hardening_modulus_pixel_bound_MPa": [H_low, H_high],
                "LAW112_tensile_direction_parameters": {"S0_MPa": S0, "A_MPa": 0.0,
                                                         "B_per_plastic_strain": 0.0,
                                                         "C_MPa": H},
                "nominal_transfer_prediction_MPa": sigma_endpoint,
                "test_rate_status": "not stated in available source figure/text",
            }
    return by_key


def source_record(role: str) -> dict[str, Any]:
    """Return hash-bound input data and explicit limits for one material role."""
    if role not in _ROLE_DATA:
        raise ValueError("role must be 'liner' or 'medium'")
    points = _source_points()
    mapping, _ = paper._read_parameters()
    return {
        "schema": SCHEMA,
        "material_role": role,
        "literature_source": "Haj-Ali et al. (2009), Figs. 1(a,b)/2(a,b), Table 1 and Table 2",
        "hardening_transfer": {
            "official_pinned_engine_source": {
                "commit": SOURCE_COMMIT,
                "path": "engine/source/materials/mat/mat112/mat112_xia_newton.F",
                "sha256": ENGINE_LAW112_SHA256,
                "equation": "sigmaY = S0 + A*tanh(B*epsp) + C*epsp",
            },
            "pinned_source_bindings": mapping["opencourant_binding"],
            "choice": "A=0, B=0, C=H for the source's bilinear uniaxial tension fit",
            "uniaxial_conversion": "H = E*Et/(E-Et), with Et from the yield-anchor to digitized-fit-endpoint tangent",
            "branches": {axis: points[(role, axis)] for axis in ("MD", "CD")},
        },
        "source_hashes_sha256": _source_hashes(),
        "elastic_and_density_mapping": _base_field_provenance(role),
        "unidentified_parameters": [
            {"parameter": "in_plane_and_out_of_plane_shear_strengths", "status": "not identified for this grade/axis set"},
            {"parameter": "LAW112 plastic Poisson ratios", "status": "synthetic zero card placeholders; not measured"},
            {"parameter": "ZD and compression hardening", "status": "zero A/B/C card placeholders; not identified"},
        ],
        "compression_mapping": {
            "status": "retained Haj-Ali Table 3 compression strengths as source modeling assumptions; not compression measurements",
            "hardening_coefficients": "A=B=C=0 explicit perfect-plastic placeholder, not an identified compression fit",
        },
        "limitations": [
            "The fit-node file contains a single pixel-read endpoint per published fitted curve, not raw specimen traces or replicate identities.",
            "Endpoint pixel bounds are visual readout bounds, not statistical confidence intervals or grade-transfer uncertainty.",
            "Uniaxial H does not identify a multiaxial Hill/LAW112 plastic hardening surface.",
            "Test rate and conditioning are not stated; the coupon rate is an explicit engineering choice and has no source-rate acceleration factor.",
            "The four native coupons verify coefficient parsing, material axes, and uniaxial transfer response only; they are not independent physical validation.",
        ],
    }


def _base_field_provenance(role: str) -> dict[str, Any]:
    _, medium_fields = paper._read_parameters()
    fields = (_liner_law112_fields(medium_fields) if role == "liner"
              else copy.deepcopy(medium_fields))
    result = {
        key: {"value": item["value"], "unit": item.get("unit"),
              "origin_class": item.get("origin_class"), "source": item.get("source"),
              **({"limitation": item["limitation"]} if "limitation" in item else {})}
        for key, item in fields.items()
    }
    # The liner mapping intentionally reuses the medium map's slot structure;
    # its numeric overrides therefore need role-correct provenance here.
    result.pop("hardening_A_B_C_fields", None)
    if role == "liner":
        source_overrides = {
            "rho_i": ("cross_source_proxy", "Popil 2017 Table 4.1, separate 42# brown kraft liner coupon, 766 kg/m^3; not the Haj-Ali 205 gsm test sheet"),
            "E1": ("published_value", "Haj-Ali 2009 Table 2, 205 gsm liner MD"),
            "E2": ("published_value", "Haj-Ali 2009 Table 2, 205 gsm liner CD"),
            "E3": ("published_estimate", "Haj-Ali 2009 Table 2, source estimate E33=E11/250"),
            "nu21": ("derived_from_published_value", "Reciprocal relation nu21=nu12*E2/E1 using liner Table 2 nu12=0.18"),
            "G12": ("published_relation_value", "Haj-Ali 2009 Table 2, in-plane shear modulus"),
            "G23": ("published_model_input", "Haj-Ali 2009 Table 2; direct-measurement basis is not documented"),
            "G13": ("published_model_input", "Haj-Ali 2009 Table 2; direct-measurement basis is not documented"),
            "S01": ("published_tensile_yield_anchor", "Haj-Ali 2009 Table 1, 205 gsm liner MD tensile yield"),
            "S02": ("published_tensile_yield_anchor", "Haj-Ali 2009 Table 1, 205 gsm liner CD tensile yield"),
            "S04": ("published_compression_model_assumption", "Haj-Ali 2009 Table 3 liner MD compression value; source model assumption, not measurement"),
            "S05": ("published_compression_model_assumption", "Haj-Ali 2009 Table 3 liner CD compression value; source model assumption, not measurement"),
            "ASIG": ("derived_compression_model_assumption", "Haj-Ali 2009 Table 3 liner ZD value based on cited relation; not direct measurement"),
            "E3C": ("law_default_mapped_to_source_estimate", "LAW112 default E3C=E3 using Haj-Ali Table 2 E33 estimate"),
        }
        for key, (origin, source) in source_overrides.items():
            result[key]["origin_class"] = origin
            result[key]["source"] = source
    return result


def _material_fields(role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if role not in _ROLE_DATA:
        raise ValueError("role must be 'liner' or 'medium'")
    mapping, medium_fields = paper._read_parameters()
    fields = (_liner_law112_fields(medium_fields) if role == "liner"
              else copy.deepcopy(medium_fields))
    points = _source_points()
    coefficients = [[0.0, 0.0, points[(role, "MD")]["plastic_hardening_modulus_MPa"]],
                    [0.0, 0.0, points[(role, "CD")]["plastic_hardening_modulus_MPa"]],
                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    fields["hardening_A_B_C_fields"] = {
        "value": coefficients,
        "unit": "LAW112 S0i-associated coefficients; tensile C in MPa",
        "origin_class": "derived_uniaxial_source_fit_estimate_plus_explicit_unidentified_placeholders",
        "source": "Directions 1-2 use source-derived H; directions 3-5 remain explicit zero placeholders because matching multiaxial hardening is unidentified.",
    }
    return mapping, fields


def _liner_law112_fields(medium_fields: dict[str, Any]) -> dict[str, Any]:
    """Map the published liner inputs into the existing LAW112 field slots.

    This is the narrow LAW112 source mapping already used by the strip
    instrument. Keep it here so the strip adapter can consume ``law_card``
    without creating a material/strip import cycle. Compression and transverse
    assumptions remain inherited from the existing medium template.
    """
    fields = copy.deepcopy(medium_fields)
    values = {
        "rho_i": 7.66e-10,       # separate Popil coupon proxy
        "E1": 4775.0, "E2": 2088.0, "E3": 19.1,
        "nu21": 0.18 * 2088.0 / 4775.0,
        "G12": 1222.0, "G23": 137.28, "G13": 166.28,
        "S01": 38.2, "S02": 18.4,
        "S04": 22.92, "S05": 11.02, "ASIG": 7.52,
        "E3C": 19.1,
    }
    for key, value in values.items():
        fields[key]["value"] = value
    return fields


def _law_card_for_fields(material_id: int, role: str, fields: dict[str, Any]) -> str:
    if isinstance(material_id, bool) or not isinstance(material_id, int) or material_id <= 0:
        raise ValueError("material_id must be a positive integer")
    value = lambda key: float(fields[key]["value"])
    plastic_nu = fields["nu1p_nu2p_nu4p_nu5p"]["value"]
    hard = fields["hardening_A_B_C_fields"]["value"]
    if len(hard) != 5 or any(len(row) != 3 for row in hard):
        raise ValueError("LAW112 requires five S0/A/B/C direction rows")
    lines = [
        f"/MAT/LAW112/{material_id}",
        f"Haj-Ali 2009 {role} tensile-fit estimate transfer; explicit unidentified branches",
        paper._float_fields(value("rho_i")),
        paper._float_fields(value("E1"), value("E2"), value("E3")) +
        paper._int_fields(int(value("Ires")), int(value("Itab")), int(value("Ismooth"))),
        paper._float_fields(value("nu21"), value("G12"), value("G23"), value("G13")),
        paper._float_fields(value("K"), value("E3C"), value("CC")),
        paper._float_fields(*map(float, plastic_nu)),
    ]
    for name, coefficients in zip(("S01", "S02", "S03", "S04", "S05"), hard):
        lines.append(paper._float_fields(value(name), *map(float, coefficients)))
    lines.append(paper._float_fields(value("ASIG"), value("BSIG"), value("CSIG")))
    lines.append(paper._float_fields(value("TAU0"), value("ATAU"), value("BTAU")))
    return "\n".join(lines)


def law_card(material_id: int, role: str) -> str:
    """Build a LAW112 material card with source-fit tensile MD/CD C values."""
    _, fields = _material_fields(role)
    return _law_card_for_fields(material_id, role, fields)


def _coupon_starter(stem: str, role: str, axis: str, fields: dict[str, Any],
                    end_time: float, strain_rate_s_1: float,
                    active_stop_time: float | None = None) -> str:
    nodes = [
        (1, 0, 0, 0), (2, 10, 0, 0), (3, 10, 10, 0), (4, 0, 10, 0),
        (5, 0, 0, 10), (6, 10, 0, 10), (7, 10, 10, 10), (8, 0, 10, 10),
    ]
    node_rows = "\n".join(f"{nid:10d}{x:20.10g}{y:20.10g}{z:20.10g}"
                           for nid, x, y, z in nodes)
    if axis == "MD":
        fixed, driven, component = (1, 4, 5, 8), (2, 3, 6, 7), "X"
        extra_groups = ((3, "y z anchor at node 1", (1,)), (4, "z rotation anchor", (4,)))
        bcs_rows = ((1, "reference face x", "100000", 1),
                    (2, "anchor node 1 y", "010000", 3),
                    (3, "anchor node 1 z", "001000", 3),
                    (4, "anchor node 4 z", "001000", 4))
    elif axis == "CD":
        fixed, driven, component = (1, 2, 5, 6), (3, 4, 7, 8), "Y"
        extra_groups = ((3, "x z anchor at node 1", (1,)), (4, "z rotation anchor", (2,)))
        bcs_rows = ((1, "reference face y", "010000", 1),
                    (2, "anchor node 1 x", "100000", 3),
                    (3, "anchor node 1 z", "001000", 3),
                    (4, "anchor node 2 z", "001000", 4))
    else:
        raise ValueError("axis must be MD or CD")
    if active_stop_time is None:
        active_stop_time = end_time
    if not math.isfinite(active_stop_time) or active_stop_time < end_time:
        raise ValueError("active_stop_time must be finite and at or after analysis end")
    velocity = strain_rate_s_1 * GAUGE_LENGTH_MM
    function = "\n".join(("/FUNCT/1", "constant uniaxial tensile velocity multiplier",
                          paper._float_fields(0.0, 1.0), paper._float_fields(active_stop_time, 1.0)))
    groups = ["/GRNOD/NODE/1\nfixed reference face\n" +
              "".join(f"{node:10d}" for node in fixed),
              "/GRNOD/NODE/2\nloaded face\n" +
              "".join(f"{node:10d}" for node in driven)]
    for gid, title, members in extra_groups:
        groups.append(f"/GRNOD/NODE/{gid}\n{title}\n" +
                      "".join(f"{node:10d}" for node in members))
    bcs = [f"/BCS/{cid}\n{title}\n   {mask[:3]} {mask[3:]}{0:10d}{gid:10d}"
           for cid, title, mask, gid in bcs_rows]
    impvel = "\n".join(("/IMPVEL/1/0", f"constant {axis} tensile velocity",
                        f"{1:10d}{component:>10}{0:10d}{0:10d}{2:10d}{0:10d}{0:10d}",
                        paper._float_fields(1.0, velocity, 0.0, active_stop_time)))
    prop = "\n".join((
        "/PROP/TYPE6/1", "fully integrated orthotropic eight-node brick",
        paper._mixed_fields(17, 2, None, 0, 0, 0, 0, -1, 0.0),
        paper._float_fields(0.0, 0.0, 0.0),
        paper._float_fields(0.0, 0.0, 0.0) + paper._int_fields(0, 20, 0),
        paper._float_fields(0.0, 0.0, 0.0, 0.0),
        paper._float_fields(0.0, 0.0, 0.0, 0.0, 0.0),
    ))
    th_nodes = "\n".join(f"{node:10d}{0:10d}" for node in range(1, 9))
    th = "\n".join((
        "/TH/BRIC/1", "native stress, strain and internal energy history",
        "".join(f"{v:<10}" for v in ("STRESS", "STRAIN", "IE")), f"{1:10d}",
        "/TH/NODE/2", "native displacement and support reaction history",
        "".join(f"{v:<10}" for v in ("DX", "DY", "DZ", "REACX", "REACY", "REACZ")),
        th_nodes,
    ))
    group_text = "\n".join(groups)
    bcs_text = "\n".join(bcs)
    return f"""#RADIOSS STARTER
# OpenCourant {paper.RELEASE}; source {SOURCE_COMMIT}
# One-element uniaxial material-transfer instrument; no physical qualification.
/BEGIN/
{stem}
{2021:10d}{0:10d}
{paper._float_fields(1.0, 1.0, 1.0)}
{paper._float_fields(1.0, 1.0, 1.0)}
/TITLE
One-element {role} {axis} tensile LAW112 transfer coupon
/ANALY
{0:41d}
{0:41d}
/DEF_SOLID
{paper._int_fields(0, 0, 2)}
{_law_card_for_fields(1, role, fields)}
{prop}
/PART/1
paper coupon instrument element
{paper._int_fields(1, 1, 0)}
/NODE
{node_rows}
/BRICK/1
{paper._int_fields(1, 1, 2, 3, 4, 5, 6, 7, 8)}
{group_text}
{bcs_text}
{function}
{impvel}
{th}
/ANIM/VERS
36
/END
"""


def _coupon_engine(stem: str, end_time: float) -> str:
    # Sufficient for root-side native runs while retaining a fixed, source-
    # independent temporal discretization. No mass scaling or inertia.
    history_dt = end_time / SAMPLE_COUNT
    animation_dt = end_time / 40.0
    initial_dt = min(1.0e-2, history_dt)
    return f"""#RADIOSS ENGINE
/IMPL/NONL/01
20 12 1e-6 1e-6 1e-6
/IMPL/DTINI
{paper._f(initial_dt)}
/IMPL/DT/1
0 0 0 0
/IMPL/SOLV/01
{1:10d}{0:10d}{3:10d}{1e-10:20g}
/IMPL/PRINT/NONLIN/1
/TFILE/4
{paper._f(history_dt)}
/ANIM/DT
0.0 {paper._f(animation_dt)}
/ANIM/BRICK/TENS/STRESS
/ANIM/BRICK/EPSP
/ANIM/BRICK/ENER
/ANIM/VECT/DISP
/TH/TITLE
/MON/ON
/RUN/{stem}/1
{paper._f(end_time)}
"""


def export_tensile_coupons(output: Path, *, strain_rate_s_1: float = ENGINEERING_STRAIN_RATE_S_1,
                           endpoint_hold_s: float = 0.0) -> list[dict[str, Any]]:
    """Export four implicit MD/CD tensile coupons; does not invoke the solver.

    ``output`` must be a pre-created, empty-or-independent suite directory.
    The engineering strain rate is explicit because no corresponding source
    test rate is reported in the accessible article record.
    """
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"suite output directory must already exist: {output}")
    if not math.isfinite(strain_rate_s_1) or strain_rate_s_1 <= 0:
        raise ValueError("strain_rate_s_1 must be finite and positive")
    if not math.isfinite(endpoint_hold_s) or endpoint_hold_s < 0:
        raise ValueError("endpoint_hold_s must be finite and non-negative")
    source_hashes = _source_hashes()
    mapping, _ = paper._read_parameters()
    all_points = _source_points()
    exported: list[dict[str, Any]] = []
    for role in ("liner", "medium"):
        _, fields = _material_fields(role)
        for axis in ("MD", "CD"):
            curve = all_points[(role, axis)]
            target_strain = float(curve["digitized_fit_endpoint"]["strain_mm_per_mm"])
            end_time = target_strain / strain_rate_s_1
            active_stop_time = end_time + endpoint_hold_s
            if not math.isfinite(active_stop_time):
                raise ValueError("analysis end plus endpoint_hold_s must be finite")
            case_id = f"law112-{role}-{axis.lower()}-tensile-fit-transfer-implicit"
            case_dir = output / case_id
            case_dir.mkdir(parents=False, exist_ok=False)
            stem = case_id.replace("-", "_")
            starter_name = f"{stem}_0000.rad"
            engine_name = f"{stem}_0001.rad"
            (case_dir / starter_name).write_text(
                _coupon_starter(stem, role, axis, fields, end_time, strain_rate_s_1,
                                active_stop_time), encoding="utf-8")
            (case_dir / engine_name).write_text(_coupon_engine(stem, end_time), encoding="utf-8")
            record = source_record(role)
            record["case"] = {"id": case_id, "axis": axis,
                              "source_curve": curve,
                              "rate_choice": {"strain_rate_s-1": strain_rate_s_1,
                                              "status": "explicit engineering choice; article rate unavailable; no rate-acceleration factor applied"},
                              "target_engineering_strain": target_strain,
                              "target_displacement_mm": target_strain * GAUGE_LENGTH_MM,
                              "expected_end_time_s": end_time,
                              "imposed_motion_window": {
                                  "analysis_end_time_s": end_time,
                                  "active_stop_time_s": active_stop_time,
                                  "additional_active_time_s": endpoint_hold_s,
                                  "purpose": "Keep the same constant-rate imposed velocity active beyond the analysis endpoint; no earlier release or rate change.",
                                  "native_source_basis": {
                                      "repository": "OpenCourant/OpenRadioss",
                                      "commit": "cb760c206e346eff12161a2268b134ef9d48a4ef",
                                      "reader_path": "starter/source/constraints/general/impvel/read_impvel.F",
                                      "reader_git_blob_sha1": "4cdc9486967a46a7d65f7f4e0f203cdad7ff1862",
                                      "reader_lines": [165, 247, 278],
                                      "engine_path": "engine/source/constraints/general/impvel/fv_imp0.F",
                                      "engine_git_blob_sha1": "da31dab9ed767d1633d36b1075c8d90eee9d8aa1",
                                      "engine_lines": [267, 270, 291, 294],
                                      "active_rule": "The implicit evaluator skips imposed velocity only when TT > STOPT; TT == STOPT remains active.",
                                  },
                              },
                              "expected_LAW112_endpoint_stress_MPa": curve["digitized_fit_endpoint"]["stress_MPa"],
                              "source_fit_endpoint_stress_MPa": curve["digitized_fit_endpoint"]["stress_MPa"],
                              "stress_prediction_basis": "source-labeled bilinear fit endpoint; the four native coupons test whether LAW112 uniaxial flow reproduces this transfer",
                              "stress_component": "SX" if axis == "MD" else "SY",
                              "fixed_face_nodes": [1, 4, 5, 8] if axis == "MD" else [1, 2, 5, 6],
                              "loaded_face_nodes": [2, 3, 6, 7] if axis == "MD" else [3, 4, 7, 8],
                              "history": {"file_suffix": "T01", "node_channel": 2, "brick_channel": 1,
                                          "reaction_channel_semantics": "native T01 REAC components require the pinned converter's impulse-window conversion; do not read raw T01 reaction values as force",
                                          "animation": ["BRICK/TENS/STRESS", "BRICK/EPSP", "BRICK/ENER", "VECT/DISP"]},
                              "material_fields": {key: value["value"] for key, value in fields.items()},
                              "pinned_parameter_map_sha256": PARAMETERS_SHA256,
                              "pinned_law112_engine_source_sha256": ENGINE_LAW112_SHA256}
            manifest = {
                "schema": COUPON_SCHEMA,
                "id": case_id,
                "starter": starter_name,
                "engines": [engine_name],
                "physical_qualification": False,
                "native_run_required": True,
                "expected_end_time_s": end_time,
                "source_hashes_sha256": source_hashes,
                "source_record": record,
                "gates": {
                    "normal_termination": True,
                    "all_outputs_finite": True,
                    "loaded_face_displacement_mm": {"expected": target_strain * GAUGE_LENGTH_MM,
                                                      "absolute_error_max_mm": 1.0e-6},
                    "final_axis_stress_MPa": {"expected": curve["digitized_fit_endpoint"]["stress_MPa"],
                                               "relative_error_max": 0.01,
                                               "comparison_basis": "source-labeled bilinear tension-fit endpoint; not independent physical validation"},
                },
                "expected_observables": ["prescribed loaded-face displacement",
                                         "converted support reaction for force-work diagnostics",
                                         "loaded-direction brick stress and strain, plastic strain, and internal energy"],
            }
            (case_dir / "source-record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            (case_dir / "case.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            exported.append(manifest)
    return exported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--strain-rate", type=float, default=ENGINEERING_STRAIN_RATE_S_1,
                        help="engineering coupon strain rate in s^-1 (source rate is unreported)")
    parser.add_argument("--endpoint-hold", type=float, default=0.0,
                        help="keep the same constant-rate imposed velocity active this many seconds after the analysis endpoint")
    args = parser.parse_args()
    print(json.dumps(export_tensile_coupons(args.output, strain_rate_s_1=args.strain_rate,
                                            endpoint_hold_s=args.endpoint_hold),
                     indent=2, sort_keys=True))
