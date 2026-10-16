#!/usr/bin/env python3
"""Export small, source-bound Radioss LAW112 paper instrument decks.

The decks exercise the pinned OpenCourant input/solver path.  Their LAW112
cards contain cross-law mappings and explicit synthetic/default values; they
are not a calibrated Xia law or a corrugated-board prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


RELEASE = "latest-20261011"
SOURCE_COMMIT = "cb760c206e346eff12161a2268b134ef9d48a4ef"
SCHEMA = "cardboard.opencourant.paper.case.v1"
ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ROOT / "matter/examples/cardboard/opencourant/paper-parameters.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _f(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("nonfinite value in generated Radioss deck")
    return f"{value:.10E}"


def _float_fields(*values: float) -> str:
    return "".join(f"{_f(v):>20}" for v in values)


def _int_fields(*values: int) -> str:
    return "".join(f"{v:10d}" for v in values)


def _mixed_fields(*values: int | float | None) -> str:
    """Radioss fixed-width mixed property row (blank entries stay blank)."""
    return "".join(" " * 10 if value is None else
                   (f"{value:10d}" if isinstance(value, int) else f"{_f(float(value)):>20}")
                   for value in values)


def _read_parameters() -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    if parameters.get("opencourant_binding", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("parameter mapping is not bound to the pinned OpenCourant source")
    for item in parameters.get("source_inputs", []):
        path = ROOT / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise ValueError(f"source input hash mismatch: {item['path']}")
    return parameters, parameters["law112_fields"]


def _law_card(stem: str, fields: dict[str, Any]) -> str:
    """LAW112 continuous yield card; values retain their JSON provenance."""
    v = lambda key: float(fields[key]["value"])
    plastic_nu = fields["nu1p_nu2p_nu4p_nu5p"]["value"]
    lines = [
        "/MAT/LAW112/1",
        "Published medium mapping plus explicit instrument-only fields",
        _float_fields(v("rho_i")),
        _float_fields(v("E1"), v("E2"), v("E3")) + _int_fields(int(v("Ires")), int(v("Itab")), int(v("Ismooth"))),
        _float_fields(v("nu21"), v("G12"), v("G23"), v("G13")),
        _float_fields(v("K"), v("E3C"), v("CC")),
        _float_fields(*map(float, plastic_nu)),
    ]
    # Itab=0 requires S0i,Ai,Bi,Ci for five directions, followed by ASIG and TAU0 rows.
    hard = fields["hardening_A_B_C_fields"]["value"]
    if hard != [[0.0, 0.0, 0.0] for _ in range(5)]:
        raise ValueError("this exporter only supports its explicitly declared zero-hardening fixture")
    for name, coefficients in zip(("S01", "S02", "S03", "S04", "S05"), hard):
        lines.append(_float_fields(v(name), *map(float, coefficients)))
    lines.append(_float_fields(v("ASIG"), v("BSIG"), v("CSIG")))
    lines.append(_float_fields(v("TAU0"), v("ATAU"), v("BTAU")))
    return "\n".join(lines)


def _starter(stem: str, *, direction: str, strain: float, parameters: dict[str, Any],
             fields: dict[str, Any], end_time: float) -> str:
    # A 10-mm cube, x along paper MD, y along CD, z through thickness.
    nodes = [
        (1, 0, 0, 0), (2, 10, 0, 0), (3, 10, 10, 0), (4, 0, 10, 0),
        (5, 0, 0, 10), (6, 10, 0, 10), (7, 10, 10, 10), (8, 0, 10, 10),
    ]
    node_rows = "\n".join(f"{nid:10d}{x:20.10g}{y:20.10g}{z:20.10g}" for nid, x, y, z in nodes)
    if direction == "MD":
        fixed_nodes, driven_nodes, axis = (1, 4, 5, 8), (2, 3, 6, 7), "X"
        # Fix only the loading-axis displacement on the reference face. The
        # two point anchors remove rigid translation/rotation while allowing
        # uniform transverse Poisson contraction in a one-element patch test.
        extra_groups = ((3, "y z anchor at node 1", (1,)), (4, "z rotation anchor", (4,)))
        bc_rows = ((1, "reference face x", "100000", 1),
                   (2, "anchor node 1 y", "010000", 3),
                   (3, "anchor node 1 z", "001000", 3),
                   (4, "anchor node 4 z", "001000", 4))
    elif direction == "CD":
        fixed_nodes, driven_nodes, axis = (1, 2, 5, 6), (3, 4, 7, 8), "Y"
        extra_groups = ((3, "x z anchor at node 1", (1,)), (4, "z rotation anchor", (2,)))
        bc_rows = ((1, "reference face y", "010000", 1),
                   (2, "anchor node 1 x", "100000", 3),
                   (3, "anchor node 1 z", "001000", 3),
                   (4, "anchor node 2 z", "001000", 4))
    else:
        raise ValueError(f"unsupported first-case direction: {direction}")
    target_displacement = strain * 10.0
    # Velocity profile is trapezoidal: 0 -> 1 over 10%, hold to 90%, then 0.
    # Its exact area is .05T + .80T + .05T = .90T.
    peak_velocity = target_displacement / (0.9 * end_time)
    function = "\n".join((
        "/FUNCT/1", "dimensionless trapezoidal velocity multiplier",
        _float_fields(0.0, 0.0),
        _float_fields(0.1 * end_time, 1.0),
        _float_fields(0.9 * end_time, 1.0),
        _float_fields(end_time, 0.0),
    ))
    grnode_parts = ["/GRNOD/NODE/1", "fixed reference face",
                    "".join(f"{n:10d}" for n in fixed_nodes),
                    "/GRNOD/NODE/2", "loaded face",
                    "".join(f"{n:10d}" for n in driven_nodes)]
    for group_id, title, node_ids in extra_groups:
        grnode_parts.extend((f"/GRNOD/NODE/{group_id}", title,
                             "".join(f"{n:10d}" for n in node_ids)))
    grnodes = "\n".join(grnode_parts)
    bcs_parts = []
    for bcs_id, title, mask, group_id in bc_rows:
        # The six translational/rotational constraint flags occupy the first
        # ten columns, followed by input-system and group identifiers.
        bcs_parts.extend((f"/BCS/{bcs_id}", title,
                          f"   {mask[:3]} {mask[3:]}{0:10d}{group_id:10d}"))
    bcs = "\n".join(bcs_parts)
    impvel = "\n".join((
        "/IMPVEL/1/0", "prescribed paper-direction velocity",
        f"{1:10d}{axis:>10}{0:10d}{0:10d}{2:10d}{0:10d}{0:10d}",
        _float_fields(1.0, peak_velocity, 0.0, end_time),
    ))
    # LAW112 is an orthotropic solid material (PROP_SOLID=2 in the pinned
    # reader), so TYPE6 is required. TYPE14 is incompatible and fails before
    # the material can be exercised. Isolid=17 is the full-integration H8C
    # brick; Ip=20 aligns its local frame to N1-N2/N1-N4, matching MD/CD axes.
    property_type6 = "\n".join((
        "/PROP/TYPE6/1",
        "full-integration orthotropic eight-node brick",
        _mixed_fields(17, 2, None, 0, 0, 0, 0, -1, 0.0),
        _float_fields(0.0, 0.0, 0.0),             # q_a, q_b, h
        _float_fields(0.0, 0.0, 0.0) + _int_fields(0, 20, 0),  # V, skew, Ip, Iorth
        _float_fields(0.0, 0.0, 0.0, 0.0),       # phi, P_x, P_y, P_z
        _float_fields(0.0, 0.0, 0.0, 0.0, 0.0), # TYPE6 bounds
    ))
    th_node_rows = "\n".join(f"{n:10d}{0:10d}" for n in range(1, 9))
    th = "\n".join((
        "/TH/BRIC/1", "element stress, strain, internal energy",
        "".join(f"{v:<10}" for v in ("STRESS", "STRAIN", "IE")),
        f"{1:10d}",
        "/TH/NODE/2", "nodal displacement and reaction history",
        "".join(f"{v:<10}" for v in ("DX", "DY", "DZ", "REACX", "REACY", "REACZ")),
        th_node_rows,
    ))
    return f"""#RADIOSS STARTER
# OpenCourant {RELEASE}; source {SOURCE_COMMIT}
# Instrument-only LAW112 mapping. No target-grade calibration claim.
/BEGIN/
{stem}
{2021:10d}{0:10d}
{_float_fields(1.0, 1.0, 1.0)}
{_float_fields(1.0, 1.0, 1.0)}
/TITLE
One eight-node paperboard LAW112 directional instrument fixture
/ANALY
{0:41d}
/DEF_SOLID
{_int_fields(0, 0, 2)}
{_law_card(stem, fields)}
{property_type6}
/PART/1
paperboard instrument element
{_int_fields(1, 1, 0)}
/NODE
{node_rows}
/BRICK/1
{_int_fields(1, 1, 2, 3, 4, 5, 6, 7, 8)}
{grnodes}
{bcs}
{function}
{impvel}
{th}
/ANIM/VERS
36
/END
"""


def _engine(stem: str, end_time: float) -> str:
    return f"""#RADIOSS ENGINE
/VERS/41
/RUN/{stem}/1
{_f(end_time)}
/DT
0.9 0.0
/ANIM/DT
0.0 {_f(end_time / 40.0)}
/TFILE/4
{_f(end_time / 40.0)}
/PRINT/-1000
/ANIM/ELEM/SIGX
/ANIM/ELEM/SIGY
/ANIM/ELEM/SIGZ
/ANIM/ELEM/ENER
/ANIM/VECT/VEL
/ANIM/VECT/DISP
/MON/ON
"""


def export_paper_cases(output: Path) -> list[dict[str, Any]]:
    """Export a runnable first LAW112 case and bounded follow-up directions.

    ``output`` must be a new suite directory.  No solver is launched here.
    """
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"suite output directory must already exist: {output}")
    mapping, fields = _read_parameters()
    sidecar_sha = _sha(PARAMETERS)
    end_time = 1.0e-3
    # Parser-first case stays below the published MD tensile-yield anchor.
    # Remaining cases deliberately probe directional sign and history paths;
    # they do not turn sparse anchors into full curves or calibration.
    cases = [
        ("law112-md-tension-instrument", "MD", 0.001),
        ("law112-md-compression-instrument", "MD", -0.001),
        ("law112-cd-tension-instrument", "CD", 0.001),
        ("law112-cd-compression-instrument", "CD", -0.001),
    ]
    exported: list[dict[str, Any]] = []
    for case_id, direction, strain in cases:
        case_dir = output / case_id
        case_dir.mkdir(parents=False, exist_ok=False)
        stem = case_id.replace("-", "_")
        starter = f"{stem}_0000.rad"
        engine = f"{stem}_0001.rad"
        (case_dir / starter).write_text(_starter(stem, direction=direction, strain=strain,
                                                  parameters=mapping, fields=fields, end_time=end_time), encoding="utf-8")
        (case_dir / engine).write_text(_engine(stem, end_time), encoding="utf-8")
        (case_dir / "parameters.json").write_bytes(PARAMETERS.read_bytes())
        modulus_key = "E1" if direction == "MD" else "E2"
        expected_stress = float(fields[modulus_key]["value"]) * strain
        stress_component = "SX" if direction == "MD" else "SY"
        case = {
            "schema": SCHEMA,
            "id": case_id,
            "starter": starter,
            "engines": [engine],
            "expected_end_time_s": end_time,
            "parameters": {"file": "parameters.json", "sha256": sidecar_sha,
                           "calibration_scope": mapping["calibration_scope"]},
            "source_commit": SOURCE_COMMIT,
            "release": RELEASE,
            "calibration_scope": mapping["calibration_scope"],
            "protocol": {"element_count": 1, "element": "8-node fully integrated solid brick",
                         "paper_axis": direction, "target_engineering_strain": strain,
                         "prescribed_face_displacement_mm": 10.0 * strain,
                         "velocity_profile": "trapezoidal, 10% ramp, 80% hold, 10% ramp-down",
                         "velocity_integral_fraction": 0.9},
            "expected_observables": [
                "solver normal termination and finite model outputs",
                "prescribed face displacement and loaded-direction reaction history",
                "element stress, strain, internal energy, and directional sign consistency",
            ],
            "gates": {
                "deck_and_engine_complete": True,
                "all_observed_values_finite": True,
                "final_loaded_face_displacement_mm": {
                    "expected": 10.0 * strain,
                    "absolute_error_max_mm": 1.0e-6,
                },
                "elastic_anchor": {
                    "sample": f"mean {stress_component} over final 1% of run",
                    "expected_stress_MPa": expected_stress,
                    "relative_error_max": 0.01,
                    "scope": "one-element linear-elastic instrument check below the mapped directional yield anchor",
                },
            },
            "physical_qualification": False,
            "limitations": mapping["nonclaims"],
        }
        (case_dir / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        exported.append(case)
    return exported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(export_paper_cases(args.output), indent=2, sort_keys=True, allow_nan=False))
