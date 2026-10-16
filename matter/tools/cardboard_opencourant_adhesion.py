#!/usr/bin/env python3
"""Export tiny OpenCourant LAW117/TYPE43 instrument-admission decks.

The material values below are deliberately synthetic.  These decks exercise
the pinned solver's LAW117 input, history, failure, unload, and compression
paths; they are not parameters for corrugated board, kraft, NSSC, or starch.

This module is a deck exporter only.  It does not launch a solver or interpret
solver output as physical qualification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SOURCE_COMMIT = "cb760c206e346eff12161a2268b134ef9d48a4ef"
RELEASE = "latest-20261011"
SCHEMA = "cardboard.opencourant.adhesion.case.v1"

# One synthetic bilinear cohesive law in mm, s, Mg, N, MPa.  The peak
# separation is TN/EN = 0.1 mm; the chosen triangular-law area is
# 0.5 * 10 MPa * 0.3 mm = 1.5 N/mm for each pure mode.  These values only
# make the numerical paths easy to exercise and have no calibration meaning.
MATERIAL = {
    "law": 117,
    "density_Mg_per_mm3": 1.0e-6,
    "normal_stiffness_MPa_per_mm": 100.0,
    "shear_stiffness_MPa_per_mm": 100.0,
    "peak_normal_MPa": 10.0,
    "peak_shear_MPa": 10.0,
    "GIC_N_per_mm": 1.5,
    "GIIC_N_per_mm": 1.5,
    "rupture_criterion": "power_law",
    "power_law_exponent": 2.0,
    "mixed_mode_gamma": 1.0,
    "failure_displacement_mm": {"mode_I": 0.3, "mode_II": 0.3},
    "failure_points_to_delete": 1,
    "compression_flag_isym": 0,
    "mass_flag": 1,
    "criterion": "displacement_based",
    "evolution": "linear",
}

END_TIME = 1.0e-3
SPLIT_TIME = 5.0e-4
ACTIVATION_TIME = 5.0e-4
NODAL_TIME_STEP_SCALE = 0.02
MINIMUM_TIME_STEP_S = 1.0e-12


def _fmt(x: float) -> str:
    return f"{x:.9g}"


def _starter(
    stem: str,
    mode: str,
    functions: list[list[tuple[float, float]]],
    *,
    activation_time: float | None = None,
) -> str:
    """Return a self-contained one-element, zero-thickness cohesive model."""
    # TYPE43 defines local t from node face 1-2-3-4 to 5-6-7-8.  The two faces
    # are initially coincident so the LAW117 separation starts at zero.
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 0.0), (6, 1.0, 0.0, 0.0),
        (7, 1.0, 1.0, 0.0), (8, 0.0, 1.0, 0.0),
    ]
    node_rows = "\n".join(f"{i:10d}{x:20.9g}{y:20.9g}{z:20.9g}" for i, x, y, z in nodes)
    funct_blocks = []
    for fid, points in enumerate(functions, start=1):
        rows = "\n".join(f"{_fmt(t):>20}{_fmt(u):>20}" for t, u in points)
        funct_blocks.append(f"/FUNCT/{fid}\nadhesion motion {fid}\n#       time (s)        displacement (mm)\n{rows}")

    # Restrict the non-driven top-face axes so this fixture has only the
    # intended uniform mode.  The driven axis is prescribed by /IMPDISP.
    fixed_mask = {"I": "110 000", "II": "011 000", "mixed": "010 000"}[mode]
    displacement_axes = {"I": [("Z", 1)], "II": [("X", 1)],
                         "mixed": [("Z", 1), ("X", 2)]}[mode]
    imp_blocks = []
    for imp_id, (axis, fid) in enumerate(displacement_axes, start=1):
        imp_blocks.append(
            f"/IMPDISP/{imp_id}\nPrescribed {axis} separation\n"
            f"# Ifunct       DIR     Iskew   Isensor   Gnod_id     Frame     Icoor\n"
            f"{fid:10d}{axis:>10}{0:10d}{0:10d}{2:10d}{0:10d}{0:10d}\n"
            f"#            Scale_x             Scale_y              Tstart               Tstop\n"
            f"{1.0:20.9g}{1.0:20.9g}{0.0:20.9g}{END_TIME:20.9g}"
        )

    functions_text = "\n".join(funct_blocks)
    imps_text = "\n".join(imp_blocks)
    activation_block = ""
    if activation_time is not None:
        group_id = 7
        activation_fields = [0, group_id, 0, 0, 0, 0, 0, 0, None, 2]
        activation_row = "".join(
            f"{'':>10}" if value is None else f"{value:10d}"
            for value in activation_fields
        )
        activation_block = (
            f"/GRBRIC/BRIC/{group_id}\n"
            "single cohesive brick activation group\n"
            f"{1:10d}\n"
            "/ACTIV/1\n"
            "activate dormant cohesive brick at translated geometry\n"
            f"{activation_row}\n"
            f"{activation_time:20.9g}{1.0e30:20.9g}"
        )
    return f"""#RADIOSS STARTER
# Generated for OpenCourant {RELEASE} at {SOURCE_COMMIT}
# Synthetic single-element LAW117/TYPE43 instrument case; no contact overlay.
/BEGIN
{stem}
2026
{'Mg':>20}{'mm':>20}{'s':>20}
{'Mg':>20}{'mm':>20}{'s':>20}
/TITLE
Synthetic LAW117 cohesive admission fixture
/ANALY
         0                   0
/MAT/LAW117/1
Synthetic only; not kraft, NSSC, or starch calibration
{MATERIAL['density_Mg_per_mm3']:20.9g}
{MATERIAL['normal_stiffness_MPa_per_mm']:20.9g}{MATERIAL['shear_stiffness_MPa_per_mm']:20.9g}{1:10d}{MATERIAL['failure_points_to_delete']:10d}{1:10d}{MATERIAL['compression_flag_isym']:10d}
{0:10d}{0:10d}{MATERIAL['peak_normal_MPa']:20.9g}{MATERIAL['peak_shear_MPa']:20.9g}{0.0:20.9g}
{MATERIAL['GIC_N_per_mm']:20.9g}{MATERIAL['GIIC_N_per_mm']:20.9g}{MATERIAL['power_law_exponent']:20.9g}{MATERIAL['power_law_exponent']:20.9g}{MATERIAL['mixed_mode_gamma']:20.9g}
{1:10d}{1:10d}{MATERIAL['failure_displacement_mm']['mode_I']:20.9g}{MATERIAL['failure_displacement_mm']['mode_II']:20.9g}{3.0:20.9g}
/PROP/TYPE43/1
Synthetic zero-thickness cohesive solid
         1
/NODE
{node_rows}
/GRNOD/NODE/1
fixed bottom face
{1:10d}{2:10d}{3:10d}{4:10d}
/GRNOD/NODE/2
driven top face
{5:10d}{6:10d}{7:10d}{8:10d}
/BCS/1
fixed bottom translations
{'111 000':>10}{0:10d}{1:10d}
/BCS/2
fix nondriven top axes
{fixed_mask:>10}{0:10d}{2:10d}
/PART/1
Synthetic cohesive connection
         1         1         0
/BRICK/1
         1         1         2         3         4         5         6         7         8
{activation_block}
{functions_text}
{imps_text}
/TH/NODE/101
bottom reaction history
{''.join(f'{v:<10}' for v in ('REACX', 'REACY', 'REACZ'))}
{1:10d}{0:10d}
{2:10d}{0:10d}
{3:10d}{0:10d}
{4:10d}{0:10d}
/TH/NODE/102
top separation history
{''.join(f'{v:<10}' for v in ('DX', 'DY', 'DZ', 'VX', 'VY', 'VZ'))}
{5:10d}{0:10d}
{6:10d}{0:10d}
{7:10d}{0:10d}
{8:10d}{0:10d}
/TH/BRIC/103
cohesive state
{''.join(f'{v:<10}' for v in ('STRESS', 'STRAIN', 'IE', 'OFF'))}
{1:10d}
/END
"""


def _engine(stem: str, run_number: int, end_time: float, *, interval: float) -> str:
    # /DT/NODA/STOP refines the critical nodal step and stops if it falls below
    # this floor. The pinned OpenCourant CFG exposes only scale and minimum
    # step for STOP; no mass-scaling ratio or other mass-scaling card is set.
    history_interval = min(1.0e-6, end_time / 1000.0)
    return f"""#RADIOSS ENGINE
/DT/NODA/STOP
{NODAL_TIME_STEP_SCALE:20.9g}{MINIMUM_TIME_STEP_S:20.9g}
/ANIM/DT
{0.0:20.9g}{interval:20.9g}
/ANIM/BRICK/DAMA
/ANIM/BRICK/TENS/STRESS
/ANIM/ELEM/ENER
/ANIM/VECT/DISP
/MON/ON
/RFILE
# restart output interval in cycles
{5000:10d}
/TFILE/4
{history_interval:20.9g}
/TH/TITLE
/RUN/{stem}/{run_number}
{end_time:20.9g}
"""


def _make_case(
    root: Path,
    *,
    case_id: str,
    mode: str,
    functions: list[list[tuple[float, float]]],
    run_times: tuple[float, ...],
    pair: str | None,
    purpose: str,
    gates: list[str],
    activation_time: float | None = None,
    additional_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    stem = case_id.replace("-", "_")[:48]
    starter = f"{stem}_0000.rad"
    (case_dir / starter).write_text(
        _starter(stem, mode, functions, activation_time=activation_time), encoding="utf-8"
    )
    engines = []
    for index, end_time in enumerate(run_times, start=1):
        name = f"{stem}_{index:04d}.rad"
        (case_dir / name).write_text(
            _engine(stem, index, end_time, interval=min(2.0e-5, end_time / 50.0)),
            encoding="utf-8",
        )
        engines.append(name)
    case = {
        "schema": SCHEMA,
        "id": case_id,
        "starter": starter,
        "engines": engines,
        "expected_end_time_s": run_times[-1],
        "parameters": {
            "units": {"length": "mm", "time": "s", "mass": "Mg", "force": "N", "stress": "MPa"},
            "material": MATERIAL,
            "time_step_control": {
                "keyword": "/DT/NODA/STOP",
                "critical_step_scale": NODAL_TIME_STEP_SCALE,
                "minimum_step_s": MINIMUM_TIME_STEP_S,
                "mass_scaling_requested": False,
                "note": "Pinned CFG STOP card has only scale/minimum fields; no mass scaling is requested. A native run must still confirm zero mass change.",
            },
            "time_history": {
                "keyword": "/TH/TITLE",
                "tfile_interval_s_by_engine_stage": [
                    min(1.0e-6, stage_end_time / 1000.0) for stage_end_time in run_times
                ],
                "force_conversion": {
                    "source_quantity": "OpenCourant T01 reaction impulse channels",
                    "converter_command": "th_to_csv <runname>T01",
                    "required_sidecar": "<runname>T01_TITLES generated by /TH/TITLE",
                    "recognized_reaction_tokens": ["REACX", "REACY", "REACZ"],
                    "conversion": "finite difference of impulse over time; inspect titled CSV columns and units before applying force/work gates",
                    "force_gate_blocked_if": "the titles sidecar is missing or the CSV contains generic varNN columns instead of titled reaction channels",
                    "converter_provenance": "th_to_csv is an external OpenCourant/Tools utility, not part of the pinned OpenCourant source tree; bind its actual binary/version in any future run receipt",
                },
            },
            "model": {"type": "one zero-thickness /PROP/TYPE43 cohesive /BRICK", "area_mm2": 1.0,
                      "initial_separation_mm": 0.0, "faces": {"bottom": [1, 2, 3, 4], "top": [5, 6, 7, 8]}},
            "motion_functions": [
                [{"time_s": t, "displacement_mm": u} for t, u in points]
                for points in functions
            ],
            "synthetic_reference": {
                "scope": "quasistatic pure-mode-I calibration of this synthetic law only",
                "area_mm2": 1.0,
                "normal_stiffness_MPa_per_mm": MATERIAL["normal_stiffness_MPa_per_mm"],
                "peak_separation_mm": MATERIAL["peak_normal_MPa"] / MATERIAL["normal_stiffness_MPa_per_mm"],
                "failure_separation_mm": MATERIAL["failure_displacement_mm"]["mode_I"],
                "source_equation": "Pinned sigeps117.F90 linear displacement evolution: delta0=TN/EN, deltaf=2*GIC/TN; D=(deltaf/delta)*(delta-delta0)/(deltaf-delta0), then T=(1-D)*EN*delta for tensile normal opening.",
                "nominal_traction_MPa_by_gap": [
                    {"gap_range_mm": [0.0, 0.1], "formula": "100*gap_mm"},
                    {"gap_range_mm": [0.1, 0.3], "formula": "15 - 50*gap_mm"},
                    {"gap_range_mm": [0.3, None], "formula": "0 after the declared failure separation"},
                ],
                "force_at_gap": [
                    {"gap_mm": 0.01, "expected_force_N": 1.0},
                    {"gap_mm": 0.1, "expected_force_N": 10.0},
                    {"gap_mm": 0.2, "expected_force_N": 5.0},
                ],
                "work_to_failure_N_mm": 1.5,
                "work_basis": "triangle under the source-derived tensile traction-gap curve; GIC=1.5 N/mm times 1 mm^2 bond area gives 1.5 N mm",
                "law_source": "OpenCourant pinned engine/source/materials/mat/mat117/sigeps117.F90, lines 171-190 and 308-327, 381-392",
                "relative_tolerance": 0.01,
                "absolute_tolerance_force_N": 1.0e-6,
                "absolute_tolerance_work_N_mm": 1.0e-6,
                "only_interpret_when_kinetic_to_internal_energy_ratio_at_check_points_le_0_01": True,
            } if mode == "I" and activation_time is None else None,
            **({
                "activation": {
                    "keyword": "/ACTIV",
                    "formulation": 2,
                    "brick_group_id": 7,
                    "brick_group_members": [1],
                    "activation_time_s": activation_time,
                    "stop_time_s": 1.0e30,
                    "entire_group_dormant_before_activation": True,
                    "expected_OFF_before": 1,
                    "expected_OFF_after": 0,
                    "reference_geometry_at_activation": "current translated geometry; this is the behavior under test, not a pre-assumed solver result",
                    "expected_pre_activation_force_N": 0.0,
                    "expected_force_jump_bound_N": 0.1,
                    "expected_post_activation_incremental_force": [
                        {"increment_mm": 0.01, "expected_force_N": 1.0},
                        {"increment_mm": 0.1, "expected_force_N": 10.0},
                    ],
                }
            } if activation_time is not None else {}),
            **(additional_parameters or {}),
        },
        "calibration_scope": {
            "classification": "synthetic solver-instrument admission only",
            "not_calibrated_for": ["kraft paper", "NSSC paper", "corrugated board", "starch adhesive"],
            "target_material_parameters_identified": False,
        },
        "analysis": {
            "purpose": purpose,
            "paired_case_id": pair,
            "observables": [
                "top-node displacement and velocity histories (DX/DY/DZ, VX/VY/VZ)",
                "sum of four constrained bottom-node reaction forces in the driven direction (N)",
                "cohesive brick global stress, strain, internal energy and deactivation flag",
                "animation damage and stress tensors",
                "engine energy history and restart state for split runs",
            ],
            "gates": gates,
            "quantitative_tolerances": {
                "force_relative": 0.01,
                "work_relative": 0.01,
                "activation_pre_force_absolute_N": 1.0e-6 if activation_time is not None else None,
                "activation_jump_max_N": 0.1 if activation_time is not None else None,
            },
            "limits": [
                "No duplicate INTER contact is placed on the coincident cohesive faces; LAW117 ISYM=0 supplies the material's own undamaged compression branch.",
                "A successful process exit is only deck/runtime admission, not material calibration or physical qualification.",
                "Check kinetic/internal-energy ratio and global energy balance before interpreting the force curve as quasi-static.",
                "The T01 reaction channels are impulses. /TH/TITLE must generate the T01_TITLES sidecar, and th_to_csv must derive the titled REACX/REACY/REACZ channels before they are treated as forces; generic varNN columns do not pass the force gate.",
                "The Engine uses /DT/NODA/STOP with a 0.02 critical-step scale, a 1e-12 s minimum-step stop floor, and no requested mass scaling. Verify the actual cycle count and zero mass change in the native output before interpreting the refined history.",
                "The official OpenCourant README and converter describe T01 reaction channels as impulses; /TH/TITLE writes the title sidecar used by th_to_csv to identify and difference recognized reaction tokens. This is an output-processing prerequisite, not independent force validation.",
            ],
        },
    }
    (case_dir / "case.json").write_text(json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return case


def export_adhesion_cases(output: Path) -> list[dict[str, Any]]:
    """Write bounded synthetic LAW117 cases and return runner case records.

    `output` is the already-created suite directory supplied by the bounded
    OpenCourant runner.  Each case is self-contained in its own subdirectory.
    """
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"suite output directory must already exist: {output}")
    tmid = SPLIT_TIME
    cases = []

    mode_i = [[(0.0, 0.0), (0.9 * END_TIME, 0.35), (END_TIME, 0.35)]]
    cases.append(_make_case(
        output, case_id="law117-mode-i-open-failure", mode="I", functions=mode_i,
        run_times=(END_TIME,), pair="law117-mode-i-open-failure-restart",
        purpose="First runnable admission case: pure mode-I opening through the synthetic linear softening and deletion range.",
        gates=[
            "Starter accepts LAW117 with the 2026 ISYM and damage-evolution fields and TYPE43 accepts one eight-node brick.",
            "After /TH/TITLE and th_to_csv produce titled reaction forces, compare the 1 mm^2 element's normal force to the source-derived synthetic reference: 1 N at 0.01 mm, 10 N at 0.1 mm, and 5 N at 0.2 mm, within 1% plus the declared absolute floor.",
            "Damage is nondecreasing and the brick active/deleted history changes consistently with the declared synthetic failure displacement.",
            "To the 0.3 mm failure separation, compare integrated force-displacement work with the synthetic LAW117 fracture work 1.5 N mm; also retain cohesive dissipation and global energy balance.",
            "Apply force/work tolerances only when kinetic/internal energy is at most 1% at the checked samples and the converted force channel has the verified reaction-force title.",
        ],
    ))
    cases.append(_make_case(
        output, case_id="law117-mode-i-open-failure-restart", mode="I", functions=mode_i,
        run_times=(tmid, END_TIME), pair="law117-mode-i-open-failure",
        purpose="Same synthetic mode-I path split across two Engine files to check restart continuation against the continuous case.",
        gates=[
            "Both ordered Engine stages terminate normally and the first stage leaves a restart consumed by the second.",
            "Interpolate continuous and split histories onto common physical times and compare displacement, reaction, damage, energy, and final element status.",
            "Keep the comparison a restart-integrity check; it does not establish material or physical fidelity.",
        ],
    ))
    mode_ii = [[(0.0, 0.0), (0.9 * END_TIME, 0.35), (END_TIME, 0.35)]]
    cases.append(_make_case(
        output, case_id="law117-mode-ii-shear-failure", mode="II", functions=mode_ii,
        run_times=(END_TIME,), pair=None,
        purpose="Pure in-plane mode-II opening/shear through the synthetic linear softening and deletion range.",
        gates=[
            "Measure tangential traction against the prescribed shear separation and retain the damage/deletion history.",
            "Verify no unintended normal separation and keep the measured dynamic energy balance visible.",
        ],
    ))
    mixed = [
        [(0.0, 0.0), (0.9 * END_TIME, 0.25), (END_TIME, 0.25)],
        [(0.0, 0.0), (0.9 * END_TIME, 0.25), (END_TIME, 0.25)],
    ]
    cases.append(_make_case(
        output, case_id="law117-mixed-mode-opening", mode="mixed", functions=mixed,
        run_times=(END_TIME,), pair=None,
        purpose="Simultaneous normal and tangential opening for the synthetic power-law mixed-mode interaction.",
        gates=[
            "Record normal and shear reaction components separately and compare the observed failure state with the declared power-law interaction inputs.",
            "Do not infer a mixed-mode adhesive calibration from this manufactured parameter set.",
        ],
    ))
    unload = [[(0.0, 0.0), (0.28 * END_TIME, 0.16), (0.52 * END_TIME, 0.04),
               (0.78 * END_TIME, 0.22), (END_TIME, 0.22)]]
    cases.append(_make_case(
        output, case_id="law117-unload-reload", mode="I", functions=unload,
        run_times=(END_TIME,), pair=None,
        purpose="Mode-I partial damage followed by unload and reload to check irreversible state/history handling.",
        gates=[
            "Damage does not decrease on unloading and the reload branch reflects retained internal history.",
            "Compare reactions at matched separation on the first and second loading branches; report dynamic effects separately.",
        ],
    ))
    close = [[(0.0, 0.0), (0.30 * END_TIME, 0.12), (0.60 * END_TIME, 0.0),
              (0.80 * END_TIME, -0.02), (END_TIME, -0.02)]]
    cases.append(_make_case(
        output, case_id="law117-open-close-compression", mode="I", functions=close,
        run_times=(END_TIME,), pair=None,
        purpose="Partial mode-I opening, closure through zero, then small normal compression under the asymmetric LAW117 branch.",
        gates=[
            "The imposed separation crosses from positive to zero to negative while the element remains active.",
            "For ISYM=0, confirm tensile damage is retained but the negative normal traction follows the undegraded compressive stiffness branch.",
            "No separate contact interface is present on the same coincident faces, so compressive reaction is not double counted.",
        ],
    ))
    activation_motion = [[
        (0.0, 0.0),
        (0.20 * END_TIME, 0.05),
        (0.49 * END_TIME, 0.05),
        (ACTIVATION_TIME, 0.05),
        (0.60 * END_TIME, 0.05),
        (END_TIME, 0.15),
    ]]
    cases.append(_make_case(
        output, case_id="law117-dormant-activate-at-translated-geometry", mode="I",
        functions=activation_motion, run_times=(END_TIME,),
        pair="law117-always-active-activation-control",
        purpose=(
            "Synthetic one-element activation instrument: the entire TYPE43 brick group is off while the prescribed top face translates to 0.05 mm, "
            "is held through /ACTIV time, then opens by a further 0.10 mm after activation. Tests reference initialization at the current geometry."
        ),
        activation_time=ACTIVATION_TIME,
        additional_parameters={
            "activation_reference": {
                "pre_activation_global_gap_mm": 0.05,
                "post_activation_increment_mm": 0.10,
                "expected_incremental_work_N_mm": 0.5,
                "force_relative_tolerance": 0.01,
                "work_relative_tolerance": 0.01,
                "force_absolute_tolerance_N": 1.0e-6,
                "work_absolute_tolerance_N_mm": 1.0e-6,
                "activation_jump_limit_N": 0.1,
                "active_control_global_gap_force_reference_N": {"0.05": 5.0, "0.15": 7.5},
                "active_control_work_to_0_15_N_mm": 0.9375,
            }
        },
        gates=[
            "At global gap 0.05 mm before Tstart, the whole /GRBRIC group is deactivated (OFF=1) and summed bottom reaction magnitude is at most 1e-6 N.",
            "After /ACTIV formulation 2 at 0.0005 s, OFF changes to active (OFF=0) and force remains within 0.1 N of its pre-activation value during the 0.05 mm hold.",
            "With the element re-referenced at activation geometry, incremental opening of 0.01 mm and 0.10 mm has nominal force 1 N and 10 N respectively, each within 1% plus 1e-6 N absolute tolerance when the kinetic/internal-energy ratio is at most 1%.",
            "The post-activation incremental work to 0.10 mm is 0.5 N mm within 1% plus 1e-6 N mm absolute tolerance; a failed gate records that this activation/reference behavior was not established.",
            "The group contains only brick 1; no mixed active/dormant group behavior is claimed or tested.",
        ],
    ))
    cases.append(_make_case(
        output, case_id="law117-always-active-activation-control", mode="I",
        functions=activation_motion, run_times=(END_TIME,),
        pair="law117-dormant-activate-at-translated-geometry",
        purpose=(
            "Always-active reference using exactly the same absolute top-face displacement function as the dormant activation case, "
            "so its pre-strain/history can expose any activation-time re-reference behavior."
        ),
        additional_parameters={
            "activation_control": {
                "activation_keyword_present": False,
                "global_gap_force_reference_N": {"0.05": 5.0, "0.15": 7.5},
                "work_to_0_15_N_mm": 0.9375,
                "force_relative_tolerance": 0.01,
                "work_relative_tolerance": 0.01,
                "force_absolute_tolerance_N": 1.0e-6,
                "work_absolute_tolerance_N_mm": 1.0e-6,
                "paired_motion_must_be_byte_identical": True,
                "expected_history_matches_dormant_case": False,
            }
        },
        gates=[
            "This control has no /ACTIV card and remains active from the start; OFF and the force history must confirm that condition.",
            "Under the common motion curve, the synthetic elastic/softening reference is 5 N at global gap 0.05 mm and 7.5 N at 0.15 mm, within 1% plus 1e-6 N absolute tolerance when quasi-static energy checks pass.",
            "The reference work to 0.15 mm is 0.9375 N mm within 1% plus 1e-6 N mm absolute tolerance; do not require its history to equal the dormant/re-activated case, whose material reference starts at activation.",
            "No mixed active/dormant element group is included.",
        ],
    ))
    return cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_adhesion_cases(args.output), indent=2, sort_keys=True, allow_nan=False))
