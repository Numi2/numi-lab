#!/usr/bin/env python3
"""Predetermined force-controlled implicit admission controls, not board calibration."""
from pathlib import Path
import json
import re
from cardboard_opencourant_adhesion import _starter


def _instrumented_engine(text: str, stem: str, end_time: float) -> str:
    """Request animation and titled time histories for an instrumented deck."""
    output_cards = (
        "/TFILE/4\n"
        f"{end_time / 100.:.9g}\n"
        "/ANIM/DT\n"
        f"{0.:.9g} {end_time / 20.:.9g}\n"
        "/ANIM/VECT/DISP\n"
        "/ANIM/ELEM/ENER\n"
        # T01 reaction channels contain impulses.  The official th_to_csv
        # converter needs this sidecar to derive titled reaction-force output.
        "/TH/TITLE\n"
    )
    if text.count(f"/RUN/{stem}/1\n") != 1:
        raise ValueError("implicit engine must contain one matching /RUN card")
    return text.replace(f"/RUN/{stem}/1\n", output_cards + f"/RUN/{stem}/1\n")


def export_implicit_cases(output: Path, *, include_instrumented: bool = False) -> list[dict]:
    """Export synthetic force controls, optionally with native output requests.

    The original controls remain no-output by default.  Instrumented variants
    have distinct IDs and request titled T01 histories plus animation fields;
    they do not change the controls' material or mechanical coefficients.
    """
    cases = []
    for material in ("elastic", "cohesive"):
        variants = [False, True] if include_instrumented else [False]
        for instrumented in variants:
            name = f"implicit-{material}-nonlinear-force-control"
            if instrumented:
                name += "-instrumented"
            stem = name.replace("-", "_")
            root = output / name
            root.mkdir(exist_ok=False)
            text = _starter(stem, "I", [[(0., 0.), (.001, 1.)]])
            # Free each upper normal DOF. Apply 0.25 N to each of four upper
            # nodes: the cohesive element is the sole normal stiffness path.
            text, n = re.subn(r"/IMPDISP/1\n.*?(?=\n/TH/)", "", text, flags=re.S)
            assert n == 1
            load = ("/CLOAD/1\nUniform normal load; 1 N total\n"
                    f"{1:10d}{'Z':>10}{0:10d}{0:10d}{2:10d}{'':10}{1.:20g}{.25:20g}\n")
            text = text.replace("/END", load + "/END")
            if material == "elastic":
                text, n = re.subn(r"/MAT/LAW117/1\n.*?(?=\n/PROP/)",
                    "/MAT/LAW1/1\nSynthetic elastic control\n"
                    f"{1e-6:20g}\n{100.:20g}{0.:20g}", text, flags=re.S)
                assert n == 1
                text, n = re.subn(r"/PROP/TYPE43/1\n.*?(?=\n/NODE)",
                    "/PROP/TYPE14/1\nFully integrated solid control\n"
                    f"{14:10d}{0:10d}{'':10}{0:10d}{'':10}{222:10d}{'':10}{0:10d}{0.:20g}\n"
                    f"{0.:20g}{0.:20g}{0.:20g}{0.:20g}{0.:20g}\n"
                    f"{0.:20g}{0.:20g}{0.:20g}{0.:20g}{0.:20g}\n"
                    f"{0:10d}{0:10d}{2:10d}", text, flags=re.S)
                assert n == 1
                # Height 1 mm and area 1 mm² give EA/L = 100 N/mm.
                lines = text.splitlines()
                start = lines.index("/NODE") + 1
                for index in range(start + 4, start + 8):
                    lines[index] = lines[index][:50] + f"{1.:20g}"
                text = "\n".join(lines) + "\n"
            text = "\n".join(line for line in text.splitlines() if line.strip()) + "\n"
            starter = f"{stem}_0000.rad"
            engine = f"{stem}_0001.rad"
            (root / starter).write_text(text)
            # No artificial inertial stiffness: this isolates actual material
            # stiffness support. Request PCG explicitly; the tested package still
            # stops at its compile-time MUMPS guard before entering the solve.
            engine_text = (
                "/IMPL/NONL/01\n"
                # The pinned parser reads this list into CHARACTER*72 first.
                "20 123 1e-6 1e-6 1e-6\n"
                "/IMPL/DTINI\n0.0001\n/IMPL/DT/1\n0 0 0 0\n"
                "/IMPL/SOLV/01\n"
                f"{1:10d}{0:10d}{3:10d}{1e-10:20g}\n"
                "/IMPL/PRINT/NONLIN/1\n/MON/ON\n"
                f"/RUN/{stem}/1\n0.001\n")
            if instrumented:
                engine_text = _instrumented_engine(engine_text, stem, .001)
            (root / engine).write_text(engine_text)
            spec = {"id": name, "starter": starter, "engines": [engine],
                    "expected_end_time_s": .001,
                    "purpose": "Bounded implicit stiffness support instrument; synthetic, not physical grade calibration",
                    "prediction": {"total_applied_force_N": 1., "mean_upper_DZ_mm": .01,
                                   "sum_bottom_reaction_magnitude_N": 1.,
                                   "elastic_work_N_mm": .005},
                    "gates": {"normal_termination": True, "finite_displacements": True,
                              "relative_displacement_error_max": .01,
                              "relative_reaction_error_max": .01,
                              "missing_stiffness_warning_allowed": False},
                    "controls": "Identical restraints, load and stiffness; LAW1 HEX8 versus LAW117 TYPE43",
                    "mass_scaling": False, "physical_qualification": False}
            if instrumented:
                spec["native_outputs"] = {
                    "time_history": {
                        "path": f"{stem}T01",
                        "force_semantics": "REACZ is stored as impulse in T01; use the generated T01_TITLES sidecar with official th_to_csv to derive reaction force",
                        "force_channel": "/TH/NODE/101 REACZ on bottom nodes 1-4; sum the signed normal reactions",
                        "displacement_channel": "/TH/NODE/102 DZ on top nodes 5-8; average the four values",
                        "energy_channel": "/TH/BRIC/103 IE for brick 1, in N mm",
                    },
                    "animation": ["/ANIM/VECT/DISP", "/ANIM/ELEM/ENER"],
                    "history_interval_s": .00001,
                    "animation_interval_s": .00005,
                    "instrumentation_only": True,
                }
            (root / "case.json").write_text(json.dumps(spec, indent=2) + "\n")
            cases.append(spec)
    return cases
