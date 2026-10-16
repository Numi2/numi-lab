#!/usr/bin/env python3
"""Recreate the Haj-Ali et al. Fig. 1-2 bilinear tension-fit nodes.

The article reports tabulated yield anchors and draws its bilinear tension
fit. Its plotted experimental markers are not exported as raw numeric data.
This script converts the four visible fit-line endpoints from the hash-bound
ResearchGate author-full-text page render and joins those pixels to the
published Table 1 yield anchors. It does not digitize the experimental marker
cloud or infer compression response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


EXPECTED_RENDER_SHA256 = "bffb53d8685c5990b2fa43f120ac0382493e893db2284b4f8c24adff6a48a6d5"
PAGE_RENDER_URL = (
    "https://www.researchgate.net/profile/Roman-Popil/publication/223896964/"
    "viewer/AS:1078658240393218@1634183573648/background/4.png"
)
TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[2]
DATA = ROOT / "matter" / "examples" / "cardboard" / "source-data"
DEFAULT_OUTPUT = DATA / "haj_ali_2009_bilinear_tension_fit_nodes.csv"

# Plot frame bounds are inclusive axis-line pixel centers in the 1041x1269
# author-page PNG. The printed tick labels and series legends are OCR/text
# overlays in the viewer, so their values are transcribed in the metadata.
CURVES = (
    {
        "series_id": "haj_ali_2009_liner_MD_bilinear_tension",
        "figure": "Fig. 1(a)",
        "component": "liner",
        "basis_weight_g_m2": 205,
        "axis": "MD",
        "yield_stress_MPa": 38.2,
        "yield_stress_resolution_MPa": 0.1,
        "yield_strain": 0.008,
        "yield_strain_resolution": 0.001,
        "table2_E_MPa": 4775,
        "plot": {"x0_px": 146, "x1_px": 493, "y_top_px": 8, "y_bottom_px": 357,
                 "x_min": 0.0, "x_max": 0.025, "y_min_MPa": 0.0, "y_max_MPa": 70.0},
        "endpoint_px": (458, 9),
    },
    {
        "series_id": "haj_ali_2009_liner_CD_bilinear_tension",
        "figure": "Fig. 1(b)",
        "component": "liner",
        "basis_weight_g_m2": 205,
        "axis": "CD",
        "yield_stress_MPa": 18.4,
        "yield_stress_resolution_MPa": 0.1,
        "yield_strain": 0.009,
        "yield_strain_resolution": 0.001,
        "table2_E_MPa": 2088,
        "plot": {"x0_px": 567, "x1_px": 918, "y_top_px": 8, "y_bottom_px": 358,
                 "x_min": 0.0, "x_max": 0.05, "y_min_MPa": 0.0, "y_max_MPa": 40.0},
        # The solid source-labeled tension-fit line terminates at x≈882,y≈49.
        # The previously selected lower point near (917,117) lies on the
        # dashed compression line and must not be used for the MD/CD tension
        # fit.
        "endpoint_px": (882, 49),
    },
    {
        "series_id": "haj_ali_2009_medium_NSSC_MD_bilinear_tension",
        "figure": "Fig. 2(a)",
        "component": "medium_NSSC",
        "basis_weight_g_m2": 126,
        "axis": "MD",
        "yield_stress_MPa": 15.7,
        "yield_stress_resolution_MPa": 0.1,
        "yield_strain": 0.0035,
        "yield_strain_resolution": 0.0001,
        "table2_E_MPa": 4472,
        "plot": {"x0_px": 146, "x1_px": 491, "y_top_px": 513, "y_bottom_px": 860,
                 "x_min": 0.0, "x_max": 0.02, "y_min_MPa": 0.0, "y_max_MPa": 40.0},
        "endpoint_px": (490, 535),
    },
    {
        "series_id": "haj_ali_2009_medium_NSSC_CD_bilinear_tension",
        "figure": "Fig. 2(b)",
        "component": "medium_NSSC",
        "basis_weight_g_m2": 126,
        "axis": "CD",
        "yield_stress_MPa": 8.06,
        "yield_stress_resolution_MPa": 0.01,
        "yield_strain": 0.0050,
        "yield_strain_resolution": 0.0001,
        "table2_E_MPa": 1612,
        "plot": {"x0_px": 575, "x1_px": 921, "y_top_px": 513, "y_bottom_px": 859,
                 "x_min": 0.0, "x_max": 0.03, "y_min_MPa": 0.0, "y_max_MPa": 15.0},
        "endpoint_px": (918, 534),
    },
)

FIELDS = (
    "source_id", "series_id", "figure", "printed_page", "component",
    "basis_weight_g_m2", "axis", "curve_label", "point_order", "point_role",
    "strain_mm_per_mm", "stress_MPa", "x_pixel", "y_pixel",
    "pixel_readout_halfwidth_px", "axis_line_halfwidth_px",
    "strain_halfwidth_mm_per_mm", "stress_halfwidth_MPa",
    "table2_elastic_modulus_MPa", "postyield_tangent_MPa",
    "postyield_tangent_low_MPa", "postyield_tangent_high_MPa",
    "uniaxial_plastic_hardening_modulus_MPa",
    "uniaxial_plastic_hardening_modulus_low_MPa",
    "uniaxial_plastic_hardening_modulus_high_MPa",
    "source_location", "observation_class", "note",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_to_strain(x_px: float, plot: dict[str, float]) -> float:
    return plot["x_min"] + (x_px - plot["x0_px"]) * (plot["x_max"] - plot["x_min"]) / (plot["x1_px"] - plot["x0_px"])


def pixel_to_stress(y_px: float, plot: dict[str, float]) -> float:
    return plot["y_min_MPa"] + (plot["y_bottom_px"] - y_px) * (plot["y_max_MPa"] - plot["y_min_MPa"]) / (plot["y_bottom_px"] - plot["y_top_px"])


def tangent_interval(curve: dict[str, object], endpoint_strain: float, endpoint_stress: float,
                     strain_pixel_halfwidth: float, stress_pixel_halfwidth: float) -> tuple[float, float, float]:
    eps_y = float(curve["yield_strain"])
    sig_y = float(curve["yield_stress_MPa"])
    eps_unc = float(curve["yield_strain_resolution"]) / 2.0
    sig_unc = float(curve["yield_stress_resolution_MPa"]) / 2.0
    eps_low, eps_high = endpoint_strain - strain_pixel_halfwidth, endpoint_strain + strain_pixel_halfwidth
    sig_low, sig_high = endpoint_stress - stress_pixel_halfwidth, endpoint_stress + stress_pixel_halfwidth
    center = (endpoint_stress - sig_y) / (endpoint_strain - eps_y)
    candidates = []
    for e_end in (eps_low, eps_high):
        for s_end in (sig_low, sig_high):
            for e_y in (eps_y - eps_unc, eps_y + eps_unc):
                for s_y in (sig_y - sig_unc, sig_y + sig_unc):
                    if e_end <= e_y:
                        raise ValueError("plot endpoint must lie beyond the published yield strain")
                    candidates.append((s_end - s_y) / (e_end - e_y))
    return center, min(candidates), max(candidates)


def rows_for_curve(curve: dict[str, object], page_hash: str) -> list[dict[str, object]]:
    plot = curve["plot"]
    x_end_px, y_end_px = curve["endpoint_px"]
    endpoint_strain = pixel_to_strain(x_end_px, plot)
    endpoint_stress = pixel_to_stress(y_end_px, plot)
    readout_px = 2.0
    axis_px = 1.0
    combined_px = readout_px + axis_px
    eps_halfwidth = combined_px * (plot["x_max"] - plot["x_min"]) / (plot["x1_px"] - plot["x0_px"])
    sig_halfwidth = combined_px * (plot["y_max_MPa"] - plot["y_min_MPa"]) / (plot["y_bottom_px"] - plot["y_top_px"])
    tangent, tangent_low, tangent_high = tangent_interval(
        curve, endpoint_strain, endpoint_stress, eps_halfwidth, sig_halfwidth)
    elastic_modulus = float(curve["table2_E_MPa"])
    if not (0.0 < tangent_low <= tangent <= tangent_high < elastic_modulus):
        raise ValueError("digitized post-yield tangent must be below the source elastic modulus")
    # For a uniaxial bilinear elastic-plastic bar, Et = E*H/(E+H), hence
    # H = E*Et/(E-Et). This converts the drawn total-stress tangent into the
    # corresponding uniaxial plastic hardening modulus; it is not an assertion
    # that Hill's full multiaxial hardening law is isotropic or fully fit.
    plastic_hardening = elastic_modulus * tangent / (elastic_modulus - tangent)
    plastic_hardening_low = elastic_modulus * tangent_low / (elastic_modulus - tangent_low)
    plastic_hardening_high = elastic_modulus * tangent_high / (elastic_modulus - tangent_high)
    metadata = {
        "source_id": "haj_ali_2009",
        "series_id": curve["series_id"],
        "figure": curve["figure"],
        "printed_page": 324,
        "component": curve["component"],
        "basis_weight_g_m2": curve["basis_weight_g_m2"],
        "axis": curve["axis"],
        "curve_label": "Bilinear tension",
        "pixel_readout_halfwidth_px": readout_px,
        "axis_line_halfwidth_px": axis_px,
        "strain_halfwidth_mm_per_mm": eps_halfwidth,
        "stress_halfwidth_MPa": sig_halfwidth,
        "table2_elastic_modulus_MPa": curve["table2_E_MPa"],
        "postyield_tangent_MPa": tangent,
        "postyield_tangent_low_MPa": tangent_low,
        "postyield_tangent_high_MPa": tangent_high,
        "uniaxial_plastic_hardening_modulus_MPa": plastic_hardening,
        "uniaxial_plastic_hardening_modulus_low_MPa": plastic_hardening_low,
        "uniaxial_plastic_hardening_modulus_high_MPa": plastic_hardening_high,
        "source_location": f"{curve['figure']}, printed p. 324; Table 1 yield node; Table 2 elastic modulus",
        "observation_class": "digitized_source_bilinear_fit_estimate_not_raw_measurement",
        "note": (
            "Only the solid source-labeled bilinear tension fit is represented. "
            "The experimental marker cloud is not digitized as raw data or separated by specimen. "
            "The derived H uses H=E*Et/(E-Et), the uniaxial bilinear elastic-plastic conversion; it does not identify full multiaxial hardening. "
            f"Page-render SHA256={page_hash}. Pixel endpoint selected at the visible end of the solid fit."
        ),
    }
    return [
        {**metadata, "point_order": 0, "point_role": "axis_origin", "strain_mm_per_mm": 0.0,
         "stress_MPa": 0.0, "x_pixel": plot["x0_px"], "y_pixel": plot["y_bottom_px"],
         "strain_halfwidth_mm_per_mm": 0.0, "stress_halfwidth_MPa": 0.0},
        {**metadata, "point_order": 1, "point_role": "published_Table_1_tensile_yield_anchor",
         "strain_mm_per_mm": curve["yield_strain"], "stress_MPa": curve["yield_stress_MPa"],
         "x_pixel": "", "y_pixel": "",
         "strain_halfwidth_mm_per_mm": float(curve["yield_strain_resolution"]) / 2,
         "stress_halfwidth_MPa": float(curve["yield_stress_resolution_MPa"]) / 2},
        {**metadata, "point_order": 2, "point_role": "digitized_bilinear_fit_endpoint",
         "strain_mm_per_mm": endpoint_strain, "stress_MPa": endpoint_stress,
         "x_pixel": x_end_px, "y_pixel": y_end_px,
         "strain_halfwidth_mm_per_mm": eps_halfwidth, "stress_halfwidth_MPa": sig_halfwidth},
    ]


def digitize(page_render: Path, output: Path) -> None:
    if not page_render.is_file():
        raise ValueError(f"page render does not exist: {page_render}")
    page_hash = sha256(page_render)
    if page_hash != EXPECTED_RENDER_SHA256:
        raise ValueError("page-render bytes are not the registered author-full-text Figure 1-2 render")
    all_rows = [row for curve in CURVES for row in rows_for_curve(curve, page_hash)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-render", type=Path, required=True,
                        help="external ResearchGate page-4 PNG from the author-posted full text")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        digitize(args.page_render, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote 12 fit nodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
