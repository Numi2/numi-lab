#!/usr/bin/env python3
"""Convert reviewed Haj-Ali Fig. 1-2 marker-center pixels to plot units.

The pixel ledger contains selected visible marker glyphs from the registered
author-page render. It is deliberately partial where symbols overlap; this
tool performs only the declared linear axis conversion and adds a finite
pixel-reading bound. It does not reconstruct raw test files, identify
specimens, or infer obscured markers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TOOL = Path(__file__).resolve()
ROOT = TOOL.parents[2]
DATA = ROOT / "matter" / "examples" / "cardboard" / "source-data"
LEDGER = DATA / "haj_ali_2009_experimental_marker_centers_v1_pixels.json"
DEFAULT_OUTPUT = DATA / "haj_ali_2009_experimental_marker_centers_v1.csv"
EXPECTED_LEDGER_SHA256 = "862bf0cc84217bc96f0bc1eeddea900b9664330d916728e7f65cb317ef5ffe6c"
EXPECTED_RENDER_SHA256 = "bffb53d8685c5990b2fa43f120ac0382493e893db2284b4f8c24adff6a48a6d5"
MARKER_CENTER_HALF_WIDTH_PX = 1.5
AXIS_LINE_HALF_WIDTH_PX = 1.0

FIELDS = (
    "source_id", "series_id", "figure", "printed_page", "component",
    "basis_weight_g_m2", "axis", "record_status", "point_id", "point_order",
    "x_pixel", "y_pixel", "strain_mm_per_mm", "stress_MPa",
    "plot_x0_px", "plot_x1_px", "plot_y_top_px", "plot_y_bottom_px",
    "strain_axis_min_mm_per_mm", "strain_axis_max_mm_per_mm",
    "stress_axis_min_MPa", "stress_axis_max_MPa",
    "marker_center_halfwidth_px", "axis_line_halfwidth_px",
    "strain_halfwidth_mm_per_mm", "stress_halfwidth_MPa",
    "unresolved_bbox_x0_px", "unresolved_bbox_y0_px",
    "unresolved_bbox_x1_px", "unresolved_bbox_y1_px",
    "source_render_sha256", "observation_class", "point_identity", "note",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixel_to_strain(x_px: float, panel: dict[str, object]) -> float:
    return float(panel["strain_min_mm_per_mm"]) + (
        (x_px - float(panel["x0_px"]))
        * (float(panel["strain_max_mm_per_mm"]) - float(panel["strain_min_mm_per_mm"]))
        / (float(panel["x1_px"]) - float(panel["x0_px"]))
    )


def pixel_to_stress(y_px: float, panel: dict[str, object]) -> float:
    return float(panel["stress_min_MPa"]) + (
        (float(panel["y_bottom_px"]) - y_px)
        * (float(panel["stress_max_MPa"]) - float(panel["stress_min_MPa"]))
        / (float(panel["y_bottom_px"]) - float(panel["y_top_px"]))
    )


def rows_from_ledger(ledger: dict[str, object], render_hash: str) -> list[dict[str, object]]:
    if ledger.get("schema") != "numi.cardboard.haj_ali_plot_marker_pixel_ledger.v1":
        raise ValueError("unsupported Haj-Ali marker-pixel ledger")
    if ledger.get("source_render_sha256") != render_hash:
        raise ValueError("pixel ledger and page-render hashes disagree")
    width, height = ledger.get("render_dimensions_px", [None, None])
    rows: list[dict[str, object]] = []
    for panel in ledger.get("panels", []):
        assert isinstance(panel, dict)
        series_id = str(panel["series_id"])
        x0, x1 = int(panel["x0_px"]), int(panel["x1_px"])
        yt, yb = int(panel["y_top_px"]), int(panel["y_bottom_px"])
        x_range = float(panel["strain_max_mm_per_mm"]) - float(panel["strain_min_mm_per_mm"])
        y_range = float(panel["stress_max_MPa"]) - float(panel["stress_min_MPa"])
        total_pixel_bound = MARKER_CENTER_HALF_WIDTH_PX + AXIS_LINE_HALF_WIDTH_PX
        strain_bound = total_pixel_bound * x_range / (x1 - x0)
        stress_bound = total_pixel_bound * y_range / (yb - yt)
        common = {
            "source_id": str(ledger["source_id"]),
            "series_id": series_id,
            "figure": str(panel["figure"]),
            "printed_page": int(ledger["printed_page"]),
            "component": str(panel["component"]),
            "basis_weight_g_m2": int(panel["basis_weight_g_m2"]),
            "axis": str(panel["axis"]),
            "plot_x0_px": x0,
            "plot_x1_px": x1,
            "plot_y_top_px": yt,
            "plot_y_bottom_px": yb,
            "strain_axis_min_mm_per_mm": panel["strain_min_mm_per_mm"],
            "strain_axis_max_mm_per_mm": panel["strain_max_mm_per_mm"],
            "stress_axis_min_MPa": panel["stress_min_MPa"],
            "stress_axis_max_MPa": panel["stress_max_MPa"],
            "source_render_sha256": render_hash,
        }
        for order, pixel in enumerate(panel["selected_marker_centers_xy_px"], start=1):
            x_px, y_px = int(pixel[0]), int(pixel[1])
            if not (x0 <= x_px <= x1 and yt <= y_px <= yb):
                raise ValueError(f"{series_id}: selected center lies outside its plot frame")
            eps, sig = pixel_to_strain(x_px, panel), pixel_to_stress(y_px, panel)
            if not math.isfinite(eps) or not math.isfinite(sig):
                raise ValueError(f"{series_id}: nonfinite axis conversion")
            rows.append({
                **common,
                "record_status": "selected_marker_center",
                "point_id": f"{series_id}-P{order:03d}",
                "point_order": order,
                "x_pixel": x_px,
                "y_pixel": y_px,
                "strain_mm_per_mm": f"{eps:.12g}",
                "stress_MPa": f"{sig:.12g}",
                "marker_center_halfwidth_px": MARKER_CENTER_HALF_WIDTH_PX,
                "axis_line_halfwidth_px": AXIS_LINE_HALF_WIDTH_PX,
                "strain_halfwidth_mm_per_mm": f"{strain_bound:.12g}",
                "stress_halfwidth_MPa": f"{stress_bound:.12g}",
                "unresolved_bbox_x0_px": "", "unresolved_bbox_y0_px": "",
                "unresolved_bbox_x1_px": "", "unresolved_bbox_y1_px": "",
                "observation_class": "digitized_plot_marker_center_estimate_not_raw_measurement",
                "point_identity": "visible_marker_glyph_only_no_specimen_or_trace_identity",
                "note": "Selected visible glyph center; finite uncertainty covers pixel center and axis-line localization only, not test scatter.",
            })
        bbox = [int(v) for v in panel["unresolved_bbox_px"]]
        if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] < int(width)
                                  and 0 <= bbox[1] < bbox[3] < int(height)):
            raise ValueError(f"{series_id}: invalid unresolved overlap region")
        rows.append({
            **common,
            "record_status": "unresolved_overlap_region",
            "point_id": f"{series_id}-UNRESOLVED-OVERPLOT",
            "point_order": "", "x_pixel": "", "y_pixel": "",
            "strain_mm_per_mm": "", "stress_MPa": "",
            "marker_center_halfwidth_px": "", "axis_line_halfwidth_px": "",
            "strain_halfwidth_mm_per_mm": "", "stress_halfwidth_MPa": "",
            "unresolved_bbox_x0_px": bbox[0], "unresolved_bbox_y0_px": bbox[1],
            "unresolved_bbox_x1_px": bbox[2], "unresolved_bbox_y1_px": bbox[3],
            "observation_class": "unresolved_plot_overplot_region_not_observation",
            "point_identity": "no_point_or_specimen_identity_assigned",
            "note": str(panel["unresolved_overlap_region_note"]),
        })
    return rows


def digitize(render: Path, output: Path, ledger_path: Path = LEDGER) -> None:
    if sha256(ledger_path) != EXPECTED_LEDGER_SHA256:
        raise ValueError("reviewed pixel ledger bytes are not the registered v1 input")
    render_hash = sha256(render)
    if render_hash != EXPECTED_RENDER_SHA256:
        raise ValueError("page-render bytes are not the registered author-page plot image")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = rows_from_ledger(ledger, render_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-render", type=Path, required=True,
                        help="hash-bound author-page PNG containing Figs. 1-2")
    parser.add_argument("--ledger", type=Path, default=LEDGER,
                        help="reviewed pixel ledger; defaults to the package ledger")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        digitize(args.page_render, args.output, args.ledger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"wrote selected plot-marker centers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
