#!/usr/bin/env python3
"""Render the C++ source-oracle trajectory CSV as a self-contained SVG."""
import argparse
import csv
import math
from pathlib import Path

COLORS = {
    "seed": "#4b5563",
    "loading": "#166534",
    "unloading": "#b45309",
    "reloading": "#2563eb",
    "new_maximum": "#7c3aed",
}
REQUIRED = (
    "path_index", "segment", "opening_mm", "traction_mpa", "damage",
    "external_work_j_m2", "free_energy_j_m2", "dissipation_j_m2",
)


def number(value: str, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {field}")
    return result


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED:
            raise ValueError("trajectory CSV columns do not match the source-oracle schema")
        rows = []
        for expected, row in enumerate(reader):
            index = int(row["path_index"])
            if index != expected:
                raise ValueError("trajectory path indices must be contiguous from zero")
            segment = row["segment"]
            if segment not in COLORS:
                raise ValueError(f"unknown source path segment {segment!r}")
            item = {key: number(row[key], key) for key in REQUIRED if key != "segment"}
            item["segment"] = segment
            if not (0.0 <= item["damage"] < 1.0):
                raise ValueError("source damage must remain in [0,1)")
            rows.append(item)
    if len(rows) < 4:
        raise ValueError("trajectory needs at least four source samples")
    return rows


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(rows, csv_path: Path) -> str:
    width, height = 1040, 940
    left, right = 92, 1005
    panel_h = 210
    panels = [
        (92, "Normal traction", "MPa", "traction_mpa", 0.0),
        (352, "Irreversible damage", "kappa", "damage", 0.0),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}.small{font-size:12px}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:3 4}.series{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}</style>',
        '<text x="92" y="37" font-size="22" font-weight="650">Bleached paperboard Mode-I source oracle</text>',
        '<text x="92" y="60" font-size="13">Tryding et al. (2023), Table 1 and Eqs. 22–24, 39, 43; seeded at kappa=0.01</text>',
    ]

    def draw_axes(top, title, ylabel, y_values, xmin, xmax, x_label, ymin=0.0):
        bottom = top + panel_h
        ymax = max(y_values)
        if ymax <= ymin:
            ymax = ymin + 1.0
        ymax *= 1.05
        out.append(f'<text x="{left}" y="{top-14}" font-size="16" font-weight="600">{esc(title)}</text>')
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = bottom - (value - ymin) / (ymax - ymin) * panel_h
            out.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
            out.append(f'<text class="small" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.4g}</text>')
        out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        out.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
        out.append(f'<text class="small" x="{(left+right)/2}" y="{bottom+31}" text-anchor="middle">{esc(x_label)}</text>')
        out.append(f'<text class="small" x="22" y="{top+panel_h/2}" transform="rotate(-90 22 {top+panel_h/2})" text-anchor="middle">{esc(ylabel)}</text>')
        return bottom, ymax

    traction_max = max(row["traction_mpa"] for row in rows)
    bottom, ymax = draw_axes(92, "Traction–opening path", "traction (MPa)",
                             [row["traction_mpa"] for row in rows],
                             0.0, max(row["opening_mm"] for row in rows),
                             "opening (mm)")
    xmax = max(row["opening_mm"] for row in rows)
    for segment in COLORS:
        group = [row for row in rows if row["segment"] == segment]
        if not group:
            continue
        points = []
        for row in group:
            x = left + row["opening_mm"] / xmax * (right-left)
            y = bottom - row["traction_mpa"] / ymax * panel_h
            points.append(f"{x:.2f},{y:.2f}")
        out.append(f'<polyline class="series" stroke="{COLORS[segment]}" points="{" ".join(points)}"/>')

    opening_max = max(row["opening_mm"] for row in rows)
    bottom, ymax = draw_axes(352, "History damage", "damage kappa",
                             [row["damage"] for row in rows], 0.0,
                             opening_max, "opening (mm)")
    for segment in COLORS:
        group = [row for row in rows if row["segment"] == segment]
        if not group:
            continue
        points = []
        for row in group:
            x = left + row["opening_mm"] / opening_max * (right-left)
            y = bottom - row["damage"] / ymax * panel_h
            points.append(f"{x:.2f},{y:.2f}")
        out.append(f'<polyline class="series" stroke="{COLORS[segment]}" points="{" ".join(points)}"/>')

    top = 612
    bottom, ymax = draw_axes(top, "Work and energy along path", "J/m^2",
                             [row[key] for row in rows for key in
                              ("external_work_j_m2", "free_energy_j_m2", "dissipation_j_m2")],
                             0.0, float(len(rows)-1), "path sample index")
    energy_colors = {
        "external_work_j_m2": "#1d4ed8",
        "free_energy_j_m2": "#be123c",
        "dissipation_j_m2": "#047857",
    }
    for key, color in energy_colors.items():
        points = []
        for index, row in enumerate(rows):
            x = left + index / (len(rows)-1) * (right-left)
            y = bottom - row[key] / ymax * panel_h
            points.append(f"{x:.2f},{y:.2f}")
        out.append(f'<polyline class="series" stroke="{color}" points="{" ".join(points)}"/>')

    legend = [
        ("loading", "loading"), ("unloading", "unloading"),
        ("reloading", "reloading"), ("new_maximum", "new maximum"),
    ]
    x = left
    y = 908
    for key, label in legend:
        out.append(f'<line x1="{x}" y1="{y}" x2="{x+22}" y2="{y}" stroke="{COLORS[key]}" stroke-width="3"/>')
        out.append(f'<text class="small" x="{x+28}" y="{y+4}">{esc(label)}</text>')
        x += 150
    for key, label, color in [
        ("external_work_j_m2", "external work", energy_colors["external_work_j_m2"]),
        ("free_energy_j_m2", "free energy", energy_colors["free_energy_j_m2"]),
        ("dissipation_j_m2", "dissipation", energy_colors["dissipation_j_m2"]),
    ]:
        out.append(f'<line x1="{x}" y1="{y}" x2="{x+22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text class="small" x="{x+28}" y="{y+4}">{esc(label)}</text>')
        x += 130
    out.append(f'<text class="small" x="92" y="930">CSV: {esc(csv_path.name)}. Source-oracle values only; not corrugated-board or starch-adhesive validation.</text>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.trajectory_csv)
    args.output_svg.write_text(render(rows, args.trajectory_csv), encoding="utf-8")
    print(f"source_oracle_svg={args.output_svg} samples={len(rows)}")


if __name__ == "__main__":
    main()
