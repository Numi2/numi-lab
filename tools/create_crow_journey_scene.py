#!/usr/bin/env python3
"""Author the crow-journey presentation geometry for native Numi rendering.

The feather surface comes from BirdFlow's public Deetjen-derived archive. The
leg meshes are deliberately simple estimated boxes matching the owning hybrid
mechanics; they are presentation geometry, not measured crow anatomy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from create_dove_window_scene import write_room, write_surface_parts, write_usda_from_obj


def recolor_crow(output: Path) -> None:
    material = output / "dove.mtl"
    text = material.read_text()
    text = text.replace("Kd 0.78 0.80 0.84", "Kd 0.025 0.032 0.045")
    text = text.replace("Kd 0.58 0.61 0.67", "Kd 0.018 0.026 0.048")
    text = text.replace("Kd 0.45 0.48 0.54", "Kd 0.030 0.020 0.052")
    material.write_text(text)


def recolor_usda(path: Path) -> None:
    text = path.read_text()
    text = text.replace("(0.78, 0.8, 0.84)", "(0.025, 0.032, 0.045)")
    text = text.replace("(0.58, 0.61, 0.67)", "(0.018, 0.026, 0.048)")
    text = text.replace("(0.45, 0.48, 0.54)", "(0.030, 0.020, 0.052)")
    text = text.replace("(0.42900000000000005, 0.44000000000000006, 0.462)", "(0.012, 0.018, 0.030)")
    text = text.replace("(0.319, 0.3355, 0.36850000000000005)", "(0.010, 0.016, 0.034)")
    text = text.replace("(0.24750000000000003, 0.264, 0.29700000000000004)", "(0.018, 0.012, 0.034)")
    path.write_text(text)


def write_box_obj(path: Path, name: str, half_extents: tuple[float, float, float]) -> None:
    x, y, z = half_extents
    vertices = (
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    )
    faces = (
        (1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
    )
    # Reuse the converter's darkest named material; the generated pack keeps
    # the estimated-leg provenance independently of this material token.
    lines = ["mtllib dove.mtl", f"o {name}", "usemtl dove_tail"]
    lines.extend(f"v {a} {b} {c}" for a, b, c in vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    write_surface_parts(args.surface, args.output)
    recolor_crow(args.output)
    for name in ("body", "left-wing", "right-wing", "tail"):
        write_usda_from_obj(args.output / f"{name}.obj", args.output / f"{name}.usda")
        recolor_usda(args.output / f"{name}.usda")

    (args.output / "crow-leg.mtl").write_text(
        "newmtl crow_leg\nKd 0.055 0.050 0.060\nKs 0.12 0.12 0.14\nNs 24\n"
    )
    parts = {
        "thigh": (0.011, 0.009, 0.0275),
        "shank": (0.010, 0.008, 0.0275),
        "foot": (0.045, 0.015, 0.008),
    }
    for part, dimensions in parts.items():
        write_box_obj(args.output / f"{part}.obj", part, dimensions)
        write_usda_from_obj(args.output / f"{part}.obj", args.output / f"{part}.usda")
        recolor_usda(args.output / f"{part}.usda")
    write_room(args.output)


if __name__ == "__main__":
    main()
