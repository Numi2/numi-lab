#!/usr/bin/env python3
"""Cook the v10 Crow navigation collision course into one visual pack."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from create_dove_window_scene import write_usda_from_obj


BOXES = (
    ("gate-left", "dove_white", (0.12, 0.12, 0.80)),
    ("gate-right", "dove_wing", (0.12, 0.12, 0.80)),
    ("slalom-a", "dove_tail", (0.10, 0.10, 0.65)),
    ("slalom-b", "floor", (0.10, 0.10, 0.65)),
    ("perch", "wall", (0.10, 0.72, 0.055)),
)


def append_box(
    lines: list[str], material: str, extents: tuple[float, float, float]
) -> None:
    x, y, z = extents
    first = 1 + sum(1 for line in lines if line.startswith("v "))
    vertices = (
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    )
    faces = (
        (0, 1, 2), (0, 2, 3), (4, 7, 6), (4, 6, 5),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    )
    lines.append(f"usemtl {material}")
    lines.extend(f"v {a} {b} {c}" for a, b, c in vertices)
    lines.extend(
        f"f {first + a} {first + b} {first + c}" for a, b, c in faces
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-body-index", type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for offset, (name, material, extents) in enumerate(BOXES):
        obj = args.output / f"{name}.obj"
        usda = args.output / f"{name}.usda"
        lines = [f"o crow_navigation_{name}"]
        append_box(lines, material, extents)
        obj.write_text("\n".join(lines) + "\n")
        write_usda_from_obj(obj, usda)
        subprocess.run([
            str(args.cooker.resolve()),
            str(usda.resolve()),
            str((args.output / f"{name}.mrvpack").resolve()),
            "--id", f"birdflow_crow_navigation_{name}",
            "--license", "NOASSERTION",
            "--provenance", "authored-v10-static-obstacle-course",
            "--link",
            f"{material}_mesh={args.first_body_index + offset}",
        ], check=True)


if __name__ == "__main__":
    main()
