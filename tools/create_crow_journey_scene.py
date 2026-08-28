#!/usr/bin/env python3
"""Author the crow-journey presentation geometry for native Numi rendering.

The feather surface comes from BirdFlow's public Deetjen-derived archive. The
leg meshes are deliberately simple estimated boxes matching the owning hybrid
mechanics; they are presentation geometry, not measured crow anatomy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess

from create_dove_window_scene import write_room, write_surface_parts, write_usda_from_obj


# The estimated crow inserts four sweep/pronation bodies ahead of the original
# dove-derived wing, tail, and leg chain. Keep this mapping next to the scene
# authoring code so recooking cannot silently bind presentation geometry to the
# carrier links again.
CROW_VISUAL_LINKS = {
    "body": ("body", "dove_white_mesh", 0),
    # The source archive uses the opposite span sign from Numi's y-left
    # convention. Name the output by the physical Numi side and swap only the
    # source mesh here.
    "left-wing": ("right-wing", "dove_wing_mesh", 5),
    "right-wing": ("left-wing", "dove_wing_mesh", 6),
    "tail": ("tail", "dove_tail_mesh", 7),
    "left-thigh": ("thigh", "dove_tail_mesh", 8),
    "left-shank": ("shank", "dove_tail_mesh", 9),
    "left-foot": ("foot", "dove_tail_mesh", 10),
    "right-thigh": ("thigh", "dove_tail_mesh", 11),
    "right-shank": ("shank", "dove_tail_mesh", 12),
    "right-foot": ("foot", "dove_tail_mesh", 13),
}


def cook_packs(cooker: Path, authored: Path, output: Path) -> None:
    cooker = cooker.resolve()
    authored = authored.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, (source, mesh, body) in CROW_VISUAL_LINKS.items():
        provenance = (
            "estimated-hybrid-collision-proxy"
            if source in {"thigh", "shank", "foot"}
            else "estimated-american-crow-hybrid-surface-v1"
        )
        subprocess.run(
            [
                str(cooker),
                f"{source}.usda",
                str(output / f"{name}.mrvpack"),
                "--id", f"birdflow_crow_{name.replace('-', '_')}",
                "--license", "NOASSERTION",
                "--provenance", provenance,
                "--link", f"{mesh}={body}",
            ],
            check=True,
            cwd=authored,
        )


def reframe_obj(
    path: Path,
    origin: tuple[float, float, float],
    rotation_x: float = 0.0,
) -> None:
    """Move root-authored vertices into an owning link's default frame."""
    cosine = math.cos(rotation_x)
    sine = math.sin(rotation_x)
    output: list[str] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            x = float(fields[1]) - origin[0]
            y = float(fields[2]) - origin[1]
            z = float(fields[3]) - origin[2]
            # Runtime maps link-local vertices by R_x(rotation_x). Apply the
            # inverse here so the default articulated pose reconstructs the
            # original contiguous BirdFlow surface exactly.
            local_y = cosine * y + sine * z
            local_z = -sine * y + cosine * z
            line = f"v {x:.8f} {local_y:.8f} {local_z:.8f}"
        output.append(line)
    path.write_text("\n".join(output) + "\n")


def recolor_crow(output: Path) -> None:
    material = output / "dove.mtl"
    text = material.read_text()
    # The sensor-fast renderer has no feather BRDF. Use readable blue-black
    # debug albedos instead of near-zero beauty-render values that collapse to
    # a silhouette in the Numi hall.
    text = text.replace("Kd 0.78 0.80 0.84", "Kd 0.30 0.36 0.48")
    text = text.replace("Kd 0.58 0.61 0.67", "Kd 0.22 0.31 0.50")
    text = text.replace("Kd 0.45 0.48 0.54", "Kd 0.32 0.22 0.50")
    material.write_text(text)


def recolor_usda(path: Path) -> None:
    text = path.read_text()
    text = text.replace("(0.78, 0.8, 0.84)", "(0.30, 0.36, 0.48)")
    text = text.replace("(0.58, 0.61, 0.67)", "(0.22, 0.31, 0.50)")
    text = text.replace("(0.45, 0.48, 0.54)", "(0.32, 0.22, 0.50)")
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
    parser.add_argument("--cooker", type=Path)
    parser.add_argument("--pack-output", type=Path)
    args = parser.parse_args()
    if (args.cooker is None) != (args.pack_output is None):
        parser.error("--cooker and --pack-output must be supplied together")
    args.output.mkdir(parents=True, exist_ok=True)

    write_surface_parts(args.surface, args.output)
    recolor_crow(args.output)
    wing_origin_y = (0.0196 + 0.0477765) * (0.91 / 0.48)
    wing_origin_z = (0.0165 - 0.0103564) * 1.15
    reframe_obj(
        args.output / "right-wing.obj",
        (0.0, wing_origin_y, wing_origin_z),
        0.35,
    )
    reframe_obj(
        args.output / "left-wing.obj",
        (0.0, -wing_origin_y, wing_origin_z),
        -0.35,
    )
    reframe_obj(
        args.output / "tail.obj",
        (
            (-0.0305 - 0.0903) * 1.15,
            (0.0 - 0.0095) * 1.15,
            (0.0180 - 0.02464) * 1.15,
        ),
    )
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
    if args.cooker is not None:
        cook_packs(args.cooker, args.output, args.pack_output)


if __name__ == "__main__":
    main()
