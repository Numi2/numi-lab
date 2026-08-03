#!/usr/bin/env python3
"""Convert official Franka Collada link meshes to Blender-readable GLB files."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


BODY_MESHES = {
    0: "link0.dae",
    1: "link1.dae",
    2: "link2.dae",
    3: "link3.dae",
    4: "link4.dae",
    5: "link5.dae",
    6: "link6.dae",
    7: "link7.dae",
    8: "hand.dae",
    9: "finger.dae",
    10: "finger.dae",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--franka-description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    source_root = (
        options.franka_description / "meshes" / "robots" / "fer" / "visual"
    )
    options.output.mkdir(parents=True, exist_ok=True)
    for body, filename in BODY_MESHES.items():
        scene = trimesh.load(source_root / filename, force="scene", process=False)
        destination = options.output / f"body-{body:02d}.glb"
        destination.write_bytes(scene.export(file_type="glb"))
        print(f"body={body} source={filename} output={destination}")


if __name__ == "__main__":
    main()
