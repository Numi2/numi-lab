#!/usr/bin/env python3
"""Reject disconnected official meshes after applying a Numi Lab pose."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh


PARENTS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 8}


def rotation(row: dict[str, str]) -> np.ndarray:
    x, y, z, w = (float(row[key]) for key in ("qx", "qy", "qz", "qw"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def nearest_vertex_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left[:: max(1, len(left) // 12000)]
    right = right[:: max(1, len(right) // 12000)]
    best = float("inf")
    for start in range(0, len(left), 512):
        delta = left[start : start + 512, None, :] - right[None, :, :]
        best = min(best, float(np.sqrt(np.min(np.sum(delta * delta, axis=2)))))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--converted-meshes", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--maximum-gap", type=float, default=0.012)
    options = parser.parse_args()

    with options.poses.open(newline="") as stream:
        rows = {
            int(row["body_index"]): row
            for row in csv.DictReader(stream)
            if int(row["step"]) == options.step
        }
    if set(rows) != set(range(11)):
        raise RuntimeError("selected pose does not contain all 11 Franka bodies")

    vertices = {}
    for body, row in rows.items():
        mesh = trimesh.load(
            options.converted_meshes / f"body-{body:02d}.glb",
            force="scene",
        ).to_mesh()
        position = np.array(
            [float(row[key]) for key in ("link_x_m", "link_y_m", "link_z_m")]
        )
        local = np.asarray(mesh.vertices, dtype=np.float64)
        local = local[np.all(np.isfinite(local), axis=1)]
        local = local[np.all(np.abs(local) < 10.0, axis=1)]
        if not len(local):
            raise RuntimeError(f"body {body} has no finite official mesh vertices")
        local = local[:: max(1, len(local) // 12000)]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            vertices[body] = local @ rotation(row).T + position

    maximum = 0.0
    for body, parent in PARENTS.items():
        gap = nearest_vertex_distance(vertices[parent], vertices[body])
        maximum = max(maximum, gap)
        print(f"parent={parent} child={body} nearest_gap_m={gap:.6f}")
        if gap > options.maximum_gap:
            raise RuntimeError(
                f"official Franka meshes disconnect at {parent}->{body}: {gap:.6f} m"
            )
    print(f"render_geometry=continuous maximum_adjacent_gap_m={maximum:.6f}")


if __name__ == "__main__":
    main()
