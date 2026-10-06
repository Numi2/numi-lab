#!/usr/bin/env python3
"""Render the actual exported boundary mesh with regional material colors."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mesh = json.loads(args.mesh.read_text())
    vertices = np.asarray(mesh["nodes_m"], dtype=float) * 1000
    cells = np.asarray(mesh["tetrahedra"], dtype=int)
    labels = np.asarray(mesh["material_indices"], dtype=int)
    if not np.isfinite(vertices).all() or labels.shape != (len(cells),):
        raise ValueError("invalid mesh positions or material labels")
    palette = ["#ad7941", "#d0a25d", "#54aeb9"]
    if labels.min() < 0 or labels.max() >= len(palette):
        raise ValueError("expected liner, medium and optional glue materials")
    polygons, colors = [], []
    y_min = vertices[:, 1].min()
    for cell, label in zip(cells, labels):
        for omit in range(4):
            triangle = np.delete(cell, omit)
            if np.all(np.abs(vertices[triangle, 1] - y_min) < 1e-8):
                polygons.append(vertices[triangle][:, [0, 2]])
                colors.append(palette[label])
    if not polygons:
        raise ValueError("no flat y-minimum boundary found")
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout="constrained")
    for ax in axes:
        ax.add_collection(PolyCollection(polygons, facecolors=colors,
            edgecolors="#483725", linewidths=.12))
        ax.set_aspect("equal")
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Height (mm)")
        ax.autoscale_view()
    axes[0].set_title("Authored corrugated board: liners, fluted medium and glue")
    bounds = vertices.max(axis=0) - vertices.min(axis=0)
    # A geometry view only: zoom, never alter geometry or magnify displacement.
    axes[1].set_xlim(vertices[:, 0].min(), vertices[:, 0].min() + .3 * bounds[0])
    axes[1].set_ylim(vertices[:, 2].min() - .05 * bounds[2], vertices[:, 2].max() + .05 * bounds[2])
    axes[1].set_title("Boundary detail — actual mesh, unchanged aspect ratio")
    present = sorted(set(labels.tolist()))
    names = ["Liner paper", "Corrugated paper", "Elastic starch glue"]
    axes[0].legend(handles=[Patch(color=palette[i], label=names[i]) for i in present],
        loc="upper center", bbox_to_anchor=(.5, 1.35), ncol=len(present))
    fig.suptitle("Literature-based geometry; manufacturing history and physical fidelity unvalidated", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    metadata = {"schema": "numi.cardboard.regional-render.v1", "physical_validation": False,
                "geometry_scale": 1, "boundary_triangles": len(polygons),
                "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
