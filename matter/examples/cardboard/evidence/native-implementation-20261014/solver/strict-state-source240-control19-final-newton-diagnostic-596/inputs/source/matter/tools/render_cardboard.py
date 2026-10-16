#!/usr/bin/env python3
"""Render an exported FEM surface; never manufacture a simulated deformation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_obj(path: Path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        items = line.split()
        if not items or items[0].startswith("#"):
            continue
        if items[0] == "v":
            vertices.append([float(x) for x in items[1:4]])
        elif items[0] == "f":
            ids = [int(x.split("/")[0]) - 1 for x in items[1:]]
            if len(ids) < 3:
                raise ValueError("surface face has fewer than three vertices")
            faces.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
    xyz = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=int)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("invalid or nonfinite OBJ positions")
    if tri.size == 0 or tri.min() < 0 or tri.max() >= len(xyz):
        raise ValueError("invalid OBJ surface indices")
    return xyz * 1000, tri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rest", type=Path)
    parser.add_argument("accepted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default="Native FEM diagnostic; physical fidelity unqualified")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rest, tri0 = read_obj(args.rest)
    accepted, tri1 = read_obj(args.accepted)
    if rest.shape != accepted.shape or not np.array_equal(tri0, tri1):
        raise ValueError("comparison requires identical surface topology")
    all_xyz = np.concatenate([rest, accepted])
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    fig = plt.figure(figsize=(13, 7.5), layout="constrained")
    for index, (xyz, title) in enumerate(((rest, "Authored geometry"), (accepted, "Exported accepted state"))):
        ax = fig.add_subplot(2, 2, index + 1, projection="3d")
        mesh = Poly3DCollection(xyz[tri0], facecolor="#c29860", edgecolor="#795c35", linewidth=.12, alpha=1)
        ax.add_collection3d(mesh)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title=title)
        ax.set_box_aspect(span)
        ax.view_init(elev=20, azim=-65)
        section = fig.add_subplot(2, 2, index + 3)
        # Display the authored y-minimum boundary, keeping identical node
        # identities for the accepted frame even if those nodes move in y.
        boundary = np.isclose(rest[:, 1], rest[:, 1].min(), atol=1e-6)
        edges = set()
        for face in tri0:
            for a, b in zip(face, np.roll(face, -1)):
                if boundary[a] and boundary[b]:
                    edges.add(tuple(sorted((int(a), int(b)))))
        for a, b in sorted(edges):
            section.plot(xyz[[a, b], 0], xyz[[a, b], 2], color="#795c35", linewidth=.45)
        section.set(xlabel="x (mm)", ylabel="z (mm)", title="Same boundary nodes; no displacement magnification")
        section.set_aspect("equal", adjustable="box")
        section.set_xlim(lo[0] - .02 * span[0], hi[0] + .02 * span[0])
        section.set_ylim(lo[2] - .05 * span[2], hi[2] + .05 * span[2])
        section.grid(alpha=.15)
    fig.suptitle(args.label, fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    metadata = {"schema": "numi.cardboard.render.v1", "label": args.label,
                "displacement_scale": 1, "vertex_count": len(rest), "triangle_count": len(tri0),
                "maximum_vertex_displacement_mm": float(np.linalg.norm(accepted - rest, axis=1).max()),
                "inputs": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.rest, args.accepted)}}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
