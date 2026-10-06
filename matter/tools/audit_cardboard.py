#!/usr/bin/env python3
"""Independently audit exported tetrahedra and accepted positions in SI units."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def positions(path):
    return np.asarray([[float(x) for x in line.split()[1:4]]
                       for line in path.read_text().splitlines()
                       if line.startswith("v ")], dtype=float)


def rotation(q):
    x, y, z, w = q
    return np.asarray([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                       [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                       [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def audit(mesh_path, accepted_path):
    mesh = json.loads(mesh_path.read_text())
    rest = np.asarray(mesh["nodes_m"], dtype=float)
    cells = np.asarray(mesh["tetrahedra"], dtype=int)
    labels = np.asarray(mesh["material_indices"], dtype=int)
    frames = np.asarray(mesh["material_frames_xyzw"], dtype=float)
    densities = np.asarray(mesh["material_densities_kg_m3"], dtype=float)
    fixed = np.asarray(mesh["fixed_nodes"], dtype=int)
    current = positions(accepted_path)
    if rest.ndim != 2 or rest.shape[1] != 3 or current.shape != rest.shape:
        raise ValueError("authored/accepted positions must have the same full node order")
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.size == 0:
        raise ValueError("expected nonempty four-node tetrahedra")
    if cells.min() < 0 or cells.max() >= len(rest) or labels.shape != (len(cells),):
        raise ValueError("cell or material indices invalid")
    if frames.shape != (len(cells), 4) or labels.min() < 0 or labels.max() >= len(densities):
        raise ValueError("material/frame data incomplete")
    if any(not np.isfinite(a).all() for a in (rest, current, frames, densities)):
        raise ValueError("nonfinite geometry or material data")
    if not (densities > 0).all() or not np.allclose(np.linalg.norm(frames, axis=1), 1, atol=2e-7, rtol=0):
        raise ValueError("invalid density or nonunit material frame")
    if fixed.size and (fixed.min() < 0 or fixed.max() >= len(rest) or len(np.unique(fixed)) != len(fixed)):
        raise ValueError("invalid fixed-node list")
    used = np.unique(cells)
    if len(used) != len(rest):
        raise ValueError("mesh contains unused nodes")

    # Compute deformation from exported positions and an independent FP64
    # rest inverse, not the native solver's determinant telemetry.
    dm = np.stack([rest[cells[:, j]] - rest[cells[:, 0]] for j in (1, 2, 3)], axis=2)
    ds = np.stack([current[cells[:, j]] - current[cells[:, 0]] for j in (1, 2, 3)], axis=2)
    determinants = np.linalg.det(dm)
    if not (determinants > 0).all():
        raise ValueError("authored tetrahedron is inverted or degenerate")
    volumes = determinants / 6
    f = ds @ np.linalg.inv(dm)
    jacobians = np.linalg.det(f)
    q = np.stack([rotation(row) for row in frames])
    fq = f @ q
    c = np.swapaxes(fq, 1, 2) @ fq
    stretches = np.sqrt(np.diagonal(c, axis1=1, axis2=2))
    true_normal = np.log(stretches)
    green = .5 * (c - np.eye(3))
    if not np.isfinite(true_normal).all() or not np.isfinite(jacobians).all() or not (jacobians > 0).all():
        raise ValueError("accepted state has nonfinite strain or inverted tetrahedra")

    parent = list(range(len(rest)))
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for cell in cells:
        for node in cell[1:]:
            parent[find(int(node))] = find(int(cell[0]))
    component_count = len({find(i) for i in range(len(rest))})
    displacement = np.linalg.norm(current - rest, axis=1)
    free = np.ones(len(rest), dtype=bool)
    free[fixed] = False
    material_rows = []
    for index in sorted(set(labels.tolist())):
        mask = labels == index
        material_rows.append({"material_index": index, "tetrahedra": int(mask.sum()),
                              "volume_m3": float(volumes[mask].sum()),
                              "mass_kg": float((volumes[mask] * densities[index]).sum()),
                              "min_J": float(jacobians[mask].min()), "max_J": float(jacobians[mask].max()),
                              "max_abs_material_log_stretch_M_C_T": np.abs(true_normal[mask]).max(axis=0).tolist(),
                              "max_abs_material_Green_shear_MC_MT_CT": np.abs(green[mask][:, [0,0,1], [1,2,2]]).max(axis=0).tolist()})
    return {"schema": "numi.cardboard.geometry-audit.v1", "status": "passed",
            "scope": "Exported geometry validity and measured deformation; no claim of physical calibration or solver convergence",
            "nodes": len(rest), "free_nodes": int(free.sum()), "fixed_nodes": len(fixed),
            "tetrahedra": len(cells), "connected_components": component_count,
            "minimum_reference_tet_volume_m3": float(volumes.min()),
            "total_mass_kg_without_adhesive": float((volumes * densities[labels]).sum()),
            "authored_extent_m": (rest.max(axis=0) - rest.min(axis=0)).tolist(),
            "max_node_displacement_m": float(displacement.max()),
            "max_free_node_displacement_m": float(displacement[free].max()) if free.any() else None,
            "materials": material_rows,
            "inputs": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (mesh_path, accepted_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("accepted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.mesh, args.accepted)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
