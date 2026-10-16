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


def tet_face_audit(rest, cells, labels, finite_glue, material_names):
    """Count conforming triangular faces and independently measure interfaces."""
    faces = {}
    for cell_index, cell in enumerate(cells):
        if len(set(map(int, cell))) != 4:
            raise ValueError(f"tetrahedron {cell_index} repeats a node")
        for opposite in range(4):
            face = tuple(sorted(int(cell[j]) for j in range(4) if j != opposite))
            faces.setdefault(face, []).append((cell_index, int(labels[cell_index])))

    incidence = {1: 0, 2: 0}
    interface_areas = {}
    interface_counts = {}
    for face, owners in faces.items():
        count = len(owners)
        if count > 2:
            raise ValueError(f"nonmanifold tetrahedral face {face} has incidence {count}")
        incidence[count] = incidence.get(count, 0) + 1
        if count != 2:
            continue
        first, second = owners[0][1], owners[1][1]
        if first == second:
            continue
        pair = tuple(sorted((first, second)))
        points = rest[np.asarray(face, dtype=int)]
        area = 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"degenerate shared face {face}")
        interface_counts[pair] = interface_counts.get(pair, 0) + 1
        interface_areas[pair] = interface_areas.get(pair, 0.0) + area

    if finite_glue:
        allowed = {
            tuple(sorted((material_names["liner"], material_names["glue"]))),
            tuple(sorted((material_names["medium"], material_names["glue"]))),
        }
        missing = allowed - set(interface_counts)
        if missing:
            raise ValueError(f"finite-glue mesh is missing required material interfaces: {sorted(missing)}")
        unexpected = set(interface_counts) - allowed
    else:
        allowed = {tuple(sorted((material_names["liner"], material_names["medium"])))}
        unexpected = set(interface_counts) - allowed
    if unexpected:
        raise ValueError(f"unexpected shared material interfaces: {sorted(unexpected)}")

    interfaces = []
    index_names = {index: role for role, index in material_names.items()}
    for pair in sorted(interface_counts):
        interfaces.append({
            "material_indices": list(pair),
            "material_names": [index_names.get(pair[0], f"material_{pair[0]}"),
                               index_names.get(pair[1], f"material_{pair[1]}")],
            "shared_face_count": interface_counts[pair],
            "shared_face_area_m2": interface_areas[pair],
        })
    return {
        "tetrahedral_faces": len(faces),
        "face_incidence_counts": {str(key): incidence.get(key, 0) for key in (1, 2)},
        "boundary_face_count": incidence.get(1, 0),
        "shared_face_count": incidence.get(2, 0),
        "material_interface_face_count": sum(interface_counts.values()),
        "material_interfaces": interfaces,
        "interface_policy": "finite glue: liner-glue and medium-glue only" if finite_glue
                            else "legacy direct liner-medium interface permitted",
    }


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

    manifest_path = mesh_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    material_names = {"liner": 0, "medium": 1, "glue": 2}
    finite_glue = len(densities) >= 3 and bool(np.any(labels == 2))
    if manifest is not None:
        rows = manifest.get("materials", {}).get("cells", [])
        by_role = {}
        for row in rows:
            name = str(row.get("name", "")).lower()
            index = int(row["index"])
            if "liner" in name:
                by_role["liner"] = index
            if "medium" in name:
                by_role["medium"] = index
            if "glue" in name:
                by_role["glue"] = index
        material_names.update(by_role)
        finite_glue = "glue" in by_role or bool(manifest.get("glue_footprints"))
    if not {"liner", "medium"}.issubset(material_names):
        raise ValueError("could not identify liner and medium material indices")

    # Compute deformation from exported positions and an independent FP64
    # rest inverse, not the native solver's determinant telemetry.
    dm = np.stack([rest[cells[:, j]] - rest[cells[:, 0]] for j in (1, 2, 3)], axis=2)
    ds = np.stack([current[cells[:, j]] - current[cells[:, 0]] for j in (1, 2, 3)], axis=2)
    determinants = np.linalg.det(dm)
    if not (determinants > 0).all():
        raise ValueError("authored tetrahedron is inverted or degenerate")
    volumes = determinants / 6
    topology = tet_face_audit(rest, cells, labels, finite_glue, material_names)
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
    result = {"schema": "numi.cardboard.geometry-audit.v1", "status": "passed",
            "scope": "Exported geometry validity and measured deformation; no claim of physical calibration or solver convergence",
            "nodes": len(rest), "free_nodes": int(free.sum()), "fixed_nodes": len(fixed),
            "tetrahedra": len(cells), "connected_components": component_count,
            "minimum_reference_tet_volume_m3": float(volumes.min()),
            "total_regional_mass_kg": float((volumes * densities[labels]).sum()),
            "authored_extent_m": (rest.max(axis=0) - rest.min(axis=0)).tolist(),
            "max_node_displacement_m": float(displacement.max()),
            "max_free_node_displacement_m": float(displacement[free].max()) if free.any() else None,
            "materials": material_rows,
            "topology": topology,
            "inputs": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (mesh_path, accepted_path)}}
    if manifest is not None:
        result["inputs"][str(manifest_path.resolve())] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        footprints = manifest.get("glue_footprints", [])
        if footprints:
            geometry = manifest.get("geometry", {})
            width = float(geometry["width_m"])
            footprint_volume = sum(float(item["cross_section_area_m2"]) * width for item in footprints)
            glue_index = material_names.get("glue")
            declared_glue_volume = next((float(row["volume_m3"]) for row in manifest["materials"]["cells"]
                                         if int(row["index"]) == glue_index), None)
            if declared_glue_volume is None:
                raise ValueError("manifest has glue footprints but no declared glue material volume")
            tolerance = max(1e-14, 1e-6 * abs(declared_glue_volume))
            if abs(footprint_volume - declared_glue_volume) > tolerance:
                raise ValueError("manifest glue volume disagrees with footprint area times board width")
            result["glue_volume_crosscheck"] = {
                "footprint_count": len(footprints),
                "board_width_m": width,
                "footprint_area_times_width_m3": footprint_volume,
                "declared_glue_volume_m3": declared_glue_volume,
                "absolute_difference_m3": abs(footprint_volume - declared_glue_volume),
                "tolerance_m3": tolerance,
            }
    return result


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
