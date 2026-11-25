#!/usr/bin/env python3
"""Convert the pinned PX4 x500_base model's visible meshes to the Mac GLB."""

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import trimesh


def pose(element):
    x, y, z, roll, pitch, yaw = map(float, element.findtext("pose", "0 0 0 0 0 0").split())
    transform = trimesh.transformations.euler_matrix(roll, pitch, yaw)
    transform[:3, 3] = (x, y, z)
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Upstream models/x500_base directory")
    parser.add_argument("output", type=Path, help="Output GLB")
    args = parser.parse_args()
    model = ElementTree.parse(args.source / "model.sdf").getroot()
    appearance = {
        "CarbonFiber": (65, 82, 104),
        "Metal": (165, 181, 190),
        "FMUK66": (237, 118, 29),
        "LandingPlastic": (88, 107, 125),
    }
    meshes = {}
    airframe = trimesh.Scene()
    rotors = {f"rotor_{index}": trimesh.Scene() for index in range(4)}
    for link in model.findall("./model/link"):
        rotor = rotors.get(link.attrib["name"])
        scene = airframe if rotor is None else rotor
        link_pose = pose(link)
        for visual in link.findall("visual"):
            mesh_spec = visual.find("geometry/mesh")
            if mesh_spec is None:
                continue
            path = args.source / "meshes" / Path(mesh_spec.findtext("uri")).name
            if path not in meshes:
                bound_parts = trimesh.load_scene(path).dump()
                converted = []
                for part in bound_parts:
                    key = part.metadata.get("name", "") or ""
                    # COLLADA splits position vertices at normal and UV seams.
                    # Weld them before decimating to preserve connected surfaces.
                    part.merge_vertices(merge_tex=True, merge_norm=True)
                    target = max(120, len(part.faces) // 12)
                    if target < len(part.faces):
                        part = part.simplify_quadric_decimation(face_count=target)
                    part.update_faces(part.nondegenerate_faces())
                    part.remove_unreferenced_vertices()
                    converted.append((key.removesuffix("-mesh"), part))
                meshes[path] = converted
            scale = np.array(list(map(float, mesh_spec.findtext("scale", "1 1 1").split())))
            transform = pose(visual) @ np.diag([*scale, 1.0])
            if rotor is None:
                transform = link_pose @ transform
            for index, (part_name, template) in enumerate(meshes[path]):
                part = template.copy()
                part.apply_transform(transform)
                normals = part.vertex_normals.copy()
                missing = np.linalg.norm(normals, axis=1) < 0.99
                if np.any(missing):
                    fallback = np.zeros_like(normals)
                    fallback[part.faces.ravel()] = np.repeat(part.face_normals, 3, axis=0)
                    normals[missing] = fallback[missing]
                    part.vertex_normals = normals
                if path.suffix == ".stl":
                    color = (192, 86, 29) if link_pose[0, 3] > 0 else (69, 98, 126)
                else:
                    color = appearance.get(part_name, (91, 105, 120))
                part.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=[*color, 255], roughnessFactor=0.68,
                        metallicFactor=0.15, doubleSided=True,
                    )
                )
                name = f"{visual.attrib['name']}-{index}"
                scene.add_geometry(part, node_name=name, geom_name=name)
    args.output.write_bytes(trimesh.exchange.gltf.export_glb(airframe, include_normals=True))
    for index, rotor in enumerate(rotors.values()):
        rotor_path = args.output.with_name(f"{args.output.stem}-rotor-{index}.glb")
        rotor_path.write_bytes(trimesh.exchange.gltf.export_glb(rotor, include_normals=True))


if __name__ == "__main__":
    main()
