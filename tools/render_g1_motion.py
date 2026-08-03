#!/usr/bin/env python3
"""Render a Numi G1 motion-retarget artifact with official Unitree meshes.

Run through Blender:
  blender --background --python tools/render_g1_motion.py -- \
    --retarget-directory /path/to/g1-retarget \
    --g1-urdf /path/to/g1_29dof_rev_1_0.urdf \
    --frames /path/to/frames
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-directory", type=Path, required=True)
    parser.add_argument("--g1-urdf", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview-frame", type=int)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _values(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != len(default):
        raise RuntimeError(f"invalid URDF vector: {text}")
    return values


def _matrix(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> Matrix:
    result = Matrix.Translation(Vector(xyz))
    result @= Matrix.Rotation(rpy[2], 4, "Z")
    result @= Matrix.Rotation(rpy[1], 4, "Y")
    result @= Matrix.Rotation(rpy[0], 4, "X")
    return result


def material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    metallic: float,
    roughness: float,
):
    result = bpy.data.materials.new(name)
    result.diffuse_color = color
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return result


def point_camera(camera, target: tuple[float, float, float]) -> None:
    camera.rotation_euler = (
        Vector(target) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()


def load_motion(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["link_names"]]
        transforms = np.asarray(
            archive["link_position_quaternion_xyzw"],
            dtype=np.float64,
        )
    if (
        transforms.ndim != 3
        or transforms.shape[0] < 2
        or transforms.shape[1] != len(names)
        or transforms.shape[2] != 7
        or not np.isfinite(transforms).all()
        or len(set(names)) != len(names)
    ):
        raise RuntimeError("retarget artifact has invalid link transforms")
    return names, transforms


def import_g1(urdf_path: Path, link_names: list[str]) -> dict[str, object]:
    description = urdf_path.parent
    robot = ET.parse(urdf_path).getroot()
    named_materials = {
        "dark": material(
            "Numi G1 graphite",
            (0.035, 0.055, 0.075, 1.0),
            metallic=0.46,
            roughness=0.22,
        ),
        "white": material(
            "Numi G1 ceramic",
            (0.72, 0.79, 0.84, 1.0),
            metallic=0.18,
            roughness=0.19,
        ),
    }
    roots: dict[str, object] = {}
    for name in link_names:
        root = bpy.data.objects.new(f"g1::{name}", None)
        bpy.context.scene.collection.objects.link(root)
        roots[name] = root

    for link in robot.findall("link"):
        name = link.attrib["name"]
        if name not in roots:
            continue
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                filename = filename.split("g1_description/", 1)[-1]
            source = (description / filename).resolve()
            if not source.is_file():
                raise RuntimeError(f"missing official G1 mesh: {source}")
            before = set(bpy.context.scene.objects)
            bpy.ops.wm.stl_import(filepath=str(source))
            imported = [obj for obj in bpy.context.scene.objects if obj not in before]
            if not imported:
                raise RuntimeError(f"Blender imported no geometry from {source}")
            origin = visual.find("origin")
            xyz = _values(
                None if origin is None else origin.attrib.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _values(
                None if origin is None else origin.attrib.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            scale = _values(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
            local = _matrix(xyz, rpy) @ Matrix.Diagonal((*scale, 1.0))
            material_element = visual.find("material")
            material_name = (
                "white"
                if material_element is None
                else material_element.attrib.get("name", "white")
            )
            surface = named_materials.get(material_name, named_materials["white"])
            for imported_index, obj in enumerate(imported):
                obj.name = f"g1::{name}::{visual_index}::{imported_index}"
                obj.parent = roots[name]
                obj.matrix_local = local
                if obj.type == "MESH":
                    obj.data.materials.clear()
                    obj.data.materials.append(surface)
                    for polygon in obj.data.polygons:
                        polygon.use_smooth = True
                    bevel = obj.modifiers.new("edge softness", "BEVEL")
                    bevel.width = 0.0008
                    bevel.segments = 2
    return roots


def apply_motion(
    roots: dict[str, object],
    names: list[str],
    transforms: np.ndarray,
) -> None:
    for frame_index, frame in enumerate(transforms, start=1):
        for link_index, name in enumerate(names):
            root = roots[name]
            value = frame[link_index]
            root.location = tuple(float(component) for component in value[:3])
            root.rotation_mode = "QUATERNION"
            root.rotation_quaternion = (
                float(value[6]),
                float(value[3]),
                float(value[4]),
                float(value[5]),
            )
            root.keyframe_insert(data_path="location", frame=frame_index)
            root.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)


def configure_scene(options: argparse.Namespace):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = options.width
    scene.render.resolution_y = options.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 18
    scene.render.film_transparent = False
    scene.render.fps = 20
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_samples = options.samples

    world = bpy.data.worlds.new("Numi motion studio")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.004, 0.008, 0.018, 1.0)
    background.inputs["Strength"].default_value = 0.12

    floor_surface = material(
        "Numi floor",
        (0.012, 0.020, 0.032, 1.0),
        metallic=0.18,
        roughness=0.26,
    )
    # Presentation shares the solver's exact z=0 support plane. Never lower
    # the visible floor to conceal contact penetration.
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Numi studio floor"
    floor.data.materials.append(floor_surface)

    ring_surface = material(
        "Numi cyan ring",
        (0.015, 0.32, 0.47, 1.0),
        metallic=0.34,
        roughness=0.2,
    )
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.73,
        minor_radius=0.006,
        major_segments=128,
        minor_segments=12,
        location=(0.0, 0.0, 0.001),
    )
    bpy.context.object.data.materials.append(ring_surface)

    bpy.ops.object.light_add(type="AREA", location=(2.1, -2.4, 3.4))
    key = bpy.context.object
    key.name = "Numi key"
    key.data.energy = 1250.0
    key.data.shape = "DISK"
    key.data.size = 2.8
    key.data.color = (0.72, 0.89, 1.0)
    point_camera(key, (0.0, 0.0, 0.85))

    bpy.ops.object.light_add(type="AREA", location=(-2.4, 1.8, 2.4))
    rim = bpy.context.object
    rim.name = "Numi rim"
    rim.data.energy = 900.0
    rim.data.size = 2.2
    rim.data.color = (0.18, 0.56, 1.0)
    point_camera(rim, (0.0, 0.0, 1.0))

    bpy.ops.object.light_add(type="AREA", location=(0.2, 2.4, 1.2))
    fill = bpy.context.object
    fill.name = "Numi fill"
    fill.data.energy = 520.0
    fill.data.size = 3.0
    fill.data.color = (1.0, 0.55, 0.25)
    point_camera(fill, (0.0, 0.0, 0.85))

    bpy.ops.object.camera_add(location=(2.35, -3.10, 1.35))
    camera = bpy.context.object
    camera.data.lens = 72.0
    camera.data.sensor_width = 36.0
    point_camera(camera, (0.0, 0.0, 0.84))
    scene.camera = camera
    return camera


def apply_camera_motion(
    camera,
    names: list[str],
    transforms: np.ndarray,
) -> None:
    pelvis_index = names.index("pelvis")
    base_location = Vector((2.35, -3.10, 1.35))
    base_pelvis_height = float(transforms[0, pelvis_index, 2])
    for frame_index, frame in enumerate(transforms, start=1):
        pelvis = Vector(tuple(float(value) for value in frame[pelvis_index, :3]))
        translation = Vector(
            (
                pelvis.x,
                pelvis.y,
                pelvis.z - base_pelvis_height,
            )
        )
        camera.location = base_location + translation
        point_camera(camera, (pelvis.x, pelvis.y, pelvis.z + 0.05))
        camera.keyframe_insert(data_path="location", frame=frame_index)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_index)


def render(options: argparse.Namespace) -> None:
    camera = configure_scene(options)
    names, transforms = load_motion(
        options.retarget_directory / "g1_motion_retarget.npz"
    )
    roots = import_g1(options.g1_urdf, names)
    apply_motion(roots, names, transforms)
    apply_camera_motion(camera, names, transforms)
    scene = bpy.context.scene
    options.frames.mkdir(parents=True, exist_ok=True)
    indices = (
        [options.preview_frame]
        if options.preview_frame is not None
        else list(range(transforms.shape[0]))
    )
    for output_index, source_index in enumerate(indices):
        if not 0 <= source_index < transforms.shape[0]:
            raise RuntimeError("preview frame is outside the retarget artifact")
        scene.frame_set(source_index + 1)
        scene.render.filepath = str(options.frames / f"g1-{output_index:04d}.png")
        bpy.ops.render.render(write_still=True)
    evidence = {
        "format": "numi.g1-presentation.v1",
        "transform_policy": "exact supplied link transforms; no floor correction",
        "dynamics_synthesized": False,
    }
    (options.frames.parent / "render-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render(arguments())
