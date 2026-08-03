#!/usr/bin/env python3
"""Render official Franka meshes from an exact Numi Lab pose trajectory.

Run this file through Blender, for example:
  blender --background --python render_gif.py -- --poses poses.csv \
    --franka-description /path/to/franka_description --frames /tmp/frames
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--franka-description", type=Path, required=True)
    parser.add_argument("--converted-meshes", type=Path)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=96)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview-frame", type=int)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
):
    result = bpy.data.materials.new(name)
    result.diffuse_color = color
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return result


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    surface,
    bevel: float = 0.02,
):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(surface)
    if bevel > 0.0:
        modifier = obj.modifiers.new("precision bevel", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def point_camera(camera, target: tuple[float, float, float]) -> None:
    camera.rotation_euler = (
        Vector(target) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()


def load_poses(path: Path) -> list[dict[int, tuple[float, ...]]]:
    frames: dict[int, dict[int, tuple[float, ...]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            body = int(row["body_index"])
            frames.setdefault(step, {})[body] = tuple(
                float(row[key])
                for key in (
                    "link_x_m",
                    "link_y_m",
                    "link_z_m",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                )
            )
    result = [frames[index] for index in sorted(frames)]
    if not result or any(set(frame) != set(BODY_MESHES) for frame in result):
        raise RuntimeError("pose stream does not contain every Franka body")
    return result


def import_franka(description: Path, converted_meshes: Path | None) -> dict[int, object]:
    mesh_root = description / "meshes" / "robots" / "fer" / "visual"
    roots = {}
    for body, filename in BODY_MESHES.items():
        source = mesh_root / filename
        if not source.is_file():
            raise RuntimeError(f"missing official Franka mesh: {source}")
        before = set(bpy.context.scene.objects)
        converted = (
            converted_meshes / f"body-{body:02d}.glb"
            if converted_meshes is not None
            else None
        )
        if converted is not None and converted.is_file():
            bpy.ops.import_scene.gltf(filepath=str(converted))
        elif hasattr(bpy.ops.wm, "collada_import"):
            bpy.ops.wm.collada_import(filepath=str(source))
        else:
            raise RuntimeError(
                "this Blender build has no Collada importer; run "
                "convert_meshes.py and pass --converted-meshes"
            )
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        root = bpy.data.objects.new(f"franka_body_{body:02d}", None)
        bpy.context.scene.collection.objects.link(root)
        top_level = [obj for obj in imported if obj.parent not in imported]
        for obj in top_level:
            # trimesh writes glTF's Y-up convention from the source Collada's
            # Z-up coordinates. Blender imports that root as (x, -z, y), so
            # cancel the transport rotation before applying Numi's Z-up pose.
            world = Matrix.Rotation(-math.pi / 2.0, 4, "X") @ obj.matrix_world
            obj.parent = root
            obj.matrix_world = world
        for obj in imported:
            if obj.type == "MESH":
                obj.name = f"franka_body_{body:02d}_{obj.name}"
        roots[body] = root

    # Preserve the official white/grey/black separation while translating the
    # legacy Collada materials into a controlled physically based response.
    for surface in bpy.data.materials:
        if surface.name.startswith("Numi"):
            continue
        color = surface.diffuse_color[:]
        surface.use_nodes = True
        shader = surface.node_tree.nodes.get("Principled BSDF")
        if shader is not None:
            shader.inputs["Base Color"].default_value = color
            shader.inputs["Roughness"].default_value = 0.31 if sum(color[:3]) > 1.5 else 0.42
            shader.inputs["Metallic"].default_value = 0.05
    return roots


def apply_pose(roots, pose: dict[int, tuple[float, ...]]) -> None:
    for body, root in roots.items():
        px, py, pz, qx, qy, qz, qw = pose[body]
        root.location = (px, py, pz)
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = (qw, qx, qy, qz)


def body_bounds(root) -> tuple[Vector, Vector]:
    points = []
    for obj in bpy.context.scene.objects:
        ancestor = obj.parent
        while ancestor is not None and ancestor != root:
            ancestor = ancestor.parent
        if ancestor == root and obj.type == "MESH":
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        raise RuntimeError(f"official mesh body has no geometry: {root.name}")
    return (
        Vector(min(point[axis] for point in points) for axis in range(3)),
        Vector(max(point[axis] for point in points) for axis in range(3)),
    )


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
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.render.film_transparent = False
    scene.render.image_settings.color_management = "FOLLOW_SCENE"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.compression = 22
    scene.render.resolution_percentage = 100
    if hasattr(scene, "eevee"):
        scene.eevee.taa_samples = options.samples

    world = bpy.data.worlds.new("Numi Lab World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.006,
        0.012,
        0.028,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.08

    floor = material("Numi graphite floor", (0.014, 0.021, 0.034, 1), 0.38, 0.12)
    platform = material("Numi blue platform", (0.025, 0.09, 0.16, 1), 0.27, 0.24)
    trim = material("Numi brushed trim", (0.18, 0.26, 0.34, 1), 0.24, 0.66)
    cyan = material("Numi cyan prop", (0.015, 0.42, 0.62, 1), 0.28, 0.18)
    orange = material("Numi orange prop", (0.95, 0.20, 0.055, 1), 0.33, 0.06)

    add_box("studio floor", (0, 0, -0.28), (4.0, 4.0, 0.10), floor, 0.06)
    add_box("robot plinth", (0, 0, -0.13), (0.59, 0.59, 0.055), platform, 0.035)
    add_box("plinth inset", (0, 0, -0.069), (0.51, 0.51, 0.006), trim, 0.01)

    add_box("cyan target", (0.66, 0.38, 0.005), (0.075, 0.075, 0.075), cyan, 0.012)
    add_box("orange target", (-0.66, 0.52, 0.035), (0.09, 0.09, 0.105), orange, 0.014)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.085, depth=0.16, location=(0.50, -0.52, 0.01))
    bpy.context.object.name = "steel target"
    bpy.context.object.data.materials.append(trim)
    bevel = bpy.context.object.modifiers.new("precision bevel", "BEVEL")
    bevel.width = 0.012
    bevel.segments = 3

    # Soft key/fill/rim rig gives the robot readable volume without clipping
    # the official white covers or losing the charcoal joint details.
    lights = (
        ("key", "AREA", (1.4, -1.3, 2.7), 520.0, 2.1, (0.80, 0.91, 1.0)),
        ("fill", "AREA", (-1.8, -0.4, 1.5), 240.0, 1.7, (0.34, 0.55, 1.0)),
        ("rim", "AREA", (0.4, 1.7, 2.2), 360.0, 1.5, (0.20, 0.68, 1.0)),
    )
    for name, kind, location, energy, size, color in lights:
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (0, 0, 0)
        point_camera(obj, (0, 0, 0.42))

    camera_data = bpy.data.cameras.new("Numi Lab cinema camera")
    camera = bpy.data.objects.new("Numi Lab cinema camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 58
    camera_data.sensor_width = 36
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = 2.25
    camera_data.dof.aperture_fstop = 7.1
    return camera


def main() -> None:
    options = arguments()
    if options.frame_count < 2:
        raise RuntimeError("--frame-count must be at least two")
    options.frames.mkdir(parents=True, exist_ok=True)
    poses = load_poses(options.poses)
    if options.start_step < 0 or options.start_step >= len(poses) - 1:
        raise RuntimeError("--start-step must select a non-final trajectory pose")
    poses = poses[options.start_step :]
    camera = configure_scene(options)
    roots = import_franka(options.franka_description, options.converted_meshes)

    frame_indices = [
        round(index * (len(poses) - 1) / (options.frame_count - 1))
        for index in range(options.frame_count)
    ]
    render_indices = (
        [max(0, min(options.frame_count - 1, options.preview_frame))]
        if options.preview_frame is not None
        else range(options.frame_count)
    )
    for output_index in render_indices:
        apply_pose(roots, poses[frame_indices[output_index]])
        if options.preview_frame is not None:
            bpy.context.view_layer.update()
            for body, root in roots.items():
                lower, upper = body_bounds(root)
                print(
                    f"visual_body={body} bounds="
                    f"{lower.x:.6f}:{lower.y:.6f}:{lower.z:.6f}:"
                    f"{upper.x:.6f}:{upper.y:.6f}:{upper.z:.6f}"
                )
        phase = 2.0 * math.pi * output_index / (options.frame_count - 1)
        angle = math.radians(-43.0 + 4.0 * math.sin(phase))
        radius = 2.55 + 0.035 * math.cos(phase)
        camera.location = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            1.16 + 0.035 * math.sin(phase),
        )
        point_camera(camera, (0.0, 0.02, 0.34))
        bpy.context.scene.render.filepath = str(
            options.frames / f"franka-{output_index:04d}.png"
        )
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
