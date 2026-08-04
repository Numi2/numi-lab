#!/usr/bin/env python3
"""Render the Numi Franka trajectory inside the supplied RoomPlan USDZ."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "franka_explore"))

from render_gif import (  # noqa: E402
    BODY_MESHES,
    apply_pose,
    body_bounds,
    import_franka,
    load_poses,
    point_camera,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--franka-description", type=Path, required=True)
    parser.add_argument("--converted-meshes", type=Path, required=True)
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=72)
    parser.add_argument("--start-step", type=int, default=174)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview-frame", type=int)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def bounds(objects: list[object]) -> tuple[Vector, Vector]:
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        raise RuntimeError("RoomPlan USDZ imported no mesh geometry")
    return (
        Vector(min(point[axis] for point in points) for axis in range(3)),
        Vector(max(point[axis] for point in points) for axis in range(3)),
    )


def neutral_room_material():
    material = bpy.data.materials.new("RoomPlan neutral surface")
    material.diffuse_color = (0.35, 0.38, 0.42, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (0.35, 0.38, 0.42, 1.0)
        shader.inputs["Roughness"].default_value = 0.72
        shader.inputs["Metallic"].default_value = 0.0
    return material


def import_room(path: Path) -> tuple[list[object], Vector, Vector]:
    if not path.is_file():
        raise RuntimeError(f"missing RoomPlan source: {path}")
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.usd_import(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("RoomPlan USDZ imported no mesh objects")
    default_surface = neutral_room_material()
    for obj in mesh_objects:
        if not obj.data.materials:
            obj.data.materials.append(default_surface)
    lower, upper = bounds(mesh_objects)
    print(
        "room_bounds="
        f"{lower.x:.3f}:{lower.y:.3f}:{lower.z:.3f}:"
        f"{upper.x:.3f}:{upper.y:.3f}:{upper.z:.3f}"
    )
    return imported, lower, upper


def configure_scene(options: argparse.Namespace, lower: Vector, upper: Vector):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = options.width
    scene.render.resolution_y = options.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 20
    scene.render.fps = 24
    scene.render.film_transparent = False
    scene.render.image_settings.color_management = "FOLLOW_SCENE"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_samples = options.samples

    world = bpy.data.worlds.new("RoomPlan Numi Lab World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.025,
        0.035,
        0.055,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22

    center = (lower + upper) * 0.5
    floor_z = lower.z
    focus = Vector((center.x, center.y, floor_z + 0.45))
    room_radius = max(3.0, (upper - lower).length * 0.35)

    lights = (
        ("room_key", (center.x - 1.3, center.y - 1.4, upper.z + 1.2), 850.0, 5.0, (1.0, 0.90, 0.78)),
        ("room_fill", (center.x + 2.0, center.y - 0.6, upper.z - 0.4), 500.0, 4.0, (0.58, 0.72, 1.0)),
        ("room_rim", (center.x, center.y + 2.0, upper.z - 0.1), 700.0, 3.0, (0.38, 0.62, 1.0)),
    )
    for name, location, energy, size, color in lights:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        point_camera(obj, tuple(focus))

    camera_data = bpy.data.cameras.new("RoomPlan Numi Lab Camera")
    camera = bpy.data.objects.new("RoomPlan Numi Lab Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 50
    camera_data.sensor_width = 36
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = room_radius
    camera_data.dof.aperture_fstop = 5.6
    return camera, focus, room_radius


def main() -> None:
    options = arguments()
    if options.frame_count < 2:
        raise RuntimeError("--frame-count must be at least two")
    options.frames.mkdir(parents=True, exist_ok=True)
    poses = load_poses(options.poses)
    if options.start_step < 0 or options.start_step >= len(poses) - 1:
        raise RuntimeError("--start-step must select a non-final trajectory pose")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    _, lower, upper = import_room(options.room)
    camera, focus, radius = configure_scene(options, lower, upper)
    roots = import_franka(options.franka_description, options.converted_meshes)

    room_center = (lower + upper) * 0.5
    robot_offset = Vector((room_center.x, room_center.y, lower.z + 0.28))
    frame_indices = [
        round(index * (len(poses) - 1 - options.start_step) / (options.frame_count - 1))
        + options.start_step
        for index in range(options.frame_count)
    ]
    render_indices = (
        [max(0, min(options.frame_count - 1, options.preview_frame))]
        if options.preview_frame is not None
        else range(options.frame_count)
    )
    for output_index in render_indices:
        apply_pose(roots, poses[frame_indices[output_index]])
        for root in roots.values():
            root.location += robot_offset
        if options.preview_frame is not None:
            bpy.context.view_layer.update()
            print(f"robot_offset={tuple(round(value, 3) for value in robot_offset)}")
            print(f"camera_focus={tuple(round(value, 3) for value in focus)}")
        phase = 2.0 * math.pi * output_index / (options.frame_count - 1)
        angle = math.radians(-38.0 + 5.0 * math.sin(phase))
        camera.location = (
            focus.x + radius * math.cos(angle),
            focus.y + radius * math.sin(angle),
            focus.z + 1.8 + 0.20 * math.sin(phase),
        )
        point_camera(camera, tuple(focus))
        if options.preview_frame is not None:
            lower_body, upper_body = body_bounds(roots[0])
            print(
                "robot_body_0_bounds="
                f"{tuple(round(value, 3) for value in lower_body)}:"
                f"{tuple(round(value, 3) for value in upper_body)}"
            )
            print(
                f"camera_location={tuple(round(value, 3) for value in camera.location)}"
            )
        bpy.context.scene.render.filepath = str(
            options.frames / f"roomplan-franka-{output_index:04d}.png"
        )
        bpy.ops.render.render(write_still=True)
        for root in roots.values():
            root.location -= robot_offset


if __name__ == "__main__":
    main()
