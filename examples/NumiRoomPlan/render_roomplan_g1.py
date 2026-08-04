#!/usr/bin/env python3
"""Render accepted Numi G1 solver states inside a RoomPlan USDZ scene.

The robot's relative link transforms are consumed exactly from the native
retarget artifact.  A single rigid presentation translation places that
motion on the imported RoomPlan floor; no dynamics or pose correction is
synthesized by this renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from render_g1_motion import import_g1, load_motion, material, point_camera  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-directory", type=Path, required=True)
    parser.add_argument("--g1-urdf", type=Path, required=True)
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview-frame", type=int)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_bounds(prefix: str) -> tuple[Vector, Vector]:
    points: list[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.name.startswith(prefix):
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        raise RuntimeError(f"no mesh geometry found for {prefix}")
    return (
        Vector(min(point[axis] for point in points) for axis in range(3)),
        Vector(max(point[axis] for point in points) for axis in range(3)),
    )


def import_room(path: Path) -> tuple[Vector, Vector]:
    if not path.is_file():
        raise RuntimeError(f"missing RoomPlan source: {path}")
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.usd_import(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("RoomPlan USDZ imported no mesh geometry")
    surface = material(
        "RoomPlan neutral surface",
        (0.34, 0.38, 0.44, 1.0),
        metallic=0.0,
        roughness=0.76,
    )
    for obj in meshes:
        if not obj.data.materials:
            obj.data.materials.append(surface)
    lower, upper = mesh_bounds("",)
    print(
        "room_bounds="
        f"{lower.x:.3f}:{lower.y:.3f}:{lower.z:.3f}:"
        f"{upper.x:.3f}:{upper.y:.3f}:{upper.z:.3f}"
    )
    return lower, upper


def configure_scene(options: argparse.Namespace, lower: Vector, upper: Vector):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = options.width
    scene.render.resolution_y = options.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 18
    scene.render.fps = 24
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_samples = options.samples

    world = bpy.data.worlds.new("Numi RoomPlan G1 World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.012, 0.020, 0.034, 1.0)
    background.inputs["Strength"].default_value = 0.32

    # The measured living-room area is open around this point.  Keeping the
    # camera on the same side of the scan's interior wall avoids looking
    # through a structural partition.
    focus = Vector((-3.90, -1.70, lower.z + 0.46))
    camera_location = Vector((-0.55, -0.95, lower.z + 1.45))

    lights = (
        ("RoomPlan key", (-2.75, -1.45, upper.z + 0.85), 1050.0, 3.2, (1.0, 0.88, 0.72)),
        ("RoomPlan fill", (-0.80, -1.10, lower.z + 1.25), 650.0, 2.8, (0.56, 0.72, 1.0)),
        ("RoomPlan rim", (-4.80, -0.20, upper.z - 0.20), 820.0, 2.6, (0.34, 0.60, 1.0)),
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

    camera_data = bpy.data.cameras.new("RoomPlan G1 Camera")
    camera = bpy.data.objects.new("RoomPlan G1 Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = camera_location
    camera_data.lens = 47.0
    camera_data.sensor_width = 36.0
    camera_data.clip_end = 100.0
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = (camera_location - focus).length
    camera_data.dof.aperture_fstop = 7.1
    point_camera(camera, tuple(focus))
    scene.camera = camera
    return camera, focus


def apply_frame(roots: dict[str, object], names: list[str], transforms: np.ndarray, index: int) -> None:
    for link_index, name in enumerate(names):
        root = roots[name]
        value = transforms[index, link_index]
        root.location = tuple(float(component) for component in value[:3])
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = (
            float(value[6]),
            float(value[3]),
            float(value[4]),
            float(value[5]),
        )


def main() -> None:
    options = arguments()
    motion_path = options.retarget_directory / "g1_motion_retarget.npz"
    if not motion_path.is_file():
        raise RuntimeError(f"missing G1 retarget artifact: {motion_path}")
    if options.preview_frame is not None and options.preview_frame < 0:
        raise RuntimeError("--preview-frame must be non-negative")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    lower, upper = import_room(options.room)
    camera, focus = configure_scene(options, lower, upper)
    names, transforms = load_motion(motion_path)
    roots = import_g1(options.g1_urdf, names)
    ceramic = bpy.data.materials.get("Numi G1 ceramic")
    if ceramic is not None:
        ceramic.diffuse_color = (0.34, 0.43, 0.52, 1.0)
        shader = ceramic.node_tree.nodes.get("Principled BSDF")
        if shader is not None:
            shader.inputs["Base Color"].default_value = (0.34, 0.43, 0.52, 1.0)
            shader.inputs["Metallic"].default_value = 0.30

    placement = bpy.data.objects.new("G1 RoomPlan placement", None)
    bpy.context.scene.collection.objects.link(placement)
    for root in roots.values():
        root.parent = placement

    # Preserve every inter-link transform from the native artifact.  Only a
    # rigid translation is introduced to place the complete robot in the scan.
    apply_frame(roots, names, transforms, 0)
    bpy.context.view_layer.update()
    first_lower, first_upper = mesh_bounds("g1::")
    first_center = (first_lower + first_upper) * 0.5
    target = Vector((focus.x, focus.y, lower.z))
    placement.location = Vector(
        (target.x - first_center.x, target.y - first_center.y, target.z - first_lower.z)
    )
    bpy.context.view_layer.update()
    placed_lower, placed_upper = mesh_bounds("g1::")
    print(
        "g1_room_offset="
        f"{placement.location.x:.4f}:{placement.location.y:.4f}:{placement.location.z:.4f}"
    )
    print(
        "g1_bounds="
        f"{placed_lower.x:.3f}:{placed_lower.y:.3f}:{placed_lower.z:.3f}:"
        f"{placed_upper.x:.3f}:{placed_upper.y:.3f}:{placed_upper.z:.3f}"
    )
    print(f"camera_location={tuple(round(value, 3) for value in camera.location)}")
    print(f"camera_focus={tuple(round(value, 3) for value in focus)}")

    options.frames.mkdir(parents=True, exist_ok=True)
    indices = (
        [options.preview_frame]
        if options.preview_frame is not None
        else list(range(transforms.shape[0]))
    )
    for output_index, source_index in enumerate(indices):
        if not 0 <= source_index < transforms.shape[0]:
            raise RuntimeError("preview frame is outside the retarget artifact")
        apply_frame(roots, names, transforms, source_index)
        bpy.context.scene.frame_set(source_index + 1)
        bpy.context.scene.render.filepath = str(
            options.frames / f"roomplan-g1-{output_index:04d}.png"
        )
        bpy.ops.render.render(write_still=True)

    evidence = {
        "format": "numi.roomplan-g1-presentation.v1",
        "room_source": str(options.room),
        "room_source_sha256": sha256(options.room),
        "g1_urdf": str(options.g1_urdf),
        "g1_urdf_sha256": sha256(options.g1_urdf),
        "retarget_artifact": str(motion_path),
        "retarget_artifact_sha256": sha256(motion_path),
        "motion_frame_count": int(transforms.shape[0]),
        "rendered_frame_count": len(indices),
        "transform_policy": "exact supplied native solver link transforms; fixed rigid RoomPlan placement translation only",
        "dynamics_synthesized": False,
        "room_floor_z": float(lower.z),
        "g1_room_offset": [float(value) for value in placement.location],
        "render_engine": "Blender EEVEE",
        "camera_focus": [float(value) for value in focus],
    }
    (options.frames.parent / "render-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
