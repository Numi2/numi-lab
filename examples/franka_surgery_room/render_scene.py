#!/usr/bin/env python3
"""Render an authored Numi Lab surgical workcell around an accepted Franka pose."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "franka_explore"))
from render_gif import (  # noqa: E402
    add_box,
    apply_pose,
    import_franka,
    load_poses,
    material,
    point_camera,
)


NEEDLE = Path("SurgicalClosure/Needle/glb/dranmar_needle.glb")
TISSUE = Path("SurgicalClosure/ClosureRobot/glb/dranmar_closure_tissue_demo.glb")
NEEDLE_THREAD = Path(
    "SurgicalClosure/NeedleThread/glb/dranmar_needle_thread_coiled.glb"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--franka-description", type=Path, required=True)
    parser.add_argument("--converted-meshes", type=Path, required=True)
    parser.add_argument("--surgery-assets-root", type=Path, required=True)
    parser.add_argument("--pose-step", type=int, default=349)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--frame-count", type=int, default=72)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def add_cylinder(
    name: str,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    surface,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 48,
):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    result = bpy.context.object
    result.name = name
    result.data.materials.append(surface)
    bevel = result.modifiers.new("precision edge", "BEVEL")
    bevel.width = min(radius * 0.18, 0.008)
    bevel.segments = 3
    return result


def add_curve(
    name: str,
    points: list[tuple[float, float, float]],
    radius: float,
    surface,
):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 8
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for handle, point in zip(spline.bezier_points, points):
        handle.co = point
        handle.handle_left_type = "AUTO"
        handle.handle_right_type = "AUTO"
    result = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(result)
    result.data.materials.append(surface)
    return result


def import_surgical_asset(path: Path, name: str):
    if not path.is_file():
        raise RuntimeError(f"missing authored surgical asset: {path}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    top_level = [obj for obj in imported if obj.parent not in imported]
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    for obj in top_level:
        # These authored GLBs use the same Z-up-to-glTF transport convention
        # as the official Franka conversion. Cancel it before scene placement.
        world = Matrix.Rotation(-math.pi / 2.0, 4, "X") @ obj.matrix_world
        obj.parent = root
        obj.matrix_world = world
    return root


def assign_asset_material(root, surface) -> None:
    for obj in bpy.context.scene.objects:
        ancestor = obj.parent
        while ancestor is not None and ancestor != root:
            ancestor = ancestor.parent
        if ancestor == root and obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(surface)


def add_room(scene):
    graphite = material("OR graphite", (0.018, 0.028, 0.041, 1), 0.34, 0.12)
    wall = material("OR wall", (0.12, 0.17, 0.21, 1), 0.48, 0.02)
    wall_inset = material("OR wall inset", (0.025, 0.12, 0.16, 1), 0.31, 0.18)
    steel = material("Surgical steel", (0.42, 0.54, 0.62, 1), 0.19, 0.82)
    drape = material("Sterile cyan drape", (0.025, 0.38, 0.48, 1), 0.52, 0.0)
    pad = material("Silicone tissue support", (0.075, 0.16, 0.20, 1), 0.62, 0.0)
    dark = material("Instrument dark", (0.015, 0.022, 0.03, 1), 0.25, 0.45)
    screen = material("Monitor screen", (0.005, 0.19, 0.25, 1), 0.24, 0.1)
    cyan = material("Numi cyan", (0.015, 0.62, 0.78, 1), 0.24, 0.18)
    white = material("Light housing", (0.75, 0.83, 0.87, 1), 0.29, 0.18)

    add_box("OR floor", (0.2, 0.1, -0.235), (2.4, 2.4, 0.10), graphite, 0.05)
    add_box("rear wall", (0.2, 1.35, 0.82), (2.4, 0.06, 1.15), wall, 0.025)
    add_box("left wall", (-1.48, 0.15, 0.82), (0.06, 1.25, 1.15), wall, 0.025)
    for x in (-0.88, -0.28, 0.32, 0.92):
        add_box("wall panel", (x, 1.282, 0.83), (0.265, 0.008, 0.78), wall_inset, 0.012)

    # Franka is bolted to a dedicated plinth beside the operating surface.
    add_cylinder("robot pedestal", 0.235, 0.12, (0.0, 0.0, -0.135), steel)
    add_cylinder("robot pedestal inset", 0.19, 0.01, (0.0, 0.0, -0.071), cyan)

    # 1.10 x 0.72 m operating table, 0.39 m high in this compact workcell.
    add_box("operating table", (0.65, 0.12, 0.315), (0.55, 0.36, 0.045), steel, 0.035)
    add_box("sterile table drape", (0.65, 0.12, 0.365), (0.535, 0.345, 0.012), drape, 0.024)
    add_box("table column", (0.67, 0.12, 0.08), (0.13, 0.16, 0.235), dark, 0.025)
    add_box("table base", (0.67, 0.12, -0.125), (0.32, 0.25, 0.035), steel, 0.025)

    # Suturing field and separate stainless instrument tray.
    add_box("tissue backing", (0.62, 0.08, 0.395), (0.105, 0.105, 0.014), pad, 0.018)
    incision = material("Incision marker", (0.24, 0.012, 0.018, 1), 0.46, 0.0)
    add_curve(
        "unsutured incision",
        [
            (0.565, 0.08, 0.421),
            (0.59, 0.075, 0.422),
            (0.62, 0.083, 0.422),
            (0.65, 0.076, 0.422),
            (0.675, 0.08, 0.421),
        ],
        0.0032,
        incision,
    )
    add_box("instrument tray", (0.68, -0.34, 0.405), (0.24, 0.105, 0.012), steel, 0.018)
    add_box("tray liner", (0.68, -0.34, 0.420), (0.22, 0.086, 0.005), dark, 0.012)
    add_box("curved needle mat", (0.845, -0.34, 0.431), (0.062, 0.055, 0.006), dark, 0.010)

    # Two visible straight instruments keep scale legible beside the 14 mm
    # curved needle; these are presentation props, not collision geometry.
    add_cylinder(
        "needle driver shaft",
        0.006,
        0.31,
        (0.68, -0.36, 0.435),
        steel,
        (0.0, math.pi / 2.0, 0.0),
    )
    add_cylinder(
        "fine forceps shaft",
        0.004,
        0.26,
        (0.69, -0.31, 0.437),
        steel,
        (0.0, math.pi / 2.0, 0.0),
    )
    add_cylinder(
        "needle driver grip",
        0.011,
        0.075,
        (0.51, -0.36, 0.435),
        dark,
        (0.0, math.pi / 2.0, 0.0),
    )

    # Back-wall monitoring and storage establish a clean operating room rather
    # than an abstract studio set.
    add_box("monitor mount", (-0.72, 1.18, 0.74), (0.035, 0.06, 0.38), steel, 0.015)
    add_box("patient monitor", (-0.72, 1.08, 1.03), (0.30, 0.055, 0.22), dark, 0.025)
    add_box("patient monitor screen", (-0.72, 1.018, 1.03), (0.27, 0.006, 0.19), screen, 0.012)
    add_curve(
        "monitor trace",
        [
            (-0.92, 1.008, 1.04),
            (-0.82, 1.008, 1.04),
            (-0.77, 1.008, 1.12),
            (-0.71, 1.008, 0.96),
            (-0.64, 1.008, 1.04),
            (-0.52, 1.008, 1.04),
        ],
        0.004,
        cyan,
    )
    add_box("supply cabinet", (1.37, 1.16, 0.40), (0.34, 0.16, 0.50), wall_inset, 0.025)
    for z in (0.12, 0.40, 0.68):
        add_box("cabinet drawer", (1.37, 0.988, z), (0.30, 0.015, 0.115), steel, 0.012)

    # Two overhead surgical lamps, plus actual area emitters aimed at the field.
    for index, x in enumerate((0.32, 0.92)):
        add_cylinder(
            f"surgical lamp {index}",
            0.22,
            0.055,
            (x, 0.20, 1.48),
            white,
            (math.pi / 2.0, 0.0, 0.0),
            72,
        )
        light_data = bpy.data.lights.new(f"surgical emitter {index}", "AREA")
        light_data.energy = 265.0
        light_data.shape = "DISK"
        light_data.size = 0.42
        light_data.color = (0.72, 0.90, 1.0)
        light = bpy.data.objects.new(f"surgical emitter {index}", light_data)
        scene.collection.objects.link(light)
        light.location = (x, 0.12, 1.42)
        point_camera(light, (0.62, 0.08, 0.38))

    return steel, drape, pad, cyan


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
    scene.render.image_settings.compression = 18
    scene.render.fps = 24
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.55

    world = bpy.data.worlds.new("Numi surgery world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.008,
        0.018,
        0.026,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.065

    add_room(scene)

    fill_data = bpy.data.lights.new("room fill", "AREA")
    fill_data.energy = 235.0
    fill_data.shape = "RECTANGLE"
    fill_data.size = 2.2
    fill_data.size_y = 1.2
    fill_data.color = (0.35, 0.62, 0.80)
    fill = bpy.data.objects.new("room fill", fill_data)
    scene.collection.objects.link(fill)
    fill.location = (-0.8, -0.9, 1.6)
    point_camera(fill, (0.55, 0.08, 0.38))

    camera_data = bpy.data.cameras.new("Numi surgical camera")
    camera = bpy.data.objects.new("Numi surgical camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 56
    camera_data.sensor_width = 36
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = 2.55
    camera_data.dof.aperture_fstop = 8.0
    return camera


def place_surgical_assets(root: Path) -> None:
    tissue_surface = material(
        "Suturable tissue phantom",
        (0.54, 0.105, 0.13, 1),
        0.58,
        0.0,
    )
    needle_surface = material(
        "Polished suture needle",
        (0.48, 0.58, 0.64, 1),
        0.14,
        0.92,
    )
    thread_surface = material(
        "Blue polypropylene suture",
        (0.012, 0.085, 0.42, 1),
        0.32,
        0.08,
    )
    needle = import_surgical_asset(root / NEEDLE, "authored curved suture needle")
    assign_asset_material(needle, needle_surface)
    needle.location = (0.845, -0.34, 0.441)
    needle.scale = (2.4, 2.4, 2.4)
    needle.rotation_euler.z = math.radians(28.0)

    tissue = import_surgical_asset(root / TISSUE, "authored suturable tissue")
    assign_asset_material(tissue, tissue_surface)
    tissue.location = (0.62, 0.08, 0.412)
    tissue.scale = (1.75, 1.75, 1.75)

    needle_thread = import_surgical_asset(
        root / NEEDLE_THREAD,
        "authored coiled suture and needle",
    )
    assign_asset_material(needle_thread, thread_surface)
    needle_thread.location = (0.62, 0.08, 0.425)
    needle_thread.scale = (1.72, 1.72, 1.72)
    needle_thread.rotation_euler.z = math.radians(-18.0)


def set_camera(camera, phase: float) -> None:
    angle = math.radians(-49.0 + 3.5 * math.sin(phase))
    radius = 2.82 + 0.035 * math.cos(phase)
    camera.location = (
        0.34 + radius * math.cos(angle),
        0.02 + radius * math.sin(angle),
        1.48 + 0.035 * math.sin(phase),
    )
    point_camera(camera, (0.42, 0.12, 0.39))


def main() -> None:
    options = arguments()
    if options.output is None and options.frames is None:
        raise RuntimeError("provide --output or --frames")
    poses = load_poses(options.poses)
    if options.pose_step < 0 or options.pose_step >= len(poses):
        raise RuntimeError("--pose-step is outside the accepted trajectory")

    camera = configure_scene(options)
    robot = import_franka(options.franka_description, options.converted_meshes)
    apply_pose(robot, poses[options.pose_step])
    place_surgical_assets(options.surgery_assets_root)

    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        set_camera(camera, 0.0)
        bpy.context.scene.render.filepath = str(options.output)
        bpy.ops.render.render(write_still=True)
    if options.frames is not None:
        options.frames.mkdir(parents=True, exist_ok=True)
        for frame in range(options.frame_count):
            phase = 2.0 * math.pi * frame / max(1, options.frame_count - 1)
            set_camera(camera, phase)
            bpy.context.scene.render.filepath = str(
                options.frames / f"surgery-{frame:04d}.png"
            )
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
