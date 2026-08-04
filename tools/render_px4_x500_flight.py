#!/usr/bin/env python3
"""Render a source-mesh PX4 X500 flight trace produced by numi drone flight."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as etree
from pathlib import Path

import bpy
from mathutils import Vector


def parse_trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as source:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(source)]
    if len(rows) < 2:
        raise RuntimeError("flight trace needs at least two accepted states")
    return rows


def point_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def import_mesh(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    if path.suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        raise RuntimeError(f"unsupported source mesh: {path}")
    objects = [object_ for object_ in bpy.context.scene.objects if object_ not in before]
    if not objects:
        raise RuntimeError(f"Blender imported no geometry from {path}")
    return objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    if "--" not in sys.argv:
        raise RuntimeError("Blender arguments must follow --")
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    rows = parse_trace(options.trace)
    mesh_root = options.source_root / "models/x500_base/meshes"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "rim.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = options.width
    scene.render.resolution_y = options.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = len(rows)
    options.frames.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(options.frames / "x500-")

    material = bpy.data.materials.new("PX4 source mesh cobalt")
    material.diffuse_color = (0.055, 0.42, 1.0, 1.0)
    frame = bpy.data.objects.new("PX4 X500 source-frame root", None)
    bpy.context.collection.objects.link(frame)
    source_sdf = etree.parse(options.source_root / "models/x500_base/model.sdf").getroot()
    for collision in source_sdf.findall("./model/link[@name='base_link']/collision"):
        size = collision.findtext("./geometry/box/size")
        pose = collision.findtext("pose")
        if size is None or pose is None:
            continue
        dimensions = tuple(float(value) for value in size.split())
        x, y, z, roll, pitch, yaw = (float(value) for value in pose.split())
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z), rotation=(roll, pitch, yaw))
        source_collision = bpy.context.object
        source_collision.name = collision.attrib.get("name", "x500 source collision")
        source_collision.dimensions = dimensions
        source_collision.data.materials.append(material)
        source_collision.parent = frame

    prop_positions = ((0.174, -0.174, 0.060), (-0.174, 0.174, 0.060), (0.174, 0.174, 0.060), (-0.174, -0.174, 0.060))
    for index, position in enumerate(prop_positions):
        prop = import_mesh(mesh_root / ("1345_prop_ccw.stl" if index < 2 else "1345_prop_cw.stl"))
        for object_ in prop:
            object_.parent = frame
            object_.location = position
            object_.scale = (0.846153846, 0.846153846, 0.846153846)
            if object_.type == "MESH":
                object_.data.materials.clear()
                object_.data.materials.append(material)

    for frame_number, row in enumerate(rows, 1):
        frame.location = (row["x_m"], row["y_m"], row["z_m"])
        frame.rotation_mode = "QUATERNION"
        frame.rotation_quaternion = (row["qw"], row["qx"], row["qy"], row["qz"])
        frame.keyframe_insert(data_path="location", frame=frame_number)
        frame.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "ground reference plane"
    ground_material = bpy.data.materials.new("matte ground")
    ground_material.diffuse_color = (0.16, 0.19, 0.25, 1.0)
    ground.data.materials.append(ground_material)
    # Follow the accepted body state: the vehicle stays legible while the
    # ground plane supplies the ascent reference.  The trace itself remains
    # untouched; this is presentation only.
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.parent = frame
    camera.location = (2.0, -2.6, 1.45)
    camera.data.lens = 58
    point_at(camera, Vector((0.0, 0.0, 0.0)))
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(3.5, -4.0, 7.0))
    bpy.context.object.data.energy = 1500
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5.0
    options.frames.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
