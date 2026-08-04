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


def import_collada_geometry(path: Path, material: bpy.types.Material) -> list[bpy.types.Object]:
    """Import the position/triangle geometry from the pinned source DAE.

    Blender 5 no longer bundles the Collada operator.  Keeping this small
    reader here lets the renderer use the upstream X500 presentation mesh
    rather than substituting a collision proxy.
    """
    root = etree.parse(path).getroot()
    namespace = {"c": root.tag.split("}")[0].removeprefix("{")}
    objects: list[bpy.types.Object] = []
    for geometry in root.findall(".//c:library_geometries/c:geometry", namespace):
        mesh = geometry.find("c:mesh", namespace)
        if mesh is None:
            continue
        sources = {source.attrib["id"]: source for source in mesh.findall("c:source", namespace)}
        vertices = mesh.find("c:vertices", namespace)
        if vertices is None:
            continue
        vertex_sources = {
            input_.attrib["semantic"]: input_.attrib["source"].removeprefix("#")
            for input_ in vertices.findall("c:input", namespace)
        }
        position_source = sources.get(vertex_sources.get("POSITION", ""))
        if position_source is None:
            continue
        values = [float(value) for value in position_source.findtext("c:float_array", "", namespace).split()]
        stride = int(position_source.find("c:technique_common/c:accessor", namespace).attrib.get("stride", "3"))
        positions = [tuple(values[index : index + 3]) for index in range(0, len(values), stride)]
        faces: list[tuple[int, int, int]] = []
        for triangles in mesh.findall("c:triangles", namespace):
            inputs = triangles.findall("c:input", namespace)
            vertex_input = next((input_ for input_ in inputs if input_.attrib["semantic"] == "VERTEX"), None)
            if vertex_input is None:
                continue
            input_stride = 1 + max(int(input_.attrib.get("offset", "0")) for input_ in inputs)
            vertex_offset = int(vertex_input.attrib.get("offset", "0"))
            indices = [int(value) for value in triangles.findtext("c:p", "", namespace).split()]
            for index in range(0, len(indices), 3 * input_stride):
                faces.append(tuple(indices[index + corner * input_stride + vertex_offset] for corner in range(3)))
        if not faces:
            continue
        data = bpy.data.meshes.new(f"{path.stem}:{geometry.attrib.get('name', geometry.attrib['id'])}")
        data.from_pydata(positions, [], faces)
        data.materials.append(material)
        object_ = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(object_)
        objects.append(object_)
    if not objects:
        raise RuntimeError(f"no triangle geometry found in {path}")
    return objects


def pose_values(text: str | None) -> tuple[float, float, float, float, float, float]:
    return tuple(float(value) for value in (text or "0 0 0 0 0 0").split())  # type: ignore[return-value]


def pbr_material(name: str, color: tuple[float, float, float, float], metallic: float, roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--last-frame", type=int)
    parser.add_argument("--sample-stride", type=int, default=1, help="render every Nth accepted trace state")
    parser.add_argument("--world-camera", action="store_true", help="hold a world-fixed camera to show accepted translation")
    parser.add_argument("--tracking-camera", action="store_true", help="track accepted translation while keeping camera attitude world-aligned")
    if "--" not in sys.argv:
        raise RuntimeError("Blender arguments must follow --")
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    rows = parse_trace(options.trace)
    if options.sample_stride < 1:
        raise RuntimeError("sample stride must be positive")
    if options.world_camera and options.tracking_camera:
        raise RuntimeError("choose either --world-camera or --tracking-camera")
    rows = rows[:: options.sample_stride]
    mesh_root = options.source_root / "models/x500_base/meshes"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.world.color = (0.012, 0.018, 0.035)
    scene.render.resolution_x = options.width
    scene.render.resolution_y = options.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 24
    if options.first_frame < 1 or options.first_frame > len(rows):
        raise RuntimeError("first frame is outside the trace")
    scene.frame_start = options.first_frame
    scene.frame_end = min(options.last_frame or len(rows), len(rows))
    if scene.frame_end < scene.frame_start:
        raise RuntimeError("last frame precedes first frame")
    options.frames.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(options.frames / "x500-")

    carbon_material = pbr_material("PX4 carbon-fibre frame", (0.012, 0.018, 0.025, 1.0), 0.32, 0.29)
    motor_material = pbr_material("PX4 machined motor", (0.11, 0.14, 0.18, 1.0), 0.92, 0.18)
    prop_material = pbr_material("PX4 composite propeller", (0.018, 0.025, 0.035, 1.0), 0.45, 0.22)
    frame = bpy.data.objects.new("PX4 X500 source-frame root", None)
    bpy.context.collection.objects.link(frame)
    rotor_spinners: dict[int, tuple[bpy.types.Object, tuple[float, float, float]]] = {}
    rotor_directions: dict[int, float] = {}
    source_vehicle_sdf = etree.parse(options.source_root / "models/x500/model.sdf").getroot()
    for plugin in source_vehicle_sdf.findall(".//plugin"):
        link_name = plugin.findtext("linkName", "")
        direction = plugin.findtext("turningDirection", "")
        if link_name.startswith("rotor_") and direction in {"ccw", "cw"}:
            rotor_directions[int(link_name.removeprefix("rotor_"))] = 1.0 if direction == "ccw" else -1.0
    source_sdf = etree.parse(options.source_root / "models/x500_base/model.sdf").getroot()
    for link in source_sdf.findall("./model/link"):
        link_name = link.attrib.get("name", "")
        rotor_index = int(link_name.removeprefix("rotor_")) if link_name.startswith("rotor_") else None
        link_pose = pose_values(link.findtext("pose"))
        for visual in link.findall("visual"):
            mesh_uri = visual.findtext("./geometry/mesh/uri")
            if mesh_uri is None:
                continue
            visual_pose = pose_values(visual.findtext("pose"))
            mesh_path = mesh_root / Path(mesh_uri).name
            source_material = motor_material if mesh_path.name.startswith("5010") else prop_material if mesh_path.suffix == ".stl" else carbon_material
            objects = import_collada_geometry(mesh_path, source_material) if mesh_path.suffix == ".dae" else import_mesh(mesh_path)
            # The source DAE motor assets retain their authoring-centimetre
            # geometry with a 0.01 scene-node transform.  Preserve that
            # transform here rather than scaling the physical airframe.
            authoring_scale = 0.01 if mesh_path.name in {"5010Base.dae", "5010Bell.dae"} else 1.0
            source_location = tuple(link_pose[index] + visual_pose[index] for index in range(3))
            source_rotation = tuple(link_pose[index + 3] + visual_pose[index + 3] for index in range(3))
            propeller = rotor_index is not None and mesh_path.suffix == ".stl"
            spinner = None
            if propeller:
                spinner = bpy.data.objects.new(f"PX4 rotor {rotor_index} spin root", None)
                bpy.context.collection.objects.link(spinner)
                spinner.parent = frame
                # The SDF revolute joint is at the rotor-link origin. Its
                # visual offset belongs below this root, otherwise the mesh
                # would orbit around its own offset instead of the motor shaft.
                spinner.location = link_pose[:3]
                spinner.rotation_euler = link_pose[3:]
                rotor_spinners[rotor_index] = (spinner, link_pose[3:])
            for object_ in objects:
                object_.parent = spinner if spinner is not None else frame
                object_.location = visual_pose[:3] if spinner is not None else source_location
                object_.rotation_euler = visual_pose[3:] if spinner is not None else source_rotation
                scale = visual.findtext("./geometry/mesh/scale")
                source_scale = tuple(float(value) for value in (scale or "1 1 1").split())
                object_.scale = tuple(authoring_scale * value for value in source_scale)
                if object_.type == "MESH":
                    object_.data.materials.clear()
                    object_.data.materials.append(source_material)

    for frame_number, row in enumerate(rows, 1):
        frame.location = (row["x_m"], row["y_m"], row["z_m"])
        frame.rotation_mode = "QUATERNION"
        frame.rotation_quaternion = (row["qw"], row["qx"], row["qy"], row["qz"])
        frame.keyframe_insert(data_path="location", frame=frame_number)
        frame.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

    if all(f"rotor{index}_rad_s" in rows[0] for index in range(4)):
        angles = [0.0, 0.0, 0.0, 0.0]
        previous_time = rows[0]["time_s"]
        for frame_number, row in enumerate(rows, 1):
            dt = 0.0 if frame_number == 1 else row["time_s"] - previous_time
            for index in range(4):
                spinner_entry = rotor_spinners.get(index)
                if spinner_entry is None:
                    continue
                spinner, base_rotation = spinner_entry
                direction = rotor_directions.get(index)
                if direction is None:
                    raise RuntimeError(f"missing PX4 turning direction for rotor {index}")
                angles[index] += direction * row[f"rotor{index}_rad_s"] * dt
                spinner.rotation_euler = (base_rotation[0], base_rotation[1], base_rotation[2] + angles[index])
                spinner.keyframe_insert(data_path="rotation_euler", frame=frame_number)
            previous_time = row["time_s"]

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "ground reference plane"
    ground_material = pbr_material("matte ground", (0.018, 0.027, 0.045, 1.0), 0.15, 0.36)
    ground.data.materials.append(ground_material)
    # Follow mode keeps detailed vehicle inspection legible. World mode keeps
    # the camera fixed, so solved translation and attitude remain visible.
    # Tracking mode follows only accepted translation: it never inherits the
    # airframe rotation, so a roll remains visible without synthesising pose.
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    if options.world_camera:
        centre = Vector((
            sum(row["x_m"] for row in rows) / len(rows),
            sum(row["y_m"] for row in rows) / len(rows),
            sum(row["z_m"] for row in rows) / len(rows),
        ))
        camera.location = centre + Vector((1.5, -2.0, 1.15))
        point_at(camera, centre)
    elif options.tracking_camera:
        camera_offset = Vector((1.45, -2.15, 0.95))
        for frame_number, row in enumerate(rows, 1):
            target = Vector((row["x_m"], row["y_m"], row["z_m"]))
            camera.location = target + camera_offset
            point_at(camera, target)
            camera.keyframe_insert(data_path="location", frame=frame_number)
            camera.keyframe_insert(data_path="rotation_euler", frame=frame_number)
    else:
        camera.parent = frame
        camera.location = (1.15, -1.45, 0.78)
        point_at(camera, Vector((0.0, 0.0, 0.0)))
    camera.data.lens = 65 if options.world_camera else 58
    scene.camera = camera
    for location, energy, size, color in (
        ((1.8, -2.4, 3.0), 950.0, 2.0, (0.62, 0.78, 1.0)),
        ((-2.0, -0.8, 2.0), 700.0, 1.6, (0.16, 0.42, 1.0)),
        ((0.2, 1.8, 1.6), 600.0, 1.2, (0.35, 0.55, 1.0)),
    ):
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.object
        if not options.world_camera and not options.tracking_camera:
            light.parent = frame
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        if options.tracking_camera:
            light_offset = Vector(location)
            for frame_number, row in enumerate(rows, 1):
                target = Vector((row["x_m"], row["y_m"], row["z_m"]))
                light.location = target + light_offset
                point_at(light, target)
                light.keyframe_insert(data_path="location", frame=frame_number)
                light.keyframe_insert(data_path="rotation_euler", frame=frame_number)
        else:
            light.location = location
            point_at(light, Vector((0.0, 0.0, 0.0)))
    options.frames.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
