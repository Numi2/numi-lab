#!/usr/bin/env python3
"""Render the saved Numi Metal pouch trajectory as a lit 3D product scene.

Run with Blender's bundled Python, for example:
  blender -b --python render_studio.py -- trajectory.json out.png --frame-index 239
  blender -b --python render_studio.py -- trajectory.json frames --animation
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


NX, NY = 25, 23
COIN_R = 0.03061 / 2
COIN_T = 0.00215


def sim_to_blender(p):
    # Simulation uses +Y up and Z through the bag. Blender uses +Z up.
    return (float(p[0]), float(p[2]), float(p[1]))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights):
        for item in list(datablocks):
            if item.users == 0:
                datablocks.remove(item)


def set_input(node, name, value):
    sock = node.inputs.get(name)
    if sock is not None:
        sock.default_value = value


def principled_material(name, color, metallic=0.0, roughness=0.3,
                         transmission=0.0, ior=1.46):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*color, 1.0))
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Transmission Weight", transmission)
    set_input(bsdf, "IOR", ior)
    return mat


def film_material():
    mat = principled_material("75 µm LDPE · clear with a soft frost", (0.84, 0.91, 0.93),
                              roughness=0.12, transmission=1.0, ior=1.47)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 190.0
    tex.inputs["Detail"].default_value = 3.0
    tex.inputs["Roughness"].default_value = 0.72
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.000025
    links.new(tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    broad = nodes.new("ShaderNodeTexNoise")
    broad.inputs["Scale"].default_value = 43.0
    broad.inputs["Detail"].default_value = 2.4
    broad.inputs["Roughness"].default_value = 0.68
    wrinkle = nodes.new("ShaderNodeBump")
    wrinkle.inputs["Strength"].default_value = 0.24
    wrinkle.inputs["Distance"].default_value = 0.00011
    links.new(broad.outputs["Fac"], wrinkle.inputs["Height"])
    links.new(wrinkle.outputs["Normal"], bsdf.inputs["Normal"])
    rough = nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.25
    rough.inputs["From Max"].default_value = 0.75
    rough.inputs["To Min"].default_value = 0.16
    rough.inputs["To Max"].default_value = 0.27
    links.new(tex.outputs["Fac"], rough.inputs["Value"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    # Real 75 µm film has almost no bulk opacity. Keep most rays transmitted,
    # and let the principled branch contribute the Fresnel glints and haze.
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.38
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return mat


def mesh_object(name, points, nx, ny, material, reverse=False):
    verts = [sim_to_blender(p) for p in points]
    faces = []
    for v in range(ny - 1):
        for u in range(nx - 1):
            a = v * nx + u
            for face in ((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)):
                faces.append(tuple(reversed(face)) if reverse else face)
    mesh = bpy.data.meshes.new(name + " mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def update_mesh(obj, points):
    for vert, p in zip(obj.data.vertices, points):
        vert.co = sim_to_blender(p)
    obj.data.update()


def interpolate(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def seam_points(front, side, alpha):
    # Narrow 4 mm welded strips, carried by the front film surface.
    cols = (0, 1) if side == "left" else (NX - 2, NX - 1)
    out = []
    for v in range(NY):
        p = interpolate(front[v * NX + cols[0]], front[v * NX + cols[1]], alpha)
        p[2] -= 0.00016
        out.append(p)
        q = list(front[v * NX + cols[0]])
        q[2] -= 0.00016
        out.append(q)
    return out


def bottom_weld_points(front):
    out = []
    for u in range(NX):
        for t in (0.20, 0.60):
            p = interpolate(front[u], front[NX + u], t)
            p[2] -= 0.00018
            out.append(p)
    return out


def arbitrary_grid_mesh(name, ncols, nrows, material):
    verts = [(0, 0, 0)] * (ncols * nrows)
    faces = []
    for v in range(nrows - 1):
        for u in range(ncols - 1):
            a = v * ncols + u
            faces.append((a, a + 1, a + ncols + 1, a + ncols))
    mesh = bpy.data.meshes.new(name + " mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def update_arbitrary_grid(obj, points):
    for vert, p in zip(obj.data.vertices, points):
        vert.co = sim_to_blender(p)
    obj.data.update()


def create_coin_material():
    mat = principled_material("Minted copper-nickel alloy", (0.52, 0.56, 0.58),
                              metallic=0.91, roughness=0.22)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 900.0
    noise.inputs["Detail"].default_value = 2.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.09
    bump.inputs["Distance"].default_value = 0.000012
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    rough = nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.2
    rough.inputs["From Max"].default_value = 0.8
    rough.inputs["To Min"].default_value = 0.20
    rough.inputs["To Max"].default_value = 0.29
    links.new(noise.outputs["Fac"], rough.inputs["Value"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def make_coin_prototype(material, relief_material, face_material, engraving_material):
    # Keep the maximum minted thickness at 2.15 mm; the bright raised rims
    # provide the last 75 µm on each face over the thinner coin core.
    rim_height = 0.000075
    bpy.ops.mesh.primitive_cylinder_add(vertices=360, radius=COIN_R,
                                        depth=COIN_T - 2.0 * rim_height,
                                        location=(0, 0, 0))
    body = bpy.context.object
    body.name = "Coin · reeded nickel-silver edge"
    mesh = body.data
    # Fine radial reeding around the rolled edge, at approximately 35 µm depth.
    for vert in mesh.vertices:
        if abs(abs(vert.co.z) - (COIN_T / 2 - rim_height)) < 1e-7:
            angle = math.atan2(vert.co.y, vert.co.x)
            radius = math.hypot(vert.co.x, vert.co.y)
            phase = math.cos(angle * 120.0)
            r = radius - 0.000035 * (0.5 - 0.5 * phase)
            vert.co.x = r * math.cos(angle)
            vert.co.y = r * math.sin(angle)
    body.data.materials.append(material)
    bevel = body.modifiers.new("Tiny rolled edge radius", "BEVEL")
    bevel.width = 0.00008
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    normal = body.modifiers.new("Weighted mint-face normals", "WEIGHTED_NORMAL")
    normal.keep_sharp = True

    # One small portrait relief, inset rim, and date on the visible face.
    ztop = COIN_T / 2 - rim_height
    bpy.ops.mesh.primitive_cylinder_add(vertices=192, radius=COIN_R * 0.79,
                                        depth=0.000035, location=(0, 0, ztop - 0.000012))
    field = bpy.context.object
    field.name = "Coin · inset satin face"
    field.data.materials.append(face_material)
    field.parent = body
    field.matrix_parent_inverse = body.matrix_world.inverted()
    face_bevel = field.modifiers.new("Face edge glint", "BEVEL")
    face_bevel.width = 0.000015
    face_bevel.segments = 2
    field.modifiers.new("Face normals", "WEIGHTED_NORMAL")

    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=12,
                                     location=(0, 0, ztop), major_radius=0.01442,
                                     minor_radius=rim_height)
    ring = bpy.context.object
    ring.name = "Coin · inset rim"
    ring.data.materials.append(relief_material)
    ring.parent = body
    ring.matrix_parent_inverse = body.matrix_world.inverted()
    for poly in ring.data.polygons:
        poly.use_smooth = True

    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=12,
                                     location=(0, 0, -ztop), major_radius=0.01442,
                                     minor_radius=rim_height)
    underside_ring = bpy.context.object
    underside_ring.name = "Coin · underside raised rim"
    underside_ring.data.materials.append(relief_material)
    underside_ring.parent = body
    underside_ring.matrix_parent_inverse = body.matrix_world.inverted()
    for poly in underside_ring.data.polygons:
        poly.use_smooth = True

    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=10,
                                     location=(0, 0, ztop + 0.000025), major_radius=0.01165,
                                     minor_radius=0.000035)
    inner_ring = bpy.context.object
    inner_ring.name = "Coin · fine engraved border"
    inner_ring.data.materials.append(engraving_material)
    inner_ring.parent = body
    inner_ring.matrix_parent_inverse = body.matrix_world.inverted()
    for poly in inner_ring.data.polygons:
        poly.use_smooth = True

    # A raised cameo and shoulders create a readable, coherent minted relief.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                                         location=(-0.00045, 0.00015, ztop + 0.000020))
    head = bpy.context.object
    head.name = "Coin · portrait cameo"
    head.scale = (0.00265, 0.0030, 0.000055)
    head.data.materials.append(relief_material)
    head.parent = body
    head.matrix_parent_inverse = body.matrix_world.inverted()
    for poly in head.data.polygons:
        poly.use_smooth = True
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=12,
                                         location=(0, -0.0044, ztop + 0.00003))
    shoulders = bpy.context.object
    shoulders.name = "Coin · portrait shoulders"
    shoulders.scale = (0.0055, 0.0017, 0.000045)
    shoulders.data.materials.append(relief_material)
    shoulders.parent = body
    shoulders.matrix_parent_inverse = body.matrix_world.inverted()
    for poly in shoulders.data.polygons:
        poly.use_smooth = True

    bpy.ops.object.text_add(location=(0, 0.0078, ztop + 0.000035))
    label = bpy.context.object
    label.name = "Coin · LIBERTY"
    label.data.body = "LIBERTY"
    label.data.align_x = "CENTER"
    label.data.size = 0.00175
    label.data.extrude = 0.000010
    label.data.bevel_depth = 0.000004
    label.data.materials.append(engraving_material)
    label.parent = body
    label.matrix_parent_inverse = body.matrix_world.inverted()

    bpy.ops.object.text_add(location=(0, -0.0090, ztop + 0.000035))
    date = bpy.context.object
    date.name = "Coin · 1964"
    date.data.body = "1964"
    date.data.align_x = "CENTER"
    date.data.size = 0.0016
    date.data.extrude = 0.000010
    date.data.bevel_depth = 0.000004
    date.data.materials.append(engraving_material)
    date.parent = body
    date.matrix_parent_inverse = body.matrix_world.inverted()

    return body


def add_area(name, location, target, power, size, size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    if size_y is not None:
        data.shape = "RECTANGLE"
        data.size = size
        data.size_y = size_y
    else:
        data.shape = "DISK"
        data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_stage():
    floor_mat = principled_material("Warm ivory studio table", (0.38, 0.30, 0.21), roughness=0.82)
    wall_mat = principled_material("Cool pale studio backdrop", (0.34, 0.41, 0.45), roughness=0.9)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0.1079))
    floor = bpy.context.object
    floor.name = "Ground plane"
    floor.data.materials.append(floor_mat)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0.36, 1.1))
    wall = bpy.context.object
    wall.name = "Pale backdrop"
    wall.rotation_euler[0] = math.radians(90)
    wall.data.materials.append(wall_mat)

    world = bpy.data.worlds.new("Soft neutral studio") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.28, 0.33, 0.36, 1)
    bg.inputs["Strength"].default_value = 0.40

    focus = (0, 0, 0.225)
    lights = [
        add_area("Key · broad silk", (-0.28, -0.40, 0.52), focus, 7, 0.28, 0.42),
        add_area("Fill · soft frontal", (0.34, -0.34, 0.34), focus, 2.5, 0.30, 0.40),
        add_area("Rim · long strip", (0.12, 0.20, 0.47), focus, 9, 0.045, 0.34),
        add_area("Edge · long strip", (-0.22, -0.15, 0.41), focus, 2.5, 0.025, 0.32),
    ]
    focus_vector = Vector(focus)
    light_offsets = [light.location - focus_vector for light in lights]

    cam_data = bpy.data.cameras.new("Front product camera")
    cam = bpy.data.objects.new("Front product camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 0.36
    set_camera_view(cam, 9.0, 38.0, focus_vector)
    bpy.context.scene.camera = cam
    environment = {"floor": floor, "wall": wall, "lights": lights,
                   "light_offsets": light_offsets}
    return cam, environment


def set_camera_view(cam, azimuth_deg, elevation_deg, focus=None):
    focus = Vector(focus) if focus is not None else Vector((0, 0, 0.225))
    distance = 1.13
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horizontal = distance * math.cos(el)
    cam.location = focus + Vector((horizontal * math.sin(az),
                                   -horizontal * math.cos(az),
                                   distance * math.sin(el)))
    cam.rotation_euler = (focus - cam.location).to_track_quat("-Z", "Y").to_euler()


def update_tracked_view(frame, cam, environment, focus_state=None, scale_state=None):
    points = frame["front"] + frame["back"] + frame["coins"]
    lower = [min(float(p[i]) for p in points) for i in range(3)]
    upper = [max(float(p[i]) for p in points) for i in range(3)]
    ground = 0.1079
    vertical_low = min(lower[1], ground)
    vertical_high = max(upper[1], ground)
    center = Vector(((lower[0] + upper[0]) * 0.5,
                     (lower[2] + upper[2]) * 0.5,
                     (lower[1] + upper[1]) * 0.5))
    x_span = upper[0] - lower[0]
    z_span = upper[2] - lower[2]
    vertical_span = vertical_high - vertical_low
    azimuth = math.radians(9.0)
    elevation = math.radians(38.0)
    horizontal_extent = x_span * math.cos(azimuth) + z_span * math.sin(azimuth)
    depth_extent = x_span * math.sin(azimuth) + z_span * math.cos(azimuth)
    projected_vertical = vertical_span * math.cos(elevation) + depth_extent * math.sin(elevation)
    aspect = bpy.context.scene.render.resolution_y / bpy.context.scene.render.resolution_x
    target_scale = max(0.42, 1.15 * horizontal_extent,
                       1.15 * projected_vertical / aspect)
    if focus_state is None:
        focus_state = center
        scale_state = target_scale
    else:
        focus_state = focus_state.lerp(center, 0.62)
        zoom_rate = 0.88 if target_scale > scale_state else 0.20
        scale_state += (target_scale - scale_state) * zoom_rate
    set_camera_view(cam, 9.0, 38.0, focus_state)
    cam.data.ortho_scale = scale_state

    wall = environment["wall"]
    wall.location = (focus_state.x, focus_state.y + 0.36, 1.1)
    for light, offset in zip(environment["lights"], environment["light_offsets"]):
        light.location = focus_state + offset
        light.rotation_euler = (focus_state - light.location).to_track_quat("-Z", "Y").to_euler()
    return focus_state, scale_state


def setup_render(width=1200, height=1400, samples=48, engine="eevee"):
    scene = bpy.context.scene
    if engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = samples
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
        try:
            cycles_preferences.compute_device_type = "METAL"
            cycles_preferences.get_devices()
            metal_devices = [device for device in cycles_preferences.devices
                             if device.type == "METAL"]
            for device in cycles_preferences.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU" if metal_devices else "CPU"
        except (AttributeError, TypeError, RuntimeError):
            # CPU rendering remains available when Blender was built without Metal.
            scene.cycles.device = "CPU"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.camera.data.lens = 58
    scene.render.filepath = "/tmp/numi_plastic_bag_render.png"


def create_scene(frame, width=1200, height=1400, samples=48, engine="eevee"):
    clear_scene()
    film = film_material()
    seam_mat = principled_material("Heat-pressed flat weld", (0.74, 0.83, 0.84),
                                   roughness=0.27, transmission=0.10)
    coin_mat = create_coin_material()
    relief_mat = principled_material("Raised coin relief", (0.74, 0.76, 0.77),
                                     metallic=0.95, roughness=0.22)
    face_mat = principled_material("Satin coin field", (0.36, 0.40, 0.42),
                                   metallic=0.93, roughness=0.30)
    engraving_mat = principled_material("Engraved coin recess", (0.12, 0.16, 0.18),
                                        metallic=0.76, roughness=0.30)

    front = mesh_object("Front LDPE film", frame["front"], NX, NY, film)
    back = mesh_object("Back LDPE film", frame["back"], NX, NY, film, reverse=True)
    welds = []
    for side in ("left", "right"):
        obj = arbitrary_grid_mesh(side.title() + " 4 mm flat heat weld", 2, NY, seam_mat)
        update_arbitrary_grid(obj, seam_points(frame["front"], side, 0.48))
        welds.append((obj, side))
    bottom_weld = arbitrary_grid_mesh("Bottom 4 mm heat weld", NX, 2, seam_mat)
    update_arbitrary_grid(bottom_weld, bottom_weld_points(frame["front"]))

    prototype = make_coin_prototype(coin_mat, relief_mat, face_mat, engraving_mat)
    coins = [prototype] if frame["coins"] else []
    if not frame["coins"]:
        for child in list(prototype.children):bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.objects.remove(prototype, do_unlink=True)
    for i in range(1, len(frame["coins"])):
        ob = bpy.data.objects.new(f"Coin {i + 1:02d} · 11.34 g", prototype.data)
        bpy.context.collection.objects.link(ob)
        for mod in prototype.modifiers:
            new_mod = ob.modifiers.new(mod.name, mod.type)
            if mod.type == "BEVEL":
                new_mod.width = mod.width
                new_mod.segments = mod.segments
                new_mod.limit_method = mod.limit_method
            elif mod.type == "WEIGHTED_NORMAL":
                new_mod.keep_sharp = mod.keep_sharp
        # Share the detailed relief children while retaining a separate transform.
        details = [o for o in bpy.context.scene.objects if o.parent == prototype]
        for detail in details:
            child = detail.copy()
            child.data = detail.data
            bpy.context.collection.objects.link(child)
            child.parent = ob
            child.matrix_parent_inverse.identity()
            child.location = detail.location.copy()
            child.rotation_euler = detail.rotation_euler.copy()
            child.scale = detail.scale.copy()
        coins.append(ob)

    update_frame(frame, front, back, welds, bottom_weld, coins)
    camera, environment = setup_stage()
    setup_render(width, height, samples, engine)
    return front, back, welds, bottom_weld, coins, camera, environment


def update_frame(frame, front, back, welds, bottom_weld, coins):
    update_mesh(front, frame["front"])
    update_mesh(back, frame["back"])
    for obj, side in welds:
        update_arbitrary_grid(obj, seam_points(frame["front"], side, 0.48))
    update_arbitrary_grid(bottom_weld, bottom_weld_points(frame["front"]))
    orientations = frame.get("coin_orientations", [[0, 0, 0, 1]] * len(coins))
    for obj, p, q in zip(coins, frame["coins"], orientations):
        obj.location = sim_to_blender(p)
        obj.rotation_mode = "QUATERNION"
        # Simulation basis maps (+X,+Y,+Z) to Blender (+X,+Z,+Y), a reflection.
        # Quaternion vector parts are axial vectors, so apply det(P) * P.
        obj.rotation_quaternion = Quaternion((float(q[3]), -float(q[0]),
                                               -float(q[2]), -float(q[1])))


def main():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-index", type=int, default=239)
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--views", action="store_true",
                        help="render front, elevated, three-quarter, side, and high views")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--animation-fps", type=int, default=60,
                        help="output frame rate; trajectory samples are selected from its recorded FPS")
    parser.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    args = parser.parse_args(raw)
    trajectory = json.loads(args.trajectory.read_text())
    frames = trajectory["frames"]
    frame_index = max(0, min(args.frame_index, len(frames) - 1))
    scene_objects = create_scene(frames[frame_index], args.width, args.height,
                                 args.samples, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.views:
        camera = scene_objects[5]
        views = (("front", 0, 16), ("front_raised", 0, 38),
                 ("three_quarter_left", -34, 28), ("three_quarter_right", 34, 28),
                 ("side", 68, 22), ("high", 0, 63))
        for name, azimuth, elevation in views:
            set_camera_view(camera, azimuth, elevation)
            scene = bpy.context.scene
            scene.render.filepath = str(args.output / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"rendered angle {name} (azimuth {azimuth}°, elevation {elevation}°)", flush=True)
        return
    if not args.animation:
        bpy.context.scene.render.filepath = str(args.output)
        bpy.ops.render.render(write_still=True)
        print(f"rendered frame {frame_index} ({frames[frame_index]['time_s']:.3f} s) to {args.output}")
        return

    scene = bpy.context.scene
    scene.render.image_settings.color_mode = "RGB"
    source_fps = float(trajectory.get("fps", 240))
    if args.animation_fps < 1 or args.animation_fps > source_fps:
        raise ValueError(f"--animation-fps must be between 1 and trajectory FPS ({source_fps:g})")
    sample_step = max(1, round(source_fps / args.animation_fps))
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = list(range(0, len(frames), sample_step))
    if sample_indices[-1] != len(frames) - 1:
        sample_indices.append(len(frames) - 1)
    camera, environment = scene_objects[5], scene_objects[6]
    focus_state = scale_state = None
    for ordinal, index in enumerate(sample_indices):
        update_frame(frames[index], *scene_objects[:5])
        focus_state, scale_state = update_tracked_view(
            frames[index], camera, environment, focus_state, scale_state)
        scene.render.filepath = str(output_dir / f"frame_{ordinal:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"rendered {ordinal + 1}/{len(sample_indices)} · {frames[index]['time_s']:.3f} s", flush=True)


if __name__ == "__main__":
    main()
