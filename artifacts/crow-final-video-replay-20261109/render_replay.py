import argparse
import os
import sys
import subprocess
import tempfile
import bpy
import numpy as np
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--layout', choices=['i', 'j'], required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--preview', type=int)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
base = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(base, 'inputs', f'reserved-final-{args.layout}-visual.npz'))
poses = data['body_poses']

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100
scene.render.fps = 25
scene.frame_start = 1
scene.frame_end = 750
shading = scene.display.shading
shading.light = 'STUDIO'
shading.studio_light = 'paint.sl'
shading.color_type = 'MATERIAL'
shading.background_type = 'WORLD'
shading.show_shadows = True
shading.show_cavity = True
shading.cavity_type = 'BOTH'
shading.curvature_ridge_factor = 1.5
shading.curvature_valley_factor = 1.3
scene.world.color = (0.72, 0.75, 0.75)
scene.render.image_settings.color_mode = 'RGB'

def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    return m

crow_mat = material('crow charcoal', (0.065, 0.085, 0.11))
gate_mat = material('gates', (0.72, 0.30, 0.13))
obstacle_mat = material('obstacles', (0.52, 0.32, 0.20))
perch_mat = material('landing perch', (0.18, 0.35, 0.28))
ground_mat = material('ground', (0.68, 0.72, 0.68))

def obj_mesh(name, path, mat):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':
                faces.append(tuple(int(p.split('/')[0]) - 1 for p in parts[1:]))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    ob.rotation_mode = 'QUATERNION'
    for poly in mesh.polygons:
        poly.use_smooth = name.startswith('crow')
    return ob

links = [(0, 'body'), (5, 'right-wing'), (6, 'left-wing'), (7, 'tail'),
         (8, 'thigh'), (9, 'shank'), (10, 'foot'),
         (11, 'thigh'), (12, 'shank'), (13, 'foot')]
objects = []
for idx, mesh_name in links:
    objects.append((idx, obj_mesh('crow-' + str(idx), os.path.join(base, 'inputs', 'surface', mesh_name + '.obj'), crow_mat)))
for idx, name in enumerate(('gate-left', 'gate-right', 'slalom-a', 'slalom-b', 'perch'), start=18):
    mat = perch_mat if name == 'perch' else gate_mat if 'gate' in name else obstacle_mat
    objects.append((idx, obj_mesh(name, os.path.join(base, 'inputs', 'course', name + '.obj'), mat)))

bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
bpy.context.object.name = 'ground'
bpy.context.object.data.materials.append(ground_mat)

camera_data = bpy.data.cameras.new('Camera')
camera = bpy.data.objects.new('Camera', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.type = 'ORTHO'
camera_data.ortho_scale = 3.2

# Smooth only the camera; the crow and collision geometry use the recorded poses.
follow = poses[:, 0, :3].copy()
for i in range(1, len(follow)):
    follow[i] = .94 * follow[i-1] + .06 * follow[i]

def update(frame):
    i = min(2999, round((frame - 1) * 2999 / 749))
    for idx, ob in objects:
        p = poses[i, idx]
        ob.location = p[:3]
        ob.rotation_quaternion = (p[6], p[3], p[4], p[5])
    target = Vector(follow[i]) + Vector((.32, 0, .08))
    camera.location = target + Vector((-1.6, -2.7, 1.55))
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()

if args.preview is not None:
    update(args.preview)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
else:
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.compression = 10
    ffmpeg = subprocess.Popen(['/opt/homebrew/bin/ffmpeg', '-loglevel', 'error', '-y',
                               '-f', 'image2pipe', '-framerate', '25', '-vcodec', 'png', '-i', '-',
                               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
                               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out], stdin=subprocess.PIPE)
    with tempfile.TemporaryDirectory(prefix='crow-render-') as tmp:
        scene.render.filepath = os.path.join(tmp, 'frame.png')
        for frame in range(1, 751):
            update(frame)
            bpy.ops.render.render(write_still=True)
            with open(scene.render.filepath, 'rb') as image:
                ffmpeg.stdin.write(image.read())
        ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError('ffmpeg failed')
