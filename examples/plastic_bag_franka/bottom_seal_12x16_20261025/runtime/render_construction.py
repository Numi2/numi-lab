"""Render exact native construction vertices; no rendering-engine dynamics."""
import sys,json,math
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_studio as studio
args=sys.argv[sys.argv.index('--')+1:]
src,out=map(Path,args[:2]);out.mkdir(parents=True,exist_ok=True)
data=json.loads(src.read_text());frame=data['frames'][0]
studio.NX=data['grid']['nu'];studio.NY=data['grid']['nv']
front,back,side_welds,bottom,coins,camera,env=studio.create_scene(frame,1100,1450,48,'eevee')
film=front.data.materials[0]
bpy.data.objects.remove(front,do_unlink=True);bpy.data.objects.remove(back,do_unlink=True)
mesh=bpy.data.meshes.new('Exact indexed native continuous-tube topology')
mesh.from_pydata([studio.sim_to_blender(p) for p in data['positions']],[],[
    tuple(reversed(t[:3])) if t[3]==1 else tuple(t[:3]) for t in data['triangles']])
mesh.update();obj=bpy.data.objects.new('Continuous LDPE tube · native indexed film',mesh)
bpy.context.collection.objects.link(obj);mesh.materials.append(film)
for poly in mesh.polygons:poly.use_smooth=True
for obj,side in side_welds:bpy.data.objects.remove(obj,do_unlink=True)
bottom.name='Bottom heat seal band · 5 mm visual width · ideal native line joint'
# Use the actual native bottom row; display a flat strip rather than a tube.
points=[];h=data['dimensions']['height_m'];dy=h/(studio.NY-1)
for y in (.0003,.0053):
    for u in range(studio.NX):
        p=studio.interpolate(frame['front'][u],frame['front'][studio.NX+u],y/dy)
        p[2]-=.00010;points.append(p)
studio.update_arbitrary_grid(bottom,points)
focus=Vector(studio.sim_to_blender([0,frame['front'][0][1]+h/2,0]))
env['wall'].location.y=1.2
scene=bpy.context.scene;scene.render.image_settings.color_mode='RGB'
for name,az,el,scale,target in [
    ('front',0,1,.52,focus),
    ('three_quarter',32,16,.54,focus),
    ('mouth',-30,50,.15,Vector(studio.sim_to_blender([-.105,frame['front'][0][1]+h-.025,0]))),
]:
    studio.set_camera_view(camera,az,el,target);camera.data.ortho_scale=scale
    scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    print('NATIVE_CONSTRUCTION_VIEW',name,flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'bottom_seal_12x16.blend'))
