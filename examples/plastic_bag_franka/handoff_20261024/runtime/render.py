"""Render native saved bag/coin poses and Numi-authored Franka FK; no render-engine physics."""
import sys,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import bpy
from mathutils import Vector,Quaternion,Matrix
sys.path.insert(0,str(Path(__file__).resolve().parent));import render_studio as studio
args=sys.argv[sys.argv.index('--')+1:];trajectory=Path(args[0]);output=Path(args[1]);mode=args[2] if len(args)>2 else 'wide';output.mkdir(parents=True,exist_ok=True)
frames=json.loads(trajectory.read_text())['frames'];objects=studio.create_scene(frames[0],960,960,32,'eevee');front,back,welds,bottom,coins,camera,env=objects;bottom.hide_render=True
asset=Path('/Users/home/Documents/Codex/2026-10-21/i-wa/work/plastic_bag_franka/franka_assets');xml=ET.parse(asset/'panda.xml').getroot()
materials={}
for m in xml.findall('asset/material'):
 rgba=list(map(float,m.get('rgba','1 1 1 1').split()));materials[m.get('name')]=studio.principled_material('Panda '+m.get('name'),rgba[:3],roughness=.33,metallic=.08)
materials['rubber']=studio.principled_material('Compliant rubber fingertip pads',(.028,.032,.037),roughness=.8)
meshes={e.get('name',Path(e.get('file')).stem):e.get('file') for e in xml.findall('asset/mesh')}
roots=[];name_indices={**{f'link{i}':i for i in range(8)},'hand':8,'left_finger':9,'right_finger':10}
for i in range(11):
 obj=bpy.data.objects.new('Panda link pose '+str(i),None);bpy.context.collection.objects.link(obj);obj.rotation_mode='QUATERNION';roots.append(obj)
for body in xml.findall('.//worldbody//body'):
 idx=name_indices.get(body.get('name'))
 if idx is None:continue
 for geom in body.findall('geom'):
  if geom.get('class')!='visual':continue
  file=asset/'assets'/meshes[geom.get('mesh')];verts=[];faces=[]
  if idx in [9,10] and file.stem=='finger_1':continue
  for line in file.read_text().splitlines():
   if line.startswith('v '):verts.append(tuple(map(float,line.split()[1:4])))
   elif line.startswith('f '):
    f=[int(v.split('/')[0]) for v in line.split()[1:]];faces.append(tuple(i-1 if i>0 else len(verts)+i for i in f))
  mesh=bpy.data.meshes.new(file.stem);mesh.from_pydata(verts,[],faces);mesh.update();obj=bpy.data.objects.new('Panda '+file.stem,mesh);bpy.context.collection.objects.link(obj);obj.parent=roots[idx]
  obj.location=tuple(map(float,geom.get('pos','0 0 0').split()));q=list(map(float,geom.get('quat','1 0 0 0').split()));obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Quaternion(q)
  mesh.materials.append(materials[geom.get('material','white')])
  for poly in mesh.polygons:poly.use_smooth=True
for idx in [9,10]:
 bpy.ops.mesh.primitive_cube_add(size=1);pad=bpy.context.object;pad.name='Stock fingertip compliant contact patch';pad.parent=roots[idx];pad.location=(0,.0015,.0445);pad.dimensions=(.017,.003,.017);pad.data.materials.append(materials['rubber'])
 bpy.ops.mesh.primitive_cube_add(size=1);carrier=bpy.context.object;carrier.name='Fingertip rigid central backing';carrier.parent=roots[idx];carrier.location=(0,.0065,.0445);carrier.dimensions=(.004,.008,.014);carrier.data.materials.append(materials['black'])
# Second robot uses the same authored meshes and contact patch geometry.
roots2=[]
for root in roots:
 copy=root.copy();copy.name=root.name+' receiver';bpy.context.collection.objects.link(copy);roots2.append(copy)
for obj in list(bpy.context.collection.objects):
 if obj.parent in roots:
  copy=obj.copy();copy.name=obj.name+' receiver';copy.parent=roots2[roots.index(obj.parent)];bpy.context.collection.objects.link(copy)
  if obj.name.startswith('Stock fingertip compliant contact patch') or obj.name.startswith('Fingertip rigid central backing'):copy.location.z+=frames[0]['robot2'].get('pad_center_local_z_m',.0445)-.0445
  if obj.name.startswith('Fingertip rigid central backing'):
   recess=frames[0]['robot2'].get('carrier_front_recess_m',0);copy.location.z-=recess/2;copy.scale=(.004,.008,.014-recess)
bpy.context.view_layer.update()
for obj in bpy.context.collection.objects:
 if obj.name.startswith('Stock fingertip compliant contact patch') or obj.name.startswith('Fingertip rigid central backing'):assert max(obj.dimensions)<.02,(obj.name,tuple(obj.dimensions))
studio.add_area('Robot softbox',(.2,-.45,1.15),(.2,-.2,.45),25,.8,.7)
C=Matrix(((1,0,0,0),(0,0,1,0),(0,1,0,0),(0,0,0,1)));scene=bpy.context.scene
env["wall"].location.y=1.2
if mode=='wide':studio.set_camera_view(camera,-35,21,studio.sim_to_blender([0,.43,.20]));camera.data.ortho_scale=1.52
elif mode=='side':studio.set_camera_view(camera,-75,26,studio.sim_to_blender([.025,.37,.065]));camera.data.ortho_scale=.86
else:studio.set_camera_view(camera,-17,18,studio.sim_to_blender([-.03,.49,.14]));camera.data.ortho_scale=.92
scene.render.image_settings.color_mode='RGB'
for ordinal,index in enumerate(range(0,len(frames),8)):
 dst=output/f'frame_{ordinal:04d}.png'
 if dst.exists():continue
 f=frames[index]
 if mode=='close' and f['time_s']>=5:studio.set_camera_view(camera,65,18,studio.sim_to_blender([-.03,.49,.14]))
 studio.update_frame(f,front,back,welds,bottom,coins)
 for r,key in [(roots,'robot'),(roots2,'robot2')]:
  for root,pose in zip(r,f[key]['links']):
   root.matrix_world=C@Matrix.Translation(Vector(pose[:3]))@Quaternion((pose[6],pose[3],pose[4],pose[5])).to_matrix().to_4x4()
 scene.render.filepath=str(output/f'frame_{ordinal:04d}.pending.png');bpy.ops.render.render(write_still=True);Path(scene.render.filepath).replace(dst);print('FRAME',ordinal+1,'time',f['time_s'],flush=True)
