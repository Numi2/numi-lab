"""Bind the saved contact/robot geometry audit to actual native GPU input buffers."""
from pathlib import Path
import json,time,struct,hashlib,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_franka_contact_finish_20261024';OUT=ROOT/'outputs/plastic_bag_franka_contact_finish_20261024'
while True:
 s=json.loads((OUT/'STATUS.json').read_text())
 if s['status']=='needs_attention':raise RuntimeError(s['error'])
 try:
  trajectory=OUT/'trajectory.json';raw=trajectory.read_bytes();fs=json.loads(raw)['frames']
  if len(fs)==1921 and fs[-1]['time_s']==8:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(25)
cp=OUT/'combined/latest_checkpoint.bin';data=cp.read_bytes();header=struct.unpack_from('<QQQdddQ',data);assert header[0]==0x4e554d4942473031 and header[1]==1920
offset=56;buffers=[]
for i in range(header[2]):
 n=struct.unpack_from('<Q',data,offset)[0];offset+=8;buffers.append(data[offset:offset+n]);offset+=n
assert offset==len(data)
path=np.frombuffer(buffers[0],dtype='<f4').reshape(-1,16);rows=np.loadtxt(W/'grip.csv',delimiter=',',skiprows=1)
expected=np.column_stack([rows[:,1:4],rows[:,7],rows[:,4:7],rows[:,8:11]]).astype('<f4')
actual=path[:,[0,1,2,3,4,5,6,8,9,10]]
assert np.array_equal(actual.view('<u4'),expected.view('<u4'))
runtime=np.frombuffer(buffers[1],dtype='<f4').reshape(2,16)
assert np.array_equal(runtime[0,:4].view('<u4'),path[1920,:4].view('<u4'))
visible=[]
for link in fs[-1]['robot']['links'][9:11]:visible.append(np.asarray(link[:3])+Rotation.from_quat(link[3:]).apply([0,0,.0445]))
visible_center=np.mean(visible,axis=0);physical_center=runtime[0,:3];center_gap=float(np.linalg.norm(visible_center-physical_center));assert center_gap<1e-7
land=fs[240];held=fs[1488];lift=float(held['front'][-1][1]-land['front'][-1][1]);coin_bottoms=[];film_bottoms=[]
for f in fs:
 centers=np.asarray(f['coins']);axis=Rotation.from_quat(f['coin_orientations']).apply(np.tile([0.,1.,0.],(10,1)))
 extent=.015305*np.linalg.norm(axis[:,[0,2]],axis=1)+.001075*np.abs(axis[:,1])
 coin_bottoms.append(float(np.min(centers[:,1]-extent-.108)))
 film_bottoms.append(float(np.asarray(f['front']+f['back'])[:,1].min()-.0000375-.108))
hold=[f for f in fs if 3.3<=f['time_s']<=6.15]
record={'trajectory_sha256':hashlib.sha256(raw).hexdigest(),'checkpoint_sha256':hashlib.sha256(data).hexdigest(),'native_checkpoint_step':header[1],'native_checkpoint_buffer_count':header[2],'gpu_input_path_poses':len(path),'gpu_input_center_gap_normal_tangent_matches_csv_bitwise':True,'final_actual_gpu_runtime_center_gap_matches_path_bitwise':True,'final_gpu_pad_center_to_visible_stock_faces_m':center_gap,'actual_opening_corner_lift_m':lift,'hold_all_frames_have_contact':all(f['grip_contact_nodes']>0 for f in hold),'held_coin_centers_min_y_m':float(np.asarray(held['coins'])[:,1].min()),'held_inside_coins':held['inside_coins'],'final_inside_coins':fs[-1]['inside_coins'],'final_grip_contact_nodes':fs[-1]['grip_contact_nodes'],'minimum_coin_table_clearance_m':min(coin_bottoms),'minimum_film_outer_surface_table_clearance_m':min(film_bottoms),'final_max_coin_linear_speed_m_s':float(np.linalg.norm(fs[-1]['coin_velocities'],axis=1).max()),'final_max_coin_angular_speed_rad_s':float(np.linalg.norm(fs[-1]['coin_angular_velocities'],axis=1).max()),'limits':['Pad center/gap binding is exact at saved output times; intermediate axis interpolation follows the native solver.','Robot joint motion is prescribed; motor dynamics are not simulated.','Material and pad properties are assumptions.']}
(OUT/'physical_grip_binding.json').write_text(json.dumps(record,indent=2))
with (OUT/'robot_film_full.log').open('w') as log:subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(trajectory)],check=True,stdout=log,stderr=subprocess.STDOUT)
mesh=json.loads(next(line.split('ROBOT_FILM_AUDIT ',1)[1] for line in (OUT/'robot_film_full.log').read_text().splitlines() if line.startswith('ROBOT_FILM_AUDIT ')))
(OUT/'robot_film_full.json').write_text(json.dumps(mesh,indent=2))
checks={'actual_lift_over_0_28_m':lift>.28,'contact_through_hold':record['hold_all_frames_have_contact'],'all_ten_coins_lifted':held['inside_coins']==10 and record['held_coin_centers_min_y_m']>.128,'released_at_end':fs[-1]['grip_contact_nodes']==0,'no_saved_non_pad_robot_film_crossing':mesh['intersection_state_count']==0,'film_table_clearance_within_0_1_um':min(film_bottoms)>=-1e-7,'coin_table_clearance_within_0_1_um':min(coin_bottoms)>=-1e-7}
(OUT/'grasp_task_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
assert all(checks.values()),checks
print('Physical GPU pad binding and full saved-state robot audit complete.',flush=True)
