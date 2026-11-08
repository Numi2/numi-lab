"""Recompute native FK for a longer visible receiver fixture; leave world pad input bytes untouched."""
from pathlib import Path
import json,subprocess,hashlib,numpy as np
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024';pad_z=.0565
rows=np.loadtxt(W/'grip2.csv',delimiter=',',skiprows=1);old=(W/'grip2.csv').read_bytes();seed=json.loads((W/'robot2.json').read_text())['frames'][0]['q']
def vals(row):
 n=row[4:7];tx=row[8:11];z=np.cross(tx,n);q=Rotation.from_matrix(np.stack([tx,n,z],axis=1)).as_quat();return [*row[1:4],*q,row[7]]
base=(W/'receiver_base.txt').read_text().split();base[7]='1';p=W/'fixture_park_request.txt';p.write_text(' '.join(base)+' '+' '.join(format(x,'.17g') for x in [*vals(rows[0]),*seed])+'\n');subprocess.run(['rsync','-az',str(p),f'macmini:{REMOTE}/'],check=True)
x=subprocess.run(['ssh','macmini',f'cd {REMOTE} && build/franka_handoff_planner {p.name} unused unused unused {pad_z}'],capture_output=True,text=True);assert x.returncode==0,x.stderr;q0=[float(x) for x in x.stdout.splitlines()[0].split()[1:]]
p=W/'fixture_receiver_targets.csv'
with p.open('w') as f:
 f.write('time_s,mode,values\n')
 for i,row in enumerate(rows):f.write(','.join([format(row[0],'.17g'),'IK' if i else 'FK',*[format(x,'.17g') for x in (vals(row) if i else q0)]])+'\n')
subprocess.run(['rsync','-az',str(p),f'macmini:{REMOTE}/'],check=True)
x=subprocess.run(['ssh','macmini',f'cd {REMOTE} && build/franka_handoff_planner receiver_base.txt {p.name} fixture_grip2.csv fixture_robot2.json {pad_z}'],capture_output=True,text=True);(O/'fixture_fk.log').write_text(x.stdout+x.stderr);assert x.returncode==0,x.stderr
for n in ['fixture_grip2.csv','fixture_robot2.json']:subprocess.run(['rsync','-az',f'macmini:{REMOTE}/{n}',str(W/n)],check=True)
b=np.loadtxt(W/'fixture_grip2.csv',delimiter=',',skiprows=1);err={'center_m':float(np.max(np.linalg.norm(rows[:,1:4]-b[:,1:4],axis=1))),'normal':float(np.max(np.linalg.norm(rows[:,4:7]-b[:,4:7],axis=1))),'tangent':float(np.max(np.linalg.norm(rows[:,8:11]-b[:,8:11],axis=1))),'half_gap_m':float(np.max(abs(rows[:,7]-b[:,7])))};assert err['center_m']<2e-8 and err['normal']<2e-7 and err['tangent']<2e-7 and err['half_gap_m']<1e-9,err
assert (W/'grip2.csv').read_bytes()==old
new=json.loads((W/'fixture_robot2.json').read_text())
for f in new['frames']:f['carrier_front_recess_m']=.010
(W/'robot2.json').write_text(json.dumps(new,separators=(',',':')))
r={'receiver_pad_center_local_z_m':pad_z,'stock_pad_center_local_z_m':.0445,'rubber_pad_forward_extension_m':.012,'carrier_front_recess_m':.010,'receiver_carrier_size_m':[.004,.008,.004],'physical_world_pad_trajectory_byte_identical':True,'world_pad_path_sha256':hashlib.sha256(old).hexdigest(),'native_fk_matching_max_errors':err,'robot2_sha256':hashlib.sha256((W/'robot2.json').read_bytes()).hexdigest(),'physical_equivalence':'World pad geometry, forces, contact coefficients, film dynamics and actuator work unchanged. Authored receiver fixture and its full native rigid FK now keep the housing behind the contact faces. Saved-state geometry must pass again.','not_stock_hardware':True};(O/'scene_geometry_binding.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
