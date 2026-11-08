"""Compare recorded native contact points with the visible stock fingertip faces."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
fs=json.loads(Path(sys.argv[1]).read_text())['frames']
path_file=Path(__file__).resolve().parent.parent/'grip.csv'
path_rows=np.loadtxt(path_file,delimiter=',',skiprows=1)
center_error=gap_error=normal_error=outside=0.;count=0;first=None;last=None;maxforce=0.;initial_force=False
authored_center_error=0.
active_face_distance=0.;spring_residual=0.;saturated_samples=0
for f in fs:
 faces=[]
 for idx in [9,10]:
  p=f['robot']['links'][idx];faces.append(np.array(p[:3])+Rotation.from_quat(p[3:]).apply([0,0,.0445]))
 center=.5*(faces[0]+faces[1]);n=faces[0]-faces[1];h=np.linalg.norm(n)/2;n/=2*h
 authored_center_error=max(authored_center_error,float(np.linalg.norm(center-f['robot']['grip_center'])))
 physical=path_rows[f['step']].astype(np.float32)
 assert abs(float(path_rows[f['step'],0])-f['time_s'])<1e-8
 center_error=max(center_error,float(np.linalg.norm(center-physical[1:4])));gap_error=max(gap_error,abs(h-float(physical[7])))
 normal_error=max(normal_error,float(np.linalg.norm(n-physical[4:7]/np.linalg.norm(physical[4:7]))))
 left=f['robot']['links'][9];rot=Rotation.from_quat(left[3:]);tx=rot.apply([1,0,0]);tz=rot.apply([0,0,1])
 nodes=f.get('grip_contacts',[])
 if f['time_s']<=1 and nodes:initial_force=True
 if nodes:
  first=f['time_s'] if first is None else first;last=f['time_s'];count+=len(nodes)
  forces=[0,0]
  for ct in nodes:
   v=np.array(ct['point'])-center
   outside=max(outside,abs(float(v@tx))-.0085,abs(float(v@tz))-.0085,abs(float(v@n))-h-.003-.0000375)
   assert ct['left_normal_N']>=0 and ct['right_normal_N']>=0
   signed=float(v@n)
   for force,distance in [(ct['left_normal_N'],signed+h),(ct['right_normal_N'],h-signed)]:
    if force<=0:continue
    active_face_distance=max(active_face_distance,abs(distance))
    if force<19.99:spring_residual=max(spring_residual,abs(distance-.0000375+force/200000.))
    else:saturated_samples+=1
   forces[0]+=ct['left_normal_N'];forces[1]+=ct['right_normal_N']
  maxforce=max(maxforce,*forces)
result={'states':len(fs),'through_s':fs[-1]['time_s'],'contact_point_samples':count,'first_contact_s':first,'last_contact_s':last,'max_physical_to_rendered_inner_face_center_error_m':center_error,'max_inner_half_gap_error_m':gap_error,'max_contact_point_outside_actual_fingertip_box_m':max(0,outside),'max_point_distance_to_its_active_inner_face_m':active_face_distance,'max_unsaturated_pad_spring_equilibrium_residual_m':spring_residual,'saturated_pad_node_samples':saturated_samples,'maximum_per_jaw_normal_force_N':maxforce,'no_grip_force_during_first_drop':not initial_force,'no_extensions':True,'all_forces_at_visible_stock_jaws':bool(center_error<1e-7 and gap_error<1e-7 and outside<2e-6),'limits':['Fingertips use authored planar compliant rubber boxes.','Arm follows prescribed joint positions; no motor dynamics or full robot mesh contact.','This checks pose/contact placement, not force calibration.']}
result.update(authored_fk_center_consistency_error_m=authored_center_error,max_physical_input_normal_error=normal_error,physical_input=str(path_file),physical_input_sha256=hashlib.sha256(path_file.read_bytes()).hexdigest(),physical_input_precision='Native path reader casts CSV pose fields to FP32; the exported forces and contact points come from the GPU contact buffer.')
result.update(force_channel_to_articulated_body={'left_normal_N':10,'right_normal_N':9},normal_force_definition='Legacy channel names denote negative-normal and positive-normal planes respectively. Values are position contact multipliers divided by dt squared; velocity impulses act at the same recorded nodes.')
print(json.dumps(result,indent=2))
