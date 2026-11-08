from pathlib import Path
import json,math
A=Path('artifacts/crow-flight-source-diagnostics-20261108')
rows=json.load(open(A/'all-contacts-results.json'))
forbidden={'crow_body','crow_left_wing','crow_right_wing','crow_tail','crow_left_thigh','crow_left_shank','crow_right_thigh','crow_right_shank'}
names=['wing.left_flap','wing.right_flap','wing.left_sweep','wing.right_sweep','wing.left_pronation','wing.right_pronation','tail.pitch',
 'leg.left_hip','leg.left_knee','leg.left_ankle','leg.right_hip','leg.right_knee','leg.right_ankle',
 'tail.yaw_moment','body.pitch_moment','head.yaw','head.pitch','beak.open']
out=[]
for run in rows:
 for e in run['first_events_by_lane']:
  if e is None:continue
  q=e['pre_q'];v=e['pre_v'];x,y,z,w=q[3:7]
  roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
  pitch=math.asin(max(-1.0,min(1.0,2*(w*y-z*x))))
  yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
  candidates=[c for c in e['contacts'] if (c['a_name'] in forbidden) != (c['b_name'] in forbidden)]
  candidates.sort(key=lambda c:abs(c['normal_impulse']),reverse=True)
  out.append({'layout':run['name'],'lane':e['lane'],'reported_control_step':e['step'],
    'pre_step_root_position_m':q[:3],'pre_step_roll_pitch_yaw_rad':[roll,pitch,yaw],
    'pre_step_root_generalized_linear_velocity_mps':v[:3],
    'pre_step_root_generalized_angular_velocity_radps':v[3:6],
    'filtered_normalized_action_for_failure_step':dict(zip(names,e['applied_filtered_action'])),
    'forbidden_contact_candidates_last_substep':candidates,
    'other_last_substep_contacts': [c for c in e['contacts'] if c not in candidates]})
(A/'first-failure-analysis.json').write_text(json.dumps({'records':out,'caveat':'Final-substep contacts only; a candidate is not proof of the unique first microstep collider. Pre-step state is the previous accepted control step. Filtered normalized actions are actuator inputs, not measured torques.'},indent=2)+'\n')
print('analyzed',len(out),'episodes; forbidden candidates',sum(bool(x['forbidden_contact_candidates_last_substep']) for x in out))
