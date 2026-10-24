"""Approach the free opening corner from above, then transfer to a stationary receiver."""
from pathlib import Path
import json,subprocess,numpy as np
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;P=O.parent/'plastic_bag_empty_franka_20261024';REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024';N=1920
s=Path('/private/tmp/numi_bag_franka_handoff_20261024/hold_progress.json').read_text();decoder=json.JSONDecoder();i=s.index('[',s.index('"frames"'))+1;ref={}
while i<len(s):
 try:f,j=decoder.raw_decode(s,i)
 except json.JSONDecodeError:break
 ref[f['step']]=f;i=j+1
ref.update({f['step']:f for f in json.loads((P/'trajectory.json').read_text())['frames'] if f['step']<=1488})
old=json.loads((W/'robot.json').read_text())['frames'];old2=json.loads((W/'robot2.json').read_text())['frames'];oldcsv=(W/'grip.csv').read_text().splitlines()
r0=Rotation.from_quat(old2[1560]['links'][8][3:]).as_matrix();normal=r0[:,1];z=np.array([0.,-1.,0.]);z-=normal*np.dot(z,normal);z/=np.linalg.norm(z);x=np.cross(normal,z);hand=Rotation.from_matrix(np.stack([x,normal,z],axis=1));hq=hand.as_quat()
def center(step):return np.array(ref[step]['front'][550])-.0075*z-.0075*x
v=(center(1489)-center(1487))*120;capcenter=center(1488)
def blend(t):t=np.clip(t,0,1);return t**3*(10+t*(-15+6*t))
park=center(1200)-.1425*z
base=(W/'receiver_base.txt').read_text().split();base[7]='1';request=W/'upright_park_request.txt';request.write_text(' '.join(base)+' '+' '.join(format(t,'.17g') for t in [*park,*hq,.02,*old2[0]['q']])+'\n')
subprocess.run(['rsync','-az',str(request),f'macmini:{REMOTE}/'],check=True)
a=subprocess.run(['ssh','macmini',f'cd {REMOTE} && build/franka_handoff_planner {request.name} unused unused unused .0475'],capture_output=True,text=True);(O/'upright_park_ik.log').write_text(a.stdout+a.stderr);assert a.returncode==0,a.stderr;parkq=[float(t) for t in a.stdout.splitlines()[0].split()[1:]]
retreat=np.array(old[-1]['q']);held=np.array(old[1488]['q'])
for arm in ['donor','receiver']:
 with (W/(arm+'_upright_targets.csv')).open('w') as out:
  out.write('time_s,mode,values\n')
  for k in range(N+1):
   t=k/240
   if arm=='donor':
    q=np.array(old[min(k,1488)]['q']);q[:7]=(1-blend(t-7))*q[:7]+blend(t-7)*retreat[:7]
    if k>=1488:q[7:]=.0000175+(.02-.0000175)*blend((t-6.55)/.2)
    mode='FK';values=q
   elif k<=1200:mode='FK';values=parkq
   else:
    if k<=1440:
     follow=blend((t-5)/.3);corner=(1-follow)*np.array(ref[1200]['front'][550])+follow*np.array(ref[k]['front'][550]);c=corner-.0075*x-z*(.15*(1-blend(t-5))+.0075*blend(t-5))
    elif k<=1488:c=center(k)
    elif t<6.4:
     u=(t-6.2)/.2;c=capcenter+v*.2*(u-u*u+u**3/3)
    else:c=capcenter+v*.2/3
    gap=.02*(1-blend((t-5.975)/.225))+.0000355*blend((t-5.975)/.225)
    mode='IK';values=[*c,*hq,gap]
   out.write(','.join([format(t,'.17g'),mode,*[format(a,'.17g') for a in values]])+'\n')
 subprocess.run(['rsync','-az',str(W/(arm+'_upright_targets.csv')),f'macmini:{REMOTE}/'],check=True)
 ext=' .0475' if arm=='receiver' else ''
 a=subprocess.run(['ssh','macmini',f'cd {REMOTE} && build/franka_handoff_planner {arm}_base.txt {arm}_upright_targets.csv {arm}_upright_grip.csv {arm}_upright_robot.json{ext}'],capture_output=True,text=True);(O/(arm+'_upright_ik.log')).write_text(a.stdout+a.stderr);assert a.returncode==0,a.stderr
 for n in [arm+'_upright_grip.csv',arm+'_upright_robot.json']:subprocess.run(['rsync','-az',f'macmini:{REMOTE}/{n}',str(W/n)],check=True)
 rows=(W/(arm+'_upright_grip.csv')).read_text().splitlines();frames=json.loads((W/(arm+'_upright_robot.json')).read_text())['frames']
 if arm=='donor':rows=oldcsv[:1490]+rows[1490:];frames=old[:1489]+frames[1489:]
 else:
  for f in frames:f['carrier_front_recess_m']=.006
 (W/('grip.csv' if arm=='donor' else 'grip2.csv')).write_text('\n'.join(rows)+'\n');(W/('robot.json' if arm=='donor' else 'robot2.json')).write_text(json.dumps({'frames':frames},separators=(',',':')))
plan={'duration_s':8,'native_hz':240,'native_states':1921,'resume_step':1200,'receiver_capture_s':6.2,'receiver_brake_until_s':6.4,'donor_release_s':[6.55,6.75],'donor_retreat_s':[7,8],'receiver_grip_per_jaw_N':.4,'receiver_total_gap_m':.000071,'receiver_soft_pad_overhang_m':.003,'receiver_hand_forward_world':z.tolist(),'receiver_carrier_front_recess_m':.006,'receiver_static_after_s':6.4,'corner_offset_in_pad_plane_m':[.0075,.0075],'coins':0,'fixed_film_nodes':0,'reason':'Receiver approaches from above with fingertips downward, clearing hanging film from its wrist and palm. Gentle finite friction pinch, no attachments.'}
for p in [W/'scene_plan.json',O/'scene_plan.json',O/'handling_plan.json']:p.write_text(json.dumps(plan,indent=2))
print(json.dumps(plan,indent=2))
