#!/usr/bin/env python3
"""Ensure the native analysis rejects synthetic conservation failures."""
import bisect,json,math,pathlib,subprocess,sys,tempfile
root=pathlib.Path(__file__).resolve().parents[2];p=root/'docs/evidence/midpoint-first-failure-20261109'
r=json.loads((p/'slider-near-rk4-h025.json').read_text());trace=r['trace'];times=[x[0] for x in trace]
wr=json.loads((p/'slider-work-history-002-near-h025.json').read_text())['work_trace']
def at(t,col):
 i=bisect.bisect_left(times,t)
 if i==0:return trace[0][col]
 if i>=len(trace):return trace[-1][col]
 a,b=trace[i-1],trace[i];return a[col]+(b[col]-a[col])*(t-a[0])/(b[0]-a[0])
def work_at(t,col):
 i=bisect.bisect_left(times,t)
 if i==0:return wr[0][col]
 if i>=len(wr):return wr[-1][col]
 a,b=wr[i-1],wr[i];return a[col]+(b[col]-a[col])*(t-a[0])/(b[0]-a[0])
h=15.625e-6;steps=[]
for i in range(640):
 t=i*h;u=(i+1)*h;j=at(u,2)-at(t,2);x=j*r['torque_impulse_x_Nms']/r['impulse_Ns'];y=j*r['torque_impulse_y_Nms']/r['impulse_Ns']
 steps.append({'step':i,'accepted':True,'gap':.00009,'midpoint_gap':.00009,'contact_potential_before_J':0.,'contact_potential_after_J':0.,'owner_wrench_work':0.,'q':at(u,3),'v':at(u,4),
  'fem_impulse':j,'owner_momentum_impulse':-j,'owner_wrench_impulse':-j,
  'fem_torque_impulse_x':x,'fem_torque_impulse_y':y,'owner_moment_impulse_x':-x,'owner_moment_impulse_y':-y,
  'support_moment_impulse_x':x,'support_moment_impulse_y':y,'fem_contact_work':0.,'owner_contact_work':0.,
  'independent_joint_residual_Ns':0.,'slider_kinematic_defect_m':0.,'fem_kinematic_defect_m':0.})
a=json.loads((p/'compact-reference-parity-n2-ordered-001.json').read_text())['contact_area_weights_m2']
active=[i for i,x in enumerate(a) if x>0]
def impulse(gap,node):
 if gap>=.0001:return 0.
 square=gap*gap;shift=square-.0001**2
 pressure=2*gap*1e16*(2*shift*math.log(square/.0001**2)+shift*shift/square)
 return h*a[node]*pressure
for i,step in enumerate(steps):
 j=step['fem_impulse'];lo=1e-12;hi=.0001
 if j>0:
  for _ in range(80):
   mid=(lo+hi)/2
   if sum(impulse(mid,node) for node in active)>j:lo=mid
   else:hi=mid
  gap=(lo+hi)/2
 else:gap=.0001
 expected=[impulse(gap,node) for node in active]
 target=work_at((i+1)*h,1)-work_at(i*h,1)
 vm=target/j if j>0 else 0.
 step['fem_work_nodes']=[[node,j*a[node]/sum(a),q,vm,gap] for node,q in zip(active,expected)]
 step['fem_contact_work']=sum(row[1]*row[3] for row in step['fem_work_nodes'])
 step['fem_law_work']=sum(row[2]*row[3] for row in step['fem_work_nodes'])
 step['fem_work_reconstruction_gate_ratio']=max(abs((row[1]-row[2])*vm)/(max(1e-12,1e-5*abs(row[2]))*abs(vm)+1e-24) for row in step['fem_work_nodes'])
 midpoint_v=.5*(at(i*h,4)+step['v'])
 step['owner_contact_work']=-j*midpoint_v
 step['owner_wrench_work']=step['owner_contact_work']

d={'schema':'numi.slider-coupled-diagnostic.v1','initial_gap':.000095,'h':h,'slider_mass_kg':.01,
 'initial_slider_velocity_mps':.02,'reference_area_m2':.0004,'reference_area_weights_m2':a,
 'activation_width_m':.0001,'contact_coefficient':1e16,'physical_residual_limit_Ns':1e-10,'steps':steps}
with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
 def run():
  f.seek(0);f.truncate();json.dump(d,f);f.flush()
  result=subprocess.run([sys.executable,str(root/'tests/contact/analyze_native_slider.py'),f.name],capture_output=True,text=True)
  return result.returncode,json.loads(result.stdout) if result.stdout else {'error':result.stderr}
 code,out=run();assert code==0,(code,out)
 # A rejected or missing step may never be upgraded to a full rollout.
 d['steps'][-1]['accepted']=False
 d['steps'][-1]['world_status']=11;d['steps'][-1]['matter_status']=10;d['steps'][-1]['failure_index']=7
 code,out=run();assert code!=0 and 'native step 639 rejected' in out['error'] and 'matter_status=10' in out['error'] and 'failure_index=7' in out['error'],('rejected step accepted or failure category lost',out)
 d['steps'][-1]['accepted']=True
 last=d['steps'].pop()
 code,out=run();assert code!=0,('incomplete trajectory accepted',out)
 d['steps'].append(last)
 for field in ('owner_momentum_impulse','owner_wrench_impulse','support_moment_impulse_x','independent_joint_residual_Ns','midpoint_gap','owner_wrench_work','fem_work_reconstruction_gate_ratio'):
  old=d['steps'][100][field];d['steps'][100][field]=0. if field=='midpoint_gap' else 2. if field=='fem_work_reconstruction_gate_ratio' else old+1e-5
  code,out=run();assert code!=0,(field,out);d['steps'][100][field]=old
 for field,check in (('slider_kinematic_defect_m','slider_midpoint_kinematics'),('fem_kinematic_defect_m','fem_midpoint_kinematics')):
  old=d['steps'][100][field];d['steps'][100][field]=1e-9
  code,out=run();assert code!=0 and not out['checks'][check],(field,out)
  d['steps'][100][field]=old
 old=d['steps'][100]['fem_work_nodes'][0][1];d['steps'][100]['fem_work_nodes'][0][1]+=1e-5
 code,out=run();assert code!=0,('corrupt node',out);d['steps'][100]['fem_work_nodes'][0][1]=old
 for field,check in (('q','slider_position'),('v','slider_velocity')):
  old=d['steps'][-1][field];d['steps'][-1][field]=old+.01
  code,out=run();assert code!=0 and not out['checks'][check],(field,out)
  d['steps'][-1][field]=old
 # At a zero-velocity node pair, opposing bad impulses cancel in total
 # and perform no work, but must still fail the direct per-node law check.
 nodes=d['steps'][100]['fem_work_nodes']
 old0,old1=nodes[0][1],nodes[1][1]
 nodes[0][1]+=1e-5;nodes[1][1]-=1e-5
 code,out=run();assert code!=0,('zero-work nodal redistribution',out)
 nodes[0][1],nodes[1][1]=old0,old1
 old=d['steps'][100]['fem_work_nodes'][0][4];d['steps'][100]['fem_work_nodes'][0][4]=old*.9
 code,out=run();assert code!=0,('corrupt gap',out);d['steps'][100]['fem_work_nodes'][0][4]=old
 # Preserve both local torque balances while corrupting the accumulated
 # reference torque; the independent total-torque gate must catch it.
 step=max(d['steps'],key=lambda x:x['fem_impulse'])
 step['fem_torque_impulse_x']+=1e-7
 step['owner_moment_impulse_x']-=1e-7
 step['support_moment_impulse_x']+=1e-7
 code,out=run();assert code!=0 and out['checks']['contact_and_support_torque'] and not out['checks']['torque_x_reference'],('corrupt total torque',out)
 step['fem_torque_impulse_x']-=1e-7
 step['owner_moment_impulse_x']+=1e-7
 step['support_moment_impulse_x']-=1e-7
 # Change all nodal velocities coherently so local work reconstruction remains
 # consistent but the cumulative history must fail against the reference.
 step=max(d['steps'],key=lambda x:x['fem_impulse'])
 for row in step['fem_work_nodes']:row[3]+=.1
 step['fem_contact_work']=sum(row[1]*row[3] for row in step['fem_work_nodes'])
 step['fem_law_work']=sum(row[2]*row[3] for row in step['fem_work_nodes'])
 step['fem_work_reconstruction_gate_ratio']=max(abs((row[1]-row[2])*row[3])/(max(1e-12,1e-5*abs(row[2]))*abs(row[3])+1e-24) for row in step['fem_work_nodes'])
 code,out=run();assert code!=0 and not out['checks']['fem_work_history'],('corrupt work history',out)
 print('synthetic analysis rejects incomplete or rejected rollouts, impulse, local and reference torque, residual, clearance, local work and cumulative-work failures')
