#!/usr/bin/env python3
"""Ensure the native analysis rejects synthetic conservation failures."""
import bisect,json,math,pathlib,subprocess,sys,tempfile
root=pathlib.Path(__file__).resolve().parents[2];p=root/'docs/evidence/midpoint-first-failure-20261109'
r=json.loads((p/'slider-near-rk4-h025.json').read_text());trace=r['trace'];times=[x[0] for x in trace]
def at(t,col):
 i=bisect.bisect_left(times,t)
 if i==0:return trace[0][col]
 if i>=len(trace):return trace[-1][col]
 a,b=trace[i-1],trace[i];return a[col]+(b[col]-a[col])*(t-a[0])/(b[0]-a[0])
h=15.625e-6;steps=[]
for i in range(640):
 t=i*h;u=(i+1)*h;j=at(u,2)-at(t,2);x=.01*j;y=-x
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
for step in steps:
 j=step['fem_impulse'];lo=1e-12;hi=.0001
 if j>0:
  for _ in range(80):
   mid=(lo+hi)/2
   if impulse(mid,active[0])>j:lo=mid
   else:hi=mid
  gap=(lo+hi)/2
 else:gap=.0001
 expected=impulse(gap,active[0])
 step['fem_work_nodes']=[[node,j if k==0 else 0.,expected if k==0 else 0.,.1,gap if k==0 else .0001] for k,node in enumerate(active)]
 step['fem_contact_work']=j*.1;step['fem_law_work']=expected*.1
 step['fem_work_reconstruction_gate_ratio']=abs((j-expected)*.1)/(max(1e-12,1e-5*abs(expected))*.1+1e-24)
d={'schema':'numi.slider-coupled-diagnostic.v1','initial_gap':.000095,'h':h,'slider_mass_kg':.01,
 'initial_slider_velocity_mps':.02,'reference_area_m2':.0004,'reference_area_weights_m2':a,
 'activation_width_m':.0001,'contact_coefficient':1e16,'physical_residual_limit_Ns':1e-10,'steps':steps}
with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
 def run():
  f.seek(0);f.truncate();json.dump(d,f);f.flush()
  result=subprocess.run([sys.executable,str(root/'tests/contact/analyze_native_slider.py'),f.name],capture_output=True,text=True)
  return result.returncode,json.loads(result.stdout) if result.stdout else {'error':result.stderr}
 code,out=run();assert code==0,(code,out)
 for field in ('owner_momentum_impulse','owner_wrench_impulse','support_moment_impulse_x','independent_joint_residual_Ns','midpoint_gap','owner_wrench_work','fem_work_reconstruction_gate_ratio'):
  old=d['steps'][100][field];d['steps'][100][field]=0. if field=='midpoint_gap' else 2. if field=='fem_work_reconstruction_gate_ratio' else old+1e-5
  code,out=run();assert code!=0,(field,out);d['steps'][100][field]=old
 old=d['steps'][100]['fem_work_nodes'][0][1];d['steps'][100]['fem_work_nodes'][0][1]+=1e-5
 code,out=run();assert code!=0,('corrupt node',out);d['steps'][100]['fem_work_nodes'][0][1]=old
 old=d['steps'][100]['fem_work_nodes'][0][4];d['steps'][100]['fem_work_nodes'][0][4]=old*.9
 code,out=run();assert code!=0,('corrupt gap',out);d['steps'][100]['fem_work_nodes'][0][4]=old
 print('baseline synthetic analysis passed; impulse, torque, residual, midpoint clearance, and both work perturbations rejected')
