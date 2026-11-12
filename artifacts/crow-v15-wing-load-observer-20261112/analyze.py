from pathlib import Path
import gzip,json,math,statistics,collections
A=Path('artifacts/crow-v15-wing-load-observer-20261112');B=Path('artifacts/crow-v15-wing-bank-turn-screen-20261112')
def flight(path):
 d={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d[x['lane'],x['step']]=[x['q'],x['v'],x['policy_action']]
 return d
baseline=flight(B/'bank.stderr.gz');observed=flight(A/'bank.stderr.gz');keys=set(baseline)|set(observed);diff=[k for k in sorted(keys) if baseline.get(k)!=observed.get(k)];assert len(keys)>=48 and not diff,(len(keys),diff[:10])
data={};errors=[]
with gzip.open(A/'bank.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_AERO_ERROR'):errors.append(l.strip())
  if l.startswith('CROW_WING_WRENCH '):
   z=l.split();t,k,e=map(int,z[1:4]);x=list(map(float,z[4:]));assert len(x)==22 and all(math.isfinite(v) for v in x) and (e,t,k) not in data;data[e,t,k]=x
assert len(data)==6*100*16 and not errors,(len(data),errors)
rows=[]
for e in range(6):
 for lo,hi in [(451,500),(701,750)]:
  vals=[data[e,t,k] for t in range(lo,hi+1) for k in range(16)];q0=[x[0] for x in vals];q1=[x[1] for x in vals];v0=[x[2] for x in vals];v1=[x[3] for x in vals]
  def stats(col):
   v=[x[col] for x in vals];return {'mean':statistics.mean(v),'min':min(v),'max':max(v),'signed_integral_1s':sum(v)*.00125}
  rows.append({'lane':e,'steps':[lo,hi],'left_flap_q_rad':[min(q0),max(q0)],'right_flap_q_rad':[min(q1),max(q1)],'left_flap_v_rad_s':[min(v0),max(v0)],'right_flap_v_rad_s':[min(v1),max(v1)],'left_force_z_N':stats(6),'right_force_z_N':stats(12),'left_com_torque_z_Nm':stats(9),'right_com_torque_z_Nm':stats(15),'root_kernel_torque_z_Nm':stats(21)})
out={'sparse_state_parity_samples':len(keys),'sparse_state_mismatches':diff,'wrench_substep_records':len(data),'errors':errors,'rows':rows,'scope':'Wrench after minus before existing flapping kernel, in world axes. Wing torques are about their body COM; root z torque excludes moments from wing-force lever arms. Source q/v precede that kernel and substep integration. Each integral uses 800 substeps at 0.00125s. Six development lanes, two selected 1s intervals; no calibrated aerodynamic ground truth, total root moment, continuous whole-run envelope, landing or qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
