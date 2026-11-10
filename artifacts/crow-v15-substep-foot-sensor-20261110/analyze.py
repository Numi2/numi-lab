from pathlib import Path
import gzip,json,struct,math
A=Path('artifacts/crow-v15-substep-foot-sensor-20261110');O=Path('artifacts/crow-v15-substep-contact-observer-20261110');name='development-reserved-final-j'
def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)['payload']
a=load(O/(name+'.json.gz'));b=load(A/(name+'.json.gz'));dif=[]
for fa,fb in zip(a['frames'],b['frames']):
 for k in ('step','q','v','actor_observation','accepted_actions','termination_reason'):
  if fa[k]!=fb[k]:
   if len(dif)<20:dif.append({'step':fa['step'],'field':k})
   break
parity=len(a['frames'])==len(b['frames']) and not dif
traces={}
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if not line.startswith('CROW_TASK_STATE 0 '):continue
  v=line.split();t=int(v[2]);
  if 4375<=t<=4410:
   bits=list(map(int,v[10:12]));traces[t]=[struct.unpack('f',struct.pack('I',u))[0] for u in bits]
prior=json.load(open(O/'analysis.json'))['toe_contacts'];expected=[]
for lo,hi in ((17,20),(21,24)):
 expected.append(sum(z['impulse'] for z in prior if z['submission']==4394 and z['lane']==0 and lo<=z['a']<=hi and z['b']==30))
measured=traces.get(4393);error=[abs(x-y) for x,y in zip(expected,measured)] if measured else None
window_diff=[]
for t in range(4376,4411):
 observed=traces.get(t-1)
 want=[sum(z["impulse"] for z in prior if z["submission"]==t and z["lane"]==0 and lo<=z["a"]<=hi and z["b"]==30) for lo,hi in ((17,20),(21,24))]
 if observed is None or max(abs(x-y) for x,y in zip(observed,want))>1e-5:window_diff.append(t)
noimpact=[t for t in (4380,4385,4390) if t in traces and max(map(abs,traces[t]))<1e-8]
res={'parity':parity,'differences':dif,'frames':[len(a['frames']),len(b['frames'])],'step4394_expected_toe_impulses':expected,'step4394_sensor_impulses':measured,'step4394_absolute_errors':error,'impulse_match':error is not None and max(error)<1e-5,'timestamp_offset':-1,'window_impulse_differences':window_diff,'zero_samples':noimpact,'scratch_samples':traces}
(A/'analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='scratch_samples'},indent=2))
