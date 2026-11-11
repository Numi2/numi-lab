from pathlib import Path
from collections import defaultdict,Counter
import gzip,json,statistics,math
D=Path('artifacts/crow-v15-perch-support-impulse-window-20261111');P=Path('artifacts/crow-v15-landing-root-balance-latch-screen-20261111/trial')
start,end=3686,5000;ts=defaultdict(lambda:[0.,0.]);pairs=Counter();states={}
for p in sorted(P.glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);t=x['step']
    if x['lane']==3 and start<=t<=end:states[t]={'position':x['q'][:3],'root_velocity':x['v'][:6],'wing_angles':x['q'][7:11],'wing_rates':x['v'][6:10]}
   elif l.startswith('CROW_SCENE_CONTACT '):
    a=l.split();t,k,e,ba,bb=map(int,a[1:6]);
    if e!=3 or not start<=t<=end or 38 not in (ba,bb):continue
    b=bb if ba==38 else ba;pairs[b]+=1
    j,j1,j2=map(float,a[6:9]);nx,ny,nz=map(float,a[13:16]);tx,ty,tz=map(float,a[16:19]);sgn=1 if ba==38 else -1
    impulse=sgn*(nz*j+tz*j1+(nx*ty-ny*tx)*j2)
    ts[t][0 if b==10 or b==13 or 17<=b<=32 else 1]+=impulse
m=.450039993491373;nom=m*9.81*.02
ratios=[ts[t][0]/nom for t in range(start,end+1)]
win=[sum(ratios[i:i+100])/100 for i in range(len(ratios)-99)]
result={'lane':3,'window_inclusive':[start,end],'sampled_control_interval_s':.02,'nominal_impulse_ns':nom,'feet_weight_fraction_mean':statistics.mean(ratios),'feet_weight_fraction_median':statistics.median(ratios),'feet_weight_fraction_min':min(ratios),'feet_weight_fraction_max':max(ratios),'feet_weight_fraction_minimum_100_interval_mean':min(win),'feet_weight_fraction_maximum_100_interval_mean':max(win),'nonfoot_perch_vertical_impulse_total_ns':sum(v[1] for v in ts.values()),'perch_contact_body_record_counts':dict(pairs),'number_of_endpoint_samples':len(states),'first_sample':states.get(start),'last_sample':states.get(end),'limitations':'Perch forces are reconstructed from logged positive-normal substep contact impulses with native tangent basis; no measurement of total external aerodynamic forces, unlogged contacts, or exact whole-body momentum. Endpoint state does not prove continuous posture. Check numerical precision, sign conventions, and momentum independently before calling this fully weight-bearing.'}
(D/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
