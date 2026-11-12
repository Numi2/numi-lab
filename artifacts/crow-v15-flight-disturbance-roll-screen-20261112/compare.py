from pathlib import Path
import collections,gzip,json,math,statistics
A=Path('artifacts/crow-v15-flight-disturbance-roll-screen-20261112');B=Path('artifacts/crow-v15-wing-normal-height-no-step-control-20261112');part='candidate-20261126-rebound-flexed/run.stderr.gz'
def read(p):
 s={};c=collections.defaultdict(list)
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);s[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);
    if t<=(1200 if e<2 else 600):c[e].append(tuple(z[1:]))
 return s,c
base,bc=read(B/part);test,tc=read(A/part);d=[]
for e in range(8):
 end=1200 if e<2 else 600
 for t in range(1,end+1):
  if (e,t) not in test or (e,t) not in base: d.append([e,t,'missing']);continue
  if any(base[e,t][k]!=test[e,t][k] for k in ('q','v','policy_action','contacts')):d.append([e,t,'state'])
 if bc[e]!=tc[e]:d.append([e,'contact'])
(A/'parity.json').write_text(json.dumps({'differences':d,'scope':'Complete 1200-step controls; first 600 steps of disturbed lanes.'},indent=2)+'\n')
def angles(q):
 x,y,z,w=q[3:7];return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
def wrapped(a):return math.atan2(math.sin(a),math.cos(a))
rows=[]
for e in range(2,8):
 ts=[t for t in range(1001,1201) if (e,t) in base and (e,t) in test]
 if len(ts)!=200:rows.append({'lane':e,'last_window_samples':len(ts),'pass':False});continue
 a=[angles(test[e,t]['q']) for t in ts];b=[angles(base[e,t]['q']) for t in ts]
 ad=[statistics.mean(abs(wrapped(x[j]-y[j])) for x,y in zip(a,b)) for j in range(3)]
 hd=statistics.mean(abs(test[e,t]['q'][2]-base[e,t]['q'][2]) for t in ts);vd=statistics.mean(math.hypot(*(test[e,t]['v'][j]-base[e,t]['v'][j] for j in (0,1))) for t in ts)
 peak=max(abs(wrapped(angles(test[e,t]['q'])[(e-2)//2]-angles(base[e,t]['q'])[(e-2)//2])) for t in range(601,801) if (e,t) in test and (e,t) in base)
 rows.append({'lane':e,'axis':['roll','pitch','yaw'][(e-2)//2],'sign':1 if e%2==0 else -1,'attitude_abs_mean_difference_rad':ad,'height_abs_mean_difference_m':hd,'horizontal_velocity_vector_mean_difference_m_s':vd,'peak_axis_deviation_12_to_16s_rad':peak,'observable_disturbance':peak>=.01,'pass_tracking_only':peak>=.01 and all(x<.10 for x in ad) and hd<.10 and vd<.15})
out={'rows':rows,'prefix_differences':len(d),'scope':'Tracking criteria only; survival, substep caps and contacts must be checked in analysis-20261126.json. Injected external torque is a prescribed test disturbance, not continuous controller authority. No biological calibration or perturbed-start qualification.'};(A/'recovery-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert not d
