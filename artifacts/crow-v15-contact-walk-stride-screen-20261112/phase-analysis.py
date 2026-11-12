from pathlib import Path
import importlib.util,math,json,collections,statistics
p=Path('artifacts/crow-v15-contact-walk-stride-screen-20261112');sp=importlib.util.spec_from_file_location('a','artifacts/crow-v15-whole-gait-audit-20261111/audit.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
import sys
seed=int(sys.argv[1]); assert seed in (20261126,20261127)
base=Path('artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low') if seed==20261127 else Path('artifacts/crow-v15-toe-preload-second-seed-20261112/candidate-20261126-rebound-low')
paths={'baseline':base,'phase':p/f'candidate-{seed}-rebound-phase'}
def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(x*x for x in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def point(b,v):
 q=rot(b[3:7],v);return [q[i]+b[i] for i in range(3)]
def angle(h,k,v):
 x=[h[i]-k[i] for i in range(3)];y=[v[i]-k[i] for i in range(3)];d=sum(x[i]*y[i] for i in range(3))/(math.sqrt(sum(i*i for i in x))*math.sqrt(sum(i*i for i in y)));return math.degrees(math.acos(max(-1,min(1,d))))
out={}
for label,path in paths.items():
 sup,sub,b,com=a.load(path);records=[];rejections=collections.Counter();maxerr=0
 for e in range(8):
  for side,sd in enumerate(('left','right')):
   samples=[]
   for t in range(201,301):
    for k in range(16):
     z=sub[e,t,k];v=z[5+side*8:13+side*8];samples.append((z[2+side]>1e-7,sum(v[:6])>1e-7,sum(v[6:])>1e-7))
   active=[i for i,v in enumerate(samples) if v[0]];groups=[]
   for i in active:
    if not groups or i-groups[-1][-1]>3:groups.append([i])
    else:groups[-1].append(i)
   events=[]
   for g in groups:
    start,end=g[0],g[-1]
    if start==0 or end==len(samples)-1:rejections['partial_contact']+=1;events.append(None);continue
    front=[i for i in range(start,end+1) if samples[i][1]];rear=[i for i in range(start,end+1) if samples[i][2]]
    if not front or not rear:rejections['missing_toe_group']+=1;events.append(None);continue
    if (end-start+1)*.00125<.04:rejections['contact_under_40ms']+=1;events.append(None);continue
    events.append({'rear_first':rear[0],'front_last':front[-1]})
   for prev,nxt in zip(events,events[1:]):
    if not prev or not nxt:rejections['incomplete_pair']+=1;continue
    start,end=prev['front_last'],nxt['rear_first'];ts=[t for t in range(201,301) if start<(t-201)*16+15<end]
    if len(ts)<3:rejections['recovery_under_three_endpoints']+=1;continue
    knee=[];ankle=[]
    for t in ts:
     thigh=b[e,t,'crow_'+sd+'_thigh'];sh=b[e,t,'crow_'+sd+'_shank'];foot=b[e,t,'crow_'+sd+'_foot'];toe=b[e,t,'crow_'+sd+'_front_toe_2_proximal'];hip=point(thigh,[0,0,.031625]);k1=point(thigh,[0,0,-.031625]);k2=point(sh,[0,0,.031625]);an=point(sh,[0,0,-.031625]);an2=point(foot,[-.02875,0,.0092]);toeprox=point(toe,[-.009,0,0]);maxerr=max(maxerr,math.dist(k1,k2),math.dist(an,an2));knee.append(angle(hip,k1,an));ankle.append(angle(k1,an,toeprox))
    ki=min(range(len(ts)),key=lambda i:knee[i]);ai=min(range(len(ts)),key=lambda i:ankle[i]);delta=(ai-ki)*.02
    records.append({'lane':e,'side':sd,'recovery_seconds':(end-start)*.00125,'sample_steps':ts,'knee_min_deg':knee[ki],'ankle_min_deg':ankle[ai],'ankle_minus_knee_minimum_seconds':delta,'order':'knee_first' if delta>0 else 'ankle_first' if delta<0 else 'same_endpoint','knee_angles_deg':knee,'ankle_proxy_angles_deg':ankle})
 out[label]={'n_recoveries':len(records),'order_counts':dict(collections.Counter(x['order'] for x in records)),'rejections':dict(rejections),'max_joint_anchor_error_m':maxerr,'records':records}
out['scope']='Development seed, 4–6s. Segmented accepted foot contact with <=2 silent substeps bridged; reject partial runs, missing toe groups and <40ms runs. Recovery from last any front-toe impulse to next rear-toe impulse. 20ms endpoint FK, no low-pass filtering, at least three samples; minima may fall outside samples and equal endpoint means unresolved order. Knee angle uses hip/knee/ankle joint anchors; ankle proxy uses knee/ankle/second front-toe base, whose anatomical mapping to measured third digit is unverified. Angles are 3D versus magpie side video. Group fragmentation, proxy foot geometry, speed, surface and species differences prevent biological validation. Criteria are engineering choices, not measured stride definition.'
(p/f'phase-analysis-{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
for name in paths:print(name,out[name]['n_recoveries'],out[name]['order_counts'],out[name]['rejections'],'maxerr',out[name]['max_joint_anchor_error_m'])
