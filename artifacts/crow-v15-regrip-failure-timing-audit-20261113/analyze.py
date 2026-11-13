import gzip,hashlib,json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); expected='9ad86cb443dab58ab480230dece348dfa0a9e82fdb6da9382dab2783cd47c6bc'; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
steps={}; events={}
with gzip.open(p,'rt') as f:
 for line in f:
  x=line.split()
  if not x or x[0] not in ('step','event'): continue
  e,t=int(x[1]),int(x[2]); key=e,t
  if x[0]=='step':
   i=x.index('q'); j=x.index('v'); q=list(map(float,x[i+1:j]));v=list(map(float,x[j+1:])); a,b,c,w=q[3:7]; pitch=math.asin(max(-1.,min(1.,2*(w*b-c*a))));steps[key]={'x_m':q[0],'z_m':q[2],'pitch_rad':pitch,'vx_m_s':v[0],'pitch_rate_rad_s':v[4]}
  else:
   z=list(map(float,x[3:]));events[key]={'release':bool(z[0]),'support_count':z[1],'up_left':z[2],'up_right':z[3],'front_left_x':z[4],'rear_left_x':z[5],'front_right_x':z[6],'rear_right_x':z[7],'finished':bool(z[8]),'clear_count':z[9],'clearance_m':z[10]}
rows=[]
for e in (0,1):
 ts=sorted(t for lane,t in events if lane==e); finish=next(t for t in ts if events[e,t]['finished']); release=next(t for t in ts if events[e,t]['release']);fall=next(t for t in ts if steps[e,t]['z_m']<.1 and t>finish)
 picks=sorted(set([release,finish,min(finish+10,fall),min(finish+30,fall),fall]));rows.append({'lane':e,'release_step':release,'return_latched_step':finish,'first_root_below_0_1m':fall,'samples':[{'step':t,**steps[e,t],**events[e,t]} for t in picks]})
print(json.dumps({'trace_sha256':expected,'lanes':rows,'limitations':'Constructed stationary perch test, sampled endpoint contact and kinematics; chronology does not establish causal mechanism or airborne landing.'},indent=2))
