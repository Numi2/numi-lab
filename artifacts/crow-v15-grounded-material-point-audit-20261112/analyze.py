from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess,sys
A=Path('artifacts/crow-v15-grounded-material-point-audit-20261112');P=Path('artifacts/crow-v15-knee-warp-contact-timing-audit-20261112');arm=sys.argv[1];assert arm in ('baseline','phase');D=P/f'candidate-20261127-rebound-{arm}-repaired';dt=.00125;tol=.0005

def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(t*t for t in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def world(b,p):
 x=rot(b[3:7],p);return [b[i]+x[i] for i in range(3)]
def local(b,p):return rot([-b[3],-b[4],-b[5],b[6]],[p[i]-b[i] for i in range(3)])
def velocity(b,p):
 r=rot(b[3:7],p);w=b[10:13];return [b[7]+w[1]*r[2]-w[2]*r[1],b[8]+w[2]*r[0]-w[0]*r[2],b[9]+w[0]*r[1]-w[1]*r[0]]
def check():
 s=math.sqrt(.5);b=[1.,2.,3.,0.,0.,s,s, .3,.4,.5,0.,0.,2.];p=[.01,0.,0.];w=world(b,p)
 assert math.dist(w,[1.,2.01,3.])<1e-12 and math.dist(local(b,w),p)<1e-12
 assert math.dist(velocity(b,p),[.28,.4,.5])<1e-12
 # A non-translating 2 rad/s spin has the analytical fixed-point velocity.
 eps=1e-5;before=b.copy();after=b.copy();before[3:7]=[0.,0.,math.sin((math.pi/2-2*eps)/2),math.cos((math.pi/2-2*eps)/2)];after[3:7]=[0.,0.,math.sin((math.pi/2+2*eps)/2),math.cos((math.pi/2+2*eps)/2)]
 fd=[(world(after,p)[i]-world(before,p)[i])/(2*eps) for i in range(3)]
 assert math.dist(fd,[-.02,0.,0.])<1e-9
 b2=b.copy();b2[0]+=.3*dt;b2[1]+=.4*dt;b2[2]+=.5*dt
 assert math.dist([world(b2,p)[i]-w[i] for i in range(3)],[.3*dt,.4*dt,.5*dt])<1e-12
check();Q={};contacts=collections.defaultdict(list)
with gzip.open(D/'replay.json.gz','rt') as f:names=json.load(f)['payload']['body_names']
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_SYNC_KIN':
   t,k,e=map(int,z[1:4]);assert e==4 and len(z)==85;idx=(t-201)*16+k;assert idx not in Q;Q[idx]=list(map(float,z[4:]))
  elif z[0]=='CROW_SYNC_CONTACT':
   t,k,e,a,b=map(int,z[1:6]);assert e==4;idx=(t-201)*16+k;body=a if a<35 else b
   if body in (11,15) or 19<=body<=34:contacts[idx,body].append((float(z[6]),list(map(float,z[7:10]))))
assert sorted(Q)==list(range(1600))
payload=''.join('4 '+str(k)+' '+' '.join(map(str,v))+'\n' for k,v in sorted(Q.items()))
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};proc=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=payload,text=True,capture_output=True,check=True,env=env)
bodies={}
for line in proc.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body':
  i=int(z[2]);name=z[3]
  if name in names and (name.endswith('_foot') or '_toe_' in name):bodies[i,name]=list(map(float,z[4:]))
assert len(bodies)==1600*18,len(bodies)
active={};tracks=[];discard=0;available=0

def finish(body):
 global discard
 tr=active.pop(body);n=tr['samples']
 if n<4:discard+=1;return
 tr.pop('last_position');tr.pop('last_velocity');tr.pop('anchor');tr['duration_s']=(n-1)*dt;tr['displacement_m']=math.hypot(tr['delta'][0],tr['delta'][1]);tr['velocity_integration_error_m']=math.dist(tr['delta'],tr['integral']);tracks.append(tr)
for i in range(1600):
 for body in (11,15,*range(19,35)):
  points=contacts.get((i,body),[]);b=bodies[i,names[body]]
  if points:available+=1
  if body in active:
   tr=active[body]
   if not points:finish(body)
   else:
    dist=min(math.dist(local(b,c[1]),tr['anchor']) for c in points)
    if dist>tol:finish(body)
    else:
     p=world(b,tr['anchor']);v=velocity(b,tr['anchor']);dx=[p[j]-tr['last_position'][j] for j in range(3)];tr['path_m']+=math.hypot(dx[0],dx[1]);tr['delta']=[tr['delta'][j]+dx[j] for j in range(3)];tr['integral']=[tr['integral'][j]+(tr['last_velocity'][j]+v[j])*.5*dt for j in range(3)];tr['samples']+=1;tr['max_local_contact_distance_m']=max(tr['max_local_contact_distance_m'],dist);tr['last_position']=p;tr['last_velocity']=v
  if points and body not in active:
   c=max(points,key=lambda x:x[0]);anchor=local(b,c[1]);active[body]={'body':names[body],'start_substep':i,'anchor':anchor,'last_position':world(b,anchor),'last_velocity':velocity(b,anchor),'samples':1,'path_m':0.,'delta':[0.,0.,0.],'integral':[0.,0.,0.],'max_local_contact_distance_m':0.}
for body in list(active):finish(body)
def summarize(xs):
 if not xs:return {'episodes':0}
 ds=sorted(x['displacement_m'] for x in xs);dur=[x['duration_s'] for x in xs];errors=sorted(x['velocity_integration_error_m'] for x in xs)
 return {'episodes':len(xs),'body_substeps_covered':sum(x['samples'] for x in xs),'total_episode_duration_s':sum(dur),'median_duration_ms':1000*statistics.median(dur),'median_endpoint_displacement_mm':1000*statistics.median(ds),'p90_endpoint_displacement_mm':1000*ds[int(.9*(len(ds)-1))],'maximum_endpoint_displacement_mm':1000*max(ds),'fraction_over_1mm':sum(d>.001 for d in ds)/len(ds),'median_velocity_consistency_error_mm':1000*statistics.median(errors),'p90_velocity_consistency_error_mm':1000*errors[int(.9*(len(errors)-1))]}
out={'arm':arm,'seed':20261127,'lane':4,'substep_seconds':dt,'contact_match_tolerance_m':tol,'min_states':4,'synthetic_transform_checks':'passed','available_body_substeps':available,'discarded_short_episodes':discard,'overall':summarize(tracks),'left':summarize([x for x in tracks if x['body'].startswith('crow_left_')]),'right':summarize([x for x in tracks if x['body'].startswith('crow_right_')]),'by_body':{n:summarize([x for x in tracks if x['body']==n]) for n in sorted(set(x['body'] for x in tracks))},'tracks':tracks,'limits':'One used development lane. Matching a contact within 0.5 mm of the initial local anchor does not show that the same material point remains in contact. Solver contact and post-substep q/v may have staggered timing. Displacement is a local-patch proximity proxy, not exact slip, stance-integrated slip, biological validity, or independent replication.'}
(A/f'analysis-{arm}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('tracks','by_body')},indent=2))
