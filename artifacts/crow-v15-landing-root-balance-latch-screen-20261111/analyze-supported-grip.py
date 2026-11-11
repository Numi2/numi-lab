from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-landing-root-balance-latch-screen-20261111');P=A/'trial';ep=json.loads((A/'analysis.json').read_text())['first_episode_navigation'];s=collections.defaultdict(lambda:[0.]*6);states={}
for p in sorted(P.glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);states[x['lane'],x['step']]=x
   if not l.startswith('CROW_SCENE_CONTACT '):continue
   a=l.split();t,k,e,ba,bb=map(int,a[1:6]);
   if t>ep['steps_by_environment'][e]+1 or 38 not in (ba,bb):continue
   b=bb if ba==38 else ba;side=0 if b==10 or 17<=b<=24 else 1 if b==13 or 25<=b<=32 else -1
   if side<0:continue
   j,j1,j2=map(float,a[6:9]);nx,ny,nz=map(float,a[13:16]);tx,ty,tz=map(float,a[16:19]);sgn=1 if ba==38 else -1
   if b in list(range(17,23))+list(range(25,31)):s[e,t][side*2]+=sgn*nx*j
   if b in (23,24,31,32):s[e,t][side*2+1]+=sgn*nx*j
   s[e,t][4+side]+=sgn*(nz*j+tz*j1+(nx*ty-ny*tx)*j2)
def longest(seq):
 best=cur=0;prev=None
 for t in sorted(seq):cur=cur+1 if prev==t-1 else 1;best=max(best,cur);prev=t
 return best
rows=[]
for e in range(8):
 both=[];supported=[]
 for (lane,t),v in s.items():
  if lane!=e or not(v[0]>.0002 and v[1]<-.0002 and v[2]>.0002 and v[3]<-.0002):continue
  both.append(t);x=states.get((e,t))
  if x:
   q=x['q'];vel=x['v'];tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(a*a for a in vel[:3]))
   if min(v[4:])>.0002 and sum(v[4:])>=.5*.45004*9.81*.02 and tilt<=.35 and speed<=.12:supported.append(t)
 rows.append({'lane':e,'opposing_intervals':len(both),'opposing_longest_intervals':longest(both),'supported_stable_intervals':len(supported),'supported_stable_longest_intervals':longest(supported),'first_episode_end':ep['steps_by_environment'][e]})
(A/'supported-grip-analysis.json').write_text(json.dumps({'rows':rows,'criteria':'Both feet opposing front/rear normal impulses >0.0002 Ns, each foot upward impulse >0.0002 Ns, combined upward impulse >= half nominal weight over 0.02 s, root tilt <=0.35 rad, speed <=0.12 m/s. Contacts are aggregated over all physics substeps; state is sampled at control endpoints. This does not establish release or continuous geometric clearance.'},indent=2)+'\n');print(rows)
