from pathlib import Path
import collections,gzip,json,math,subprocess,sys
A=Path('artifacts/crow-v15-two-segment-toe-damping-walk-screen-20261111/d120');B=Path('build/crow-v15-two-segment-gait-geometry-20261111');baseline=Path('artifacts/crow-v15-two-segment-walk-screen-20261111');positive=Path('artifacts/crow-v15-two-segment-toe-damping-walk-screen-20261111/d120')
def source(p):
 state={};sub=collections.defaultdict(lambda:[0.]*5);cover=collections.Counter()
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);key=x['lane'],x['step']
    if 101<=x['step']<=301:state[key]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);
    if 101<=t<=301:
     cover[e,t,k]+=1
     for i in range(5):sub[e,t][i]+=float(a[6+i])
 return state,sub,cover
def compute(state,label):
 path=B/(label+'-states.txt')
 with path.open('w') as f:
  for (e,t),x in sorted(state.items()):f.write(f'{e} {t} '+' '.join(map(str,x['q']+x['v']))+'\n')
 p=subprocess.run([str(B/'probe'),str(path)],capture_output=True,text=True,check=True);out={}
 for line in p.stdout.splitlines():
  a=line.split();out[int(a[0]),int(a[1]),a[2]]=list(map(float,a[3:]))
 return out
def gap(p,half):
 x,y,z,w=p[3:];row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));return p[2]-sum(abs(a)*b for a,b in zip(row,half))
def names(side):return ['crow_'+side+'_foot']+['crow_'+side+'_'+n+'_'+segment for n in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for segment in ('proximal','distal')]
def bouts(flags,start_step):
 ans=[];start=None
 for i,z in enumerate(flags+[False]):
  if z and start is None:start=i
  if not z and start is not None:
   if i-start>=3:ans.append([start_step+start,start_step-1+i])
   start=None
 return ans
bs,_,_=source(baseline/'zero16.stderr.gz');bp=compute(bs,'baseline');replay=json.load(gzip.open(baseline/'zero16.replay.json.gz','rt'))['payload'];assert replay['environment']==1;inds={n:replay['body_names'].index(n)*13 for side in ('left','right') for n in names(side)};err=[]
for f in replay['frames']:
 t=f['step']+1
 if 101<=t<=201:
  for n,i in inds.items():
   p=bp[1,t,n];r=f['body_states'][i:i+7];err.extend(abs(a-b) for a,b in zip(p,r))
cal={'replay_environment':1,'compared_components':len(err),'maximum_absolute_pose_error':max(err) if err else None,'pass':len(err)==101*18*7 and max(err)<=1e-5};(A/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n');print('calibration',cal,flush=True)
if len(sys.argv)<2:raise SystemExit(0)
assert sys.argv[1]=='positive';assert cal['pass'];ss,sub,cov=source(positive/'stderr.gz');poses=compute(ss,'positive');rows=[]
for e in range(8):
 sides={};events=[]
 for side,idx in [('left',0),('right',1)]:
  flags=[];gaps=[]
  first=201
  for t in range(first,first+101):
   ns=names(side);avail=all((e,t,n) in poses for n in ns);g=min((gap(poses[e,t,n],[.01,.01725,.0092] if n.endswith('_foot') else [.009,.0025,.0092]) for n in ns),default=-1.) if avail else -1.;gaps.append(g);v=sub[e,t];flags.append(avail and g>.001 and v[2+idx]<=1e-7 and v[1-idx]>=.5*.45004*9.81*.02 and v[4]<=1e-7 and all(cov[e,t,k]==1 for k in range(16)))
  b=bouts(flags,first);sides[side]={'bouts':b,'qualified_intervals':sum(flags),'maximum_minimum_gap_m':max(gaps)};events.extend((x[0],side) for x in b)
 events.sort();alternating=all(events[i][1]!=events[i-1][1] for i in range(1,len(events)))
 rows.append({'lane':e,'sides':sides,'swing_sequence':events,'alternating':alternating,'pass':len(sides['left']['bouts'])>=3 and len(sides['right']['bouts'])>=3 and alternating})
report={'calibration':cal,'lanes':rows,'scope':'endpoint vertical clearance plus all-substep impulse, development conditions only'};(A/'positive-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
