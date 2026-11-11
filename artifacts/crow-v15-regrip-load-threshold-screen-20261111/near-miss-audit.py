from pathlib import Path
import gzip,json,math

def rot(q):
 x,y,z,w=q
 return [[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)],[2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],[2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def gap(toe,ground):
 a=toe[:3];P=rot(toe[3:]);Q=rot(ground[3:]);b=[ground[i]-.055*Q[2][i] for i in range(3)];d=[b[i]-a[i] for i in range(3)];axes=P+Q+[cross(p,q) for p in P for q in Q];g=[]
 for u in axes:
  n=math.sqrt(dot(u,u))
  if n>1e-8:
   u=[x/n for x in u];g.append(abs(dot(d,u))-sum(abs(dot(p,u))*z for p,z in zip(P,[.018,.0025,.0092]))-sum(abs(dot(q,u))*z for q,z in zip(Q,[.025,.72,.055])))
 return max(g)
A=Path('artifacts/crow-v15-regrip-load-threshold-screen-20261111');p=Path('artifacts/crow-v15-regrip-load-threshold-screen-20261111/results.txt.gz');lane=3;ground=None;poses={};force={};traces={};contacts={}
with gzip.open(p,'rt') as f:
 for line in f:
  v=line.split()
  if len(v)<3 or v[0] not in ('groundpose','pose','toe_horizontal','trace','allcontact') or v[1]!=str(lane):continue
  if v[0]=='groundpose':ground=[float(x) for x in v[3:]];continue
  t=int(v[3])
  if v[0]=='pose':poses.setdefault(t,[]).append((v[4],[float(x) for x in v[5:]]))
  elif v[0]=='toe_horizontal':force.setdefault(t,{})[int(v[4])]=float(v[5])
  elif v[0]=='trace':traces[t]=v[4:]
  elif v[0]=='allcontact':
   an,bn=v[4:6];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;up=max((1 if body==bn else -1)*nz*imp,0);c=contacts.setdefault(t,{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):c[side]+=up
   else:c['other']+=imp
assert len(poses)==len(traces)==1500 and ground is not None
worst=[];terminal=[]
for t in range(1500):
 h=force.get(t,{});p=poses[t];opp=len(h)==8 and all(sum(h[k] for k in (base,base+1,base+2))>=.0002 and h[base+3]<=-.0002 for base in (0,4));v=traces[t];c=contacts.get(t,{'left':0.,'right':0.,'other':0.});tilt=math.acos(max(-1.,min(1.,1.-2*(float(v[9])**2+float(v[10])**2))));speed=math.sqrt(sum(float(v[j])**2 for j in (8,11,12)));support=min(c['left'],c['right'])>.0002/16 and c['left']+c['right']>=.5*.45004*9.81*.02/16 and c['other']<=1e-7 and tilt<=.35 and speed<=.12
 if opp:
  for toe,pos in p:worst.append({'step':t,'toe':toe,'gap_m':gap(pos,ground),'forces_Ns':h,'contact':c,'trace':v})
 if t>=1400 and not (opp and support):terminal.append({'step':t,'opposition':opp,'support':support,'forces_Ns':h,'contact':c,'tilt_rad':tilt,'speed_m_s':speed,'trace':v})
worst.sort(key=lambda x:x['gap_m']);out={'lane':lane,'worst_10':worst[:10],'terminal_violations':terminal,'scope':'post-hoc endpoint geometry; support sampled at last substep, force all 16; one developmental start'};(A/'near-miss-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
