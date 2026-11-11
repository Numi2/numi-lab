from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-com-velocity-balance-screen-20261111');B=Path('artifacts/crow-v15-grip-perturbed-acquisition-20261111')
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=np.array(toe[:3]);P=rot(toe[3:]);Q=rot(ground[3:]);b=np.array(ground[:3])+Q@np.array([0,0,-.055]);d=b-a;axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(P.T@u)*[.018,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(g))
def read(path):
 trace={};feet={};other={};horizontal={};poses={};grounds={};releases={};completed=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   v=line.split()
   if not v:continue
   if v[0]=='foot_release_requested':releases[int(v[1])]=int(v[2]);continue
   if v[0]=='case_result':completed.append(int(v[1]));continue
   if v[0]=='groundpose':grounds[int(v[1])]=[float(x) for x in v[3:]];continue
   if v[0] not in ('trace','foot_interval','other_interval','toe_horizontal','pose'):continue
   l,t=int(v[1]),int(v[3]);key=l,t
   if v[0]=='trace':trace[key]=v[4:]
   elif v[0]=='foot_interval':feet[key]=[float(v[4]),float(v[5])]
   elif v[0]=='other_interval':other[key]=float(v[4])
   elif v[0]=='toe_horizontal':horizontal.setdefault(key,{})[int(v[4])]=float(v[5])
   elif v[0]=='pose':poses.setdefault(key,[]).append([float(x) for x in v[5:]])
 return trace,feet,other,horizontal,poses,grounds,releases,completed
trace,feet,other,horiz,poses,grounds,releases,completed=read(A/'results.txt.gz');prior,*_=read(B/'results.txt.gz');mapping=[1,2,3,7];diff=[]
for l in range(1,5):
 for t in range(900):
  if trace.get((l,t))!=prior.get((mapping[l-1],t)):diff.append([l,t])
for l in range(5,17):
 if trace.get((l,0))!=prior.get((mapping[(l-1)%4],0)):diff.append([l,0])
def longest(v):
 best=cur=0
 for a in v:cur=cur+1 if a else 0;best=max(best,cur)
 return best
rows=[]
for l in range(1,17):
 flags=[];supp=[];gaps=[];miss=[];nonfoot=0;tiltmax=0.;speedmax=0.
 for t in range(900):
  key=l,t;v=trace.get(key);ft=feet.get(key);o=other.get(key);h=horiz.get(key,{});pp=poses.get(key,[]);ground=grounds.get(l)
  if v is None or ft is None or o is None or len(h)!=8 or len(pp)!=8 or ground is None:miss.append(t);flags.append(False);supp.append(False);continue
  tilt=math.acos(max(-1.,min(1.,1.-2.*(float(v[9])**2+float(v[10])**2))));speed=math.sqrt(sum(float(v[j])**2 for j in (8,11,12)));tiltmax=max(tiltmax,tilt);speedmax=max(speedmax,speed);nonfoot+=o>1e-7;support=min(ft)>.0002 and sum(ft)>=.5*.45004*9.81*.02 and o<=1e-7 and tilt<=.35 and speed<=.12;opposed=all(sum(h[k] for k in (base,base+1,base+2))>=.0002 and h[base+3]<=-.0002 for base in (0,4));supp.append(support);flags.append(support and opposed)
  if opposed:gaps.extend(gap(p,ground) for p in pp)
 first50=next((t-49 for t in range(49,len(flags)) if all(flags[t-49:t+1])),None);firstloss=next((t for t in range(1,len(supp)) if supp[t-1] and not supp[t]),None);v=trace.get((l,899));ok=not diff and not miss and l in completed and first50 is not None
 rows.append({'lane':l,'mode':(l-1)//4,'start':(l-1)%4,'samples_missing':miss,'first_50_consecutive_supported_opposition':first50,'longest_supported_opposition':longest(flags),'release_requested_step':releases.get(l),'first_support_loss_after_support':firstloss,'nonfoot_contact_samples':nonfoot,'minimum_opposed_pose_gap_m':min(gaps) if gaps else None,'terminal_root_x_m':float(v[6]) if v else None,'terminal_root_z_m':float(v[7]) if v else None,'maximum_tilt_rad':tiltmax,'maximum_speed_m_s':speedmax,'acquisition_pass':ok})
out={'parity_differences':diff[:30],'parity_difference_count':len(diff),'completed':completed,'lanes':rows,'scope':'four matched development starts; no complete grip, landing or route qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('control differences',len(diff),'completed',len(completed))
for mode in range(4):
 for x in rows[mode*4:(mode+1)*4]:print('mode',mode,'start',x['start'],'acquired',x['acquisition_pass'],'first50',x['first_50_consecutive_supported_opposition'],'release',x['release_requested_step'],'terminalz',x['terminal_root_z_m'],'nonfoot',x['nonfoot_contact_samples'],'gap',x['minimum_opposed_pose_gap_m'])
