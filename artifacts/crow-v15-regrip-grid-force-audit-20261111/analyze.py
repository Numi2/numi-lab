from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-regrip-grid-force-audit-20261111'); B=Path('artifacts/crow-v15-regrip-regulation-grid-20261111')
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=np.array(toe[:3]);P=rot(toe[3:]);Q=rot(ground[3:]);b=np.array(ground[:3])+Q@np.array([0,0,-.055]);d=b-a;axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(P.T@u)*[.018,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(g))
ref={}; trace={};horiz={};poses={};grounds={};allc={};oldopp={}
with gzip.open(B/'results.txt.gz','rt') as f:
 for line in f:
  v=line.split()
  if v and v[0]=='trace':ref[(int(v[1]),int(v[3]))]=v[4:]
with gzip.open(A/'results.txt.gz','rt') as f:
 for line in f:
  v=line.split()
  if not v:continue
  e=int(v[1]) if v[0] in ('trace','toe_horizontal','interval_contact','pose','groundpose','allcontact') else 0
  if v[0]=='groundpose':grounds[e]=[float(z) for z in v[3:]];continue
  if not e:continue
  t=int(v[3]);key=e,t
  if v[0]=='trace':trace[key]=v[4:]
  elif v[0]=='toe_horizontal':horiz.setdefault(key,{})[int(v[4])]=float(v[5])
  elif v[0]=='interval_contact':oldopp.setdefault(key,{})[int(v[4])]=float(v[5])
  elif v[0]=='pose':poses.setdefault(key,[]).append([float(z) for z in v[5:]])
  elif v[0]=='allcontact':
   an,bn=v[4:6];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;up=max((1 if body==bn else -1)*nz*imp,0);d=allc.setdefault(key,{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):d[side]+=up
   else:d['other']+=imp
assert len(trace)==len(ref)==24000 and trace==ref, f'trace parity failed: {len(trace)} {len(ref)}'
def longest(v):
 best=cur=0
 for x in v:cur=cur+1 if x else 0;best=max(best,cur)
 return best
out=[]
for e in range(1,17):
 rec=[];gaps=[]
 for t in range(1500):
  key=e,t;v=trace[key];h=horiz.get(key,{});p=poses.get(key,[]);o=oldopp.get(key,{});c=allc.get(key,{'left':0.,'right':0.,'other':0.});q0,q1=float(v[9]),float(v[10]);tilt=math.acos(max(-1.,min(1.,1.-2*(q0*q0+q1*q1))));speed=math.sqrt(sum(float(v[j])**2 for j in (8,11,12)));support=min(c['left'],c['right'])>.0002/16 and c['left']+c['right']>=.5*.45004*9.81*.02/16 and c['other']<=1e-7 and tilt<=.35 and speed<=.12;force=len(h)==8 and all(sum(h[k] for k in (base,base+1,base+2))>=.0002 and h[base+3]<=-.0002 for base in (0,4));old=len(o)==8 and all(sum(o[k] for k in (base,base+1,base+2))>.0002 and o[base+3]>.0002 for base in (0,4));rec.append((force,support,old))
  if force:
   assert len(p)==8 and len(grounds[e])==7;gaps.extend(gap(x,grounds[e]) for x in p)
 out.append({'lane':e,'terminal_horizontal_supported':sum(x[0] and x[1] for x in rec[-100:]),'terminal_original_cone':sum(x[2] for x in rec[-100:]),'longest_horizontal_supported':longest([x[0] and x[1] for x in rec]),'minimum_gap_m_during_opposition':min(gaps) if gaps else None})
r={'exact_trace_parity_lanes':16,'samples_per_lane':1500,'threshold_Ns':.0002,'lanes':out,'scope':'diagnostic only; no qualification or prospective criterion change'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
