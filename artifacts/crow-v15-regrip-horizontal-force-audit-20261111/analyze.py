from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-regrip-horizontal-force-audit-20261111');B=Path('artifacts/crow-v15-single-foot-release-20261111');trace={};reference={};horiz={};poses={};grounds={};allc={};oldopp={}
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=np.array(toe[:3]);P=rot(toe[3:]);Q=rot(ground[3:]);b=np.array(ground[:3])+Q@np.array([0,0,-.055]);d=b-a;axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(P.T@u)*[.018,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(g))
with gzip.open(B/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if v and v[0]=='trace':reference[int(v[3])]=v[4:]
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='trace':trace[int(v[3])]=v[4:]
  elif v[0]=='toe_horizontal':horiz.setdefault(int(v[3]),{})[int(v[4])]=float(v[5])
  elif v[0]=='interval_contact':oldopp.setdefault(int(v[3]),{})[int(v[4])]=float(v[5])
  elif v[0]=='pose':poses.setdefault(int(v[3]),[]).append([float(z) for z in v[5:]])
  elif v[0]=='groundpose':grounds=[float(z) for z in v[3:]]
  elif v[0]=='allcontact':
   t=int(v[3]);an,bn=v[4:6];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;up=max((1 if body==bn else -1)*nz*imp,0);d=allc.setdefault(t,{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):d[side]+=up
   else:d['other']+=imp
diff=[t for t in range(1500) if trace.get(t)!=reference.get(t)];assert len(trace)==len(reference)==1500 and not diff
records=[];gaps=[]
for t in range(1500):
 v=trace[t];h=horiz.get(t,{});p=poses.get(t,[]);o=oldopp.get(t,{});c=allc.get(t,{'left':0.,'right':0.,'other':0.});q0,q1=float(v[9]),float(v[10]);tilt=math.acos(max(-1.,min(1.,1.-2*(q0*q0+q1*q1))));speed=math.sqrt(sum(float(v[j])**2 for j in (8,11,12)));support=min(c['left'],c['right'])>.0002/16 and c['left']+c['right']>=.5*.45004*9.81*.02/16 and c['other']<=1e-7 and tilt<=.35 and speed<=.12;force=len(h)==8 and all(sum(h[k] for k in (base,base+1,base+2))>=.0002 and h[base+3]<=-.0002 for base in (0,4));old=len(o)==8 and all(sum(o[k] for k in (base,base+1,base+2))>.0002 and o[base+3]>.0002 for base in (0,4));records.append((force,support,old));
 if force:
  assert len(p)==8 and len(grounds)==7;gaps.extend(gap(x,grounds) for x in p)
def longest(v):
 best=cur=0
 for x in v:cur=cur+1 if x else 0;best=max(best,cur)
 return best
r={'reference_samples':len(trace),'reference_differences':len(diff),'last_100_horizontal_opposition':sum(x[0] for x in records[-100:]),'last_100_horizontal_opposition_and_support':sum(x[0] and x[1] for x in records[-100:]),'last_100_original_normal_cone_gate':sum(x[2] for x in records[-100:]),'longest_horizontal_opposition_and_support':longest([x[0] and x[1] for x in records]),'minimum_gap_m_during_horizontal_opposition':min(gaps) if gaps else None,'force_threshold_Ns_per_group':.0002,'scope':'signed normal plus friction impulse over all substeps; independent diagnostic, does not change original acceptance gate'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
