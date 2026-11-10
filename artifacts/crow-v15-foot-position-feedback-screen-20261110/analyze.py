from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-foot-position-feedback-screen-20261110');O=Path('artifacts/crow-v15-bilateral-perturbation-screen-20261110');trace={};allc={};toe={};outcomes={};old={};poses={};grounds={}
with gzip.open(O/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if v and v[0]=='trace' and int(v[1]) in (0,1) and int(v[2])==4:old[(int(v[1]),int(v[2]),int(v[3]))]=v[4:]
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='pose':poses.setdefault((int(v[1]),int(v[2]),int(v[3])),[]).append([float(z) for z in v[5:]])
  elif v[0]=='groundpose':grounds[(int(v[1]),int(v[2]))]=[float(z) for z in v[3:]]
  elif v[0]=='trace':trace[(int(v[1]),int(v[2]),int(v[3]))]=v[4:]
  elif v[0]=='case_result':outcomes[(int(v[1]),int(v[2]))]=' '.join(v[3:])
  elif v[0]=='contact' and int(v[2])==0:
   i,a,t=int(v[1]),int(v[2]),int(v[3]);side='left' if '_left_' in v[4] else 'right';part='rear' if 'rear_toe' in v[4] else 'front';nx=float(v[5]);imp=float(v[7]);d=toe.setdefault((i,a,t),{s:{'front':0.,'rear':0.} for s in ('left','right')})
   if (part=='front' and nx<-.5) or (part=='rear' and nx>.5):d[side][part]+=imp
  elif v[0]=='interval_contact' and int(v[2])>=1:
   i,a,t,k=int(v[1]),int(v[2]),int(v[3]),int(v[4]);side='left' if k<4 else 'right';part='rear' if k%4==3 else 'front';d=toe.setdefault((i,a,t),{s:{'front':0.,'rear':0.} for s in ('left','right')});d[side][part]+=float(v[5])
  elif v[0]=='allcontact':
   i,a,t=int(v[1]),int(v[2]),int(v[3]);an,bn=v[4],v[5];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;sgn=1. if body==bn else -1.;up=max(sgn*nz*imp,0.);d=allc.setdefault((i,a,t),{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):d[side]+=up
   else:d['other']+=imp
keys=set(old)|{k for k in trace if k[0] in (0,1) and k[1]==4};parity_diff=[k for k in sorted(keys) if old.get(k)!=trace.get(k)];rows=[]
def longest(xs):
 best=cur=0
 for x in xs:cur=cur+1 if x else 0;best=max(best,cur)
 return best
for i in range(2):
 for a in (4,5,6):
  rec=[]
  for t in range(200):
   v=trace.get((i,a,t));
   if v is None:continue
   phase=int(v[0]);qx,qy=float(v[9]),float(v[10]);vel=[float(v[8]),float(v[11]),float(v[12])];tilt=math.acos(max(-1.,min(1.,1.-2.*(qx*qx+qy*qy))));speed=math.sqrt(sum(q*q for q in vel));c=allc.get((i,a,t),{'left':0.,'right':0.,'other':0.});p=toe.get((i,a,t),{s:{'front':0.,'rear':0.} for s in ('left','right')});grip=all(p[s][k]>(.0000125 if a==0 else .0002) for s in p for k in ('front','rear'));support=min(c['left'],c['right'])>(.0002/[16,16,16,16,16,16,16][a]) and c['left']+c['right']>=.5*.45004*9.81*(.02/[16,16,16,16,16,16,16][a]) and c['other']<=1e-7 and tilt<=.35 and speed<=.12;rec.append({'step':t,'grip':grip,'support':support,'phase':phase,'max_toe_abs_rad':max(map(abs,map(float,v[13:21]))),'up':c['left']+c['right'],'other':c['other'],'tilt':tilt,'speed':speed})
  grips=[r['step'] for r in rec if r['grip']];released=[r['step'] for r in rec if r['phase']==4 and not r['grip']];tail=rec[-25:];passed=not parity_diff and len(rec)==200 and longest([r['grip'] for r in rec])>=3 and longest([r['grip'] and r['support'] for r in rec])>=3 and bool(released) and len(tail)==25 and all(r['support'] and r['max_toe_abs_rad']<0.05 for r in tail);rows.append({'case':i,'arm':a,'outcome':outcomes.get((i,a),'missing'),'samples':len(rec),'longest_opposing_grip':longest([r['grip'] for r in rec]),'longest_supported_opposing_grip':longest([r['grip'] and r['support'] for r in rec]),'longest_supported':longest([r['support'] for r in rec]),'terminal_25_supported':len(tail)==25 and all(r['support'] for r in tail),'first_release_without_grip':released[0] if released else None,'terminal_max_toe_abs_rad':max((r['max_toe_abs_rad'] for r in tail),default=None),'strict_pass':passed})
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=np.array(toe[:3]);A=rot(toe[3:]);B=rot(ground[3:]);b=np.array(ground[:3])+B@np.array([0,0,-.055]);d=b-a;axes=[*A.T,*B.T]+[np.cross(A[:,i],B[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(A.T@u)*[.018,.0025,.0092])-sum(abs(B.T@u)*[.025,.72,.055]))
 return float(max(g))
for row in rows:
 i,a=row['case'],row['arm']; ds=[];missing=0
 for t in range(200):
  d=toe.get((i,a,t),{s:{'front':0.,'rear':0.} for s in ('left','right')})
  if not all(d[z][k]>(.0000125 if a==0 else .0002) for z in d for k in ('front','rear')):continue
  pp=poses.get((i,a,t),[]);gg=grounds.get((i,a))
  if len(pp)!=8 or gg is None:missing+=1;continue
  ds.extend(gap(p,gg) for p in pp)
 row['opposed_grip_pose_samples']=len(ds)//8;row['missing_grip_pose_samples']=missing;row['minimum_grip_sat_gap_m']=min(ds) if ds else None
out={'baseline_parity_exact':not parity_diff,'baseline_parity_differences':parity_diff[:20], 'rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('Baseline parity:',not parity_diff,'differences',len(parity_diff))
for a in (4,5,6):
 z=[r for r in rows if r['arm']==a];print('arm',a,'strict pass',sum(r['strict_pass'] for r in z),'opposed 3+',sum(r['longest_opposing_grip']>=3 for r in z),'terminal support',sum(r['terminal_25_supported'] for r in z),'minimum grip gap m',min((r['minimum_grip_sat_gap_m'] for r in z if r['minimum_grip_sat_gap_m'] is not None),default=None),'missing pose steps',sum(r['missing_grip_pose_samples'] for r in z))
