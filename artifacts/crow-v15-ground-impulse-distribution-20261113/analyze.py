from pathlib import Path
import collections,gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-ground-impulse-distribution-20261113';A.mkdir(exist_ok=True)
p=R/'crow-v15-pronation-phase-screen-20261113/push-wing-20261126-rebound-phase/run.stderr.gz';D=collections.defaultdict(lambda:{'normal':0.,'absolute_tangent_bound':0.,'contacts':0,'above_ratio_0_59':0,'normal_above_ratio_0_59':0.})
with gzip.open(p,'rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0]!='CROW_GROUND_CONTACT':continue
  t,k,e,a,b=map(int,z[1:6]);
  if e<2 or t>365:continue
  body=a if b==35 else b
  data=list(map(float,z[8:]));assert len(data)==16
  nx,ny,nz=data[4:7];tx,ty,tz=data[7:10];mu0,mu1,Jn,Jt,Jb,pre=data[10:]
  assert Jn>0
  key=(e,'348-355' if t<=355 else '356-365',body);d=D[key];ratio=math.hypot(Jt,Jb)/Jn
  d['normal']+=Jn;d['absolute_tangent_bound']+=math.hypot(Jt,Jb);d['contacts']+=1
  if ratio>=.59:d['above_ratio_0_59']+=1;d['normal_above_ratio_0_59']+=Jn
rows=[]
for (e,window,body),v in sorted(D.items()):rows.append({'lane':e,'ticks':window,'ground_body':body,**v})
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc distribution of positive ground-contact normal impulses and tangential magnitudes by body. Contact friction parameters are recorded, but the 0.59 ratio is descriptive and not proof of sliding or friction saturation; sign, basis, body velocity and solver law require separate interpretation. Correlated used-seed lanes only.','rows':rows},indent=2)+'\n')
for e in range(2,8):
 for w in ('348-355','356-365'):
  x=[r for r in rows if r['lane']==e and r['ticks']==w]; print(e,w,'body normal',[(r['ground_body'],round(r['normal'],4)) for r in x],'high_ratio_normal',round(sum(r['normal_above_ratio_0_59'] for r in x),4),'total',round(sum(r['normal'] for r in x),4))
