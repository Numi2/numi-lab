from pathlib import Path
import gzip,json,math,sys,collections
A=Path('artifacts/crow-v15-full-pronation-observer-20261113');P=A/sys.argv[1]
def scan(path):
 tar={};eff=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);
    if e>=6:tar[e,t]=list(map(float,z[3:]))
   elif z[0]=='CROW_PRONATION_EFFORT':
    t,sub,e=map(int,z[1:4]);
    if e>=6 and 650<=t<=1150:
     a=list(map(float,z[4:]));assert len(a)==14;eff[e].extend((abs(a[0]),abs(a[7])))
 return tar,eff
new,ne=scan(P/'run.stderr.gz');old,oe=scan(Path(sys.argv[2]));out={}
def smooth(a,b,x):
 x=max(0.,min(1.,(x-a)/(b-a)));return x*x*(3.-2.*x)
for e in (6,7):
 errors=[]
 for t in range(600,1201):
  sec=(t-2)*.02;start=7.42 if e==6 else 7.46;phase=2*math.pi*6.4*(sec-start);env=smooth(12.,12.5,sec)*(1-smooth(23.5,24.,sec));g=smooth(-.2,.2,.55*math.sin(phase)-.83*math.cos(phase));delta=(1 if e==6 else -1)*.04*env*g
  if (e,t) not in new or (e,t) not in old:continue
  errors.extend([abs((new[e,t][10]-old[e,t][10])-delta),abs((new[e,t][11]-old[e,t][11])+delta)])
 out[str(e)]={'target_samples':len(errors),'max_abs_target_delta_error_rad':max(errors,default=None),'baseline_saturation_fraction':sum(x>=.1499 for x in oe[e])/len(oe[e]) if oe[e] else None,'candidate_saturation_fraction':sum(x>=.1499 for x in ne[e])/len(ne[e]) if ne[e] else None,'observed_effort_values_each':len(ne[e]),'target_match':len(errors)==1202 and max(errors)<=.0003}
(P/'target-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
