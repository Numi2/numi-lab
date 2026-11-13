from pathlib import Path
import gzip,json,math,sys,statistics
A=Path('artifacts/crow-v15-phase-pronation-turn-20261113');P=A/sys.argv[1]
def scan(path):
 out={}
 with gzip.open(path,'rt') as f:
  for l in f:
   z=l.split()
   if z and z[0]=='CROW_PRONATION_EFFORT':
    t,sub,e=map(int,z[1:4])
    if e in (6,7) and 650<=t<=1150:
     k=e,t,sub;assert k not in out;out[k]=list(map(float,z[4:]));assert len(out[k])==14
 return out
new=scan(P/'run.stderr.gz');old=scan(Path(sys.argv[2]));assert set(new)==set(old) and len(new)==2*501*16
out={}
for e in (6,7):
 vals=[];targets=[];tracking=[]
 for k in new:
  if k[0]!=e:continue
  a,b=new[k],old[k];vals.append((a[2]-a[9])-(b[2]-b[9]));targets.append((a[1]-a[8])-(b[1]-b[8]));tracking.extend([a[1]-a[2],a[8]-a[9]])
 out[str(e)]={'samples':len(vals),'mean_differential_position_change_rad':statistics.mean(vals),'mean_differential_target_change_rad':statistics.mean(targets),'differential_position_change_range_rad':[min(vals),max(vals)],'joint_tracking_rms_rad':math.sqrt(statistics.mean(x*x for x in tracking)),'joint_tracking_mean_abs_rad':statistics.mean(map(abs,tracking))}
(P/'achieved-pronation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
