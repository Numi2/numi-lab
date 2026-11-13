from pathlib import Path
import gzip,json,math,statistics
A=Path(__file__).resolve().parent
rows={}
for line in (A/'results.tsv').read_text().splitlines():
 x=line.split();key=tuple(map(int,x[:3]));assert key not in rows;rows[key]=list(map(float,x[3:]));assert len(rows[key])==9 and all(map(math.isfinite,rows[key]))
poses={};events={}
with gzip.open(A.parent/'crow-v15-hiproll-grip-release-20261113/trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x[0]=='pose' and x[3] in ('crow_left_foot','crow_right_foot'):poses[(int(x[1]),int(x[2]),x[3])]=list(map(float,x[4:7]))
  if x[0]=='event':events[(int(x[1]),int(x[2]))]=[float(v) for v in x[3:]]
keys={(a,b) for a,b,c in rows if c==0};assert len(keys)==494 and len(rows)==5*len(keys) and len(poses)==2*len(keys)
errors=[]
for lane,t in keys:
 for name,ofs in [('crow_left_foot',3),('crow_right_foot',6)]:errors.extend(abs(a-b) for a,b in zip(rows[lane,t,0][ofs:ofs+3],poses[lane,t,name]))
assert max(errors)<1e-6
out={'samples':len(keys),'foot_coordinate_checks':len(errors),'foot_max_abs_error_m':max(errors),'lanes':{}}
for lane in (0,1):
 ks=sorted(t for e,t in keys if e==lane); rel=[t for t in ks if events[lane,t][0]];row={'release_start':min(rel) if rel else None,'release_end':max(rel) if rel else None,'moments':{}}
 for t in [100,150,180,200,220,230,240]:
  if t not in ks:continue
  x=rows[lane,t,0];r={ 'com_m':x[:3],'left_foot_m':x[3:6],'right_foot_m':x[6:9], 'com_minus_right_foot_m':[x[i]-x[i+6] for i in range(3)],'release':bool(events[lane,t][0])}
  for label,p,m in [('left',1,2),('right',3,4)]:
   a=rows[lane,t,p];b=rows[lane,t,m];r[label+'_d_com_minus_right_foot_m_per_rad']=[((a[i]-a[i+6])-(b[i]-b[i+6]))/.02 for i in range(3)]
  row['moments'][str(t)]=r
 out['lanes'][str(lane)]=row
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
