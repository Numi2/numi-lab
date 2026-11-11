from pathlib import Path
from collections import defaultdict,Counter
import gzip,json,statistics
D=Path('artifacts/crow-v15-perch-hip-feedback-repeat-20261111');P=D/'trial';lanes=[3,4,7]
s=json.loads((D/'supported-grip-analysis.json').read_text())['rows'];windows={e:(5000-s[e]['supported_stable_longest_intervals']+1,5000) for e in lanes};ts=defaultdict(float);others=defaultdict(float)
for p in sorted(P.glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for l in f:
   if not l.startswith('CROW_SCENE_CONTACT '):continue
   a=l.split();t,k,e,ba,bb=map(int,a[1:6]);
   if e not in windows or not windows[e][0]<=t<=windows[e][1] or 38 not in (ba,bb):continue
   b=bb if ba==38 else ba;j,j1,j2=map(float,a[6:9]);nx,ny,nz=map(float,a[13:16]);tx,ty,tz=map(float,a[16:19]);sgn=1 if ba==38 else -1
   impulse=sgn*(nz*j+tz*j1+(nx*ty-ny*tx)*j2)
   if b==10 or b==13 or 17<=b<=32:ts[e,t]+=impulse
   else:others[e]+=impulse
nom=.450039993491373*9.81*.02;rows=[]
for e in lanes:
 start,end=windows[e];ratios=[ts[e,t]/nom for t in range(start,end+1)];win=[sum(ratios[i:i+100])/100 for i in range(len(ratios)-99)]
 rows.append({'lane':e,'window_inclusive':[start,end],'duration_s':len(ratios)*.02,'mean_foot_to_nominal_gravity_impulse':statistics.mean(ratios),'minimum_100_interval_mean':min(win),'maximum_100_interval_mean':max(win),'nonfoot_perch_impulse_total_ns':others[e]})
(D/'support-windows-analysis.json').write_text(json.dumps({'rows':rows,'limitations':'Window average is useful evidence, not an external-force and whole-body momentum closure; upright and speed are sampled at endpoints; teacher is on; no release or held-out layout.'},indent=2)+'\n')
for r in rows:print(r)
