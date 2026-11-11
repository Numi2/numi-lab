from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-contact-pitch-feedback-control-20261111/autonomous-20261126');d=json.loads((A/'run.stdout').read_text());first=d['first_episode_navigation'];ends=first['steps_by_environment'];coverage=collections.Counter();values=collections.defaultdict(lambda:[0.]*7);overflow=[]
with gzip.open(A/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append((e,t,k))
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,a,b=map(int,z[1:6])
   if t>ends[e] or 40 not in (a,b):continue
   body=b if a==40 else a;sgn=1. if a==40 else -1.;normal,tu,tv=map(float,z[6:9]);nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]);up=sgn*(nz*normal+tz*tu+(nx*ty-ny*tx)*tv)
   side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1
   v=values[e,t]
   if side<0:v[6]+=normal;continue
   v[side]+=up
   if side==0:v[2]+=normal
   if body>=19:
    digit=((body-19)%8)//2
    if side==1:v[4 if digit==3 else 3]+=sgn*nx*normal
    elif side==0 and digit==3:v[5]+=sgn*nx*normal
nom=.45006*9.81*.02
rows=[]
for e,end in enumerate(ends):
 start=None;best=0;run=0;first_run=None
 for t in range(1,end+1):
  x=values[e,t];complete=all(coverage[e,t,k]==1 for k in range(16));right=x[1]>=.5*nom and x[3]>1e-6 and x[4]<-1e-6 and x[6]<=1e-7
  clear=complete and x[2]<=1e-7 and right
  run=run+1 if clear else 0;best=max(best,run)
  if run>=10 and first_run is None:first_run=t-9
 perch=json.loads((A/'perch-analysis.json').read_text())['rows'][e]
 rows.append({'lane':e,'first_episode_steps':end,'physical_failure':first['physical_failure_by_environment'][e],'longest_unloaded_left_right_supported_intervals':best,'first_ten_interval_release_start':first_run,'final_25_bilateral_support':perch['last_25_half_weight_bilateral_opposed'],'complete_substep_coverage':all(coverage[e,t,k]==1 for t in range(1,end+1) for k in range(16))})
out={'rows':rows,'overflow_count':len(overflow),'scope':'A release here means zero left foot/toe normal impulse across all 16 substeps while the right foot has at least half body-weight upward impulse and opposing front/rear normals. This does not establish physical clearance, preceding bilateral stability, subsequent regrip or biological accuracy. Only first episodes are counted.'}
(A/'release-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
