from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-perch-contact-toe-drive-dev-20261111');P=A/'trial';ep=json.loads((A/'analysis.json').read_text())['first_episode_navigation'];counts=collections.Counter();first={};steps=collections.defaultdict(lambda:[0.]*4);names=json.load(gzip.open(P/'replay.json.gz','rt'))['payload']['body_names']
for path in sorted(P.glob('stderr-*.gz')):
 with gzip.open(path,'rt') as f:
  for line in f:
   if not line.startswith('CROW_SCENE_CONTACT '):continue
   a=line.split();t,k,e,bodya,bodyb=map(int,a[1:6]);end=ep['steps_by_environment'][e]
   if t>end+1 or 38 not in (bodya,bodyb):continue
   bird=bodyb if bodya==38 else bodya;name=names[bird];normal=float(a[6]);nx=float(a[13]);sgn=1 if bodya==38 else -1
   counts[e,name]+=1;first.setdefault(e,{'step':t,'substep':k,'bird':name,'other':'crow_course_perch','normal_impulse':normal,'point_m':list(map(float,a[9:12]))})
   side=0 if name.startswith('crow_left_') else 1 if name.startswith('crow_right_') else -1
   if side>=0 and 'toe' in name:
    if 'front_toe' in name:steps[e,t][side*2]+=sgn*nx*normal
    if 'rear_toe' in name:steps[e,t][side*2+1]+=sgn*nx*normal
rows=[]
for e in range(8):
 opposing=[t for (lane,t),v in steps.items() if lane==e and v[0]>.0002 and v[1]<-.0002 and v[2]>.0002 and v[3]<-.0002]
 rows.append({'lane':e,'first_perch_contact':first.get(e),'perch_contact_substeps_by_body':{n:c for (lane,n),c in counts.items() if lane==e},'intervals_with_opposing_toe_normals_both_feet':len(opposing)})
(A/'perch-analysis.json').write_text(json.dumps({'lanes':rows,'limitations':'Opposing normal impulses are a partial grip check only; weight support, stability, sustained contact and release require independent checks.'},indent=2)+'\n');print(rows)
