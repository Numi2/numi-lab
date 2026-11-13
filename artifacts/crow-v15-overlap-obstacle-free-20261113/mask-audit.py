from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-overlap-obstacle-free-20261113');P=A/'push-wing-20261126-rebound-clear';rows=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  z=l.split()
  if z and z[0]=='CROW_SCENE_MASK':rows.append({'name':z[1],'old':int(z[2]),'new':int(z[3])})
expected={'crow_course_gate_left/collision','crow_course_gate_right/collision','crow_course_slalom_a/collision','crow_course_slalom_b/collision','crow_course_perch/collision'};course=[r for r in rows if r['name'].startswith('crow_course_')];ground=[r for r in rows if r['name']=='locomotion_ground/collision'];other=[r for r in rows if r not in course];bind=json.loads((A/'policy-binding.json').read_text());valid=(set(x['name'] for x in course)==expected and len(course)==5 and all(x['old']!=0 and x['new']==0 for x in course) and len(ground)==1 and ground[0]['old']!=0 and ground[0]['new']==ground[0]['old'] and all(x['new']==x['old'] for x in other) and all(x['identical_layers_and_transforms'] for x in bind['reports']));out={'valid':valid,'shapes':rows,'actor_rebinding':bind['reports']};(A/'mask-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));
if not valid:raise SystemExit('mask or actor audit failed')
