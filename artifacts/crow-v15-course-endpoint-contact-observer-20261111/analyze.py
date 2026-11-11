from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-course-endpoint-contact-observer-20261111');B=Path('artifacts/crow-v15-course-collisions-enabled-20261111/enabled');x={};r={};xs={};rs={};scene=collections.defaultdict(list);endpoints=[]
def read(p,states,subs,extract):
 with gzip.open(p/'stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    z=json.loads(l.split(' ',1)[1]);states[z['lane'],z['step']]=z
    if extract:
     for c in z['contacts']:
      if c['impulses'][0]>0 and ('crow_course_' in c['a_name'] or 'crow_course_' in c['b_name']):endpoints.append((z['lane'],z['step'],c))
   elif l.startswith(('CROW_TWO_TOE_SUBSTEP ','CROW_TOE_V_SUBSTEP ')):
    a=l.split();subs[(a[0],)+tuple(map(int,a[1:4]))]=a[4:]
   elif extract and l.startswith('CROW_SCENE_CONTACT '):
    a=l.split();t,k,e=int(a[1]),int(a[2]),int(a[3]);scene[e,t,k].append((int(a[4]),int(a[5]),float(a[6])))
read(A/'trace',x,xs,True);read(B,r,rs,False);different=[]
for key in x.keys()|r.keys():
 a=x.get(key);b=r.get(key)
 if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action')):different.append(key)
subdiff=[k for k in xs.keys()|rs.keys() if xs.get(k)!=rs.get(k)];matched=[]
for e,t,c in endpoints:
 for a,b,imp in scene[e,t,15]:
  if (a,b)==(c['a'],c['b']) and abs(imp-c['impulses'][0])<=max(1e-8,abs(imp)*1e-5):matched.append((e,t,c['a_name'],c['b_name'],imp));break
result={'state_counts':[len(x),len(r)],'different_q_v_actions':len(different),'substep_record_counts':[len(xs),len(rs)],'different_substep_records':len(subdiff),'endpoint_scene_contacts':len(endpoints),'matching_last_substep_contacts':len(matched),'first_matches':matched[:5],'limitations':'Artificial initial overlap, with contact terminations and reset; tests observer identity only, not route performance, physical biological realism, or supported landing.'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
