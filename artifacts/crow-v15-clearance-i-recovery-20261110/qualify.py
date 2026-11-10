from pathlib import Path
import gzip,json,sys,math
R=Path.cwd();A=R/'artifacts/crow-v15-clearance-i-recovery-20261110';name=sys.argv[1];takeover=int(sys.argv[2]);kind='clear-i-original';ref=R/'artifacts/crow-v15-clearance-corrective-eval-20261110'/(kind+'.json.gz')
with gzip.open(ref,'rt') as f:old=json.load(f)['payload']['frames']
with gzip.open(A/(name+'.json.gz'),'rt') as f:new=json.load(f)['payload']['frames']
diff=[];n=0
for x,y in zip(old,new):
 if x['step']>=takeover:break
 n+=1
 for k in ('step','q','v','actor_observation','accepted_actions'):
  if x[k]!=y[k] and len(diff)<8:diff.append({'step':x['step'],'field':k})
summary=json.loads((A/(name+'.stdout')).read_text());first=summary['first_episode_navigation'];last=-1;run=0;maxrun=0;nav5=None;fatal=None;nonfinite=[];nonfoot=0;nav=0;clear=None
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_CONTACT_TRACE ') and fatal is None:fatal=json.loads(line.split(' ',1)[1])
  elif line.startswith('CROW_CLEARANCE_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if clear is None or x['end_step']>clear['end_step']:clear=x
  elif line.startswith('CROW_SUPPORT_SAMPLE ') and nav>=5:
   x=json.loads(line.split(' ',1)[1])
   for c in x['contacts']:
    a,b=c['a_name'],c['b_name'];body=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
    if body is not None and not ('_foot' in body or '_toe' in body) and c['impulses'][0]>1e-8:nonfoot+=1
  elif line.startswith('CROW_TASK_STATE '):
   v=line.split();t=int(v[2]);
   if t<last:break
   last=t;nav=float(v[4]);left,right=float(v[5]),float(v[6]);
   if not all(math.isfinite(float(v[k])) for k in (4,5,6,7,8,9)):nonfinite.append(t)
   if nav>=5 and nav5 is None:nav5=t
   run=run+1 if nav>=5 and left>.1 and right>.1 and left+right>2.2 else 0;maxrun=max(run,maxrun)
eligible=n==takeover-1 and not diff and nav5 is not None and first['physical_failure_count']==0 and summary['failed_environment_steps']==0 and not nonfinite and nonfoot==0 and run>=25 and clear is not None and clear['minimum_m']>=.05
r={'student_prefix_frames':n,'exact_prefix_equal':n==takeover-1 and not diff,'first_differences':diff,'nav5_step':nav5,'first_episode':first,'first_fatal_contact':fatal,'minimum_accepted_clearance':clear,'solver_failed_steps':summary['failed_environment_steps'],'nonfoot_perch_contact_samples':nonfoot,'nonfinite_task_samples':nonfinite,'longest_bilateral_support_samples':maxrun,'terminal_bilateral_support_samples':run,'eligible_for_development_routing_prefix_training':eligible,'limitations':'Teacher privileged after takeover, development only; endpoint clearance and support are not continuous substep guarantees. Landing labels excluded.'};(A/(name+'-qualification.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
