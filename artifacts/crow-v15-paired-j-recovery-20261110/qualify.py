from pathlib import Path
import gzip,json,sys,math
R=Path.cwd();a=R/'artifacts/crow-v15-paired-corrective-eval-20261110';b=R/'artifacts/crow-v15-paired-j-recovery-20261110';side=sys.argv[1];name=side+'-paired-recovery'
with gzip.open(a/(side+'-hard.json.gz'),'rt') as f:old=json.load(f)['payload']['frames']
with gzip.open(b/(name+'.json.gz'),'rt') as f:new=json.load(f)['payload']['frames']
diff=[];n=0
for x,y in zip(old,new):
 if x['step']>=2050:break
 n+=1
 for k in ('step','q','v','actor_observation','accepted_actions'):
  if x[k]!=y[k] and len(diff)<8:diff.append({'step':x['step'],'field':k})
summary=json.loads((b/(name+'.stdout')).read_text());first=summary['first_episode_navigation'];last=-1;run=0;maxrun=0;nav5=None;firstcollision=None;nonfinite=[];nonfoot=0;nav=0
with gzip.open(b/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_CONTACT_TRACE ') and firstcollision is None:firstcollision=json.loads(line.split(' ',1)[1])
  if line.startswith('CROW_SUPPORT_SAMPLE ') and nav>=5:
   sample=json.loads(line.split(' ',1)[1])
   for c in sample['contacts']:
    an,bn=c['a_name'],c['b_name'];body=an if bn=='crow_course_perch' else bn if an=='crow_course_perch' else None
    if body is not None and not ('_foot' in body or '_toe' in body) and c['impulses'][0]>1e-8:nonfoot+=1
  if line.startswith('CROW_TASK_STATE '):
   v=line.split();t=int(v[2]);
   if t<last:break
   last=t;nav=float(v[4]);left,right=float(v[5]),float(v[6]);
   if not all(math.isfinite(float(v[k])) for k in (4,5,6,7,8,9)):nonfinite.append(t)
   if nav>=5 and nav5 is None:nav5=t
   run=run+1 if nav>=5 and left>.1 and right>.1 and left+right>2.2 else 0;maxrun=max(maxrun,run)
r={'student_prefix_frames':n,'exact_prefix_equal':n==2049 and not diff,'first_differences':diff,'nav5_step':nav5,'first_episode':first,'first_fatal_contact':firstcollision,'solver_failed_steps':summary['failed_environment_steps'],'nonfoot_perch_contact_samples':nonfoot,'nonfinite_task_samples':nonfinite,'longest_bilateral_support_samples':maxrun,'terminal_bilateral_support_samples':run,'eligible_for_development_routing_prefix_training':n==2049 and not diff and nav5 is not None and first['physical_failure_count']==0 and summary['failed_environment_steps']==0 and not nonfinite and nonfoot==0 and run>=25,'limitations':'Teacher privileged after 2050, development layout; endpoint support is not strict continuous opposing-toe grip. Landing labels excluded.'}
(b/(name+'-qualification.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
