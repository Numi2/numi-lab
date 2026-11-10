from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-anchored-i-recovery-20261110')
with gzip.open('artifacts/crow-v15-anchored-eval-20261110/i-anchored.json.gz','rt') as f:off=json.load(f)['payload']['frames']
with gzip.open(A/'development-i-anchored-recovery.json.gz','rt') as f:recovery=json.load(f)['payload']['frames']
diff=[];count=0
for a,b in zip(off,recovery):
 if a['step']>=2100:break
 count+=1
 for k in ('step','q','v','actor_observation','accepted_actions'):
  if a[k]!=b[k] and len(diff)<10:diff.append({'step':a['step'],'field':k})
run=0;last=-1
with gzip.open(A/'development-i-anchored-recovery.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_TASK_STATE '): continue
  v=line.split();t=int(v[2])
  if t<last: break
  last=t;left,right=float(v[5]),float(v[6]);run=run+1 if left>.1 and right>.1 and left+right>2.2 else 0
r=json.loads((A/'analysis.json').read_text());out={'student_prefix_frames':count,'exact_prefix_equal':count==2099 and not diff,'first_differences':diff,'five_waypoints':r['nav5_step'] is not None,'physical_failure':r['first_episode']['physical_failure_count'],'solver_failed_steps':r['failed_environment_steps'],'nonfoot_perch_contact_samples':r['nonfoot_contact_samples'],'longest_bilateral_loaded_samples':r['longest_bilateral_load'],'terminal_bilateral_loaded_samples':run,'eligible_for_development_routing_prefix_training':count==2099 and not diff and r['nav5_step'] is not None and r['first_episode']['physical_failure_count']==0 and r['failed_environment_steps']==0 and r['nonfoot_contact_samples']==0 and run>=25,'limitations':'One privileged recovery after 2100 student steps, used layout; restrict routing data to student-visited and physically reached waypoint-prefix states. Whole-sequence strict opposing-toe support remains unqualified; exclude landing as a full-sequence qualified demonstration.'}
(A/'qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
