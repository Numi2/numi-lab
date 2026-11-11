from pathlib import Path
import gzip,json,math
R=Path.cwd();A=R/'artifacts/crow-v15-approach-observation-diagnostic-20261111/autonomous-20261126';ref=R/'artifacts/crow-v15-student-state-landing-eval-20261111/autonomous-20261126'
def read(path):
 out={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1])
    if x['lane']==0:out[x['step']]=x
 return out
a=read(ref);b=read(A);missing=[];different=[]
for t in range(1,3801):
 if t not in a or t not in b:missing.append(t);continue
 if any(a[t][k]!=b[t][k] for k in ('q','v','policy_action','contacts')):different.append(t)
with gzip.open(A/'replay.json.gz','rt') as f:p=json.load(f)['payload']
assert p['environment']==0 and p['body_names'][40]=='crow_course_perch'
frames=p['frames'];records=[]
for frame in frames:
 if frame['outcomes']['navigation_waypoints_reached']<5:continue
 body=frame['body_states'];perch=body[40*13:40*13+3];q=frame['q'];v=frame['v'];goal=[perch[0],perch[1]+.55,perch[2]+.12];distance=math.hypot(goal[0]-q[0],goal[1]-q[1]);speed=math.hypot(v[0],v[1]);act=frame['accepted_actions'];command=[.10+.30*act[4],.30*act[5],.15*act[6],.975+.325*act[7]]
 records.append({'frame':frame['step'],'root_position_m':q[:3],'final_goal_m':goal,'horizontal_goal_distance_m':distance,'horizontal_speed_m_s':speed,'velocity_m_s':v[:3],'candidate_latent_4_to_7_mapped_commands':command,'autonomous_phase_one_geometric_predicate':distance<.08 and speed<.08})
assert records
samples=[x for x in records if x['frame']%100==0 or x is records[0] or x is records[-1]]
out={'parity':{'compared':3800-len(missing),'first_missing':missing[:1],'first_difference':different[:1],'different_samples':len(different)},'post_fifth_frames':len(records),'minimum_horizontal_goal_distance_m':min(x['horizontal_goal_distance_m'] for x in records),'first_phase_one_predicate':next((x['frame'] for x in records if x['autonomous_phase_one_geometric_predicate']),None),'samples':samples,'scope':'One-lane development replay. Goal geometry is inspected for diagnosis only; the actor has an explicit final waypoint observation. Mapped latent outputs are not applied joint targets. Parity is endpoint q/v, policy outputs and contacts, not every substep.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
