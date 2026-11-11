from pathlib import Path
import gzip,json,math
R=Path.cwd();rep=R/'artifacts/crow-v15-approach-observation-diagnostic-20261111/autonomous-20261126/replay.json.gz'
with gzip.open(rep,'rt') as f:replay=json.load(f)['payload']
assert replay['environment']==0 and replay['body_names'][40]=='crow_course_perch';goalx=replay['frames'][0]['body_states'][40*13]
paths={'autonomous':R/'artifacts/crow-v15-autonomous-approach-balance-control-20261111/autonomous-20261126','teacher_reference':R/'artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126'};out={}
for mode,path in paths.items():
 runs=[0,0,0];best=[0,0,0];hits=[]
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   x=json.loads(line.split(' ',1)[1]);t=x['step'];q=x['q'];v=x['v']
   if x['lane']!=0 or not 3200<=t<=3385:continue
   sides=set()
   for c in x['contacts']:
    if c['impulses'][0]<=1e-7:continue
    names=(c['a_name'],c['b_name'])
    if 'crow_course_perch' not in names:continue
    n=names[0] if names[1]=='crow_course_perch' else names[1]
    for side,prefix in enumerate(('crow_left_','crow_right_')):
     if n.startswith(prefix) and ('toe' in n or 'foot' in n):sides.add(side)
   err=goalx-q[0]-.012;speed=math.hypot(v[0],v[1]);both=len(sides)==2;conditions=[both,both and abs(err)<.025 and speed<.08,both and abs(err)<.04 and speed<.08]
   for i,z in enumerate(conditions):runs[i]=runs[i]+1 if z else 0;best[i]=max(best[i],runs[i])
   if conditions[1]:hits.append(t)
 out[mode]={'longest_both_feet_endpoint_contact':best[0],'longest_with_goal_x_error_under_025m_and_speed_under_08':best[1],'longest_with_goal_x_error_under_04m_and_speed_under_08':best[2],'first_tight_step':hits[:1]}
out['scope']='Lane 0, steps 3200–3385 on one used course and seed. Endpoint contacts only. Goal x is taken from the matched lane-zero replay, and the teacher reference remains privileged.';out['final_goal_x_m']=goalx
p=paths['autonomous']/'alignment-audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
