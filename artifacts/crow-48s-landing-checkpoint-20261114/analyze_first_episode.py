"""Audit a full-window contact trace and discard everything at/after first done."""
import collections, gzip, json, math, sys
from pathlib import Path

run=Path(__file__).parent/sys.argv[1]
payload=json.loads((run/'replay.json').read_text())['payload']
done=[(f['step'],f['termination_reason']) for f in payload['frames'] if f['done']]
first_done=done[0][0] if done else len(payload['frames'])+1
state={}; intervals=collections.defaultdict(lambda:{'up':[0.,0.],'front_x':[0.,0.],'rear_x':[0.,0.],
                                                     'nonfoot_scene':0.,'ground_normal':0.,'substeps':set()})
with gzip.open(run/'stderr.gz','rt') as stream:
    for line in stream:
        if line.startswith('CROW_SUPPORT_SAMPLE '):
            s=json.loads(line.split(' ',1)[1]);state[s['step']]=s
        elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
            p=line.split();intervals[int(p[1])]['substeps'].add(int(p[2]))
        elif line.startswith('CROW_GROUND_CONTACT '):
            p=line.split();intervals[int(p[1])]['ground_normal']+=float(p[20])
        elif line.startswith('CROW_SCENE_CONTACT '):
            p=line.split();step,a,b=int(p[1]),int(p[4]),int(p[5]);imp=list(map(float,p[6:9]))
            if 40 not in (a,b):continue
            normal=list(map(float,p[13:16]));tan=list(map(float,p[16:19]));body=b if a==40 else a;sign=1. if a==40 else -1.
            side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1
            rec=intervals[step]
            if side<0:rec['nonfoot_scene']+=imp[0];continue
            rec['up'][side]+=sign*(normal[2]*imp[0]+tan[2]*imp[1]+(normal[0]*tan[1]-normal[1]*tan[0])*imp[2])
            if body>=19:
                digit=((body-19)%8)//2
                rec['rear_x' if digit==3 else 'front_x'][side]+=sign*normal[0]*imp[0]
rows=[];passed=[];failures=collections.Counter()
for step,rec in sorted(intervals.items()):
    if step<1 or step>=first_done or step+1 not in state:continue
    q,v=state[step+1]['q'],state[step+1]['v']
    tilt=math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in v[:3]))
    checks={'all_16_substeps':len(rec['substeps'])==16,'both_up':min(rec['up'])>.0002 and sum(rec['up'])>=.5*.45004*9.81*.02,
            'left_opposed':rec['front_x'][0]>=.0002 and rec['rear_x'][0]<=-.0002,
            'right_opposed':rec['front_x'][1]>=.0002 and rec['rear_x'][1]<=-.0002,
            'no_nonfoot':rec['nonfoot_scene']<=1e-7,'no_ground':rec['ground_normal']<=1e-7,
            'tilt':tilt<=.35,'speed':speed<=.12}
    for k,ok in checks.items():
        if not ok:failures[k]+=1
    if all(checks.values()):passed.append(step)
    rows.append({'step':step,'tilt':tilt,'speed':speed,**{k:rec[k] for k in ('up','front_x','rear_x','nonfoot_scene','ground_normal')},'checks':checks})
streaks=[]
for t in passed:
    if not streaks or t!=streaks[-1][-1]+1:streaks.append([t])
    else:streaks[-1].append(t)
jumps=[];prev=None
for t,s in sorted(state.items()):
    if prev:
        dist=math.sqrt(sum((s['q'][i]-prev['q'][i])**2 for i in range(3)))
        if dist>.03:jumps.append({'before':prev['step'],'after':t,'distance_m':dist})
    prev=s
out={'first_done':done[0] if done else None,'all_done':done,'root_jumps':jumps,'audited_first_episode_intervals':len(rows),
     'streaks':[[s[0],s[-1],len(s)] for s in streaks],'failed_gates':dict(failures),'intervals':rows,
     'continuous_all_steps':not done and not jumps,'note':'Only intervals strictly before the first replay done are analyzed; sample state index is interval+1.'}
(run/'first-episode-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='intervals'},indent=2))
