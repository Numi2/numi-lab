import collections, gzip, json, math
from pathlib import Path

run = Path(__file__).parent / 'near-perch-landingmode-500'
state = {}
interval = collections.defaultdict(lambda: {'up': [0., 0.], 'front_x': [0., 0.], 'rear_x': [0., 0.], 'nonfoot_scene': 0., 'ground_normal': 0., 'substeps': set()})
with gzip.open(run / 'stderr.gz', 'rt') as stream:
    for line in stream:
        if line.startswith('CROW_SUPPORT_SAMPLE '):
            s = json.loads(line.split(' ', 1)[1]); state[s['step']] = s
        elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
            p = line.split(); interval[int(p[1])]['substeps'].add(int(p[2]))
        elif line.startswith('CROW_GROUND_CONTACT '):
            p = line.split(); interval[int(p[1])]['ground_normal'] += float(p[20])
        elif line.startswith('CROW_SCENE_CONTACT '):
            p = line.split(); step, a, b = int(p[1]), int(p[4]), int(p[5])
            imp = list(map(float, p[6:9])); normal = list(map(float, p[13:16])); tangent = list(map(float, p[16:19]))
            if 40 not in (a, b):
                continue
            body = b if a == 40 else a
            sign = 1. if a == 40 else -1.
            side = 0 if body == 11 or 19 <= body <= 26 else 1 if body == 15 or 27 <= body <= 34 else -1
            record = interval[step]
            if side < 0:
                record['nonfoot_scene'] += imp[0]; continue
            cross_z = normal[0]*tangent[1] - normal[1]*tangent[0]
            record['up'][side] += sign*(normal[2]*imp[0] + tangent[2]*imp[1] + cross_z*imp[2])
            if body >= 19:
                digit = ((body-19) % 8)//2
                record['rear_x' if digit == 3 else 'front_x'][side] += sign*normal[0]*imp[0]

passed, failures, rows = [], collections.Counter(), []
for step, rec in sorted(interval.items()):
    if step < 348 or step + 1 not in state: continue
    s = state[step + 1]; q, v = s['q'], s['v']
    tilt = math.acos(max(-1., min(1., 1-2*(q[3]**2+q[4]**2))))
    speed = math.sqrt(sum(x*x for x in v[:3]))
    checks = {'all_16_substeps': len(rec['substeps']) == 16,
              'both_up': min(rec['up']) > .0002 and sum(rec['up']) >= .5*.45004*9.81*.02,
              'left_opposed': rec['front_x'][0] >= .0002 and rec['rear_x'][0] <= -.0002,
              'right_opposed': rec['front_x'][1] >= .0002 and rec['rear_x'][1] <= -.0002,
              'no_nonfoot': rec['nonfoot_scene'] <= 1e-7,
              'no_ground': rec['ground_normal'] <= 1e-7,
              'tilt': tilt <= .35, 'speed': speed <= .12}
    for key, value in checks.items():
        if not value: failures[key] += 1
    if all(checks.values()): passed.append(step)
    rows.append({'step':step,'tilt':tilt,'speed':speed, **{k:rec[k] for k in ('up','front_x','rear_x','nonfoot_scene','ground_normal')},'checks':checks})
streaks = []
for step in passed:
    if not streaks or step != streaks[-1][-1] + 1: streaks.append([step])
    else: streaks[-1].append(step)
out = {'audited_intervals':len(rows),'qualification_streaks':[[s[0],s[-1],len(s)] for s in streaks],
       'failed_gate_counts':dict(failures),'intervals':rows,
       'limitation':'Full-substep audit starts at observer step 348; state sample indexed step + 1. Capture transient and altered scene remain separate limitations.'}
(run / 'hold-analysis.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k != 'intervals'}, indent=2))
