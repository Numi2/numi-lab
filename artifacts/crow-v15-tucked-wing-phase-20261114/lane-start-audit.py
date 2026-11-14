from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-tucked-wing-phase-20261114');d=json.load(gzip.open(A/'push-window.json.gz','rt'));s={(r['lane'],r['step']):r for r in d['state']};rows=[]
for e in range(8):
 rows.append({'lane':e,'phase_quadrant':e%4,'first_recorded_q_root_xyz':s[e,1]['q'][:3],'first_recorded_v_root_xyz':s[e,1]['v'][:3],'pre_push_q_root_xyz':s[e,368]['q'][:3],'pre_push_v_root_xyz':s[e,368]['v'][:3]})
(A/'lane-start-audit.json').write_text(json.dumps({'rows':rows,'limitations':'First recorded states, not reset inputs. Correlated lanes have different starting state and may traverse different positions before push; differences cannot isolate numerical or controller variability.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
