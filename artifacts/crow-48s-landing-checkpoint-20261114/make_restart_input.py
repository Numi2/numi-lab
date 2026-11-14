import json
from pathlib import Path
p=Path(__file__).resolve().parent
x=json.loads((p/'recorded-state.json').read_text());f=x['last_three_recorded_frames'][-1];pnames=x['context']['body_names']
assert pnames[35:]==['locomotion_ground','crow_course_gate_left','crow_course_gate_right','crow_course_slalom_a','crow_course_slalom_b','crow_course_perch']
values=f['q']+f['v']+f['body_states'][35*13:41*13]
assert len(values)==159
(p/'restart-input.txt').write_text('\n'.join(format(v,'.17g') for v in values)+'\n')
