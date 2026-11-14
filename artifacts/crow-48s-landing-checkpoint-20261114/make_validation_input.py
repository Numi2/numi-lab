import json
from pathlib import Path
p=Path(__file__).resolve().parent
x=json.loads((p/'recorded-state.json').read_text()); f=x['last_three_recorded_frames'][-1]; names=x['context']['body_names']; stride=x['context']['body_state_stride']
assert len(names)==41 and stride==13 and len(f['q'])==41 and len(f['v'])==40
with (p/'validation-input.txt').open('w') as o:
    for values in (f['q'],f['v']): o.write(' '.join(format(v,'.17g') for v in values)+'\n')
    for i,n in enumerate(names[:35]): o.write(n+' '+' '.join(format(v,'.17g') for v in f['body_states'][i*stride:i*stride+7])+'\n')
