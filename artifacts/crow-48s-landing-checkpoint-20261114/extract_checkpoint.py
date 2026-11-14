"""Preserve the recorded end of lane 5 without claiming a full native checkpoint."""
import gzip, hashlib, json
from pathlib import Path

out=Path(__file__).resolve().parent
source=out.parent/'crow-latest-multiview-20261114/latest-replay.json.gz'
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='272e358377f13145acab957e87af73b77f9c05f4733aa8d66e76ee9636425a49'
data=json.loads(gzip.decompress(raw)); p=data['payload']; frames=p.pop('frames')
assert p['environment']==5 and p['frame_count']==2400 and p['timestep_seconds']==0.02
assert [f['step'] for f in frames[-3:]]==[2398,2399,2400]
checkpoint={'schema':'numi.crow.recorded-state-checkpoint.v1',
 'classification':'recorded accepted-frame snapshot; not a native full-resident checkpoint',
 'source_replay_gzip_sha256':hashlib.sha256(raw).hexdigest(),
 'source_payload_sha256':data['payload_sha256'],
 'context':p,'last_three_recorded_frames':frames[-3:],
 'limitations':['solver warm-start/contact caches, resident GPU state, hidden controller/task state and RNG state not captured',
                'state epochs must be checked against observer trace before physical initialization',
                'the recorded flight disabled all five course collision masks',
                'native v15 life-checkpoint restore is not established'],
 'frame_time_s':48.0}
(out/'recorded-state.json').write_text(json.dumps(checkpoint,indent=2,allow_nan=False)+'\n')
print('wrote',out/'recorded-state.json','with 3 original frames')
