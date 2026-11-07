"""Check recorded actor observation timing and contrast delayed action source."""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

here=Path(__file__).resolve().parent
prior=here.parent/'g1-replay-gates-20261107'
obs=list(csv.DictReader((here/'observations.tsv').open(), delimiter='\t'))
rows=[line.split() for line in (prior/'recorded.trace.tsv').read_text().splitlines() if not line.startswith('#')]
raw=(prior/'actions.f32').read_bytes()
actions=struct.unpack('<'+'f'*(len(raw)//4),raw)
result=json.loads((prior/'recorded.stdout').read_text())
terminal=result['first_termination_step_by_environment'][0]
assert len(obs)==1000 and len(rows)==1000 and terminal==373
errors=[]; position_errors=[]; velocity_outliers=[]; nonzero_previous=0; observed_nonzero=0; mismatch=0
for step in range(2,terminal+1):
    row=rows[step-2]; assert int(row[0])==step-1
    x,y,z,w=map(float,row[4:8])
    prediction=[-2*(x*z-y*w),-2*(y*z+x*w),-(1-2*(x*x+y*y))]
    observed=[float(obs[step-1]['gravity_'+c]) for c in 'xyz']
    errors.append(max(abs(a-b) for a,b in zip(prediction,observed)))
    position_errors.append(abs(float(obs[step-1]['ankle_position_error_rad'])-float(row[13])))
    if step>=3:
        direct=(float(rows[step-2][13])-float(rows[step-3][13]))/.02
        actual=float(obs[step-1]['ankle_finite_difference_velocity_rad_s'])
        if abs(direct-actual)>1e-5:
            velocity_outliers.append({'step':step,'actor_rad_s':actual,'pose_difference_rad_s':direct})
        # At sampled pre-step n, the completed n-1 frame was written after
        # raw action n-1 entered slot 1; delay 1 refers to raw action n-2.
        expected=actions[((step-2-1)*8)*29+5]
        measured=float(obs[step-1]['delayed_ankle_action'])
        nonzero_previous += expected != 0.0
        observed_nonzero += measured != 0.0
        mismatch += abs(expected-measured)>1e-6
assert max(errors)<1e-6
report={'environment':0,'first_terminal_step':terminal,
        'prior_state_gravity_samples':len(errors),
        'max_gravity_error':max(errors),
        'max_joint_position_error':max(position_errors),
        'velocity_difference_outliers_gt_1e_5':velocity_outliers,
        'delay_comparisons':terminal-2,
        'nonzero_expected_delayed_actions':nonzero_previous,
        'nonzero_observed_delayed_actions':observed_nonzero,
        'delay_mismatches_gt_1e_6':mismatch,
        'actions_sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(report,indent=2,sort_keys=True))
