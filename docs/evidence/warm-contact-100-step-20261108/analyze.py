import json
import math
from pathlib import Path

p = Path(__file__).resolve().parent
out = {}
for label in ('main', 'candidate'):
    data = json.loads((p / (label + '.json')).read_text())
    cases = []
    for index, case in enumerate(data['cases']):
        rows = case['steps']
        accepted = [step for step in rows if step['accepted']]
        rejected = [i + 1 for i, step in enumerate(rows) if not step['accepted']]
        cases.append({
            'case': index, 'steps': len(rows), 'accepted': len(accepted),
            'first_rejected_step': rejected[0] if rejected else None,
            'replay': all(step['replay'] for step in rows),
            'rejected_restore': all(step['restored'] for step in rows if not step['accepted']),
            'max_linear_error': max((step['linear_error'] for step in accepted), default=0),
            'max_angular_error': max((step['angular_error'] for step in accepted), default=0),
            'max_normal_impulse': max((abs(step['normal_impulse']) for step in accepted), default=0),
            'max_residual': max((abs(step['residual']) for step in accepted), default=0),
            'finite_accepted': all(all(math.isfinite(v) for v in step['v'] + step['q']) for step in accepted),
        })
    out[label] = cases
candidate = out['candidate']
rows = json.loads((p / 'candidate.json').read_text())['cases']
stationary = rows[3]['steps']
static_error = max(abs(step['normal_impulse'] - .0981) for step in stationary)
near_energy_error = max(abs(step['energy'] - .5) for step in rows[2]['steps'])
out['gate'] = {
    'near_miss': candidate[2]['accepted'] == 100 and candidate[2]['replay'] and candidate[2]['max_normal_impulse'] == 0 and near_energy_error < 1e-6,
    'supported_stationary': candidate[3]['accepted'] == 100 and candidate[3]['replay'] and candidate[3]['max_linear_error'] < 2e-6 and static_error < .0002,
    'supported_sliding': candidate[4]['accepted'] == 100 and candidate[4]['replay'] and candidate[4]['finite_accepted'] and candidate[4]['max_linear_error'] < 2e-6,
    'static_impulse_max_error': static_error,
    'near_miss_energy_max_error': near_energy_error,
}
(p / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out['gate'], indent=2))
