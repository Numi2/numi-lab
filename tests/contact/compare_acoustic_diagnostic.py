#!/usr/bin/env python3
"""Early-bin comparison against the approximate 1D acoustic diagnostic."""
import json,pathlib
e=pathlib.Path('docs/evidence/contact-acoustic-diagnosis-20261107')
s=pathlib.Path('docs/evidence/contact-surface-diagnosis-20261107')
a=json.loads((e/'h1us.json').read_text());b=json.loads((e/'h0.5us.json').read_text())
force=max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/max(b['force_bins_N'])
impulse=abs(a['impulse_Ns']-b['impulse_Ns'])/abs(b['impulse_Ns'])
rows=[]
for n in (2,4,8,16):
 x=json.loads((s/f'n{n}-h5e-07-nodal.json').read_text());f=x['force_bins_N'][4]
 rows.append({'n':n,'first_bin_force_N':f,'relative_to_1d_acoustic':abs(f-b['force_bins_N'][4])/b['force_bins_N'][4]})
z=json.loads((s/'n16-h5e-07-gaussian.json').read_text())
result={'schema':'numi.contact.acoustic-diagnostic-comparison.v1',
 'acoustic_first_bin_N':b['force_bins_N'][4],
 'acoustic_temporal_force_relative':force,'acoustic_temporal_impulse_relative':impulse,
 'temporal_gate':force<=.0025 and impulse<=.0025,
 'nodal':rows,'n16_gaussian_first_bin_N':z['force_bins_N'][4],
 'pilot_informed_prediction_decreasing':all(x['relative_to_1d_acoustic']>y['relative_to_1d_acoustic'] for x,y in zip(rows[1:],rows[2:])),
 'pilot_informed_prediction_n16_below_5pct':rows[-1]['relative_to_1d_acoustic']<.05,
 'limits':['1D linear half-space, not the 3D finite block; free side effects are omitted',
 'prediction was selected after an exploratory pilot',
 '2.5% or 5% diagnostic agreement does not meet the 0.25% scientific force gate']}
print(json.dumps(result,indent=2))
assert result['temporal_gate']
