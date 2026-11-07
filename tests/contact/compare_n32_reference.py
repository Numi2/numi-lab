#!/usr/bin/env python3
"""Guard fixed-law 32-mesh spatial comparison behind temporal/validity checks."""
import json,pathlib,sys
e=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence/contact-n32-reference-20261107')
a=json.loads((e/'n32-h2e-06.json').read_text());b=json.loads((e/'n32-h1e-06.json').read_text())
p=pathlib.Path('docs/evidence/contact-mesh-reference-20261107')
r16=json.loads((p/'n16-h1e-06.json').read_text());r16fine=json.loads((p/'n16-h5e-07.json').read_text())
def diff(x,y):
 return {'binned_force':max(abs(i-j) for i,j in zip(x['force_bins_N'],y['force_bins_N']))/max(y['force_bins_N']),
         'impulse':abs(x['impulse_Ns']-y['impulse_Ns'])/abs(y['impulse_Ns'])}
def valid(x):
 return x['min_gap_m']>1e-7 and abs(x['final_momentum_Ns']-(x['impulse_Ns']-x['mass_kg']*9.81*x['duration']))<1e-12
for x,n in ((a,32),(b,32),(r16,16),(r16fine,16)):
 assert x['mesh_divisions']==n and abs(x['mass_kg']-.0088)<1e-13 and abs(x['area_m2']-.0004)<1e-13
temporal=diff(a,b);prior=diff(r16,r16fine)
passed=all((temporal['binned_force']<=.0025,temporal['impulse']<=.0025,
            prior['binned_force']<=.0025,prior['impulse']<=.0025,
            valid(a),valid(b),valid(r16),valid(r16fine)))
spatial=diff(r16,b) if passed else None
result={'schema':'numi.contact.n32-reference.v1','n32_temporal_2us_vs_1us':temporal,
 'n16_temporal_1us_vs_0.5us':prior,'n32_clearance_and_momentum':valid(a) and valid(b),
 'temporal_and_validity_gate':passed,'n16_vs_n32_at_1us':spatial,
 'spatial_gate':bool(spatial and spatial['binned_force']<=.0025 and spatial['impulse']<=.0025),
 'limits':['fixed nodal quadrature; no Gaussian n32 run',
           'same finite-width law, not a hard-contact limit or physical material validation',
           'nonlinear trajectory stability inferred from retained validity and temporal tests, not reference-state row bound alone']}
print(json.dumps(result,indent=2))
