#!/usr/bin/env python3
"""Guard same-rule and same-mesh dynamic comparisons by temporal gates."""
import json,pathlib,sys
e=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence/contact-n32-gaussian-20261107')
g=pathlib.Path('docs/evidence/contact-surface-diagnosis-20261107')
n=pathlib.Path('docs/evidence/contact-n32-reference-20261107')
read=lambda p:json.loads(p.read_text())
coarse=read(e/'n32-h2e-06-gaussian.json');fine=read(e/'n32-h1e-06-gaussian.json')
g16=read(g/'n16-h1e-06-gaussian.json');g16f=read(g/'n16-h5e-07-gaussian.json')
n32=read(n/'n32-h1e-06.json')
def diff(x,y):
 return {'binned_force':max(abs(i-j) for i,j in zip(x['force_bins_N'],y['force_bins_N']))/max(y['force_bins_N']),
 'impulse':abs(x['impulse_Ns']-y['impulse_Ns'])/abs(y['impulse_Ns'])}
def valid(x):
 return x['min_gap_m']>1e-7 and abs(x['final_momentum_Ns']-(x['impulse_Ns']-x['mass_kg']*9.81*x['duration']))<1e-12
for x,N in ((coarse,32),(fine,32),(g16,16),(g16f,16),(n32,32)):
 assert x['mesh_divisions']==N and abs(x['mass_kg']-.0088)<1e-13 and abs(x['area_m2']-.0004)<1e-13
a,b=diff(coarse,fine),diff(g16,g16f)
passed=all((a['binned_force']<=.0025,a['impulse']<=.0025,
 b['binned_force']<=.0025,b['impulse']<=.0025,
 all(valid(x) for x in (coarse,fine,g16,g16f,n32))))
spatial=diff(g16,fine) if passed else None
same=diff(n32,fine) if passed else None
out={'schema':'numi.contact.n32-gaussian.v1','n32_temporal_2us_vs_1us':a,
 'n16_temporal_1us_vs_0.5us':b,'temporal_and_validity_gate':passed,
 'gaussian_n16_vs_n32_at_1us':spatial,'nodal_vs_gaussian_n32_at_1us':same,
 'spatial_gate':bool(spatial and spatial['binned_force']<=.0025 and spatial['impulse']<=.0025),
 'primary_quadrature_hypothesis_below_half_nodal_difference':bool(spatial and spatial['binned_force']<.12425294953506019/2),
 'limits':['diagnostic prediction is not the scientific gate',
  'fixed regularized law, not a hard-contact or physical material reference',
  'a failed primary-cause prediction does not isolate a unique remaining source']}
print(json.dumps(out,indent=2))
