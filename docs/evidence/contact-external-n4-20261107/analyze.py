import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'coarse.json').read_text());b=json.loads((p/'fine.json').read_text())
c=json.loads((root/'docs/evidence/contact-mesh-reference-20261107/n4-h5e-07.json').read_text())
d=json.loads((root/'docs/evidence/contact-external-dynamics-20261107/fine.json').read_text())
peak=max(b['force_bins_N']);imp=b['impulse_Ns']
def compare(x):return {'force_bin_relative':max(abs(q-r) for q,r in zip(x['force_bins_N'],b['force_bins_N']))/peak,
 'impulse_relative':abs(x['impulse_Ns']-imp)/imp}
out={'gate':.0025,'coarse_vs_fine':compare(a),'rk4_vs_fine':compare(c),'n2_vs_n4_external':compare(d),
 'minimum_sampled_gaps_m':[a['min_sampled_gap_m'],b['min_sampled_gap_m']],
 'momentum_imbalance_over_impulse':[a['momentum_error_Ns']/imp,b['momentum_error_Ns']/imp]}
out['temporal_gate_pass']=max(out['coarse_vs_fine'].values())<=.0025
out['independent_integration_gate_pass']=max(out['rk4_vs_fine'].values())<=.0025
out['validity_gate_pass']=min(out['minimum_sampled_gaps_m'])>0 and max(out['momentum_imbalance_over_impulse'])<=.0025
out['spatial_gate_pass']=max(out['n2_vs_n4_external'].values())<=.0025
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
