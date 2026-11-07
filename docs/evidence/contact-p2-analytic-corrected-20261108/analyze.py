import json,pathlib
p=pathlib.Path(__file__).resolve().parent
out={'meshes':[],'diagnostic_force_threshold':.0025,'impact_force_gate':.0025,'impact_impulse_gate':.0025}
for n in (2,4,8,16):
 d=json.loads((p/f'n{n}.json').read_text());a=d['orders']['1'];b=d['orders']['2']
 out['meshes'].append({'n':n,'p1_volume_error':a['volume_energy_relative_error'],'p2_volume_error':b['volume_energy_relative_error'],
 'p1_force_error':a['force_relative_error'],'p2_force_error':b['force_relative_error'],
 'p1_surface_energy_error':a['surface_energy_relative_error'],'p2_surface_energy_error':b['surface_energy_relative_error'],
 'p2_better_volume':b['volume_energy_relative_error']<a['volume_energy_relative_error'],
 'p2_better_force':b['force_relative_error']<a['force_relative_error'],
 'p1_fixed_state_force_pass':a['force_relative_error']<=.0025,'p2_fixed_state_force_pass':b['force_relative_error']<=.0025})
out['prediction_pass']=all(x['p2_better_volume'] and x['p2_better_force'] for x in out['meshes'] if x['n']>=4)
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
