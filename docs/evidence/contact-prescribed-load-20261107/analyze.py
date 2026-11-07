import csv,json,pathlib
p=pathlib.Path(__file__).resolve().parent
def load(n,h):
 with (p/f'n{n}-h{h}us.csv').open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
out={'diagnostic_displacement_scale_m':1e-4,'diagnostic_threshold':.0025,
     'impact_force_gate':.0025,'impact_impulse_gate':.0025,'meshes':{}}
for n in (16,32):
 coarse,fine=load(n,2),load(n,1)
 delta=max(abs(a['surface_mean_displacement_m']-b['surface_mean_displacement_m']) for a,b in zip(coarse,fine))
 balance=max(abs(r['momentum_Ns']-(r['prescribed_impulse_Ns']-.0088*9.81*r['time_s'])) for r in fine)
 impulse=fine[-1]['prescribed_impulse_Ns']
 out['meshes'][str(n)]={'temporal_max_displacement_difference_m':delta,
    'temporal_difference_in_activation_distances':delta/1e-4,
    'temporal_check_pass':delta/1e-4<=.0025,
    'momentum_balance_over_input_impulse':balance/impulse,
    'momentum_check_pass':balance/impulse<=.0025,'input_impulse_Ns':impulse}
a,b=load(16,1),load(32,1)
delta,index=max((abs(x['surface_mean_displacement_m']-y['surface_mean_displacement_m']),i) for i,(x,y) in enumerate(zip(a,b)))
out['spatial']={'max_mean_surface_displacement_difference_m':delta,'time_s':a[index]['time_s'],
 'difference_in_activation_distances':delta/1e-4,'prediction_exceeds_threshold':delta/1e-4>.0025,
 'max_rms_displacement_difference_m':max(abs(x['surface_displacement_rms_m']-y['surface_displacement_rms_m']) for x,y in zip(a,b))}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
