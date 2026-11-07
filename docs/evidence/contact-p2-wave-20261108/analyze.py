import json,pathlib
p=pathlib.Path(__file__).resolve().parent
a=json.loads((p/'wave.json').read_text())
def get(degree,mass):
 return next(x for x in a['results'] if x['n']==8 and x['degree']==degree and x['mass']==mass)
p2=get(2,'consistent')['initial_acceleration_mass_relative_error']
p1c=get(1,'consistent')['initial_acceleration_mass_relative_error']
p1l=get(1,'lumped')['initial_acceleration_mass_relative_error']
out={'p2_acceleration_error':p2,'p1_consistent_acceleration_error':p1c,'p1_lumped_acceleration_error':p1l,
 'prediction_pass':p2<p1c and p2<p1l,
 'all_mass_solve_residuals_pass':all(x['mass_solve_relative_residual']<=1e-9 for x in a['results'])}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
