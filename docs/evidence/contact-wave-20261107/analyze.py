import csv,json,math,pathlib
p=pathlib.Path(__file__).resolve().parent
def load(n,mode,h,amp=10):
 with (p/f'n{n}_m{mode}_h{h}us_a{amp}nm.csv').open() as f:
  return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
def zero(rows,column):
 for a,b in zip(rows,rows[1:]):
  if a[column]>0>=b[column]:
   return a['time']+(b['time']-a['time'])*a[column]/(a[column]-b[column])
 raise ValueError('first zero not observed')
out={'runs':{},'temporal':{},'impact_force_gate':0.0025,'impact_impulse_gate':0.0025}
for n in (8,16):
 for mode in (2,8):
  fine=load(n,mode,1);coarse=load(n,mode,2)
  exact=.02/(2*mode*math.sqrt(9000/1100))
  key=f'n{n}_mode{mode}'
  out['runs'][key]={
    'peak_normalized_l2':max(r['normalized_l2'] for r in fine),
    'final_normalized_l2':fine[-1]['normalized_l2'],
    'first_projection_zero_seconds':zero(fine,'projection'),
    'exact_continuum_zero_seconds':exact,
    'sampled_analytic_zero_seconds':zero(fine,'analytical_projection'),
    'first_zero_delay_fraction':zero(fine,'projection')/exact-1,
    'min_sampled_gap_m':min(r['min_gap'] for r in fine),
    'max_sampled_strain':max(r['max_strain'] for r in fine),
    'max_sampled_stress_nonlinearity':max(r['max_stress_nonlinearity'] for r in fine)}
  delta=max(abs(a['projection']-b['projection']) for a,b in zip(fine,coarse))
  out['temporal'][key]={'max_projection_difference':delta,'pass':delta<=.0025}
a=load(16,8,1);b=load(16,8,1,5)
delta=max(abs(x['projection']-y['projection']) for x,y in zip(a,b))
out['amplitude_halving']={'max_projection_difference':delta,'pass':delta<=.0025}
out['high_mode_errors_larger_than_low']=all(out['runs'][f'n{n}_mode8']['peak_normalized_l2']>out['runs'][f'n{n}_mode2']['peak_normalized_l2'] for n in (8,16))
out['high_mode_peak_error_decreases_with_resolution']=out['runs']['n16_mode8']['peak_normalized_l2']<out['runs']['n8_mode8']['peak_normalized_l2']
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
