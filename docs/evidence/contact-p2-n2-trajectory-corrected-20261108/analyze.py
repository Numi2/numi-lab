import json,pathlib
p=pathlib.Path(__file__).resolve().parent
a=json.loads((p/'coarse.json').read_text());b=json.loads((p/'fine.json').read_text())
peak=max(b['force_bins_N']);imp=abs(b['impulse_Ns'])
out={'threshold':.0025,'peak_bin_N':peak,'fine_impulse_Ns':imp,
 'temporal_force':max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/peak,
 'temporal_impulse':abs(a['impulse_Ns']-b['impulse_Ns'])/imp,'runs':{}}
for x in (a,b):
 r=x['trace'];energy_scale=.5*.0088*(9.81*.01)**2
 work_scale=max(max(abs(z[10]) for z in r),max(abs(z[9]) for z in r),1e-12)
 out['runs'][x['variant']]={'min_gap_m':min(min(z[3],z[4]) for z in r),
  'min_J':min(min(z[5],z[6]) for z in r),
  'quadrature_force_relative':max(abs(z[1]-z[2]) for z in r)/peak,
  'momentum_relative':max(abs(z[7]-z[8]+.0088*9.81*z[0]) for z in r)/imp,
  'energy_drift_relative':max(abs(z[11]-r[0][11]) for z in r)/energy_scale,
  'contact_work_balance_relative':max(abs(z[9]+z[10]-r[0][10]) for z in r)/work_scale,
  'contact_work_scale_J':work_scale}
r=b['trace'];work_scale=max(max(abs(z[10]) for z in r),max(abs(z[9]) for z in r),1e-12)
out['temporal_contact_work']=abs(a['contact_work_J']-b['contact_work_J'])/work_scale
out['temporal_pass']=max(out['temporal_force'],out['temporal_impulse'],out['temporal_contact_work'])<=.0025
out['quadrature_pass']=all(x['quadrature_force_relative']<=.0025 for x in out['runs'].values())
out['other_gates_pass']=all(x['min_gap_m']>0 and x['min_J']>0 and max(x['momentum_relative'],x['energy_drift_relative'],x['contact_work_balance_relative'])<=.0025 for x in out['runs'].values())
out['reference_prerequisite_pass']=out['temporal_pass'] and out['quadrature_pass'] and out['other_gates_pass']
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
