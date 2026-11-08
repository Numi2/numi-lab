import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'coarse.json').read_text());b=json.loads((p/'fine.json').read_text())
full=json.loads((root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json').read_text())
peak=max(full['force_bins_N']);imp=abs(full['impulse_Ns']);t0=.0045;E=.5*.0088*(9.81*t0)**2
out={'peak_bin_N':peak,'retained_bin4_N':full['force_bins_N'][4],
 'temporal_bin_relative':abs(a['bin4_N']-b['bin4_N'])/peak,
 'retained_bin_relative':abs(b['bin4_N']-full['force_bins_N'][4])/peak,'runs':{}}
for x in (a,b):
 r=x['trace'];W=max(max(abs(z[11]) for z in r),max(abs(z[10]) for z in r),1e-12)
 v={'min_gap_m':min(min(z[4:7]) for z in r),'min_J':min(z[7] for z in r),
 'duffy_reference_relative':max(abs(z[2]-z[3]) for z in r)/peak,
 'order18_relative':max(abs(z[1]-z[3]) for z in r)/peak,
 'momentum_relative':max(abs(z[8]-r[0][8]-z[9]+.0088*9.81*(z[0]-t0)) for z in r)/imp,
 'energy_relative':max(abs(z[12]-r[0][12]) for z in r)/E,
 'contact_work_relative':max(abs(z[10]+z[11]-r[0][11]) for z in r)/W}
 v['pass']=v['min_gap_m']>0 and v['min_J']>0 and v['duffy_reference_relative']<=.0005 and max(v['order18_relative'],v['momentum_relative'],v['energy_relative'],v['contact_work_relative'])<=.0025
 out['runs'][x['variant']]=v
out['segment_checks_pass']=out['temporal_bin_relative']<=.0025 and out['retained_bin_relative']<=.0025 and all(v['pass'] for v in out['runs'].values())
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
