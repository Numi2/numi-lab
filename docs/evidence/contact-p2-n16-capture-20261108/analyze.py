import json,math,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'fine.json').read_text())
ref=json.loads((root/'docs/evidence/contact-p2-n16-onset-20261108/fine.json').read_text())
full=json.loads((root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json').read_text())
peak=max(full['force_bins_N']);imp=abs(full['impulse_Ns']);t0=.0045;E=.5*.0088*(9.81*t0)**2
r=a['trace'];W=max(max(abs(z[11]) for z in r),max(abs(z[10]) for z in r),1e-12)
out={'n8_peak_bin_N':peak,'max_force_reproduction_relative':max(abs(z[1]-v[1]) for z,v in zip(r,ref['trace']))/peak,
 'bin_reproduction_relative':abs(a['bin4_N']-ref['bin4_N'])/peak,
 'min_sampled_gap_m':min(min(z[4:7]) for z in r),'min_sampled_J':min(z[7] for z in r),
 'duffy_48_64_relative':max(abs(z[2]-z[3]) for z in r)/peak,
 'order18_64_relative':max(abs(z[1]-z[3]) for z in r)/peak,
 'momentum_relative':max(abs(z[8]-r[0][8]-z[9]+.0088*9.81*(z[0]-t0)) for z in r)/imp,
 'energy_relative':max(abs(z[12]-r[0][12]) for z in r)/E,
 'contact_work_relative':max(abs(z[10]+z[11]-r[0][11]) for z in r)/W}
out['shapes_and_finite_pass']=len(a['displacements'])==6 and len(a['velocities'])==6 and all(len(v)==a['dofs'] and all(len(w)==3 and all(math.isfinite(x) for x in w) for w in v) for v in a['displacements']+a['velocities']) and len(r)==6 and all(abs(x[0]-y[0])<1e-15 for x,y in zip(r,ref['trace']))
out['capture_pass']=out['shapes_and_finite_pass'] and max(out['max_force_reproduction_relative'],out['bin_reproduction_relative'])<=1e-7 and out['min_sampled_gap_m']>0 and out['min_sampled_J']>0 and out['duffy_48_64_relative']<=.0005 and max(out['order18_64_relative'],out['momentum_relative'],out['energy_relative'],out['contact_work_relative'])<=.0025
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
