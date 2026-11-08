import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
read=lambda x:json.loads(x.read_text())
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
af=p/'fine.json';rf=root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json'
a=read(af);ref=read(rf);peak=max(ref['force_bins_N']);imp=abs(ref['impulse_Ns']);r=a['trace'];t0=.0045
energy=.5*.0088*(9.81*.01)**2
work=max(max(abs(z[11]) for z in r),max(abs(z[10]) for z in r),1e-12)
v={'restart_sha256':sha(af),'reference_sha256':sha(rf),'peak_N':peak,'impulse_Ns':imp,
 'force_bin_difference_relative':max(abs(x-y) for x,y in zip(a['force_bins_N'],ref['force_bins_N']))/peak,
 'impulse_difference_relative':abs(a['impulse_Ns']-ref['impulse_Ns'])/imp,
 'work_difference_relative':abs(a['contact_work_J']-ref['contact_work_J'])/work,
 'min_sampled_gap_m':min(min(z[4:7]) for z in r),'min_sampled_J':min(z[7] for z in r),
 'duffy_relative':max(abs(z[2]-z[3]) for z in r)/peak,
 'order18_relative':max(abs(z[1]-z[3]) for z in r)/peak,
 'momentum_relative':max(abs(z[8]-r[0][8]-z[9]+.0088*9.81*(z[0]-t0)) for z in r)/imp,
 'energy_relative':max(abs(z[12]-r[0][12]) for z in r)/energy,
 'work_balance_relative':max(abs(z[10]+z[11]-r[0][11]) for z in r)/work,
 'local_mass_solve_relative':max(z[13] for z in r)}
v['shape_pass']=len(a['force_bins_N'])==10 and len(r)==56 and abs(r[0][0]-.0045)<1e-15 and abs(r[-1][0]-.01)<1e-15
v['prerequisite_pass']=v['shape_pass'] and v['min_sampled_gap_m']>0 and v['min_sampled_J']>0 and v['duffy_relative']<=.0005 and max(v[x] for x in ('force_bin_difference_relative','impulse_difference_relative','work_difference_relative','order18_relative','momentum_relative','energy_relative','work_balance_relative','local_mass_solve_relative'))<=.0025
(p/'analysis.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
if not v['prerequisite_pass']:raise SystemExit('prerequisite failed')
