import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
read=lambda x:json.loads(x.read_text())
a=read(p/'coarse.json');b=read(p/'fine.json')
ref=read(root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json')
pre=read(root/'docs/evidence/contact-p2-n8-restart-full-20261108/analysis.json')
if not pre['prerequisite_pass']:raise SystemExit('n8 prerequisite failed')
peak=max(b['force_bins_N']);imp=abs(b['impulse_Ns'])
energy=.5*.0088*(9.81*.01)**2;t0=.0045
work=max(max(abs(z[11]) for z in b['trace']),max(abs(z[10]) for z in b['trace']),1e-12)
v={'peak_N':peak,'impulse_Ns':imp,'temporal_force_relative':max(abs(x-y) for x,y in zip(a['force_bins_N'],b['force_bins_N']))/peak,
 'temporal_impulse_relative':abs(a['impulse_Ns']-b['impulse_Ns'])/imp,
 'temporal_work_relative':abs(a['contact_work_J']-b['contact_work_J'])/work,'runs':{}}
for x in (a,b):
 r=x['trace'];w=max(max(abs(z[11]) for z in r),max(abs(z[10]) for z in r),1e-12)
 q={'shape_pass':len(x['force_bins_N'])==10 and len(r)==56 and abs(r[0][0]-.0045)<1e-15 and abs(r[-1][0]-.01)<1e-15,
  'min_sampled_gap_m':min(min(z[4:7]) for z in r),'min_sampled_J':min(z[7] for z in r),
  'duffy_relative':max(abs(z[2]-z[3]) for z in r)/peak,
  'order18_relative':max(abs(z[1]-z[3]) for z in r)/peak,
  'momentum_relative':max(abs(z[8]-r[0][8]-z[9]+.0088*9.81*(z[0]-t0)) for z in r)/imp,
  'energy_relative':max(abs(z[12]-r[0][12]) for z in r)/energy,
  'work_balance_relative':max(abs(z[10]+z[11]-r[0][11]) for z in r)/w,
  'local_mass_solve_relative':max(z[13] for z in r)}
 q['pass']=q['shape_pass'] and q['min_sampled_gap_m']>0 and q['min_sampled_J']>0 and q['duffy_relative']<=.0005 and max(q[k] for k in ('order18_relative','momentum_relative','energy_relative','work_balance_relative','local_mass_solve_relative'))<=.0025
 v['runs'][x['variant']]=q
v['numerical_prerequisite_pass']=max(v['temporal_force_relative'],v['temporal_impulse_relative'],v['temporal_work_relative'])<=.0025 and all(x['pass'] for x in v['runs'].values())
if v['numerical_prerequisite_pass']:
 f=max(abs(x-y) for x,y in zip(ref['force_bins_N'],b['force_bins_N']))/peak
 i=abs(ref['impulse_Ns']-b['impulse_Ns'])/imp
 v['n8_to_n16_spatial']={'force_bin_relative':f,'impulse_relative':i,'pass':f<=.0025 and i<=.0025}
(p/'analysis.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
