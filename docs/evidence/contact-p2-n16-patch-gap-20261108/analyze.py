import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'transfer.json').read_text())
prior=json.loads((root/'docs/evidence/contact-p2-n16-transfer-inset-20261108/transfer.json').read_text())
peak=a['peak_N'];r48=a['results']['48'];r64=a['results']['64'];expected_area=(.02/4)**2
get=lambda order,mesh,tag:next(s['force_N'] for s in prior['samples'] if s['order']==order and s['mesh']==mesh and s['index']==3 and s['kind']==tag)
v={'area_max_relative':max(abs(x-expected_area)/expected_area for r in (r48,r64) for x in r['areas_m2']),
 'total_area_max_relative':max(abs(r['area_sum_m2']-.02**2)/(.02**2) for r in (r48,r64)),
 'constant_force_max_relative':max(abs(r['constant_force_N']-r['constant_expected_N'])/abs(r['constant_expected_N']) for r in (r48,r64)),
 'min_sampled_gap_m':min(v['min_gap_m'] for r in (r48,r64) for v in r['measures'].values()),
 'actual_total_reproduction_relative':abs(sum(r64['measures']['actual']['force_N'])-get(64,16,'original'))/peak,
 'transferred_total_reproduction_relative':abs(sum(r64['measures']['1e-12']['force_N'])-get(64,8,'from16_eps1e-12'))/peak,
 'max_patch_duffy_relative':max(abs(x-y)/peak for name in r64['measures'] for x,y in zip(r48['measures'][name]['force_N'],r64['measures'][name]['force_N'])),
 'max_patch_inset_relative':max(abs(x-y)/peak for x,y in zip(r64['measures']['1e-12']['force_N'],r64['measures']['1e-13']['force_N']))}
v['controls_pass']=max(v['area_max_relative'],v['total_area_max_relative'],v['constant_force_max_relative'])<=1e-10 and v['min_sampled_gap_m']>0 and max(v['actual_total_reproduction_relative'],v['transferred_total_reproduction_relative'],v['max_patch_inset_relative'])<=1e-7 and v['max_patch_duffy_relative']<=.0005
v['patches']=[]
for i,(aa,bb) in enumerate(zip(r64['measures']['actual']['force_N'],r64['measures']['1e-12']['force_N'])):
 v['patches'].append({'x_index':i//4,'y_index':i%4,'actual_N':aa,'transferred_N':bb,'difference_N':bb-aa,'difference_relative_to_peak':(bb-aa)/peak,'mean_gap_difference_m':r64['gap_difference_mean_m'][i],'rms_gap_difference_m':r64['gap_difference_rms_m'][i],'max_abs_gap_difference_m':r64['gap_difference_max_abs_m'][i]})
v['sum_actual_N']=sum(r64['measures']['actual']['force_N']);v['sum_transferred_N']=sum(r64['measures']['1e-12']['force_N'])
(p/'analysis.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
