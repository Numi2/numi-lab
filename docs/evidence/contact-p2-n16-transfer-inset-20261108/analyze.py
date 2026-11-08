import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'transfer.json').read_text())
n8=json.loads((root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json').read_text())
n16=json.loads((root/'docs/evidence/contact-p2-n16-capture-20261108/fine.json').read_text())
peak=a['peak_N']
by={(r['mesh'],r['index'],r['kind'],r['order']):r for r in a['samples']}
if len(by)!=72:raise ValueError('unexpected or repeated rows')
out={'peak_n8_full_bin_N':peak,'max_polynomial_abs_m':max(a['polynomial_controls'].values()),
 'min_sampled_gap_m':min(r['min_gap_m'] for r in a['samples']),'max_duffy_difference_relative':0.,
 'max_reproduction_difference_relative':0.,'max_inset_difference_relative':0.,'samples':[]}
for n,d in ((8,n8),(16,n16)):
 for j in range(6):
  source=24-n
  orig=by[n,j,'original',64]['force_N']
  saved=d['trace'][45+j if n==8 else j][3]
  out['max_reproduction_difference_relative']=max(out['max_reproduction_difference_relative'],abs(orig-saved)/peak)
  hi=by[n,j,f'from{source}_eps1e-12',64]['force_N']
  lo=by[n,j,f'from{source}_eps1e-13',64]['force_N']
  out['max_inset_difference_relative']=max(out['max_inset_difference_relative'],abs(hi-lo)/peak)
  for tag in ('original',f'from{source}_eps1e-12',f'from{source}_eps1e-13'):
   out['max_duffy_difference_relative']=max(out['max_duffy_difference_relative'],abs(by[n,j,tag,48]['force_N']-by[n,j,tag,64]['force_N'])/peak)
  source_orig=by[source,j,'original',64]['force_N']
  out['samples'].append({'time_s':.0045+j*.0001,'source_mesh':source,'destination_mesh':n,
   'source_force_N':source_orig,'transferred_force_N':hi,'destination_actual_force_N':orig,
   'transfer_change_relative_to_peak':(hi-source_orig)/peak})
out['controls_pass']=out['max_polynomial_abs_m']<=1e-12 and out['min_sampled_gap_m']>0 and out['max_duffy_difference_relative']<=.0005 and max(out['max_reproduction_difference_relative'],out['max_inset_difference_relative'])<=1e-7
out['max_abs_transfer_change_relative_by_direction']={}
for src,dst in ((8,16),(16,8)):
 vals=[abs(s['transfer_change_relative_to_peak']) for s in out['samples'] if s['source_mesh']==src and s['destination_mesh']==dst]
 out['max_abs_transfer_change_relative_by_direction'][f'{src}_to_{dst}']=max(vals)
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
