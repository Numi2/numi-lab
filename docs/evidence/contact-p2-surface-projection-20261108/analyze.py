import json,math,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
projection=json.loads((p/'projection.json').read_text())
paths={4:root/'docs/evidence/contact-p2-n4-order18-20261108/fine.json',8:root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json'}
data={n:json.loads(path.read_text()) for n,path in paths.items()}
peak=max(data[8]['force_bins_N'])
out={'n8_peak_bin_N':peak,'free_fall_activation_s':math.sqrt(2*.0001/9.81),'directions':{},'episodes':{}}
for name in ('4_to_8','8_to_4'):
 xs=[r['directions'][name] for r in projection['samples']]
 out['directions'][name]={'max_projection_difference_relative':max(abs(x['original_force_N']-x['projected_force_N']) for x in xs)/peak,
 'max_reproduction_difference_relative':max(abs(x['original_force_N']-x['recorded_duffy64_force_N']) for x in xs)/peak,
 'min_sampled_gap_m':min(min(x['original_min_gap_m'],x['projected_min_gap_m']) for x in xs)}
for n in (4,8):
 r=data[n]['trace'];on=False;episodes=[]
 for i,row in enumerate(r):
  active=row[1]>1e-10
  if active and not on:start=i;on=True
  if on and (not active or i==len(r)-1):
   episodes.append({'first_active_sample_s':r[start][0],'end_sample_s':r[i][0],
    'impulse_between_sample_bounds_Ns':r[i][9]-r[max(0,start-1)][9]});on=False
 out['episodes'][str(n)]=episodes
out['control_pass']=out['directions']['4_to_8']['max_projection_difference_relative']<=.0005
out['reproduction_pass']=max(x['max_reproduction_difference_relative'] for x in out['directions'].values())<=1e-10
out['prediction_pass']=out['directions']['8_to_4']['max_projection_difference_relative']>.0025
out['sampled_episode_impulse_difference_relative']=abs(data[4]['impulse_Ns']-data[8]['impulse_Ns'])/abs(data[8]['impulse_Ns'])
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
