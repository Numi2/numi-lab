import json,pathlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
a=json.loads((p/'sweep.json').read_text())
b=json.loads((root/'docs/evidence/contact-p2-n2-trajectory-corrected-20261108/fine.json').read_text())
peak=max(b['force_bins_N']);out={'peak_bin_N':peak,'differences':{},'reproduction':{}}
for k in ('triangle8','triangle12','triangle14','triangle16','triangle18','duffy48'):
 diff=[abs(x['force_N'][k]-x['force_N']['duffy64'])/peak for x in a['samples']]
 i=max(range(len(diff)),key=diff.__getitem__)
 out['differences'][k]={'max_relative_to_peak':diff[i],'time_s':a['samples'][i]['time_s']}
for k,j in (('triangle8',1),('triangle12',2)):
 out['reproduction'][k]=max(abs(x['force_N'][k]-y[j]) for x,y in zip(a['samples'],b['trace']))/peak
out['min_sampled_gap_m']=min(min(s['min_gap_m'].values()) for s in a['samples'])
out['reference_pass']=out['differences']['duffy48']['max_relative_to_peak']<=.0005
out['reproduction_pass']=max(out['reproduction'].values())<=1e-10
out['prediction_pass']=all(out['differences'][k]['max_relative_to_peak']<=.0025 for k in ('triangle16','triangle18'))
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
