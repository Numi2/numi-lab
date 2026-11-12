from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-wing-steering-without-body-moments-20261112')
def read(arm):
 o={}
 with gzip.open(A/(arm+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if t<450:
     if (e,t) in o:raise ValueError('duplicate sample')
     o[e,t]=[x['q'],x['v'],x['policy_action']]
 return o
b=read('baseline');e=read('enhanced');keys=set(b)|set(e);diff=[k for k in sorted(keys) if b.get(k)!=e.get(k)]
x=json.loads((A/'baseline-analysis.json').read_text());y=json.loads((A/'enhanced-analysis.json').read_text())
rows=[]
for i in range(6):
 a=x['rows'][i];c=y['rows'][i]
 rows.append({'lane':i,'heading_target_rad':a['target_heading_rad'],'baseline_final_heading_error_rad':a['final_sample_heading_error_rad'],'enhanced_final_heading_error_rad':c['final_sample_heading_error_rad'],'enhanced_first_episode_steps':c['first_episode_steps'],'enhanced_failure':c['physical_failure'],'enhanced_ground_substeps':c['ground_contact_substeps_from_step400'],'enhanced_missing_substeps':c['missing_or_duplicate_substeps']})
out={'before_intervention':{'samples':len(keys),'baseline_samples':len(b),'enhanced_samples':len(e),'mismatches':diff[:20],'pass':bool(keys) and len(b)==len(e)==len(keys) and not diff},'rows':rows,'limits':'Sparse samples, previously used development seed and obstacle-free commands. Actuator source suppression applies only after 9 s, only to two identified body-moment channels.'}
(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
