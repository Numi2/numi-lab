from pathlib import Path
import json
A=Path('artifacts/crow-v15-materialized-contact-feedback-20261112')
b=json.loads(Path('artifacts/crow-v15-substep-slip-observer-20261112/substep-contact-speed.json').read_text());c=json.loads((A/'substep-contact-speed-fine.json').read_text());u=json.loads((A/'contact-analysis-fine-20261127.json').read_text());d=json.loads((A/'displacement-fine.json').read_text())
index=lambda obj:{(x['lane'],x['side']):x for x in obj['rows']}; x=index(b);y=index(c);rows=[]
for e in range(8):
 ratios={side:y[e,side]['integral_of_mean_contact_speed_m']/x[e,side]['integral_of_mean_contact_speed_m'] for side in ('left','right')}
 q=u['rows'][e]['phase'];dx=d['rows'][e]['candidate']['dx_m'];qual=e>=2 and all(v<=.75 for v in ratios.values()) and q['both_unloaded_substeps']==0 and q['toe_cap_substeps']==0 and q['nonfoot_ground_impulse']==0 and dx>=.05
 rows.append({'lane':e,'gain':[0,0,.5,-.5,1,-1,2,-2][e],'contact_integral_ratio':ratios,'both_unloaded_substeps':q['both_unloaded_substeps'],'toe_cap_substeps':q['toe_cap_substeps'],'nonfoot_ground_impulse':q['nonfoot_ground_impulse'],'dx_m':dx,'pass_screen':qual})
out={'baseline':'crow-v15-substep-slip-observer-20261112/substep-contact-speed.json','condition':'at least 25% reduction for each foot, no simultaneous unload, no toe cap, no nonfoot contact, >= 0.05 m progress','rows':rows};(A/'fine-evaluation.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows:print(row)
