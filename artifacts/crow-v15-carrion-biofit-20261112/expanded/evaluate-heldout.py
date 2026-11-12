from pathlib import Path
import hashlib,json,numpy as np
A=Path(__file__).resolve().parent
fit=json.loads((A/'fit-training.json').read_text()); idx=json.loads((A/'source-sample-index.json').read_text());selected=next(r for r in fit['candidates'] if r['name']==fit['selected']); subjects=[]
for i in (8,9,10):
 rows=[r for r in idx['rows'] if r['individual_id']==i]
 if not rows:subjects.append({'individual_id':i,'count':0,'status':'no usable flying windows in selected clip'});continue
 peak=float(np.median([r['peak_hz'] for r in rows]));conc=float(np.median([r['band_concentration'] for r in rows]));subjects.append({'individual_id':i,'count':len(rows),'peak_hz':peak,'concentration':conc,'absolute_peak_difference_hz':abs(selected['median_peak_hz']-peak),'absolute_concentration_difference':abs(selected['median_concentration']-conc),'score':abs(selected['median_peak_hz']-peak)+abs(selected['median_concentration']-conc)/.25})
out={'training_commit':'96d91e799','training_result_sha256':hashlib.sha256((A/'fit-training.json').read_bytes()).hexdigest(),'selected':selected['name'],'sim_median_peak_hz':selected['median_peak_hz'],'sim_median_concentration':selected['median_concentration'],'held_out_subjects':subjects,'scope':'Only two of three held-out carrion crows supplied selected flying windows. Tail feather tag and simulated tail COM are different sensors; no flapping-only annotation, independent simulation replicate, or biological qualification.'}
(A/'heldout.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
