from pathlib import Path
import gzip,json
p=Path(__file__).resolve().parent
with gzip.open(p/'push-window.json.gz','rt') as f:d=json.load(f)
rows=[]
for e in (6,7):
 for s in d['state']:
  if s['lane']==e and 369<=s['step']<=377: rows.append({'lane':e,'observer_step':s['step'],'root_height_m':s['q'][2],'root_vertical_velocity_m_s':s['v'][2]})
out={'source_sha256':d['source_sha256'],'rows':rows,'scope':'observer q/v labels are inferred to lag contact intervals by one frame; root velocity is not whole-body COM velocity'}
(p/'upward-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
