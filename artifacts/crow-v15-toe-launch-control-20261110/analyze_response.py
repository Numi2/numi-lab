from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-toe-launch-control-20261110');out={}
for arm in ('control','push'):
 lanes=[{} for _ in range(8)]
 for line in gzip.open(A/(arm+'.stderr.gz'),'rt'):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   d=json.loads(line.split(' ',1)[1]);lanes[d['lane']][d['step']]=d
 out[arm]=[{'lane':i,'start_z':l[248]['q'][2],'peak_z':max(l[t]['q'][2] for t in range(249,261)),'peak_vz':max(l[t]['v'][2] for t in range(249,261)),'first_qualifying_unloaded_upward':next((t for t in range(249,261) if l[t]['v'][2]>0 and all(not (c['impulses'][0]>1e-7 and 'locomotion_ground' in (c['a_name'],c['b_name']) and ('_toe' in c['a_name'] or '_toe' in c['b_name'] or '_foot' in c['a_name'] or '_foot' in c['b_name'])) for c in l[t]['contacts'])),None)} for i,l in enumerate(lanes)]
(A/'launch-response.json').write_text(json.dumps({'arms':out,'scope':'post-hoc secondary analysis, endpoint measurements; not prespecified gate'},indent=2)+'\n')
