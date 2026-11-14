from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-wing-target-release-audit-20261114');S=Path('artifacts/crow-v15-early-flight-boost-20261114');data={}
for arm in ('push','hold'):
 with gzip.open(S/(arm+'-window.json.gz'),'rt') as f:x=json.load(f)
 data[arm]={(v[0],v[1]):v[2:] for v in x['target']}
rows=[]
for e in (6,7):
 missing=[t for t in range(335,408) if any((e,t) not in data[a] for a in ('push','hold'))]
 ds=[abs(data['push'][e,t][j]-data['hold'][e,t][j]) for t in range(335,408) if t not in missing for j in (8,9,10,11)]
 rows.append({'lane':e,'first':335,'last':407,'count':len(ds),'missing':missing,'max_abs_difference_rad':max(ds,default=None),'exact':not missing and len(ds)==292 and max(ds,default=1)==0})
out={'rows':rows,'note':'Supplemental post-run check; held-leg lane 6 first recontact is at interval 407. Later differences remain in the full-window report.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
