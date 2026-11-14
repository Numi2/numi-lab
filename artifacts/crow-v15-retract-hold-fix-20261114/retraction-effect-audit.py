from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-retract-hold-fix-20261114');rows=[]
for arm in ('push','hold'):
 d=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'));v={(x[0],x[1]):x[2:] for x in d['target']}
 for a,b in ((0,1),(2,3),(4,5),(6,7)):
  missing=[t for t in range(370,391) if (a,t) not in v or (b,t) not in v]
  differences=[{'t':t,'max_abs_leg_target_difference_rad':max(abs(v[a,t][k]-v[b,t][k]) for k in range(13,19))} for t in range(370,391) if (a,t) in v and (b,t) in v and max(abs(v[a,t][k]-v[b,t][k]) for k in range(13,19))>1e-7]
  rows.append({'arm':arm,'hold_lane':a,'retract_lane':b,'window':[370,390],'missing':missing,'differences':differences})
out={'rows':rows,'scope':'Cross-lane logged target comparison; initial positions differ by lane and sensor/latch internals were not observed. An unchanged output does not prove which condition delayed release.'}
(A/'retraction-effect-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
