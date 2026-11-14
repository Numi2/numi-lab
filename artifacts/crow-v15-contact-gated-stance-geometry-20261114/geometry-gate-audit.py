from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-contact-gated-stance-geometry-20261114');g={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='stanceGeometry':g[int(z[1]),int(z[2])]={'active':int(z[3]),'offset_rad':float(z[4])}
assert len(g)==900
out={'lanes':{str(e):{'first_active_zero_based':next((t for t in range(450) if g[e,t]['active']),None),'peak_offset_rad':max(g[e,t]['offset_rad'] for t in range(450))} for e in (0,1)},'scope':'Controller latch is driven by preceding endpoint contact; observer epoch alignment is not independently verified.'}
(A/'geometry-gate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
