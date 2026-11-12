from pathlib import Path
import gzip,hashlib,json
A=Path('artifacts/crow-v15-rear-toe-damping-perch-screen-20261112');P=Path('artifacts/crow-v15-release-perturbation-screen-20261112/release')
def digest(p):
 h=hashlib.sha256()
 with gzip.open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
rows={}
for n in ('trace.log.gz','substeps.log.gz'):
 old=digest(P/n);new=digest(A/'baseline'/n);rows[n]={'previous_decompressed_sha256':old,'baseline_decompressed_sha256':new,'exact':old==new}
(A/'baseline-parity.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
