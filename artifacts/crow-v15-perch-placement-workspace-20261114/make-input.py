from pathlib import Path
import json,hashlib
A=Path('artifacts/crow-v15-perch-placement-workspace-20261114');P=Path('artifacts/crow-v15-stance-toe-workspace-20261114');m=json.loads((P/'inputs.json').read_text())['records'];source=[line.split() for line in (P/'input.tsv').read_text().splitlines()];assert len(m)==len(source)==513
records=[]
with (A/'input.tsv').open('w') as f:
 for delta in (-.004,-.002,0.,.002,.004):
  for k in range(1,513):
   z=source[k].copy();n=len(records);z[1]=str(n);z[2]=str(float(z[2])+delta);f.write(' '.join(z)+'\n');records.append({'id':n,'offset_m':delta,'kind':m[k]['kind'],'angles':m[k]['angles'],'prior_id':k})
(A/'inputs.json').write_text(json.dumps({'source_input_sha256':hashlib.sha256((P/'input.tsv').read_bytes()).hexdigest(),'records':records},indent=2)+'\n')
