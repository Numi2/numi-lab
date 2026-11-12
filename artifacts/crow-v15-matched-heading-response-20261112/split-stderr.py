from pathlib import Path
import hashlib,json
D=Path('artifacts/crow-v15-matched-heading-response-20261112/candidate-20261126-rebound-hip');src=D/'run.stderr.gz';h=hashlib.sha256();parts=[]
with src.open('rb') as f:
 i=0
 while True:
  chunk=f.read(32*1024*1024)
  if not chunk:break
  h.update(chunk);p=D/f'run.stderr.gz.part{i:03d}';p.write_bytes(chunk);parts.append({'name':p.name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()});i+=1
assert hashlib.sha256(b''.join((D/x['name']).read_bytes() for x in parts)).digest()==h.digest()
(D/'stderr-parts.json').write_text(json.dumps({'whole_sha256':h.hexdigest(),'parts':parts},indent=2)+'\n')
print('verified compressed stream parts',len(parts),h.hexdigest())
