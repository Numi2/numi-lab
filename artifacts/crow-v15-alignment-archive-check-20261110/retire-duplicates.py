from pathlib import Path
import gzip,hashlib,json
R=Path.cwd();B=R/'build/crow-v15-integrated-route-baseline-20261110';A=R/'artifacts';dirs=['crow-v15-leg-push-sweep-20261110','crow-v15-leg-push-timing-20261110','crow-v15-landing-support-alignment-20261110','crow-v15-landing-forward-offset-20261110'];items=[]
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for d in dirs:
 for arc in (A/d).glob('diagnostic-reserved-final-j-*.*.gz'):
  if not arc.name.endswith(('.json.gz','.tsv.gz')):continue
  raw=B/arc.name[:-3]
  if not raw.is_file():continue
  with raw.open('rb') as f:dr=digest(f)
  with gzip.open(arc,'rb') as f:da=digest(f)
  if dr==da:
   size=raw.stat().st_size;raw.unlink();items.append({'path':str(raw.relative_to(R)),'bytes':size,'sha256':dr,'verified_archive':str(arc.relative_to(R))})
print(json.dumps({'retired':items,'bytes':sum(x['bytes'] for x in items),'note':'Only byte-identical raw duplicates of independently restored archives are retired.'},indent=2))
