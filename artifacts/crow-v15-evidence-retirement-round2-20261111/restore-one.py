from pathlib import Path
import gzip,hashlib,json,os,shutil,sys
R=Path.cwd();manifest=json.loads((R/'artifacts/crow-v15-evidence-retirement-round2-20261111/result.json').read_text());wanted=sys.argv[1] if len(sys.argv)>1 else ''
rows=[x for x in manifest['files_retired'] if x['raw_path'].endswith(wanted)]
if len(rows)!=1:raise SystemExit('Specify a unique raw path suffix from result.json')
r=rows[0];src=R/r['remote_object'];target=Path(r['raw_path']);tmp=target.with_name(target.name+'.restoring')
if target.exists() or tmp.exists():raise SystemExit('Refusing to overwrite an existing file')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
assert digest(src)==r['remote_compressed_sha256'];target.parent.mkdir(parents=True,exist_ok=True)
try:
 with gzip.open(src,'rb') as f,tmp.open('xb') as out:shutil.copyfileobj(f,out)
 assert digest(tmp)==r['raw_sha256'];tmp.rename(target)
except BaseException:
 tmp.unlink(missing_ok=True);raise
print('Restored',target)
