from pathlib import Path
import gzip,hashlib,json,shutil
R=Path.cwd();B=R/'build/crow-four-expert-eval-20261109';A=R/'artifacts/crow-four-expert-trace-archive-20261109';rows=[]
for p in sorted(B.glob('*/*-trace.tsv')):
 relative=p.relative_to(B);out=A/(str(relative)+'.gz');out.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as f,out.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:shutil.copyfileobj(f,z)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert hashlib.sha256(gzip.decompress(out.read_bytes())).hexdigest()==h
 result=R/'artifacts/crow-four-expert-eval-20261109'/relative.parent/(relative.name.removesuffix('-trace.tsv'))/'results.json'
 rows.append({'raw_path':str(p.relative_to(R)),'archive_path':str(out.relative_to(R)),'sha256':h,'bytes':p.stat().st_size,'completed_result':str(result.relative_to(R)) if result.exists() else None})
(A/'manifest.json').write_text(json.dumps({'rows':rows,'scope':'single-lane diagnostic traces; entries without completed_result are incomplete stopped diagnostics'},indent=2)+'\n')
print(len(rows),'traces archived',sum(x['bytes'] for x in rows),'raw bytes')
