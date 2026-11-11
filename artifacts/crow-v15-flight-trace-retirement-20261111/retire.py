from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');raw=R/'build/crow-v15-toe-flight-16substep-20261111/18647329.tsv';arc='artifacts/crow-v15-toe-flight-16substep-20261111/18647329.trace.tsv.gz'
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip();subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor','0bd23003',rev],check=True)
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
p=subprocess.Popen(['git','-C',str(C),'show',rev+':'+arc],stdout=subprocess.PIPE)
with gzip.GzipFile(fileobj=p.stdout,mode='rb') as f:archived=digest(f)
if p.wait():raise SystemExit('remote restore failed')
with raw.open('rb') as f:local=digest(f)
assert archived==local;entry={'remote_commit':rev,'archive':arc,'retired':str(raw.relative_to(R)),'sha256':local,'bytes':raw.stat().st_size};raw.unlink();(R/'artifacts/crow-v15-flight-trace-retirement-20261111/retired.json').write_text(json.dumps(entry,indent=2)+'\n');print(entry)
