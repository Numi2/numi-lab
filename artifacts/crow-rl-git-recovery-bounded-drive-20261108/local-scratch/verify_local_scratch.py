from pathlib import Path
import gzip,hashlib,json,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-bounded-drive-20261108';D=G/'local-scratch';prefix=str(D.relative_to(R))+'/'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree();export=[]
with Objects() as objects:
 def read(name):return objects.read(entries[prefix+name][1])
 raw=read('manifest.json');m=json.loads(raw);assert m['schema']=='crow-local-scratch-recovery-v1';seen={}
 for f in m['files']:
  key=f['sha256'],f['bytes']
  if key in seen:continue
  data=gzip.decompress(read(f['blob']));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];seen[key]=True
 for p,(_,oid) in entries.items():
  if p.startswith(prefix):
   data=objects.read(oid);export.append({'path':p[len(prefix):],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob':oid})
report={'archive_commit':head,'remote_commit':remote,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'summary':m['summary'],'verified_source_unique_contents':len(seen),'archive_files':export,'scientific_qualification_claimed':False}
if '--stdout' in sys.argv:print(json.dumps(report))
else:
 (G/'local-scratch-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='archive_files'}))
