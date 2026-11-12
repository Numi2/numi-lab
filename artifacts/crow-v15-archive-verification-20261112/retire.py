from pathlib import Path
import json,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-archive-verification-20261112';data=json.loads((A/'verified.json').read_text());assert data['files_verified']==22 and len(data['files'])==22
paths=[]
for x in data['files']:
 p=R/x['raw'];assert p.is_file() and p.stat().st_size==x['raw_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['raw_sha256'];paths.append(p)
opened=subprocess.run(['lsof','-F','n',*[str(x) for x in paths]],text=True,capture_output=True);assert opened.returncode==1 and not opened.stdout,(opened.returncode,opened.stdout[:500],opened.stderr[:500])
for p in paths:p.unlink()
assert all(not p.exists() for p in paths)
out={'remote_commit':data['remote_commit'],'retired_files':len(paths),'retired_bytes':data['total_raw_bytes'],'verification':'Independent filtered remote clone, git-stored manifest hash, downloaded compressed blob hash, decompressed exact raw SHA-256, local archive hash and no open handles checked before retirement. Only named generated duplicate traces/replays removed.'};(A/'retired.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
