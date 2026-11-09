from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd();check=json.loads((R/'artifacts/crow-gate-remote-restore-20261109/results.json').read_text())
assert len(check['rows'])==6 and all(x['matches_manifest'] for x in check['rows'])
removed=[]
for row in check['rows']:
 parts=Path(row['path']).parts
 raw=R/'build'/parts[1]/(parts[2]+'.rollout')
 assert raw.is_file() and raw.stat().st_size==row['raw_bytes']
 h=hashlib.sha256()
 with raw.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==row['raw_sha256']
 p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True)
 assert p.returncode==1 and not p.stdout,('file in use or lsof failed',str(raw),p.returncode,p.stdout,p.stderr)
 os.remove(raw)
 removed.append({'path':str(raw.relative_to(R)),'bytes':row['raw_bytes'],'sha256':row['raw_sha256']})
(R/'artifacts/crow-gate-remote-restore-20261109/retired.json').write_text(json.dumps({'remote_revision':check['remote_revision'],'removed_raw_copies':removed,'total_bytes':sum(x['bytes'] for x in removed),'note':'Compressed research records remain in the repository and were independently streamed from GitHub and hash checked before removal.'},indent=2)+'\n')
print('retired bytes',sum(x['bytes'] for x in removed))
