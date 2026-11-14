from pathlib import Path
import hashlib,json,os,subprocess,sys
p=Path(__file__).resolve().parent;r=p.parent;d=r/'crow-v15-matched-wing-leg-ablation-20261114';v=json.loads((r/'crow-v15-matched-wing-verification-20261114/verification.json').read_text());entries=[]
for arm in ('push','hold'):
 path=d/(arm+'-20261126-rebound-paired500')/'run.stderr.gz';expected=v[arm];assert expected['gzip_eof'] and path.is_file() and not path.is_symlink();s=path.stat();assert s.st_nlink==1 and s.st_size==expected['bytes'];h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected['sha256'];q=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True);assert q.returncode==1 and not q.stdout;s2=path.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);entries.append({'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest(),'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((p/'preflight.json').read_text());assert pre['files']==entries
 for x in entries:
  s=Path(x['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
 for x in entries:os.unlink(x['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
