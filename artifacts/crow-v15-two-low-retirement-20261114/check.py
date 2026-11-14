import hashlib,json,os,subprocess,sys
from pathlib import Path
spec=json.loads(sys.argv[1]);retire=len(sys.argv)>2
entries=[]
for item in spec:
 p=Path(item['path']);assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==item['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==item['sha256'];r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 entries.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':item['sha256'],'lsof_clear':True})
if retire:
 pre=json.loads(sys.argv[2]);assert pre['files']==entries
 for e in entries:
  s=Path(e['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(e['inode'],e['bytes'],e['mtime_ns'])
 for e in entries:os.unlink(e['path'])
print(json.dumps({'retired':retire,'files':entries,'total_bytes':sum(e['bytes'] for e in entries)},indent=2))
