from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve();verified=json.loads(Path(sys.argv[2]).read_text());files=[]
for x in verified:
 assert x['gzip_eof'] is True and x['source_commit']=='c1019e75f8020455c77d14dbf1aaeaa1d19a2b7d'
 p=root/x['path'];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==x['sha256'];r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 files.append({'path':str(p),'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest(),'lsof_clear':True})
if len(sys.argv)>3:
 pre=json.loads(Path(sys.argv[3]).read_text());assert pre['files']==files
 for x in files:
  p=Path(x['path']);s=p.stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
 for x in files:Path(x['path']).unlink()
print(json.dumps({'retired':len(sys.argv)>3,'files':files,'total_bytes':sum(x['bytes'] for x in files)},indent=2))
