import hashlib,json,os,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]);expected='ea15cbcda7974f603ea48bd2df331552ee017e19d9b41c19f4eb945d1c85c6bb';assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1;h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
assert h.hexdigest()==expected
r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
z=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
out={'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True,'retired':False}
if len(sys.argv)>2:
 assert json.loads(sys.argv[2])==out;os.unlink(p);out['retired']=True
print(json.dumps(out,indent=2))
