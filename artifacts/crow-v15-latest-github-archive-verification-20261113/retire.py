from pathlib import Path
import hashlib,json,os,subprocess,sys
A=Path(__file__).resolve().parent;R=A.parents[1];verified=json.loads((A/'verification.json').read_text());entries=[]
for x in verified['files']:
 p=R/x['path'];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==x['sha256'] and x['gzip_eof'];r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 entries.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':x['sha256'],'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((A/'preflight.json').read_text());assert pre['files']==entries
 for e in entries:
  p=Path(e['path']);s=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(e['inode'],e['bytes'],e['mtime_ns'])
 for e in entries:os.unlink(e['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
