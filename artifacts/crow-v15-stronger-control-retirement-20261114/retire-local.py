from pathlib import Path
import hashlib,json,os,subprocess,sys
A=Path(__file__).resolve().parent;R=A.parent;D=R/'crow-v15-stronger-disturbance-20261114';spec=[('control-20261127-rebound-observer-control','crow-v15-stronger-control-github-verification-20261114')];targets=[]
for folder,ver in spec:
 p=D/folder;m=json.loads((p/'stderr-parts.json').read_text());v=json.loads((R/ver/'verification.json').read_text());assert (v['sha256'],v['bytes'],v['gzip_eof'])==(m['sha256'],m['bytes'],True);targets.append((p/'run.stderr.gz',m['bytes'],m['sha256']));targets += [(p/x['file'],x['bytes'],x['sha256']) for x in m['parts']]
entries=[]
for p,n,expected in targets:
 assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==n;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==expected;r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 entries.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((A/'preflight.json').read_text());assert pre['files']==entries
 for e in entries:
  p=Path(e['path']);s=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(e['inode'],e['bytes'],e['mtime_ns'])
 for e in entries:os.unlink(e['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
