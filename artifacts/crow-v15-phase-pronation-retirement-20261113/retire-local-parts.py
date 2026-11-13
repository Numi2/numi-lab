from pathlib import Path
import hashlib,json,os,subprocess
A=Path(__file__).resolve().parent;P=A.parent/'crow-v15-phase-pronation-turn-20261113/signed-turn-20261126-rebound-phase';m=json.loads((P/'stderr-parts.json').read_text());items=[]
for x in m['parts']:
 p=P/x['file'];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==x['sha256'];r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 items.append((p,s,x))
for p,s,x in items:
 t=p.stat();assert (t.st_ino,t.st_size,t.st_mtime_ns)==(s.st_ino,s.st_size,s.st_mtime_ns)
for p,s,x in items:os.unlink(p)
(A/'local-parts-receipt.json').write_text(json.dumps({'source_commit':'6ea957345c977da40c951ea12613cfc3401d0468','independent_github_verification':'../crow-v15-phase-pronation-github-verification-20261113/verification.json','removed':[{'name':p.name,'bytes':s.st_size,'inode':s.st_ino,'sha256':x['sha256']} for p,s,x in items]},indent=2)+'\n');print('retired exact verified local parts',sum(s.st_size for p,s,x in items))
