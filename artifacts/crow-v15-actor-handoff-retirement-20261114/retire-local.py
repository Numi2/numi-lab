from pathlib import Path
import hashlib,json,os,subprocess,sys
P=Path('artifacts/crow-v15-actor-handoff-retirement-20261114');D=Path.cwd()/'artifacts/crow-v15-actor-command-flight-20261114/turn-20261126-rebound-actor';m=json.loads(subprocess.check_output(['git','show','HEAD:artifacts/crow-v15-actor-command-flight-20261114/turn-20261126-rebound-actor/stderr-parts.json'],text=True));v=json.loads(subprocess.check_output(['git','show','HEAD:artifacts/crow-v15-actor-command-flight-20261114/github-verification.json'],text=True));assert (m['bytes'],m['sha256'],m['parts'])==(v['bytes'],v['sha256'],v['parts']) and v['gzip_eof'];targets=[(D/m['file'],m['bytes'],m['sha256'])]+[(D/x['file'],x['bytes'],x['sha256']) for x in m['parts']];out=[]
for path,n,expected in targets:
 assert path.is_file() and not path.is_symlink();s=path.stat();assert s.st_nlink==1 and s.st_size==n and s.st_blocks>0;flags=subprocess.check_output(['ls','-ldO',str(path)],text=True);assert 'dataless' not in flags;h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected;r=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout;s2=path.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);out.append({'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((P/'local-preflight.json').read_text());assert pre['files']==out
 for x in out:
  s=Path(x['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
 for x in out:os.unlink(x['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':out,'total_bytes':sum(x['bytes'] for x in out)},indent=2))
