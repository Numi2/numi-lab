from pathlib import Path
import hashlib,json,os,subprocess,sys
p=Path('artifacts/crow-v15-matched-nonleg-retirement-20261114');v=json.loads(subprocess.check_output(['git','show','HEAD:artifacts/crow-v15-matched-nonleg-verification-20261114/verification.json'],text=True));roots=[Path.cwd(),Path('/Users/home/Documents/Codex/2026-11-07/c/crow-research-staging')];entries=[]
for root in roots:
 for arm in ('push','hold'):
  path=root/'artifacts/crow-v15-matched-nonleg-ablation-20261114'/(arm+'-20261126-rebound-nonleg')/'run.stderr.gz';x=v[arm];assert x['gzip_eof'] and path.is_file() and not path.is_symlink();s=path.stat();assert s.st_nlink==1 and s.st_size==x['bytes'] and s.st_blocks>0;assert 'dataless' not in subprocess.check_output(['ls','-ldO',str(path)],text=True);h=hashlib.sha256()
  with path.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==x['sha256'];q=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True);assert q.returncode==1 and not q.stdout;s2=path.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);entries.append({'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest(),'lsof_clear':True})
if '--retire' in sys.argv:
 assert json.loads((p/'local-preflight.json').read_text())['files']==entries
 for x in entries:
  s=Path(x['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
 for x in entries:os.unlink(x['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
