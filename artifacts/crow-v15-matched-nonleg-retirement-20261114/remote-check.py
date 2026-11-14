from pathlib import Path
import hashlib,json,os,subprocess,sys
v=json.loads(sys.argv[1]);entries=[];base=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-matched-nonleg-ablation-20261114')
for arm in ('push','hold'):
 p=base/(arm+'-20261126-rebound-nonleg')/'run.stderr.gz';x=v[arm];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==x['sha256'];q=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert q.returncode==1 and not q.stdout;s2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);entries.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest(),'lsof_clear':True})
if len(sys.argv)>2:
 assert json.loads(sys.argv[2])['files']==entries
 for x in entries:
  s=Path(x['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
 for x in entries:os.unlink(x['path'])
print(json.dumps({'retired':len(sys.argv)>2,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
