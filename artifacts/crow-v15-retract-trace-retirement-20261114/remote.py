from pathlib import Path
import subprocess,json,hashlib,os,sys
v=json.loads(sys.argv[1]);out=[]
for arm in ('push','hold'):
 x=v['arms'][arm];p=Path('/Users/n/Developer/numi-crow-rl-20261107')/x['path'];assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_blocks>0 and s.st_size==x['bytes'];assert 'dataless' not in subprocess.check_output(['ls','-ldO',str(p)],text=True);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==x['sha256'];r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout;s2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns);out.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest(),'lsof_clear':True})
if len(sys.argv)>2:
 assert json.loads(sys.argv[2])['files']==out
 for x in out:
  s=Path(x['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
 for x in out:os.unlink(x['path'])
print(json.dumps({'files':out,'retired':len(sys.argv)>2},indent=2))
