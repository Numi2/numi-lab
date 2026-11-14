import hashlib,json,os,subprocess,sys
from pathlib import Path
names={'turn':('turn-20261126-rebound-earlier','1ae4021f4bec661c2c0955fb960bdf94ba6dbd4fcc146302d554717c56dc951e')}
folder,expected=names[sys.argv[1]]
p=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-earlier-height-capture-20261114')/folder/'run.stderr.gz'
assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==expected
r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
s2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns)
out={'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True,'retired':False}
if len(sys.argv)>2:
 pre=json.loads(sys.argv[2]);assert pre==out;os.unlink(p);out['retired']=True
print(json.dumps(out,indent=2))
