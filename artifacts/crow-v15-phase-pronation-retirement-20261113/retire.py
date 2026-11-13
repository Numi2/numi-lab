import hashlib,json,os,subprocess,sys
from pathlib import Path
p=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-phase-pronation-turn-20261113/signed-turn-20261126-rebound-phase/run.stderr.gz')
expected='dbe2f164b24d51d243502e5323463a4d703e4388b205449a9d976da024b90217'
assert p.is_file() and not p.is_symlink(); s=p.stat(); assert s.st_nlink==1
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==expected
handles=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
assert handles.returncode==1 and not handles.stdout,(handles.returncode,handles.stdout,handles.stderr)
now=p.stat();assert (now.st_ino,now.st_size,now.st_mtime_ns)==(s.st_ino,s.st_size,s.st_mtime_ns)
if '--retire' in sys.argv:
 pre=json.loads(Path(sys.argv[sys.argv.index('--retire')+1]).read_text()); assert (pre['inode'],pre['bytes'],pre['mtime_ns'],pre['sha256'])==(s.st_ino,s.st_size,s.st_mtime_ns,expected);os.unlink(p)
print(json.dumps({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'remote_archive_commit':'6ea957345c977da40c951ea12613cfc3401d0468','lsof_clear':True,'retired':'--retire' in sys.argv},indent=2))
