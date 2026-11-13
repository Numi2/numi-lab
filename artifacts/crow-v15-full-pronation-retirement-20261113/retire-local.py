from pathlib import Path
import hashlib,json,os,subprocess,sys
A=Path(__file__).resolve().parent;R=A.parent;phase=R/'crow-v15-phase-pronation-turn-20261113/signed-turn-20261126-rebound-phase';full=R/'crow-v15-full-pronation-observer-20261113/signed-turn-20261126-rebound-observer'
pm=json.loads((phase/'stderr-parts.json').read_text());fm=json.loads((full/'stderr-parts.json').read_text());pv=json.loads((R/'crow-v15-phase-pronation-github-verification-20261113/verification.json').read_text());fv=json.loads((R/'crow-v15-full-pronation-github-verification-20261113/verification.json').read_text());rv=json.loads((R/'crow-v15-phase-pronation-turn-20261113/reference-verification.json').read_text());assert (pv['sha256'],pv['bytes'],pv['gzip_eof'])==(pm['sha256'],pm['bytes'],True) and (fv['sha256'],fv['bytes'],fv['gzip_eof'])==(fm['sha256'],fm['bytes'],True) and rv['gzip_eof']
targets=[(phase/'run.stderr.gz',pm['bytes'],pm['sha256']),(full/'run.stderr.gz',fm['bytes'],fm['sha256']),(Path('/tmp/crow-leg-longflight-reference-20261126.gz'),rv['bytes'],rv['sha256'])]+[(full/x['file'],x['bytes'],x['sha256']) for x in fm['parts']];entries=[]
for p,n,expected in targets:
 assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and s.st_size==n;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==expected;r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 entries.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((A/'local-preflight.json').read_text());assert pre['files']==entries
 for e in entries:
  p=Path(e['path']);s=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(e['inode'],e['bytes'],e['mtime_ns'])
 for e in entries:os.unlink(e['path'])
 print(json.dumps({'retired':True,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
else:print(json.dumps({'retired':False,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
