from pathlib import Path
import hashlib,json,os,subprocess,sys
A=Path(__file__).resolve().parent; R=A.parent; D=R/'crow-v15-altitude-bias-correction-20261114'; arm='altitude'
p=D/'altitude-20261126-rebound-corrected'; m=json.loads((p/'stderr-parts.json').read_text()); v=json.loads((R/'crow-v15-corrected-height-verification-20261114'/'altitude-verification.json').read_text())
assert (v['sha256'],v['bytes'],v['gzip_eof'])==(m['sha256'],m['bytes'],True)
assert v['parts']==m['parts']
targets=[(p/'run.stderr.gz',m['bytes'],m['sha256'])]+[(p/x['file'],x['bytes'],x['sha256']) for x in m['parts']]; entries=[]
for path,n,expected in targets:
 assert path.is_file() and not path.is_symlink(); s=path.stat(); assert s.st_nlink==1 and s.st_size==n; h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 assert h.hexdigest()==expected
 r=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True); assert r.returncode==1 and not r.stdout
 s2=path.stat(); assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns)
 entries.append({'path':str(path),'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':expected,'lsof_clear':True})
if '--retire' in sys.argv:
 pre=json.loads((A/(arm+'-preflight.json')).read_text()); assert pre['files']==entries
 for e in entries:
  s=Path(e['path']).stat(); assert (s.st_ino,s.st_size,s.st_mtime_ns)==(e['inode'],e['bytes'],e['mtime_ns'])
 for e in entries: os.unlink(e['path'])
print(json.dumps({'retired':'--retire' in sys.argv,'files':entries,'total_bytes':sum(x['bytes'] for x in entries)},indent=2))
