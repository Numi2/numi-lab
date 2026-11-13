from pathlib import Path
import hashlib,json,os,subprocess,sys
m=json.load(open(sys.argv[1]));rows=m['rows'];assert len(rows)==12
allowed=('/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-mirrored-leg-wing-longflight-20261113/','/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-mirrored-height-switch-descent-20261113/','/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-leg-flight-height-switch-20261113/','/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-bank-damping-sensitivity-20261113/')
def check(r):
 p=Path(r['path']);assert any(str(p).startswith(a) for a in allowed) and str(p).endswith(('-trace.tsv','-replay.json'));assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1 and (s.st_ino,s.st_size,s.st_mtime_ns)==(r['inode'],r['bytes'],r['mtime_ns']) and r['gzip_eof_verified'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==r['sha256']==r['restored_sha256'];l=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout
for r in rows:check(r)
retired=[]
for r in rows:
 check(r);Path(r['path']).unlink();retired.append({'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'commit':r['commit']})
print(json.dumps({'retired':retired,'bytes_retired':sum(r['bytes'] for r in rows),'df':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)},indent=2))
