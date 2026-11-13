import hashlib,json,os,subprocess,sys
from pathlib import Path
m=json.load(sys.stdin);rows=m['rows'];allowed=('/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-low-wing-effort-screen-20261113/','/Users/n/Developer/numi-crow-rl-20261107/build/crow-v15-early-low-wing-screen-20261113/')
def check(r):
 p=Path(r['path']);assert any(str(p).startswith(x) for x in allowed);assert str(p).endswith(('-trace.tsv','-replay.json'));assert not p.is_symlink() and p.is_file()
 s=p.stat();assert s.st_ino==r['inode'] and s.st_size==r['size_bytes'] and int(s.st_mtime)==r['mtime_epoch']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==r['sha256']==r['restored_sha256']
 check=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
 assert check.returncode==1 and not check.stdout,(str(p),check.stdout,check.stderr)
for r in rows:check(r)
retired=[]
for r in rows:
 check(r);os.unlink(r['path']);retired.append({'path':r['path'],'sha256':r['sha256'],'size_bytes':r['size_bytes']})
df=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
print(json.dumps({'verified_archive_commit':rows[0]['commit'],'retired':retired,'bytes_retired':sum(x['size_bytes'] for x in retired),'after_df_k':df},indent=2))
