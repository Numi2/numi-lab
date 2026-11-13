import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/Users/n/Developer/numi-crow-rl-20261107').resolve();manifest=Path('/tmp/crow-space-verified-batch1.jsonl');assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='7515dc3684042c5e25c67888c82061b9bf4d9b3a2c57cff93a878c380a09cad1'
records=[json.loads(x) for x in manifest.read_text().splitlines()];tracked=set(subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0'));r=subprocess.run(['lsof','-nP','-Fn'],capture_output=True,text=True);assert r.returncode in (0,1);opened={x[1:] for x in r.stdout.splitlines() if x.startswith('n/')}
receipt=Path('/tmp/crow-space-retired-batch1.jsonl');receipt.write_text('');before=shutil=None;fs=os.statvfs('/System/Volumes/Data');print('Before available bytes',fs.f_bavail*fs.f_frsize,flush=True)
for x in records:
 if x.get('status')!='verified':continue
 p=root/x['path'];out={'path':x['path'],'sha256':x['sha256'],'bytes':x['bytes']}
 try:
  assert x['path'] not in tracked and str(p) not in opened and not p.is_symlink() and p.resolve().is_relative_to(root)
  st=p.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev,st.st_mode)==(x['bytes'],x['mtime_ns'],x['inode'],x['device'],x['mode']) and st.st_nlink==1
  att=subprocess.run(['xattr',str(p)],capture_output=True,text=True);assert att.returncode==0 and not att.stdout.strip()
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==x['sha256'];st2=p.stat();assert (st2.st_size,st2.st_mtime_ns,st2.st_ino,st2.st_dev)==(st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  p.unlink();out['status']='retired'
 except Exception as e:out.update(status='skipped',reason=repr(e))
 with receipt.open('a') as f:f.write(json.dumps(out,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
fs=os.statvfs('/System/Volumes/Data');a=[json.loads(z) for z in receipt.read_text().splitlines()];print('After available bytes',fs.f_bavail*fs.f_frsize,'retired',sum(z['status']=='retired' for z in a),'skipped',sum(z['status']!='retired' for z in a),'logical bytes',sum(z['bytes'] for z in a if z['status']=='retired'),flush=True)
