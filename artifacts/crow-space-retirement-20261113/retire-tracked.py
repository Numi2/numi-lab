import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/Users/n/Developer/numi-crow-rl-20261107').resolve();batch=sys.argv[1];expected=sys.argv[2];manifest=Path('/tmp/crow-space-verified-'+batch+'.jsonl');assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
records=[json.loads(x) for x in manifest.read_text().splitlines()];head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head=='4ab800743a06cc955774b1812849c720ae19d24f'
indexlines=subprocess.check_output(['git','-C',str(root),'ls-files','--stage','-z']).decode().split('\0');index={z.split('\t',1)[1]:z.split()[1] for z in indexlines if '\t' in z and z.split()[2]=='0'};hl=subprocess.check_output(['git','-C',str(root),'ls-tree','-r',head,'--','artifacts'],text=True).splitlines();ht={z.split('\t',1)[1]:z.split()[2] for z in hl if '\t' in z};flags={z[2:]:z[:1] for z in subprocess.check_output(['git','-C',str(root),'ls-files','-t','-z']).decode().split('\0') if len(z)>2}
r=subprocess.run(['lsof','-nP','-Fn'],capture_output=True,text=True);assert r.returncode in (0,1);opened={x[1:] for x in r.stdout.splitlines() if x.startswith('n/')};receipt=Path('/tmp/crow-space-retired-'+batch+'.jsonl');receipt.write_text('');fs=os.statvfs('/System/Volumes/Data');print('Before available bytes',fs.f_bavail*fs.f_frsize,flush=True)
for i,x in enumerate(records):
 if x.get('status')!='verified':continue
 p=root/x['path'];out={'path':x['path'],'sha256':x['sha256'],'bytes':x['bytes'],'git_blob':x['git_blob']};marked=False
 try:
  assert index.get(x['path'])==x['git_blob']==ht.get(x['path']) and flags.get(x['path'])=='H' and str(p) not in opened and not p.is_symlink() and p.resolve().is_relative_to(root)
  st=p.stat();assert (st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev,st.st_mode)==(x['bytes'],x['mtime_ns'],x['inode'],x['device'],x['mode']) and st.st_nlink==1
  att=subprocess.run(['xattr',str(p)],capture_output=True,text=True);assert att.returncode==0 and not att.stdout.strip()
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==x['sha256'];st2=p.stat();assert (st2.st_size,st2.st_mtime_ns,st2.st_ino,st2.st_dev)==(st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  l=subprocess.run(['lsof','-nP','--',str(p)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout and not l.stderr
  subprocess.run(['git','-C',str(root),'update-index','--skip-worktree','--',x['path']],check=True,stdout=subprocess.DEVNULL);marked=True
  st3=p.stat();assert (st3.st_size,st3.st_mtime_ns,st3.st_ino,st3.st_dev)==(st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  p.unlink();out['status']='retired'
 except Exception as e:
  if marked and p.exists():subprocess.run(['git','-C',str(root),'update-index','--no-skip-worktree','--',x['path']],check=True)
  out.update(status='skipped',reason=repr(e))
 with receipt.open('a') as f:f.write(json.dumps(out,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 if (i+1)%20==0:print('Processed',i+1,'of',len(records),flush=True)
fs=os.statvfs('/System/Volumes/Data');a=[json.loads(z) for z in receipt.read_text().splitlines()];print('After available bytes',fs.f_bavail*fs.f_frsize,'retired',sum(z['status']=='retired' for z in a),'skipped',sum(z['status']!='retired' for z in a),'logical bytes',sum(z['bytes'] for z in a if z['status']=='retired'),flush=True)
