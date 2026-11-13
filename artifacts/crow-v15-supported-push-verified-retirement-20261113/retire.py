import json,os,hashlib,subprocess,time
p='/tmp/crow-retirement-new-verified.json';entries=json.load(open(p));root='/Users/n/Developer/numi-crow-rl-20261107/build/'
for e in entries:
 path=e['path'];assert path.startswith(root) and '/crow-v15-supported-' in path
 s=os.stat(path,follow_symlinks=False);assert (s.st_ino,s.st_mtime_ns,s.st_size)==(e['inode'],e['mtime_ns'],e['bytes'])
 with open(path,'rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
 assert h==e['sha256']==e['restored_sha256'] and e['restored_bytes']==s.st_size
 r=subprocess.run(['/usr/sbin/lsof','-t','--',path],capture_output=True,text=True)
 assert r.returncode==1 and not r.stdout,(path,r.returncode,r.stdout,r.stderr)
before=subprocess.check_output(['df','-k','/'],text=True);removed=[]
for e in entries:
 path=e['path'];s=os.stat(path,follow_symlinks=False);assert (s.st_ino,s.st_mtime_ns,s.st_size)==(e['inode'],e['mtime_ns'],e['bytes'])
 r=subprocess.run(['/usr/sbin/lsof','-t','--',path],capture_output=True,text=True);assert r.returncode==1 and not r.stdout
 os.unlink(path);removed.append(path)
after=subprocess.check_output(['df','-k','/'],text=True)
r={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'manifest_verified_commit':entries[0]['verified_commit'],'logical_bytes_removed':sum(e['bytes'] for e in entries),'removed':removed,'df_before':before,'df_after':after,'scope':'Exact manifest entries only, independently restored and verified from immutable GitHub commit, metadata and SHA-256 checked, open handles checked twice. No claim about APFS physical recovery.'}
open('/tmp/crow-retirement-new-receipt.json','w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
