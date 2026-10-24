from pathlib import Path
import os,json,hashlib,subprocess,shutil
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'incremental-delta-clone-reclamation-20261024-007'
assert not D.exists();D.mkdir()
F=E/'source-route-frame-owner-394verifyunused'
M=F/'publication-freeze-052/file-list-052.json'
base=Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base)=='8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042'
data0=base.read_bytes();before=shutil.disk_usage(E).free;records=[]
for rec in json.loads(M.read_text())['files']:
 if not rec['path'].endswith('.nhtissue') or rec['bytes']!=len(data0):continue
 p=F/rec['path'];assert p.is_file() and not p.is_symlink() and sha(p)==rec['sha256']
 check=subprocess.run(['/usr/sbin/lsof','-n','-F','n','--',str(p)],text=True,capture_output=True)
 if check.stdout.strip():records.append({'path':str(p),'skipped':'open'});continue
 data=p.read_bytes();chunk=65536;diffs=[(i,data[i:i+chunk]) for i in range(0,len(data),chunk) if data[i:i+chunk]!=data0[i:i+chunk]]
 assert sum(len(v) for _,v in diffs)<len(data)/3
 tmp=p.with_name('.clone007-'+p.name);assert not tmp.exists();st=p.stat()
 subprocess.run(['/bin/cp','-c','-p',str(base),str(tmp)],check=True)
 os.chmod(tmp,st.st_mode|0o200)
 with tmp.open('r+b') as f:
  for off,b in diffs:f.seek(off);f.write(b)
  f.flush();os.fsync(f.fileno())
 assert sha(tmp)==rec['sha256'] and sha(p)==rec['sha256']
 os.chmod(tmp,st.st_mode);os.utime(tmp,ns=(st.st_atime_ns,st.st_mtime_ns));os.replace(tmp,p)
 assert sha(p)==rec['sha256']
 records.append({'path':str(p),'sha256_before_after':sha(p),'logical_bytes':len(data),'rewritten_bytes':sum(len(v) for _,v in diffs)})
after=shutil.disk_usage(E).free
report={'source_manifest_path':str(M),'source_manifest_sha256':sha(M),'manifest_path_resolution':'paths resolved against source-route-frame-owner-394verifyunused; original root_relative_to label was wrong and retained',
 'method':'APFS clone plus exact changed64KiBblocks; identical bytes, paths, modes and mtimes; no deletion',
 'before_available_bytes':before,'after_available_bytes':after,'recovered_available_bytes':after-before,'qualification_12GiB_free':after>=12*1024**3,'files':records}
(D/'reclamation-report.json').write_text(json.dumps(report,indent=2)+'\n')
(D/'executed_reclaim007.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report))
