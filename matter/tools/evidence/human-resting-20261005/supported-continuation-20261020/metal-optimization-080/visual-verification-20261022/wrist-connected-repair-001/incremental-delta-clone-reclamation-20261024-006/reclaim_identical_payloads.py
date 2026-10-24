from pathlib import Path
import json,hashlib,subprocess,os,shutil
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'incremental-delta-clone-reclamation-20261024-006'
if D.exists():raise RuntimeError('refuse overwrite')
D.mkdir()
base=Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base)=='8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042'
raw=base.read_bytes();chunk=65536;before=shutil.disk_usage(E).free;records=[]
for run in ('003','004','005'):
 p=E/'source-route-frame-owner-394verifyunused/fixed-boundary-source-fit-001'/('run-'+run)/'candidate-8874-local-position-fit.nhtissue'
 if not p.is_file() or p.is_symlink():continue
 check=subprocess.run(['/usr/sbin/lsof','-n','-F','n','--',str(p)],text=True,capture_output=True)
 if check.stdout.strip():records.append({'path':str(p),'skipped':'open file'});continue
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest();assert len(data)==len(raw)
 diffs=[(i,data[i:i+chunk]) for i in range(0,len(data),chunk) if data[i:i+chunk]!=raw[i:i+chunk]]
 assert sum(len(v) for _,v in diffs)<len(data)/3
 tmp=p.with_name('.delta-clone-006-'+p.name);assert not tmp.exists();st=p.stat()
 subprocess.run(['/bin/cp','-c','-p',str(base),str(tmp)],check=True)
 with tmp.open('r+b') as f:
  for off,block in diffs:f.seek(off);f.write(block)
  f.flush();os.fsync(f.fileno())
 assert sha(tmp)==h and sha(p)==h
 os.chmod(tmp,st.st_mode);os.utime(tmp,ns=(st.st_atime_ns,st.st_mtime_ns));os.replace(tmp,p)
 assert sha(p)==h
 records.append({'path':str(p),'sha256_before_after':h,'logical_bytes':len(data),'rewritten_bytes':sum(len(v) for _,v in diffs)})
after=shutil.disk_usage(E).free
report={'method':'APFS clone plus exact changed64KiB blocks; identical paths and bytes, no source or evidence deleted','base':str(base),'base_sha256':sha(base),'before_available_bytes':before,'after_available_bytes':after,'recovered_available_bytes':after-before,'qualification_12GiB_free':after>=12*1024**3,'files':records}
(D/'reclamation-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

