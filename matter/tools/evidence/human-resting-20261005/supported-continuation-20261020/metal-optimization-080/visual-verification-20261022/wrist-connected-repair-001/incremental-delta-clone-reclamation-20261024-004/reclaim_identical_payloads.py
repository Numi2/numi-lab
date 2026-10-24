from pathlib import Path
import json,hashlib,subprocess,os,shutil,time
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'incremental-delta-clone-reclamation-20261024-004';D.mkdir(exist_ok=True)
base=Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base)=='8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042'
rel=['source-route-frame-owner-394verifyunused/fixed-boundary-remesh-001/candidate.nhtissue', 'source-route-frame-owner-394verifyunused/fixed-boundary-remesh-005/candidate.nhtissue', 'source-route-frame-owner-394verifyunused/fixed-boundary-remesh-007/candidate.nhtissue', 'source-route-frame-owner-394verifyunused/topology-flips-012/candidate-topology.nhtissue', 'source-route-frame-owner-394verifyunused/topology-flips-013/candidate-topology.nhtissue', 'source-route-frame-owner-394verifyunused/topology-flips-014/candidate-topology.nhtissue']
before=shutil.disk_usage(E).free;records=[]
raw=base.read_bytes();chunk=65536
for name in rel:
 p=E/name
 if not p.exists():continue
 opencheck=subprocess.run(['/usr/sbin/lsof','-n','-F','n','--',str(p)],text=True,capture_output=True)
 if opencheck.stdout.strip():records.append({'path':str(p),'skipped':'open file'});continue
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
 assert abs(len(data)-len(raw)) < 1024
 diffs=[(i,data[i:i+chunk]) for i in range(0,len(data),chunk) if data[i:i+chunk]!=raw[i:i+chunk]]
 if sum(len(v) for _,v in diffs)>len(data)//3:records.append({'path':str(p),'skipped':'insufficient shared bytes'});continue
 tmp=p.with_name('.delta-clone-004-'+p.name);assert not tmp.exists();st=p.stat()
 subprocess.run(['/bin/cp','-c','-p',str(base),str(tmp)],check=True)
 with tmp.open('r+b') as stream:
  for off,v in diffs:stream.seek(off);stream.write(v)
  stream.truncate(len(data))
  stream.flush();os.fsync(stream.fileno())
 assert sha(tmp)==h and sha(p)==h
 os.chmod(tmp,st.st_mode);os.utime(tmp,ns=(st.st_atime_ns,st.st_mtime_ns))
 os.replace(tmp,p);assert sha(p)==h
 records.append({'path':str(p),'sha256_before_after':h,'bytes':len(data),'rewritten_chunks':len(diffs),'rewritten_bytes':sum(len(v) for _,v in diffs)})
after=shutil.disk_usage(E).free
r={'method':'APFS copy-on-write clone plus exact changed64KiB chunks; no logical file or evidence bytes deleted','base':str(base),'base_sha256':sha(base),'before_available_bytes':before,'after_available_bytes':after,'recovered_available_bytes':after-before,'files':records}
(D/'reclamation-report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)

