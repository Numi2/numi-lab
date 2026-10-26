from pathlib import Path
import json,hashlib,tarfile,decimal,subprocess,os,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';P=D/'publication-127'
(P/'evidence').mkdir(parents=True,exist_ok=True);(P/'sources').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (D/'D/postfit-backtrack-403/run-001/backtrack-373/root-independent-construction-verification.json').is_file()
names=['D/postfit-backtrack-408','barycentric-connected-clearance-170/surface-witness-basis-408','D/surface-witness-basis-408','D/self-witness-refresh-407','D/self-witness-refresh-408','D/refresh_self_witnesses_407.py','D/refresh_self_witnesses_408.py','D/refresh_self_witnesses_409.py','D/test_refresh_self_witnesses_409.py','D/fixed-bone-guards-407','D/point-depth-barriers-357/candidate-407-6c115d-near-001','D/source-outward-proposal-300/candidate-407-6c115d-run002','D/coherent-point-targets-390/candidate-407-6c115d','D/local-two-parent-retriangulation-406','D/local-two-parent-retriangulation-409']
files=sorted({p for n in names for p in ([D/n] if (D/n).is_file() else (D/n).rglob('*')) if p.is_file() and not p.is_symlink() and '__pycache__'not in p.parts and p.suffix!='.pyc'})
total=sum(p.stat().st_size for p in files);assert total<800*1024**2,total
members=[];archive=P/'evidence/connected-clearance-407-409.tar.xz'
with tarfile.open(archive,'w:xz',preset=6,format=tarfile.PAX_FORMAT)as tf:
 for p in files:
  st=p.stat();rel=p.relative_to(D).as_posix()
  removable=False
  e=dict(archive_path=rel,source_path=str(p),sha256=sha(p),bytes=st.st_size,mode=st.st_mode&0o7777,uid=st.st_uid,gid=st.st_gid,mtime_ns=st.st_mtime_ns,source_device=st.st_dev,source_inode=st.st_ino,retire_after_remote_verification=removable)
  ti=tf.gettarinfo(str(p),arcname=rel);ti.pax_headers['mtime']=str(decimal.Decimal(st.st_mtime_ns)/decimal.Decimal(1000000000))
  with p.open('rb')as f:tf.addfile(ti,f)
  assert sha(p)==e['sha256'];members.append(e)
manifest=dict(archive_path=str(archive.relative_to(P)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,member_count=len(members),members=members,exclusions=[],format='POSIX PAX tar.xz; XZ preset6')
(P/'evidence/archive-manifest-407-409.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Streaming restoration verification before publication.
by={e['archive_path']:e for e in members};seen=set()
with tarfile.open(archive,'r:xz')as tf:
 for t in tf:
  e=by[t.name];assert t.isfile()and t.name not in seen
  data=tf.extractfile(t).read();assert len(data)==e['bytes']and hashlib.sha256(data).hexdigest()==e['sha256']
  assert (t.mode,t.uid,t.gid)==(e['mode'],e['uid'],e['gid'])
  assert int(decimal.Decimal(t.pax_headers['mtime'])*1000000000)==e['mtime_ns'];seen.add(t.name)
assert seen==set(by)

print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=len(members),bytes=total)))
