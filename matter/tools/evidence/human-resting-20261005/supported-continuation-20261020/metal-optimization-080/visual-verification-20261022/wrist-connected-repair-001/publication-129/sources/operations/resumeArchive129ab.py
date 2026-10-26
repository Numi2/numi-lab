from pathlib import Path
import json,hashlib,tarfile,decimal,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'publication-129'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
groups={
'417-418':['D/postfit-backtrack-417','barycentric-connected-clearance-170/surface-witness-basis-417','D/surface-witness-basis-417','D/point-depth-refresh-417','D/fixed-bone-guard-refresh-417','D/connected-continuation-prep-417','D/self-witness-refresh-417','D/postfit-backtrack-418','D/remaining-clearance-locality-417'],
'420-422':['D/postfit-backtrack-420','D/local-two-parent-retriangulation-422']}
for tag,names in groups.items():
 start=time.monotonic();files=sorted({p for n in names for p in ([D/n]if(D/n).is_file()else(D/n).rglob('*'))if p.is_file()and not p.is_symlink()and'__pycache__'not in p.parts and'.pytest_cache'not in p.parts and p.suffix!='.pyc'})
 archive=P/'evidence'/('connected-clearance-'+tag+'.tar.xz');members=[]
 if archive.exists():
  with tarfile.open(archive,'r:xz')as tf:
   for t in tf:
    p=D/t.name;st=p.stat();data=tf.extractfile(t).read()
    assert t.isfile()and len(data)==st.st_size and hashlib.sha256(data).hexdigest()==sha(p)
    assert(t.mode,t.uid,t.gid)==(st.st_mode&0o7777,st.st_uid,st.st_gid)
    assert int(decimal.Decimal(t.pax_headers['mtime'])*1000000000)==st.st_mtime_ns
    members.append(dict(archive_path=t.name,source_path=str(p),sha256=sha(p),bytes=st.st_size,mode=st.st_mode&0o7777,uid=st.st_uid,gid=st.st_gid,mtime_ns=st.st_mtime_ns,source_device=st.st_dev,source_inode=st.st_ino,retire_after_remote_verification=False))
  assert {e['source_path']for e in members}=={str(p)for p in files}
 else:
  with tarfile.open(archive,'w:xz',preset=3,format=tarfile.PAX_FORMAT)as tf:
   for p in files:
    st=p.stat();e=dict(archive_path=p.relative_to(D).as_posix(),source_path=str(p),sha256=sha(p),bytes=st.st_size,mode=st.st_mode&0o7777,uid=st.st_uid,gid=st.st_gid,mtime_ns=st.st_mtime_ns,source_device=st.st_dev,source_inode=st.st_ino,retire_after_remote_verification=False)
    ti=tf.gettarinfo(str(p),arcname=e['archive_path']);ti.pax_headers['mtime']=str(decimal.Decimal(st.st_mtime_ns)/decimal.Decimal(1000000000))
    with p.open('rb')as f:tf.addfile(ti,f)
    assert sha(p)==e['sha256'];members.append(e)
 assert archive.stat().st_size<64*1024**2,archive.stat().st_size
 manifest=dict(archive_path=str(archive.relative_to(P)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,member_count=len(members),members=members,exclusions=[],format='POSIX PAX tar.xz; XZ preset3')
 (P/'evidence'/('archive-manifest-'+tag+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 by={e['archive_path']:e for e in members};seen=set()
 with tarfile.open(archive,'r:xz')as tf:
  for t in tf:
   e=by[t.name];data=tf.extractfile(t).read();assert t.isfile()and t.name not in seen and len(data)==e['bytes']and hashlib.sha256(data).hexdigest()==e['sha256']
   assert(t.mode,t.uid,t.gid)==(e['mode'],e['uid'],e['gid'])and int(decimal.Decimal(t.pax_headers['mtime'])*1000000000)==e['mtime_ns'];seen.add(t.name)
 assert seen==set(by)
 print(json.dumps(dict(tag=tag,sha256=sha(archive),bytes=archive.stat().st_size,members=len(members),restored_bytes=sum(e['bytes']for e in members),wall_seconds=time.monotonic()-start)),flush=True)

