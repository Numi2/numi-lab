from pathlib import Path
import json,hashlib,tarfile,decimal,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'publication-128'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
groups={
'410-412':['D/postfit-backtrack-410','barycentric-connected-clearance-170/surface-witness-basis-410','D/surface-witness-basis-410','D/point-depth-barriers-410','D/fixed-bone-guards-410','D/source-outward-proposal-300/candidate-410-a661a0','D/coherent-point-targets-390/candidate-410-a661a0','D/self-witness-refresh-410','D/self-witness-refresh-412','D/point-depth-barriers-412','D/fixed-bone-guards-412','D/connected-continuation-prep-412','D/prospective-self-guards-412','D/surface-witness-basis-412','barycentric-connected-clearance-170/surface-witness-basis-412','D/point-depth-refresh-413','D/fixed-bone-guard-refresh-413'],
'413-416':['D/surface-witness-basis-413','barycentric-connected-clearance-170/surface-witness-basis-413','D/postfit-backtrack-413','D/local-two-parent-retriangulation-414','D/surface-witness-basis-415','barycentric-connected-clearance-170/surface-witness-basis-415','D/postfit-backtrack-415','D/local-two-parent-retriangulation-416']}
for tag,names in groups.items():
 start=time.monotonic();files=sorted({p for n in names for p in ([D/n]if(D/n).is_file()else(D/n).rglob('*'))if p.is_file()and not p.is_symlink()and'__pycache__'not in p.parts and'.pytest_cache'not in p.parts and p.suffix!='.pyc'})
 archive=P/'evidence'/('connected-clearance-'+tag+'.tar.xz');assert not archive.exists();members=[]
 with tarfile.open(archive,'w:xz',preset=6,format=tarfile.PAX_FORMAT)as tf:
  for p in files:
   st=p.stat();e=dict(archive_path=p.relative_to(D).as_posix(),source_path=str(p),sha256=sha(p),bytes=st.st_size,mode=st.st_mode&0o7777,uid=st.st_uid,gid=st.st_gid,mtime_ns=st.st_mtime_ns,source_device=st.st_dev,source_inode=st.st_ino,retire_after_remote_verification=False)
   ti=tf.gettarinfo(str(p),arcname=e['archive_path']);ti.pax_headers['mtime']=str(decimal.Decimal(st.st_mtime_ns)/decimal.Decimal(1000000000))
   with p.open('rb')as f:tf.addfile(ti,f)
   assert sha(p)==e['sha256'];members.append(e)
 assert archive.stat().st_size<50*1024**2,archive.stat().st_size
 manifest=dict(archive_path=str(archive.relative_to(P)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,member_count=len(members),members=members,exclusions=[],format='POSIX PAX tar.xz; XZ preset6')
 (P/'evidence'/('archive-manifest-'+tag+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 by={e['archive_path']:e for e in members};seen=set()
 with tarfile.open(archive,'r:xz')as tf:
  for t in tf:
   e=by[t.name];data=tf.extractfile(t).read();assert t.isfile()and t.name not in seen and len(data)==e['bytes']and hashlib.sha256(data).hexdigest()==e['sha256']
   assert(t.mode,t.uid,t.gid)==(e['mode'],e['uid'],e['gid'])and int(decimal.Decimal(t.pax_headers['mtime'])*1000000000)==e['mtime_ns'];seen.add(t.name)
 assert seen==set(by)
 print(json.dumps(dict(tag=tag,sha256=sha(archive),bytes=archive.stat().st_size,members=len(members),restored_bytes=sum(e['bytes']for e in members),wall_seconds=time.monotonic()-start)),flush=True)

