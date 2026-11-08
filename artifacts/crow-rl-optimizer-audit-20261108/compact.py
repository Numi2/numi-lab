from pathlib import Path
import importlib.util,json,stat,subprocess,gzip,hashlib
r=Path.cwd();base=r/'artifacts/crow-rl-optimizer-audit-20261108';spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();byoid={oid:p for p,(_,oid)in entries.items()if p.startswith('artifacts/crow-')and p.endswith('.gz')};records=[];mapping={};dirs=[]
with a.Objects()as obj:
 for p in (base/'blobs').iterdir():
  data=p.read_bytes();oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if oid not in byoid:continue
  old=byoid[oid];blob=obj.read(oid)
  if data!=blob:raise RuntimeError('Compressed Git blob differs')
  raw=gzip.decompress(blob);s=p.lstat();rec={'path':str(p.relative_to(r)),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p),'existing_git_path':old,'git_blob':oid,'decoded_sha256':a.digest(raw)};records.append(rec);mapping[rec['path']]=old
manifest=json.loads((base/'manifest.json').read_text())
for f in manifest['files']:
 for chunk in f['backing'].get('chunks',[]):
  if chunk['path']in mapping:chunk['path']=mapping[chunk['path']]
# Check every candidate hash, mode, timestamp and handle before unlinking.
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(base/'blobs')],capture_output=True,text=True,timeout=60)
if handles.returncode not in (0,1)or handles.stdout or handles.stderr:raise RuntimeError('Open archive blob handles')
for rec in records:
 p=r/rec['path'];s=p.lstat()
 if s.st_size!=rec['bytes']or stat.S_IMODE(s.st_mode)!=rec['mode']or s.st_mtime_ns!=rec['mtime_ns']or a.file_hash(p)!=rec['sha256']:raise RuntimeError('Duplicate changed')
for rec in records:(r/rec['path']).unlink()
manifest['summary']['new_compressed_bytes_before_duplicate_compaction']=manifest['summary']['new_compressed_bytes'];manifest['summary']['new_compressed_bytes']=sum(p.stat().st_size for p in (base/'blobs').iterdir());a.write_json(base/'manifest.json',manifest)
a.write_json(base/'archive-compaction.json',{'status':'removed_duplicate_checkout_chunks','files':records,'compressed_checkout_bytes_removed':sum(x['bytes']for x in records),'all_backing_references_rebound_to_identical_existing_Git_blobs':True,'open_handles':0})
print(json.dumps({'removed_chunks':len(records),'removed_bytes':sum(x['bytes']for x in records),'new_compressed_bytes':manifest['summary']['new_compressed_bytes']}))
