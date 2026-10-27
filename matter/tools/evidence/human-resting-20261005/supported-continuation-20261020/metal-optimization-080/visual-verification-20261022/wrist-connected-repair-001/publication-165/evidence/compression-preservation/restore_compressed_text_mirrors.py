from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys

def digest(path):
 h=hashlib.sha256(); n=0
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
 return h.hexdigest(),n

def restore_one(item,destination,compressed_root=None):
 destination=Path(destination)
 compressed=Path(item['compressed_path']) if compressed_root is None else Path(compressed_root)/item.get('package_compressed_relative_path',item['relative_path']+'.zst')
 if not compressed.is_file() or compressed.is_symlink():raise RuntimeError(f'compressed payload missing/linked: {compressed}')
 h,n=digest(compressed)
 if h!=item['compressed_sha256'] or n!=item['compressed_bytes']:raise RuntimeError(f'compressed payload hash/size mismatch: {compressed}')
 if destination.exists():
  if destination.is_file() and digest(destination)==(item['original_sha256'],item['original_bytes']):return {'path':str(destination),'status':'already_restored_verified','sha256':item['original_sha256'],'bytes':item['original_bytes']}
  raise RuntimeError(f'restore destination exists with different content: {destination}')
 destination.parent.mkdir(parents=True,exist_ok=True)
 temp=destination.with_name(destination.name+f'.restore-{os.getpid()}.tmp')
 if temp.exists():raise RuntimeError(f'restore temp exists: {temp}')
 proc=subprocess.Popen(['zstd','-q','-d','-c',str(compressed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 hobj=hashlib.sha256();written=0
 try:
  with temp.open('xb') as f:
   while True:
    block=proc.stdout.read(1024*1024)
    if not block:break
    f.write(block);hobj.update(block);written+=len(block)
   f.flush();os.fsync(f.fileno())
  err=proc.stderr.read();rc=proc.wait()
  if rc:raise RuntimeError(f'zstd decode failed: {err.decode(errors="replace")[:300]}')
  h=hobj.hexdigest()
  if h!=item['original_sha256'] or written!=item['original_bytes']:raise RuntimeError(f'restored original hash/size mismatch: {destination}')
  os.chmod(temp,item['original_mode'])
  os.utime(temp,ns=(item['original_atime_ns'],item['original_mtime_ns']))
  os.replace(temp,destination)
  dfd=os.open(destination.parent,os.O_RDONLY)
  try:os.fsync(dfd)
  finally:os.close(dfd)
  post_hash,post_bytes=digest(destination);st=destination.stat()
  if post_hash!=item['original_sha256'] or post_bytes!=item['original_bytes'] or (st.st_mode&0o777)!=item['original_mode'] or st.st_mtime_ns!=item['original_mtime_ns']:
   raise RuntimeError(f'post-restore verification failed: {destination}')
  return {'path':str(destination),'status':'restored_and_verified','sha256':post_hash,'bytes':post_bytes,'mode':st.st_mode&0o777,'mtime_ns':st.st_mtime_ns}
 finally:
  if proc.poll() is None:proc.kill();proc.wait()
  if temp.exists():temp.unlink()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--receipt',required=True);ap.add_argument('--destination-root');ap.add_argument('--compressed-root');ap.add_argument('--only',action='append',default=[]);a=ap.parse_args()
 receipt=Path(a.receipt);d=json.loads(receipt.read_text());root=Path(a.destination_root) if a.destination_root else None
 rows=d['files'];wanted=set(a.only)
 if wanted:rows=[x for x in rows if x['relative_path'] in wanted or x['original_path'] in wanted]
 if not rows:raise RuntimeError('no matching receipt entries')
 results=[]
 for x in rows:
  dest=Path(x['original_path']) if root is None else root/x['relative_path']
  results.append(restore_one(x,dest,a.compressed_root))
 print(json.dumps({'receipt':str(receipt),'restored_count':len(results),'results':results},indent=2))
if __name__=='__main__':main()
