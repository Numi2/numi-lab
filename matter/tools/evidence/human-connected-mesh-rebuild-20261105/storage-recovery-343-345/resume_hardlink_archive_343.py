
"""Resume independent restoration of the completed343 archive, including hard links."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,shutil,os,collections
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'human-failed326-text-343'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=json.loads((D/'human-failed326-text-343.json').read_text())
download=D/'independent-download';asset=download/spec['asset']
assert sha(asset)==spec['archive_sha256']
assert sha(download/'human-failed326-text-343.json')==sha(D/'human-failed326-text-343.json')
rows=spec['files'];expected={r['relative']:r for r in rows};inode_count=collections.Counter((r['device'],r['inode']) for r in rows)
restore=D/'hardlink-verified-restore';restore.mkdir(exist_ok=False)
retained={};restored={};links=0
decoder=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','-q','-c',str(asset)],stdout=subprocess.PIPE)
with tarfile.open(fileobj=decoder.stdout,mode='r|') as archive:
 for member in archive:
  assert member.name in expected and member.name not in restored,member.name
  row=expected[member.name];target=restore/'member.bin'
  if member.isfile():
   with archive.extractfile(member) as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,1<<20)
  else:
   assert member.islnk() and member.linkname in restored,member.name
   source=restored[member.linkname]
   assert source['sha256']==row['sha256'] and source['size']==row['size']
   assert member.linkname in retained
   os.link(retained[member.linkname],target);links+=1
  assert target.stat().st_size==row['size'] and sha(target)==row['sha256']
  restored[member.name]=row
  if inode_count[(row['device'],row['inode'])]>1:
   held=restore/('verified-'+str(len(restored)));target.rename(held);retained[member.name]=held
  else:target.unlink()
decoder.stdout.close();assert decoder.wait()==0
assert set(restored)==set(expected)
restoration=dict(status='PASS_INDEPENDENT_DOWNLOAD_SEQUENTIAL_FILE_AND_HARDLINK_RESTORE',files=len(restored),hardlink_members=links,all_sizes_and_sha256_verified=True)
(D/'hardlink-restore-before-retirement.json').write_text(json.dumps(restoration,indent=2)+'\n')
# Check every source before retiring any; archive and complete independent restore exist.
for row in rows:
 p=Path(row['path']);s=p.stat()
 assert subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True).returncode==1
 assert (s.st_ino,s.st_mtime_ns,s.st_size)==(row['inode'],row['mtime_ns'],row['size'])
 assert sha(p)==row['sha256']
before=shutil.disk_usage(W).free
for row in rows:Path(row['path']).unlink()
shutil.rmtree(restore)
old_restore=D/'verified-restore'
if old_restore.exists():shutil.rmtree(old_restore)
asset.unlink();(D/spec['asset']).unlink();os.sync()
receipt=dict(status='PASS_INDEPENDENT_DOWNLOAD_ALL_MEMBERS_AND_RESTORE',source_files=rows,archive_sha256=spec['archive_sha256'],archive_bytes=spec['archive_size'],
 remote_asset='https://github.com/Numi2/macmini-evidence-archive/releases/download/numi-20261104-cold-human-evidence/'+spec['asset'],
 restore_verification=restoration,free_before_retirement=before,free_after_retirement=shutil.disk_usage(W).free,
 retained_failed_attempt=str(W/'human-failed326-text-343-execution.json'))
out=D/'human-failed326-text-343-verification.json';out.write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['/opt/homebrew/bin/gh','release','upload','numi-20261104-cold-human-evidence',str(out),'--repo','Numi2/macmini-evidence-archive'],check=True)
print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))
