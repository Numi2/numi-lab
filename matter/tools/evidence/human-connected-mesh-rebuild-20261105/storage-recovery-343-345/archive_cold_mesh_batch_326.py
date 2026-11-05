"""Archive superseded64c native export inputs after verified restoration; current321 plan contains no such paths."""
from pathlib import Path
import os,stat,json,hashlib,subprocess,tarfile,shutil,time
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
D=Path(__import__('sys').argv[1]);D.mkdir(exist_ok=False)
selection=Path(__import__('sys').argv[2])
files=[Path(p) for p in json.loads(selection.read_text())['paths']]
assert files
repo='Numi2/macmini-evidence-archive';tag='numi-20261104-cold-human-evidence'
asset=D/(D.name+'.tar.zst');manifest=D/(D.name+'.json')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rows=[]
active=subprocess.check_output(['ps','-axo','command='],text=True)
for p in files:
 s=p.stat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
 assert subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True).returncode==1
 rows.append({'path':str(p),'relative':str(p).lstrip('/'),'size':s.st_size,'bytes':s.st_size,'sha256':sha(p),'inode':s.st_ino,'mode':stat.S_IMODE(s.st_mode),'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'retire':True})
with asset.open('xb') as f:
 proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-T1','-3','-q','-c'],stdin=subprocess.PIPE,stdout=f)
 with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT,dereference=True) as t:
  for row in rows:t.add(row['path'],arcname=row['relative'],recursive=False)
 proc.stdin.close();assert proc.wait()==0
spec={'asset':asset.name,'archive_sha256':sha(asset),'archive_size':asset.stat().st_size,'files':rows}
manifest.write_text(json.dumps(spec,indent=2)+'\n')
gh='/opt/homebrew/bin/gh'
subprocess.run([gh,'release','upload',tag,str(asset),str(manifest),'--repo',repo],check=True)
download=D/'independent-download';download.mkdir()
subprocess.run([gh,'release','download',tag,'--repo',repo,'--pattern',asset.name,'--pattern',manifest.name,'--dir',str(download)],check=True)
assert sha(download/asset.name)==spec['archive_sha256'] and sha(download/manifest.name)==sha(manifest)

# Restore every member from the independently downloaded archive, one at a time,
# bounding local recovery staging to one file while preserving a normal full manifest.
restore=D/'verified-restore';restore.mkdir()
expected={row['relative']:row for row in rows};restored=[]
decoder=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','-q','-c',str(download/asset.name)],stdout=subprocess.PIPE)
with tarfile.open(fileobj=decoder.stdout,mode='r|') as archive:
 for member in archive:
  assert member.isfile() and member.name in expected,member.name
  row=expected[member.name];target=restore/'member.bin'
  with archive.extractfile(member) as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,1<<20)
  assert target.stat().st_size==row['size'] and sha(target)==row['sha256']
  restored.append(member.name);target.unlink()
decoder.stdout.close();assert decoder.wait()==0
assert sorted(restored)==sorted(expected)
restoration={'status':'PASS_INDEPENDENT_DOWNLOAD_SEQUENTIAL_FILE_RESTORE','files':len(restored),'all_sizes_and_sha256_verified':True}
before=shutil.disk_usage(W).free
for row in rows:
 p=Path(row['path']);s=p.stat()
 assert subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True).returncode==1
 assert (s.st_ino,s.st_mtime_ns,s.st_size)==(row['inode'],row['mtime_ns'],row['size']) and sha(p)==row['sha256']
 p.unlink()
shutil.rmtree(restore);(download/asset.name).unlink();asset.unlink();os.sync()
receipt={'status':'PASS_INDEPENDENT_DOWNLOAD_ALL_MEMBERS_AND_RESTORE','source_files':rows,'archive_sha256':spec['archive_sha256'],'archive_bytes':spec['archive_size'],'remote_asset':'https://github.com/'+repo+'/releases/download/'+tag+'/'+asset.name,'restore_verification':restoration,'free_before_retirement':before,'free_after_retirement':shutil.disk_usage(W).free}
p=D/(D.name+'-verification.json');p.write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run([gh,'release','upload',tag,str(p),'--repo',repo],check=True)
print(json.dumps(receipt),flush=True)
