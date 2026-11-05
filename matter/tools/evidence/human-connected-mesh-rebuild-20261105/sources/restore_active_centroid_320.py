from pathlib import Path
import json,hashlib,subprocess,tarfile,shutil,os,time
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
D=W/'old-mesh-preparation-308-recovery'
m=json.loads((D/'human-old-mesh-preparation-308.json').read_text())
row=next(x for x in m['files'] if x['path'].endswith('/combined-source-centroid-slots.csv'))
target=Path(row['path']);assert not target.exists()
out=W/'compiler-rebuild-320/restored-centroid-308';out.mkdir(exist_ok=False)
asset=out/m['asset']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
subprocess.run(['/opt/homebrew/bin/gh','release','download','numi-20261104-cold-human-evidence','--repo','Numi2/macmini-evidence-archive','--pattern',m['asset'],'--dir',str(out)],check=True)
assert sha(asset)==m['archive_sha256']
decoder=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','-q','-c',str(asset)],stdout=subprocess.PIPE)
seen=0;partial=target.with_suffix('.restoring')
with tarfile.open(fileobj=decoder.stdout,mode='r|') as t:
 for member in t:
  if member.name!=row['relative']:continue
  assert member.isfile() and member.size==row['size'];seen+=1
  with t.extractfile(member) as a,partial.open('xb') as b:shutil.copyfileobj(a,b,1<<20)
decoder.stdout.close();assert decoder.wait()==0 and seen==1
assert partial.stat().st_size==row['size'] and sha(partial)==row['sha256']
os.chmod(partial,row['mode']);os.utime(partial,ns=(row['mtime_ns'],row['mtime_ns']))
os.replace(partial,target)
receipt={'status':'PASS_EXACT_RESTORATION_FOR_ACTIVE_COMPILER_INPUT','archive_sha256':m['archive_sha256'],'manifest_sha256':sha(D/'human-old-mesh-preparation-308.json'),'restored_input':row,'independent_archive_download':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
asset.unlink();print(json.dumps(receipt),flush=True)
