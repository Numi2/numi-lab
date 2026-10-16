#!/usr/bin/env python3
"""Remove only one pre-inventoried build-host tree after verifying its archive."""
import argparse, hashlib, json, os, shutil, stat, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/Users/home/cardboard-implementation-20261014')
MANIFEST=HERE/'archive-manifest.json'; INV=HERE/'preinventory.json'
POINTER='RESTORE_FROM_COMPACTION.txt'
FIELDS=('kind','mode','uid','gid','size','mtime_ns','flags','nlink','xattrs','sha256')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 return json.loads(subprocess.check_output(['python3',str(HERE/'inventory_build_host.py'),str(root)],text=True))
def require(cond,msg):
 if not cond: raise SystemExit(msg)
def main(key):
 manifest=json.loads(MANIFEST.read_text()); invraw=INV.read_bytes()
 require(hashlib.sha256(invraw).hexdigest()==manifest['preinventory']['sha256'],'preinventory hash mismatch')
 spec=manifest['targets'][key]; archive=HERE/spec['archive']; target=BASE/spec['relative_target']
 require(sha(archive)==spec['archive_sha256'],'archive hash mismatch')
 require(target.is_dir() and not target.is_symlink(),'target is not a real directory')
 current=inventory(target); pre=json.loads(invraw)['targets'][key]; expected=pre['entries']
 require(not current.get('symlinks') and not current.get('other_entries'),'unexpected symlink or unsupported entry type')
 require(pre['summary']['acl_entry_count']==0 and current['acl_entry_count']==0,'unexpected ACL entry')
 A={r['path']:r for r in expected};B={r['path']:r for r in current['entries']}
 require(set(A)==set(B),'tree paths changed since pre-inventory')
 for p,a in A.items():
  for f in FIELDS: require(a.get(f)==B[p].get(f),f'pre-delete mismatch at {p} field {f}')
 # No target file may be held open while we remove it.
 cp=subprocess.run(['/usr/sbin/lsof','+D',str(target)],text=True,capture_output=True)
 require(cp.returncode in (0,1) and not cp.stdout.strip(),f'open handle found under target: {cp.stdout[:2000]}')
 pointer=target/POINTER
 require(not pointer.exists(),'restore pointer already exists')
 text=(f'Archived build-host tree {key}.\nArchive: {archive}\n'
       f'Archive SHA-256: {spec["archive_sha256"]}\n'
       f'Pre-inventory SHA-256: {manifest["preinventory"]["sha256"]}\n'
       f'Restore: python3 {HERE}/restore_build_host.py {key}\n')
 with pointer.open('x') as f:
  f.write(text);f.flush();os.fsync(f.fileno())
 fd=os.open(target,os.O_RDONLY);os.fsync(fd);os.close(fd)
 # Verify every file again immediately before unlinking; delete only manifest paths.
 files=[r for r in expected if r['kind']=='file']
 for row in files:
  p=target/row['path'];st=p.lstat()
  require(stat.S_ISREG(st.st_mode) and not p.is_symlink(),f'not a regular file: {p}')
  require(sha(p)==row['sha256'],f'file changed immediately before removal: {row["path"]}')
  p.unlink()
 dirs=sorted((r for r in expected if r['kind']=='directory' and r['path']!='.'),key=lambda r:(-r['path'].count('/'),r['path']))
 for row in dirs:
  p=target/row['path'];require(p.is_dir() and not p.is_symlink(),f'not a real directory: {p}')
  os.rmdir(p)
 remaining=inventory(target)
 require(not remaining.get('symlinks') and not remaining.get('other_entries'),'unexpected item remained')
 require({r['path'] for r in remaining['entries']}=={'.',POINTER},'target is not pointer-only after removal')
 require((target/POINTER).read_text()==text,'restore pointer changed')
 out={'schema':'numi.cardboard.buildhost-removal-receipt.v1','target_key':key,'target':str(target),
      'archive':str(archive),'archive_sha256':spec['archive_sha256'],'preinventory_sha256':manifest['preinventory']['sha256'],
      'removed_regular_files':len(files),'removed_logical_file_bytes':sum(r['size'] for r in files),
      'removed_allocated_file_bytes':sum(r['allocated_bytes'] for r in files),'removed_directories':len(dirs),
      'remaining_entries':[r['path'] for r in remaining['entries']],'pointer':str(pointer)}
 receipt=HERE/f'removal-{key}.json';receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 with receipt.open('rb') as f: os.fsync(f.fileno())
 fd=os.open(HERE,os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('target',choices=['004','005']);main(p.parse_args().target)
