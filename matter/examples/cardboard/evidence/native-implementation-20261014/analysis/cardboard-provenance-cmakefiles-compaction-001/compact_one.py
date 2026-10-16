#!/usr/bin/env python3
"""Fail-closed one-tree archive, full restore verification, and exact CMakeFiles removal."""
import argparse, gzip, hashlib, json, os, re, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/Users/home/numi-cardboard-20261006')
INVENTORY=HERE/'inventory_cmakefiles.py'
POINTER='RESTORE_FROM_COMPACTION.txt'
EXCLUDED={'build-cardboard-provenance-027','build-cardboard-provenance-154','build-cardboard-provenance-157','build-cardboard-provenance-current'}
FIELDS=('kind','mode','uid','gid','size','mtime_ns','flags','nlink','xattrs','sha256')
MIN_FREE=300*1024**2

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def disk():
    u=shutil.disk_usage(BASE)
    return {'total':u.total,'used':u.used,'free':u.free}
def inv(path):
    return json.loads(subprocess.check_output([sys.executable,str(INVENTORY),str(path)],text=True))
def fail(s): raise SystemExit(s)
def require(ok,s):
    if not ok: fail(s)
def identify(name,allow_pointer=False):
    require(re.fullmatch(r'build-cardboard-provenance-[0-9]{3}(?:[a-z]|-full)?',name) is not None, f'bad build dir name: {name}')
    require(name not in EXCLUDED, f'explicitly excluded build: {name}')
    require(not ('current' in name), f'current build excluded: {name}')
    root=BASE/name; target=root/'CMakeFiles'
    require(root.is_dir() and not root.is_symlink(), f'build root missing or symlink: {root}')
    require(target.is_dir() and not target.is_symlink(), f'CMakeFiles missing or symlink: {target}')
    if not allow_pointer: require(not (target/POINTER).exists(), f'restore pointer already present: {target/POINTER}')
    return root,target

def git_tracked(name):
    cp=subprocess.run(['/usr/bin/git','-C',str(BASE),'ls-files','--full-name','--',f'{name}/CMakeFiles'],text=True,capture_output=True,check=True)
    return [x for x in cp.stdout.splitlines() if x]
def lsof_gate(target):
    cp=subprocess.run(['/usr/sbin/lsof','+D',str(target)],text=True,capture_output=True)
    require(cp.returncode in (0,1), f'lsof failed for {target}: {cp.stderr[:1000]}')
    require(not cp.stdout.strip(), f'open handles under {target}: {cp.stdout[:1000]}')
    return {'command':['/usr/sbin/lsof','+D',str(target)],'exit_code':cp.returncode,'open_handles':0}
def process_gate(name):
    # Record active build commands and refuse if one names this exact tree.
    cp=subprocess.run(['/bin/ps','-axo','pid=,comm=,args='],text=True,capture_output=True,check=True)
    tools={'cmake','make','ninja','clang','clang++','metal','xcodebuild','ld'}
    hits=[]
    for line in cp.stdout.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)<3 or parts[1].split('/')[-1] not in tools: continue
        if name in parts[2]: hits.append(line.strip())
    require(not hits, 'active build-tool process mentions target: '+repr(hits[:5]))
    return {'command':['/bin/ps','-axo','pid=,comm=,args='],'target_mentions_from_build_tools':hits}
def target_specs(name):
    root,target=identify(name)
    tracked=git_tracked(name)
    require(not tracked, f'tracked files exist under CMakeFiles: {tracked[:10]}')
    process=process_gate(name); openfiles=lsof_gate(target)
    before=disk(); data=inv(target)
    require(not data['symlinks'],f'symlinks present; skip: {data["symlinks"][:5]}')
    require(not data['other_entries'],f'non-file/directory entries present: {data["other_entries"][:5]}')
    require(data['acl_entry_count']==0, f'ACLs present, unsupported in this batch: {data["acl_entry_count"]}')
    require(all(r.get('flags',0)==0 for r in data['entries']), 'nonzero file flags; skip target')
    require(all(r.get('nlink',1)==1 for r in data['entries'] if r['kind']=='file'), 'hard-linked files; skip target')
    allocated=sum(r.get('allocated_bytes',0) for r in data['entries'] if r['kind']=='file')
    # Keep >=300 MiB spare while archive and full extracted verification tree coexist.
    require(before['free'] >= 2*allocated+MIN_FREE, f'insufficient temp headroom: free={before["free"]}, allocated={allocated}, floor={MIN_FREE}')
    return root,target,data,tracked,process,openfiles,before,allocated

def apply_metadata(rows,root):
    root=Path(root)
    for row in sorted(rows,key=lambda r:(r['kind']=='directory',-r['path'].count('/'))):
        p=root if row['path']=='.' else root/row['path']
        st=p.lstat()
        require(not stat.S_ISLNK(st.st_mode),f'restore unexpectedly made symlink: {p}')
        if st.st_uid!=row['uid'] or st.st_gid!=row['gid']: os.chown(p,row['uid'],row['gid'],follow_symlinks=False)
        os.chmod(p,row['mode'],follow_symlinks=False)
        st=p.stat(follow_symlinks=False)
        os.utime(p,ns=(st.st_atime_ns,row['mtime_ns']),follow_symlinks=False)

def prune_ditto_empty_scratch(restored,expected):
    expected_paths={r['path'] for r in expected['entries']}
    scratch=Path(restored)/'CMakeScratch'
    if 'CMakeScratch' not in expected_paths and scratch.is_dir():
        require(not any(scratch.iterdir()),'ditto created unexpected nonempty CMakeScratch')
        scratch.rmdir()

def compare_inventory(expected,actual):
    require(not actual.get('symlinks') and not actual.get('other_entries'),'restored tree has unsupported entries')
    A={r['path']:r for r in expected['entries']}; B={r['path']:r for r in actual['entries']}
    require(set(A)==set(B),f'path mismatch missing={sorted(set(A)-set(B))[:4]} extra={sorted(set(B)-set(A))[:4]}')
    for p,a in A.items():
        for f in FIELDS: require(a.get(f)==B[p].get(f),f'metadata/hash mismatch {p} {f}: {a.get(f)!r} != {B[p].get(f)!r}')
    require(expected['acl_entry_count']==actual['acl_entry_count'],'ACL entry count differs')
    require(expected['normalized_acl_listing_sha256']==actual['normalized_acl_listing_sha256'],'ACL listing differs')
    require(expected['xattr_names']==actual['xattr_names'],'xattr names differ')
    require(expected['xattr_entry_count']==actual['xattr_entry_count'],'xattr count differs')

def archive_path(name): return HERE/'archives'/f'{name}-CMakeFiles.zip'
def inv_path(name): return HERE/'inventories'/f'{name}-CMakeFiles.json.gz'
def receipt_path(name,phase): return HERE/'receipts'/f'{name}-{phase}.json'

def prepare(name):
    root,target,data,tracked,process,openfiles,before,allocated=target_specs(name)
    archive=archive_path(name); ip=inv_path(name)
    require(not archive.exists() and not ip.exists(), 'archive or inventory already exists; refusing overwrite')
    raw=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode()
    with open(ip,'wb') as rawout:
        with gzip.GzipFile(fileobj=rawout,mode='wb',compresslevel=9,mtime=0) as f: f.write(raw)
    invsha=sha(ip)
    cp=subprocess.run(['/usr/bin/ditto','-c','-k','--sequesterRsrc','--keepParent','--extattr','--acl','--rsrc','--qtn',str(target),str(archive)],text=True,capture_output=True)
    require(cp.returncode==0, f'ditto archive failed: {cp.stderr[:2000]}')
    archive_sha=sha(archive); archive_bytes=archive.stat().st_size
    after_archive=disk()
    require(after_archive['free']>=MIN_FREE, f'archive creation crossed space floor: {after_archive["free"]}')
    with tempfile.TemporaryDirectory(prefix=f'{name}-restore-',dir=str(HERE)) as td:
        td=Path(td)
        cp=subprocess.run(['/usr/bin/ditto','-x','-k','--extattr','--acl','--rsrc','--qtn',str(archive),str(td)],text=True,capture_output=True)
        require(cp.returncode==0, f'ditto extraction failed: {cp.stderr[:2000]}')
        restored=td/'CMakeFiles'
        require(restored.is_dir() and not restored.is_symlink(),'archive does not restore exact CMakeFiles root')
        prune_ditto_empty_scratch(restored,data)
        apply_metadata(data['entries'],restored)
        got=inv(restored); compare_inventory(data,got)
        test={'restored_path_count':len(got['entries']),'file_count':sum(r['kind']=='file' for r in got['entries']),
              'directory_count':sum(r['kind']=='directory' for r in got['entries']),'acl_entry_count':got['acl_entry_count'],
              'xattr_entry_count':got['xattr_entry_count'],'path_and_metadata_hash_match':True}
    after_test=disk()
    record={'schema':'numi.cardboard.provenance-cmakefiles-prepare.v1','build':name,'target':str(target),
      'archive':str(archive),'archive_sha256':archive_sha,'archive_bytes':archive_bytes,
      'inventory':str(ip),'inventory_gzip_sha256':invsha,'inventory_uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
      'inventory_entries':len(data['entries']),'allocated_file_bytes':allocated,'logical_file_bytes':sum(r['size'] for r in data['entries'] if r['kind']=='file'),
      'tracked_files':tracked,'process_gate':process,'lsof_gate':openfiles,'restore_test':test,
      'birthtime_limit':'Filesystem-created birthtime is recorded but changes on extraction; all required file bytes, mode, uid/gid, nanosecond mtime, flags, link count, xattrs, ACL, and path inventory were compared.',
      'df_before':before,'df_after_archive':after_archive,'df_after_restore_test_cleanup':after_test,
      'archive_command':['/usr/bin/ditto','-c','-k','--sequesterRsrc','--keepParent','--extattr','--acl','--rsrc','--qtn',str(target),str(archive)],
      'restore_command':['/usr/bin/ditto','-x','-k','--extattr','--acl','--rsrc','--qtn',str(archive),'<temp>']}
    out=receipt_path(name,'prepare'); out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record,sort_keys=True))

def remove(name):
    root,target=identify(name); recpath=receipt_path(name,'prepare'); require(recpath.is_file(),f'prepare receipt absent: {recpath}')
    record=json.loads(recpath.read_text()); archive=archive_path(name); ip=inv_path(name)
    require(sha(archive)==record['archive_sha256'],'archive hash changed since restore test')
    require(sha(ip)==record['inventory_gzip_sha256'],'inventory hash changed')
    with gzip.open(ip,'rb') as f: data=json.loads(f.read())
    tracked=git_tracked(name); require(not tracked,f'tracked files now appear: {tracked[:10]}')
    process=process_gate(name); openfiles=lsof_gate(target)
    actual=inv(target); compare_inventory(data,actual)
    rows=data['entries']; expected={r['path']:r for r in rows}
    files=[r for r in rows if r['kind']=='file']; dirs=[r for r in rows if r['kind']=='directory' and r['path']!='.']
    before_remove=disk()
    pointer=target/POINTER; require(not pointer.exists(),'restore pointer already exists')
    text=(f'Archived CMakeFiles for {name}.\nArchive: {archive}\nArchive SHA-256: {record["archive_sha256"]}\n'
          f'Inventory: {ip}\nInventory SHA-256: {record["inventory_gzip_sha256"]}\n'
          f'Restore: python3 {HERE}/compact_one.py restore {name}\n')
    with pointer.open('x') as f: f.write(text);f.flush();os.fsync(f.fileno())
    # Re-check every file hash immediately before unlink; only inventory paths are eligible.
    for row in files:
        p=target/row['path']; st=p.lstat()
        require(stat.S_ISREG(st.st_mode) and not p.is_symlink(),f'not regular: {p}')
        require(sha(p)==row['sha256'],f'file changed before removal: {row["path"]}')
        p.unlink()
    for row in sorted(dirs,key=lambda r:(-r['path'].count('/'),r['path'])):
        p=target/row['path'];require(p.is_dir() and not p.is_symlink(),f'not directory: {p}')
        os.rmdir(p)
    remaining=inv(target)
    require({r['path'] for r in remaining['entries']}=={'.',POINTER},'target not pointer-only')
    require((target/POINTER).read_text()==text,'pointer mismatch')
    after=disk()
    result={'schema':'numi.cardboard.provenance-cmakefiles-removal.v1','build':name,'target':str(target),
      'archive_sha256':record['archive_sha256'],'inventory_sha256':record['inventory_gzip_sha256'],
      'removed_files':len(files),'removed_logical_bytes':sum(r['size'] for r in files),
      'removed_allocated_bytes':sum(r['allocated_bytes'] for r in files),'removed_directories':len(dirs),
      'pointer':str(pointer),'process_gate':process,'lsof_gate':openfiles,'df_before_remove':before_remove,'df_after_remove':after,
      'measured_free_increase_since_prepare':after['free']-record['df_before']['free'],'remaining_entries':['.',POINTER]}
    receipt_path(name,'remove').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))

def restore(name):
    root,target=identify(name,allow_pointer=True); rec=json.loads(receipt_path(name,'prepare').read_text())
    archive=archive_path(name); ip=inv_path(name)
    require(sha(archive)==rec['archive_sha256'],'archive hash mismatch')
    require(sha(ip)==rec['inventory_gzip_sha256'],'inventory hash mismatch')
    with gzip.open(ip,'rb') as f:data=json.loads(f.read())
    require(target.is_dir() and not target.is_symlink(),'target missing/not real')
    pointer=target/POINTER; require(pointer.is_file() and rec['archive_sha256'] in pointer.read_text(),'restore pointer missing/mismatch')
    current=inv(target)
    require(not current.get('symlinks') and not current.get('other_entries'),'target has unsupported entries; refusing restore')
    expected={r['path']:r for r in data['entries']}; current_rows={r['path']:r for r in current['entries']}
    require(set(current_rows)<=set(expected)|{POINTER},'target has unexpected paths; refusing restore')
    for rel,row in current_rows.items():
        if rel==POINTER: continue
        exp=expected[rel]
        require(row['kind']==exp['kind'],f'partial target path kind differs: {rel}')
        if row['kind']=='file': require(row.get('sha256')==exp.get('sha256'),f'partial target file differs: {rel}')
    require(POINTER in current_rows,'restore pointer absent; refusing restore')
    with tempfile.TemporaryDirectory(prefix=f'{name}-restore-',dir=str(HERE)) as td:
        td=Path(td); cp=subprocess.run(['/usr/bin/ditto','-x','-k','--extattr','--acl','--rsrc','--qtn',str(archive),str(td)],text=True,capture_output=True,check=True)
        restored=td/'CMakeFiles'; require(restored.is_dir(),'missing archive CMakeFiles'); prune_ditto_empty_scratch(restored,data)
        apply_metadata(data['entries'],restored); compare_inventory(data,inv(restored))
        shutil.rmtree(target); os.replace(restored,target)
        compare_inventory(data,inv(target))
    print(f'restored and reverified {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','remove','restore']);p.add_argument('build')
    a=p.parse_args(); {'prepare':prepare,'remove':remove,'restore':restore}[a.action](a.build)
