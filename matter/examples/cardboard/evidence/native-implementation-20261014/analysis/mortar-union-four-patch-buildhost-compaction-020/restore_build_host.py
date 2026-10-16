#!/usr/bin/env python3
"""Restore one archived build-host tree; fail closed on unexpected current contents."""
import argparse, hashlib, json, os, shutil, stat, subprocess, tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/Users/home/cardboard-implementation-20261014')
MANIFEST=HERE/'archive-manifest.json'
INV=HERE/'preinventory.json'
POINTER='RESTORE_FROM_COMPACTION.txt'
FIELDS=('kind','mode','uid','gid','size','mtime_ns','flags','nlink','xattrs','sha256')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def die(s): raise SystemExit(s)
def inventory(root):
    return json.loads(subprocess.check_output(['python3',str(HERE/'inventory_build_host.py'),str(root)],text=True))
def apply_metadata(rows,root):
    root=Path(root)
    for row in sorted(rows,key=lambda r:(r['kind']=='directory',-r['path'].count('/'))):
        p=root if row['path']=='.' else root/row['path']
        st=p.lstat()
        if stat.S_ISLNK(st.st_mode): die(f'refuse symlink during restore: {p}')
        if st.st_uid!=row['uid'] or st.st_gid!=row['gid']:
            os.chown(p,row['uid'],row['gid'],follow_symlinks=False)
        os.chmod(p,row['mode'],follow_symlinks=False)
        st=p.stat(follow_symlinks=False)
        os.utime(p,ns=(st.st_atime_ns,row['mtime_ns']),follow_symlinks=False)
def validate(root,expected):
    got=inventory(root)
    if got.get('symlinks') or got.get('other_entries'):
        die('restored tree contains symlinks or unsupported entry types')
    A={x['path']:x for x in expected}; B={x['path']:x for x in got['entries']}
    if set(A)!=set(B): die(f'restored path inventory differs: missing={sorted(set(A)-set(B))[:5]} extra={sorted(set(B)-set(A))[:5]}')
    for p,a in A.items():
        for f in FIELDS:
            if a.get(f)!=B[p].get(f): die(f'restored field mismatch {p} {f}')
    if got['acl_entry_count']!=0: die('unexpected ACL entries; this archive has no source ACLs')
    if got['xattr_entry_count']!=sum(len(x.get('xattrs',{})) for x in expected):
        die('restored extended-attribute count mismatch')
    return got

def main(key, target_override=None):
    manifest=json.loads(MANIFEST.read_text()); invbytes=INV.read_bytes()
    if hashlib.sha256(invbytes).hexdigest()!=manifest['preinventory']['sha256']: die('preinventory hash mismatch')
    spec=manifest['targets'].get(key)
    if not spec: die('unknown target key')
    target=Path(target_override).resolve() if target_override else BASE/spec['relative_target']
    if target.name!='build-host': die('target root must be named build-host')
    archive=HERE/spec['archive']
    if sha(archive)!=spec['archive_sha256']: die('archive SHA-256 mismatch')
    inv=json.loads(invbytes)['targets'][key]; expected=inv['entries']; expected_map={r['path']:r for r in expected}
    if inv['summary']['acl_entry_count']!=0: die('this restore tool is scoped to the two no-ACL build-host trees')
    if not target.is_dir() or target.is_symlink(): die('target must exist as a real directory with restore pointer')
    pointer=target/POINTER
    if not pointer.is_file() or spec['archive_sha256'] not in pointer.read_text(): die('missing/mismatched restore pointer')
    current=inventory(target)
    if current.get('symlinks') or current.get('other_entries'):
        die('target contains symlinks or unsupported entry types; refusing replacement')
    current_map={r['path']:r for r in current['entries']}
    allowed=set(expected_map)|{POINTER}
    if not set(current_map)<=allowed: die('unexpected path in target; refusing replacement')
    for p,row in current_map.items():
        if p==POINTER: continue
        exp=expected_map[p]
        if row['kind']!=exp['kind']: die(f'target kind changed: {p}')
        if row['kind']=='file' and row.get('sha256')!=exp.get('sha256'): die(f'target file changed: {p}')
    with tempfile.TemporaryDirectory(prefix='buildhost-restore-',dir=str(HERE)) as td:
        td=Path(td)
        subprocess.run(['/usr/bin/ditto','-x','-k','--extattr','--acl','--rsrc','--qtn',str(archive),str(td)],check=True)
        extracted=td/'build-host'
        if not extracted.is_dir(): die('archive did not contain build-host root')
        unwanted=extracted/'CMakeFiles'/'CMakeScratch'
        if 'CMakeFiles/CMakeScratch' not in expected_map and unwanted.is_dir():
            if any(unwanted.iterdir()): die('unexpected nonempty CMakeScratch during restore')
            unwanted.rmdir()
        apply_metadata(expected,extracted)
        validate(extracted,expected)
        shutil.rmtree(target)
        os.replace(extracted,target)
        validate(target,expected)
    print(f'restored and verified {key}: {target}')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('target',choices=['004','005']);ap.add_argument('--target-root',help='non-production restore destination used for a recovery drill');args=ap.parse_args();main(args.target,args.target_root)
