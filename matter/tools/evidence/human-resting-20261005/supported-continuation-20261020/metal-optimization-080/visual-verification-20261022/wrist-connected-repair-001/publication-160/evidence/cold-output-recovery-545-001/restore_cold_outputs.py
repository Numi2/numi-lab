#!/usr/bin/env python3
"""Restore outputs from cold-output-recovery-545-001 after validating zstd and hashes."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

def digest(path):
    h=hashlib.sha256(); n=0
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b); n+=len(b)
    return h.hexdigest(),n

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def restore_copy(receipt,item,copy,archive_root,destination_root):
    archive=archive_root/item['archive_relative_path']
    if not archive.is_file() or archive.is_symlink(): raise RuntimeError(f'missing/linked archive: {archive}')
    if digest(archive)!=(item['archive_sha256'],item['archive_bytes']): raise RuntimeError(f'archive SHA/size mismatch: {archive}')
    dest=destination_root/copy['relative_to_workspace']
    if dest.exists():
        if digest(dest)!=(copy['sha256'],copy['bytes']): raise RuntimeError(f'restore target exists with different contents: {dest}')
        st=dest.stat()
        if st.st_mtime_ns!=copy['mtime_ns'] or (st.st_mode&0o777)!=copy['mode']:
            raise RuntimeError(f'restore target exists with different metadata: {dest}')
        return {'path':str(dest),'status':'already-restored-and-verified','sha256':copy['sha256'],'bytes':copy['bytes']}
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+f'.restore-{os.getpid()}.tmp')
    if tmp.exists(): raise RuntimeError(f'restore temporary path exists: {tmp}')
    proc=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    h=hashlib.sha256(); n=0
    try:
        with tmp.open('xb') as f:
            while True:
                b=proc.stdout.read(1024*1024)
                if not b: break
                f.write(b); h.update(b); n+=len(b)
            f.flush(); os.fsync(f.fileno())
        err=proc.stderr.read(); rc=proc.wait()
        if rc: raise RuntimeError(f'zstd restore failed: {err.decode(errors="replace")[:300]}')
        if (h.hexdigest(),n)!=(copy['sha256'],copy['bytes']): raise RuntimeError(f'restored SHA/size mismatch: {dest}')
        os.chmod(tmp,copy['mode']); os.utime(tmp,ns=(copy['atime_ns'],copy['mtime_ns']))
        os.replace(tmp,dest); fsync_dir(dest.parent)
        st=dest.stat()
        if digest(dest)!=(copy['sha256'],copy['bytes']) or (st.st_mode&0o777)!=copy['mode'] or st.st_mtime_ns!=copy['mtime_ns']:
            raise RuntimeError(f'post-restore SHA/size/mode/mtime mismatch: {dest}')
        return {'path':str(dest),'status':'restored-and-verified','sha256':copy['sha256'],'bytes':copy['bytes'],
                'mode':st.st_mode&0o777,'mtime_ns':st.st_mtime_ns}
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
        if tmp.exists(): tmp.unlink()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',required=True)
    ap.add_argument('--destination-root',required=True,help='Empty root that will contain D/ and publication-142/143/')
    ap.add_argument('--archive-root',help='Defaults to the directory containing the receipt')
    ap.add_argument('--restore-all',action='store_true')
    ap.add_argument('--only',action='append',default=[],help='Published-relative path, workspace-relative copy path, or absolute original path; repeatable')
    a=ap.parse_args(); receipt_path=Path(a.receipt); receipt=json.loads(receipt_path.read_text())
    archive_root=Path(a.archive_root) if a.archive_root else receipt_path.parent
    destination_root=Path(a.destination_root)
    wanted=set(a.only)
    if a.restore_all and wanted: raise RuntimeError('use --restore-all or --only, not both')
    rows=[]
    for item in receipt['files']:
        if wanted and not any(w in (item['published_path'],c['relative_to_workspace'],c['path']) for w in wanted for c in item['copies']): continue
        rows.append(item)
    if not rows: raise RuntimeError('no matching recovery items')
    if not a.restore_all and not wanted: raise RuntimeError('choose --restore-all or at least one --only path')
    results=[]
    for item in rows:
        for copy in item['copies']:
            results.append(restore_copy(receipt,item,copy,archive_root,destination_root))
    print(json.dumps({'restored_member_count':len(rows),'restored_copy_count':len(results),'results':results},indent=2))
if __name__=='__main__': main()
