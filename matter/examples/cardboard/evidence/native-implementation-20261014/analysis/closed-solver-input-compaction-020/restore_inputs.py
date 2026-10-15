#!/usr/bin/env python3
"""Restore one compacted solver input tree from its verified deterministic tar.gz."""
from pathlib import Path, PurePosixPath
import argparse, gzip, hashlib, json, os, shutil, stat, tarfile

BASE=Path('/Users/home/cardboard-implementation-20261014')


def sha_file(path):
    h=hashlib.sha256(); n=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block); n+=len(block)
    return h.hexdigest(),n


def restore(pointer_path):
    pointer_path=Path(pointer_path).resolve(strict=True)
    pointer=json.loads(pointer_path.read_text())
    if pointer.get('schema')!='numi.cardboard.closed-solver-input-compaction.pointer.v1':
        raise RuntimeError('unsupported pointer schema')
    case=Path(pointer['case']).resolve(strict=True)
    archive=Path(pointer['archive_path']).resolve(strict=True)
    outroot=(BASE/'analysis/closed-solver-input-compaction-020').resolve(strict=True)
    archive.relative_to(outroot/'archives')
    ah,asz=sha_file(archive)
    if ah!=pointer['archive_sha256'] or asz!=pointer['archive_bytes']:
        raise RuntimeError('archive hash/size mismatch')
    manifest_path=Path(pointer['before_manifest_path']).resolve(strict=True)
    manifest=json.loads(manifest_path.read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=pointer['before_manifest_sha256']:
        raise RuntimeError('before inventory hash mismatch')
    if manifest['case']!=str(case): raise RuntimeError('case binding mismatch')
    target=case/'inputs'
    if not target.is_dir() or target.is_symlink(): raise RuntimeError('expected preserved inputs directory')
    if any(p.is_file() or p.is_symlink() for p in target.rglob('*')):
        raise RuntimeError('refusing to overwrite existing input content')
    dirs={row['path']:row for row in manifest['directories']}
    files={row['path']:row for row in manifest['files']}
    seen_dirs=set();seen_files=set()
    with tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            pp=PurePosixPath(member.name)
            if pp.is_absolute() or '..' in pp.parts or not pp.parts or pp.parts[0]!='inputs':
                raise RuntimeError(f'unsafe archive member {member.name}')
            rel='/'.join(pp.parts[1:])
            if member.isdir():
                if rel not in dirs or rel in seen_dirs or member.mode!=dirs[rel]['mode']:
                    raise RuntimeError(f'unexpected directory member {member.name}')
                dest=target/rel if rel else target
                dest.mkdir(parents=True,exist_ok=True)
                seen_dirs.add(rel)
            elif member.isfile():
                if rel not in files or rel in seen_files:
                    raise RuntimeError(f'unexpected file member {member.name}')
                row=files[rel]
                if member.size!=row['bytes'] or member.mode!=row['mode']:
                    raise RuntimeError(f'extent/mode mismatch {member.name}')
                dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                src=tar.extractfile(member)
                if src is None: raise RuntimeError(f'cannot read {member.name}')
                h=hashlib.sha256();n=0
                with src,dest.open('xb') as out:
                    for block in iter(lambda:src.read(1024*1024),b''):
                        h.update(block);n+=len(block);out.write(block)
                    out.flush();os.fsync(out.fileno())
                if h.hexdigest()!=row['sha256'] or n!=row['bytes']:
                    raise RuntimeError(f'content hash mismatch {member.name}')
                os.chmod(dest,row['mode']);os.utime(dest,ns=(row['mtime_ns'],row['mtime_ns']))
                seen_files.add(rel)
            else:
                raise RuntimeError(f'unsupported archive member type {member.name}')
    if seen_dirs!=set(dirs) or seen_files!=set(files):
        raise RuntimeError('archive member set incomplete')
    for row in manifest['directories']:
        dest=target/row['path'] if row['path'] else target
        os.chmod(dest,row['mode']);os.utime(dest,ns=(row['mtime_ns'],row['mtime_ns']))
    pred=json.loads((case/'prediction.json').read_text())
    if pred.get('inputs_sha256')!=manifest['inputs_sha256']:
        raise RuntimeError('prediction map differs from archived input map')
    for row in manifest['files']:
        p=target/row['path'];h,n=sha_file(p)
        if h!=row['sha256'] or n!=row['bytes'] or stat.S_IMODE(p.stat().st_mode)!=row['mode']:
            raise RuntimeError(f'postrestore verification failed {row["path"]}')
        if pred['inputs_sha256'].get(row['path'])!=h:
            raise RuntimeError(f'prediction input hash mismatch {row["path"]}')
    print(json.dumps({'restored_case':str(case),'files':len(files),'directories':len(dirs),'bytes':sum(x['bytes'] for x in files),'archive_sha256':ah,'verified':True},sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pointer');a=p.parse_args();restore(a.pointer)
