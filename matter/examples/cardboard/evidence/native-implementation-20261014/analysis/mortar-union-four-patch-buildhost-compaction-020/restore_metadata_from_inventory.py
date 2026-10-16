#!/usr/bin/env python3
"""Restore exact mode/owner/mtime for a ditto-restored generated build tree."""
import argparse, json, os, stat
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('preinventory'); p.add_argument('target_key'); p.add_argument('restored_root'); a=p.parse_args()
source=json.load(open(a.preinventory)); rows=source['targets'][a.target_key]['entries']; root=Path(a.restored_root).resolve()
# Directories last, deepest first, so child operations do not alter their restored mtimes.
rows=sorted(rows,key=lambda r:(r['kind']=='directory', -r['path'].count('/')))
for row in rows:
    rel=row['path']; path=root if rel=='.' else root/rel
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode): raise SystemExit(f'refuse symlink: {path}')
    if st.st_uid!=row['uid'] or st.st_gid!=row['gid']:
        os.chown(path,row['uid'],row['gid'],follow_symlinks=False)
    os.chmod(path,row['mode'],follow_symlinks=False)
    current=path.stat(follow_symlinks=False)
    os.utime(path,ns=(current.st_atime_ns,row['mtime_ns']),follow_symlinks=False)
print(f'restored mode/uid/gid/mtime for {len(rows)} entries under {root}; birthtime is filesystem-created and not restored')
