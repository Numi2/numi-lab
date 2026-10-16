#!/usr/bin/env python3
"""Inventory a build-host tree without following symlinks; uses macOS xattr/ls."""
import argparse, collections, hashlib, json, os, re, stat, subprocess
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def xattr_values(root, attr):
    cp=subprocess.run(['/usr/bin/xattr','-p','-r','-v','-x',attr,str(root)],check=True,text=True,capture_output=True)
    result={}; current=None; parts=[]
    for line in cp.stdout.splitlines():
        if line.startswith('/') and line.endswith(': '):
            if current is not None: result[current]=bytes.fromhex(' '.join(parts)).hex()
            current=line[:-2]; parts=[]
        elif line.strip(): parts.append(line.strip())
    if current is not None: result[current]=bytes.fromhex(' '.join(parts)).hex()
    return result

def inventory(root):
    root=Path(root).resolve()
    x=subprocess.run(['/usr/bin/xattr','-r','-v',str(root)],check=True,text=True,capture_output=True)
    xnames=collections.defaultdict(set)
    for line in x.stdout.splitlines():
        if ': ' in line:
            path,attr=line.rsplit(': ',1); xnames[path].add(attr)
    values={a:xattr_values(root,a) for a in sorted({a for names in xnames.values() for a in names})}
    acl=subprocess.run(['/bin/ls','-leR',str(root)],check=True,text=True,capture_output=True)
    acl_text=acl.stdout.replace(str(root),'$ROOT')
    acl_entries=sum(bool(re.match(r'^\s*\d+: ',line)) for line in acl_text.splitlines())
    rows=[]; symlinks=[]; others=[]
    for base,dirs,files in os.walk(root,topdown=True,followlinks=False):
        parent=Path(base)
        for name in list(dirs):
            p=parent/name
            if p.is_symlink(): symlinks.append(p.relative_to(root).as_posix()); dirs.remove(name)
        for p in [parent/name for name in dirs]+[parent/name for name in files]:
            st=p.lstat(); reg=stat.S_ISREG(st.st_mode); directory=stat.S_ISDIR(st.st_mode)
            if not reg and not directory:
                others.append({'path':p.relative_to(root).as_posix(),'mode':st.st_mode}); continue
            row={'path':p.relative_to(root).as_posix(),'kind':'file' if reg else 'directory',
                 'mode':stat.S_IMODE(st.st_mode),'uid':st.st_uid,'gid':st.st_gid,'size':st.st_size,
                 'allocated_bytes':st.st_blocks*512,'mtime_ns':st.st_mtime_ns,
                 'birthtime_ns':int(getattr(st,'st_birthtime',0)*1_000_000_000),
                 'flags':getattr(st,'st_flags',0),'nlink':st.st_nlink,
                 'xattrs':{a:values[a].get(str(p),'') for a in sorted(xnames.get(str(p),set()))}}
            if reg: row['sha256']=sha256(p)
            rows.append(row)
    st=root.lstat()
    rows.append({'path':'.','kind':'directory','mode':stat.S_IMODE(st.st_mode),'uid':st.st_uid,'gid':st.st_gid,
        'size':st.st_size,'allocated_bytes':st.st_blocks*512,'mtime_ns':st.st_mtime_ns,
        'birthtime_ns':int(getattr(st,'st_birthtime',0)*1_000_000_000),'flags':getattr(st,'st_flags',0),'nlink':st.st_nlink,
        'xattrs':{a:values[a].get(str(root),'') for a in sorted(xnames.get(str(root),set()))}})
    rows.sort(key=lambda r:r['path'])
    return {'schema':'numi.cardboard.buildhost-tree-inventory.v1','root':str(root),
        'entries':rows,'acl_entry_count':acl_entries,'normalized_acl_listing_sha256':hashlib.sha256(acl_text.encode()).hexdigest(),
        'xattr_names':sorted({a for ns in xnames.values() for a in ns}),'xattr_entry_count':sum(map(len,xnames.values())),
        'symlinks':symlinks,'other_entries':others}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--output'); args=ap.parse_args()
    data=inventory(args.root); raw=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if args.output: Path(args.output).write_text(raw)
    else: print(raw,end='')
