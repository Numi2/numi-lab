#!/usr/bin/env python3
"""Archive and restore-verify only old generated provenance-build CMakeFiles trees."""
import argparse, gzip, hashlib, importlib.util, json, os, re, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path

R=Path('/Users/home/numi-cardboard-20261006')
B=Path('/Users/home/cardboard-implementation-20261014')
HERE=Path(__file__).resolve().parent
ARCH=HERE/'archives'; INVS=HERE/'inventories'; RECEIPTS=HERE/'receipts'
POINTER='RESTORE_FROM_COMPACTION.txt'
MIN_FREE=512*1024**2
F_FIELDS=('kind','mode','uid','gid','size','mtime_ns','flags','nlink','xattrs','sha256')
RX=re.compile(r'^build-cardboard-provenance-(\d{3})$')
EXCLUDED_BUILDS={'build-cardboard-provenance-027'}

_spec=importlib.util.spec_from_file_location('inventory_cmakefiles',str(HERE/'inventory_cmakefiles.py'))
_invmod=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_invmod)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def df_record():
    cp=subprocess.run(['/bin/df','-k','/System/Volumes/Data'],text=True,capture_output=True,check=True)
    line=cp.stdout.splitlines()[-1]
    cols=line.split()
    return {'free_kib':int(cols[3]),'free_bytes':int(cols[3])*1024,'raw':cp.stdout.rstrip()}
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def candidate_roots():
    out=[]
    for build in sorted(R.iterdir()):
        m=RX.fullmatch(build.name)
        if not m: continue
        n=int(m.group(1))
        if not (23<=n<=153) or build.name in EXCLUDED_BUILDS: continue
        if build.is_symlink() or not build.is_dir(): continue
        for rel,kind in [('CMakeFiles','root'),('matter/CMakeFiles','matter')]:
            target=build/rel
            if target.is_symlink() or not target.is_dir(): continue
            if (target/POINTER).exists(): continue
            files=dirs=logical=allocated=0; symlinks=[]; special=[]
            for base,ds,fs in os.walk(target,topdown=True,followlinks=False):
                bp=Path(base)
                for name in list(ds):
                    p=bp/name; st=p.lstat()
                    if stat.S_ISLNK(st.st_mode): symlinks.append(str(p)); ds.remove(name)
                    elif stat.S_ISDIR(st.st_mode): dirs+=1; allocated+=st.st_blocks*512
                    else: special.append(str(p))
                for name in fs:
                    p=bp/name; st=p.lstat()
                    if stat.S_ISREG(st.st_mode): files+=1; logical+=st.st_size; allocated+=st.st_blocks*512
                    else: special.append(str(p))
            st=target.lstat(); dirs+=1; allocated+=st.st_blocks*512
            if not files: continue
            out.append({'key':f'{build.name}__{kind}','build':build.name,'kind':kind,
                        'relative_target':str(target.relative_to(R)),'target':str(target),
                        'file_count':files,'directory_count':dirs,'logical_file_bytes':logical,
                        'allocated_bytes':allocated,'symlinks':symlinks,'special_entries':special})
    return sorted(out,key=lambda x:(-x['allocated_bytes'],x['key']))
def git_tracked(rel):
    cp=subprocess.run(['/usr/bin/git','-C',str(R),'ls-files','--full-name','-z','--',rel],capture_output=True,check=True)
    return [x.decode(errors='replace') for x in cp.stdout.split(b'\0') if x]
def process_gate(build,target):
    cp=subprocess.run(['/bin/ps','-axo','pid=,comm=,args='],text=True,capture_output=True,check=True)
    tools={'cmake','make','ninja','clang','clang++','cc','c++','ld','metal','xcodebuild','swift','swiftc','xcrun',
           'numi-matter-cardboard-paper-material-check','numi-matter-cardboard-paper-material-probe'}
    hits=[]
    for line in cp.stdout.splitlines():
        q=line.strip().split(None,2)
        if len(q)<3 or q[1].split('/')[-1] not in tools: continue
        args=q[2]
        if str(target) in args or build in args:
            hits.append(line.strip())
    require(not hits,'active build/probe process refers to target: '+repr(hits[:5]))
    return {'process_snapshot_command':['/bin/ps','-axo','pid=,comm=,args='],'target_matches':hits}
def lsof_gate(target):
    cp=subprocess.run(['/usr/sbin/lsof','+D',str(target)],text=True,capture_output=True)
    require(cp.returncode in (0,1),'lsof failed for target: '+cp.stderr[:1000])
    rows=[x for x in cp.stdout.splitlines() if x.strip() and not x.startswith('COMMAND')]
    require(not rows,'open handles under target: '+repr(rows[:10]))
    return {'command':['/usr/sbin/lsof','+D',str(target)],'exit_code':cp.returncode,'open_handle_rows':rows}
def inventory(path):
    data=_invmod.inventory(path)
    require(not data.get('symlinks'),'symlinks are not eligible: '+repr(data['symlinks'][:5]))
    require(not data.get('other_entries'),'special entries are not eligible: '+repr(data['other_entries'][:5]))
    require(all(x.get('nlink',1)==1 for x in data['entries'] if x['kind']=='file'),'hard-linked file is not eligible')
    require(all(x.get('flags',0)==0 for x in data['entries']),'nonzero file flags are not eligible')
    return data
def compare(expected,actual):
    require(not actual.get('symlinks') and not actual.get('other_entries'),'restored tree has symlink/special path')
    A={x['path']:x for x in expected['entries']}; C={x['path']:x for x in actual['entries']}
    require(set(A)==set(C),f'path mismatch missing={sorted(set(A)-set(C))[:5]} extra={sorted(set(C)-set(A))[:5]}')
    for p,a in A.items():
        for field in F_FIELDS:
            require(a.get(field)==C[p].get(field),f'mismatch {p} {field}: {a.get(field)!r}!={C[p].get(field)!r}')
    require(expected['acl_entry_count']==actual['acl_entry_count'],'ACL count mismatch')
    require(expected['normalized_acl_listing_sha256']==actual['normalized_acl_listing_sha256'],'ACL listing mismatch')
    require(expected['xattr_names']==actual['xattr_names'],'xattr names mismatch')
    require(expected['xattr_entry_count']==actual['xattr_entry_count'],'xattr count mismatch')
def inv_file(key): return INVS/f'{key}.json.gz'
def archive_file(key): return ARCH/f'{key}.tar.gz'
def receipt_file(key): return RECEIPTS/f'{key}.json'
def load_inv_file(path):
    with gzip.open(path,'rb') as f: return json.loads(f.read())
def write_json_atomic(path,obj):
    path=Path(path); tmp=path.with_name(path.name+'.partial')
    raw=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
    with open(tmp,'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    return hashlib.sha256(raw).hexdigest()
def apply_metadata(rows,root):
    root=Path(root)
    for row in sorted(rows,key=lambda r:(r['kind']=='directory',-r['path'].count('/'))):
        p=root if row['path']=='.' else root/row['path']
        st=p.lstat(); require(not stat.S_ISLNK(st.st_mode),f'restore symlink: {p}')
        if st.st_uid!=row['uid'] or st.st_gid!=row['gid']: os.chown(p,row['uid'],row['gid'],follow_symlinks=False)
        os.chmod(p,row['mode'],follow_symlinks=False)
        # bsdtar --mac-metadata restores xattrs/extended ACLs; exact comparison follows.
        st=p.stat(follow_symlinks=False); os.utime(p,ns=(st.st_atime_ns,row['mtime_ns']),follow_symlinks=False)
def make_inventory_archive(key,target,preflight_row):
    require(not inv_file(key).exists() and not archive_file(key).exists() and not receipt_file(key).exists(),f'output already exists for {key}')
    build=preflight_row['build']; rel=preflight_row['relative_target']; target=Path(target)
    require(target==R/rel and target.name=='CMakeFiles' and target.is_dir() and not target.is_symlink(),'target path fails strict scope binding')
    require(not (target/POINTER).exists(),'target already has restore pointer; skip')
    tracked=git_tracked(rel); require(not tracked,'tracked paths present: '+repr(tracked[:5]))
    proc=process_gate(build,target); open0=lsof_gate(target); df0=df_record()
    data=inventory(target)
    allocated=sum(x['allocated_bytes'] for x in data['entries'] if x['kind']=='file')
    logical=sum(x['size'] for x in data['entries'] if x['kind']=='file')
    require(df0['free_bytes']>=MIN_FREE+2*logical,'insufficient per-tree archive/restore headroom')
    raw=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode()
    ip=inv_file(key); temp_ip=ip.with_name(ip.name+'.partial')
    with open(temp_ip,'xb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=9,mtime=0) as gz: gz.write(raw)
        f.flush();os.fsync(f.fileno())
    os.replace(temp_ip,ip); inv_sha=sha(ip); raw_inv_sha=hashlib.sha256(raw).hexdigest()
    arc=archive_file(key); temp_arc=arc.with_name(arc.name+'.partial')
    cmd=['/usr/bin/tar','--format=pax','--mac-metadata','--acls','--fflags','--xattrs','--numeric-owner','-czf',str(temp_arc),'-C',str(target.parent),target.name]
    cp=subprocess.run(cmd,text=True,capture_output=True)
    require(cp.returncode==0,'tar create failed: '+cp.stderr[:2000])
    os.replace(temp_arc,arc); arc_sha=sha(arc); arc_size=arc.stat().st_size; after_arc=df_record()
    with tempfile.TemporaryDirectory(prefix=key+'-',dir=str(HERE/'restore-test')) as td:
        td=Path(td)
        xcmd=['/usr/bin/tar','--mac-metadata','--acls','--fflags','--xattrs','--numeric-owner','-xzf',str(arc),'-C',str(td)]
        xp=subprocess.run(xcmd,text=True,capture_output=True)
        require(xp.returncode==0,'tar extract failed: '+xp.stderr[:2000])
        restored=td/'CMakeFiles'
        require(restored.is_dir() and not restored.is_symlink(),'archive did not restore expected CMakeFiles root')
        apply_metadata(data['entries'],restored)
        got=inventory(restored); compare(data,got)
        test={'restore_path_count':len(got['entries']),'files':sum(r['kind']=='file' for r in got['entries']),
              'directories':sum(r['kind']=='directory' for r in got['entries']),'xattr_names':got['xattr_names'],
              'xattr_entries':got['xattr_entry_count'],'acl_entries':got['acl_entry_count'],'exact_paths_hashes_metadata':True}
    after_restore=df_record()
    # Recheck source unchanged before authorizing any deletion.
    got_src=inventory(target); compare(data,got_src)
    proc2=process_gate(build,target); open1=lsof_gate(target); tracked2=git_tracked(rel)
    require(not tracked2,'tracked paths appeared before delete')
    record={'schema':'numi.cardboard.closed-cmakefiles-compaction-tree.v1','key':key,'build':build,
            'target':str(target),'relative_target':rel,'archive':str(arc),'archive_sha256':arc_sha,'archive_bytes':arc_size,
            'inventory':str(ip),'inventory_gzip_sha256':inv_sha,'inventory_uncompressed_sha256':raw_inv_sha,
            'inventory_entries':len(data['entries']),'file_count':test['files'],'directory_count':test['directories'],
            'logical_file_bytes':logical,'allocated_file_bytes':allocated,'tracked_files_before':tracked,
            'tracked_files_before_delete':tracked2,'process_before':proc,'process_before_delete':proc2,
            'lsof_before_archive':open0,'lsof_before_delete':open1,'restore_test':test,
            'birthtime_limit':'Original birthtime is recorded in the inventory; archive extraction creates new birthtimes. File hashes, relative paths, modes, uid/gid, nanosecond mtimes, flags, links, extended attributes, and ACL listing were verified exactly.',
            'df_before_archive':df0,'df_after_archive':after_arc,'df_after_restore_test_cleanup':after_restore,
            'archive_command':cmd,'restore_command':xcmd}
    recsha=write_json_atomic(receipt_file(key),record)
    return record,recsha,data
def remove_target(key,preflight_row):
    target=R/preflight_row['relative_target']; build=preflight_row['build']
    recp=receipt_file(key); require(recp.is_file(),'no successful archive/restore prepare receipt')
    rec=json.loads(recp.read_text()); require(sha(archive_file(key))==rec['archive_sha256'],'archive changed after restore test')
    require(sha(inv_file(key))==rec['inventory_gzip_sha256'],'inventory changed')
    data=load_inv_file(inv_file(key)); require(data['root']==str(target),'inventory root mismatch')
    tracked=git_tracked(preflight_row['relative_target']); require(not tracked,'tracked paths present before removal')
    proc=process_gate(build,target); openf=lsof_gate(target); current=inventory(target); compare(data,current)
    before=df_record()
    pointer=target/POINTER
    require(not pointer.exists(),'restore pointer already present')
    record_text=(f'Archived generated CMakeFiles for {build}.\nTarget-relative path: {preflight_row["relative_target"]}\n'
      f'Archive: {archive_file(key)}\nArchive SHA-256: {rec["archive_sha256"]}\n'
      f'Inventory: {inv_file(key)}\nInventory SHA-256: {rec["inventory_gzip_sha256"]}\n'
      f'Restore: python3 {HERE}/compact_trees.py restore {key}\n')
    with open(pointer,'x') as f: f.write(record_text);f.flush();os.fsync(f.fileno())
    files=[x for x in data['entries'] if x['kind']=='file']
    dirs=[x for x in data['entries'] if x['kind']=='directory' and x['path']!='.']
    # Only inventoried files are unlinked, after an immediate per-file SHA/type recheck.
    for row in files:
        p=target/row['path']; st=p.lstat()
        require(stat.S_ISREG(st.st_mode) and not p.is_symlink(),f'not a regular file at removal: {p}')
        require(sha(p)==row['sha256'],f'file changed immediately before removal: {p}')
        p.unlink()
    for row in sorted(dirs,key=lambda x:(-x['path'].count('/'),x['path'])):
        p=target/row['path']; require(p.is_dir() and not p.is_symlink(),f'not directory at removal: {p}'); os.rmdir(p)
    remain=inventory(target); require({x['path'] for x in remain['entries']}=={'.',POINTER},'target not pointer-only after removal')
    require((target/POINTER).read_text()==record_text,'restore pointer content differs')
    after=df_record()
    result={'schema':'numi.cardboard.closed-cmakefiles-removal.v1','key':key,'target':str(target),
            'archive_sha256':rec['archive_sha256'],'inventory_gzip_sha256':rec['inventory_gzip_sha256'],
            'removed_files':len(files),'removed_logical_bytes':sum(x['size'] for x in files),
            'removed_allocated_bytes':sum(x['allocated_bytes'] for x in files),
            'removed_directories':len(dirs),'retained_pointer':str(pointer),
            'process_before_delete':proc,'lsof_before_delete':openf,'df_before_remove':before,'df_after_remove':after,
            'measured_df_free_change_bytes':after['free_bytes']-before['free_bytes'],'remaining_paths':['.',POINTER]}
    result_sha=write_json_atomic(RECEIPTS/f'{key}-removed.json',result)
    return result,result_sha
def restore_target(key):
    rp=receipt_file(key); require(rp.is_file(),'prepare receipt missing')
    rec=json.loads(rp.read_text()); target=Path(rec['target']); arc=Path(rec['archive']); ip=Path(rec['inventory'])
    require(sha(arc)==rec['archive_sha256'],'archive hash mismatch')
    require(sha(ip)==rec['inventory_gzip_sha256'],'inventory hash mismatch')
    data=load_inv_file(ip); require(target.is_dir() and not target.is_symlink(),'target missing/not real')
    ptr=target/POINTER; require(ptr.is_file() and rec['archive_sha256'] in ptr.read_text(),'restore pointer missing/hash mismatch')
    current=inventory(target); require(not current['symlinks'] and not current['other_entries'],'unsupported current target content')
    cur={x['path']:x for x in current['entries']}; expect={x['path']:x for x in data['entries']}
    require(set(cur)<=set(expect)|{POINTER},'unexpected path in pointer target')
    require(set(cur)=={'.',POINTER},'restore currently supports pointer-only target, refuses partial target')
    with tempfile.TemporaryDirectory(prefix=key+'-restore-',dir=str(HERE/'restore-test')) as td:
        td=Path(td); cmd=['/usr/bin/tar','--mac-metadata','--acls','--fflags','--xattrs','--numeric-owner','-xzf',str(arc),'-C',str(td)]
        cp=subprocess.run(cmd,text=True,capture_output=True); require(cp.returncode==0,'restore tar failed: '+cp.stderr[:1500])
        restored=td/'CMakeFiles'; require(restored.is_dir(),'restored CMakeFiles absent')
        apply_metadata(data['entries'],restored); compare(data,inventory(restored))
        pointer_text=ptr.read_text(); ptr.unlink(); os.rmdir(target); os.replace(restored,target)
        compare(data,inventory(target))
    return {'key':key,'target':str(target),'restore_verified':True,'archive_sha256':rec['archive_sha256'],'restored_at':time.time()}
def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='action',required=True)
    sp.add_parser('preflight')
    q=sp.add_parser('compact');q.add_argument('key')
    r=sp.add_parser('restore');r.add_argument('key')
    args=ap.parse_args()
    if args.action=='preflight':
        rows=candidate_roots(); df=df_record()
        # Record the prior compaction pointers and deliberately excluded range.
        all_roots=[]
        for build in sorted(R.iterdir()):
            m=RX.fullmatch(build.name)
            if not m or not (23<=int(m.group(1))<=153): continue
            for rel in ('CMakeFiles','matter/CMakeFiles'):
                p=build/rel
                if p.is_dir() and not p.is_symlink(): all_roots.append({'path':str(p),'pointer_present':(p/POINTER).exists()})
        doc={'schema':'numi.cardboard.closed-cmakefiles-preflight.v1','root':str(R),'range':[23,153],
          'excluded_builds':sorted(EXCLUDED_BUILDS),'excluded_suffixes':[154,157],'excluded_current':True,
          'df_at_preflight':df,'in_scope_real_roots':len(all_roots),'prior_pointer_roots':sum(x['pointer_present'] for x in all_roots),
          'eligible_unarchived_nonempty_roots':len(rows),'eligible_file_count':sum(x['file_count'] for x in rows),
          'eligible_directory_count':sum(x['directory_count'] for x in rows),
          'eligible_logical_file_bytes':sum(x['logical_file_bytes'] for x in rows),
          'eligible_allocated_bytes':sum(x['allocated_bytes'] for x in rows),'targets':rows}
        digest=write_json_atomic(HERE/'preflight.json',doc);print(json.dumps({'preflight_sha256':digest,**{k:doc[k] for k in ('df_at_preflight','in_scope_real_roots','prior_pointer_roots','eligible_unarchived_nonempty_roots','eligible_file_count','eligible_logical_file_bytes','eligible_allocated_bytes')}},sort_keys=True))
    elif args.action=='compact':
        doc=json.loads((HERE/'preflight.json').read_text());row=next((x for x in doc['targets'] if x['key']==args.key),None);require(row is not None,'key not in sealed preflight target list')
        rec,recsha,data=make_inventory_archive(args.key,row['target'],row)
        removal,rmsha=remove_target(args.key,row)
        print(json.dumps({'key':args.key,'prepare_receipt_sha256':recsha,'removal_receipt_sha256':rmsha,'archive_bytes':rec['archive_bytes'],'removed_allocated_bytes':removal['removed_allocated_bytes'],'df_after':removal['df_after_remove']},sort_keys=True))
    elif args.action=='restore': print(json.dumps(restore_target(args.key),sort_keys=True))
if __name__=='__main__': main()
