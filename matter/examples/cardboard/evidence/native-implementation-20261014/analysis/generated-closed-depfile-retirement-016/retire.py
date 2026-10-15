#!/usr/bin/env python3
"""Retire only inventoried closed generated CMake .d files in builds 133-136."""
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, time

BASE=pathlib.Path('/Users/home/numi-cardboard-20261006')
NUMBERS=(133,134,135,136)
TARGETS={n: BASE/f'build-cardboard-provenance-{n}' for n in NUMBERS}
OUT=pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/generated-closed-depfile-retirement-016')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def tree_candidates():
    result={}
    for n,root in TARGETS.items():
        cm=root/'CMakeFiles'
        if not root.is_dir() or not cm.is_dir() or root.name != f'build-cardboard-provenance-{n}':
            raise RuntimeError(f'expected build/CMakeFiles path missing: {root}')
        rows=[]
        for p in cm.rglob('*.d'):
            st=p.lstat()
            if pathlib.Path(p).is_symlink() or not p.is_file():
                raise RuntimeError(f'non-regular dependency candidate: {p}')
            rp=p.resolve(strict=True)
            if cm.resolve() not in rp.parents or rp.suffix != '.d':
                raise RuntimeError(f'out-of-scope candidate: {p}')
            rows.append({'path':str(p),'sha256':sha(p),'bytes':st.st_size,
                         'allocated_bytes':st.st_blocks*512,'device':st.st_dev,
                         'inode':st.st_ino,'mode':oct(st.st_mode & 0o7777)})
        result[str(n)]=sorted(rows,key=lambda x:x['path'])
    return result

def build_artifacts():
    result={}
    for n,root in TARGETS.items():
        receipt=root/'source-build.json'
        d=json.loads(receipt.read_text())
        if d.get('success') is not True or pathlib.Path(d.get('build_dir','')) != root:
            raise RuntimeError(f'build receipt not successful/bound: {receipt}')
        check={}
        for category,items in [('artifacts',d.get('artifacts_sha256',{})),
                               ('metadata',d.get('build_metadata_sha256',{}))]:
            check[category]={}
            for raw,want in items.items():
                q=pathlib.Path(raw) if raw.startswith('/') else root/raw
                got=sha(q) if q.is_file() else 'MISSING'
                check[category][str(q)]={'expected_sha256':want,'observed_sha256':got,
                                         'matches':got==want,'bytes':q.stat().st_size if q.is_file() else None}
                if got != want: raise RuntimeError(f'{category} hash mismatch: {q}')
        result[str(n)]={'source_build_receipt':str(receipt),
            'source_build_receipt_sha256':sha(receipt),
            'source_snapshot':d.get('source_snapshot'),'targets':d.get('targets'),
            'artifacts':check['artifacts'],'metadata':check['metadata']}
    return result

def statvfs(path='/Users/home'):
    s=os.statvfs(path)
    return {'path':path,'f_bsize':s.f_bsize,'f_frsize':s.f_frsize,
            'f_blocks':s.f_blocks,'f_bfree':s.f_bfree,'f_bavail':s.f_bavail,
            'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize}

def safety_scan():
    ps=subprocess.run(['ps','-Ao','pid,ppid,stat,etime,command'],capture_output=True,text=True,check=True)
    path_matches=[]
    for line in ps.stdout.splitlines()[1:]:
        if any(str(root) in line for root in TARGETS.values()): path_matches.append(line)
    # lsof is scoped to only the CMakeFiles trees being considered. Empty output
    # means there are no open files below those trees at the scan instant.
    lsof_rows=[]; lsof_runs=[]
    for n,root in TARGETS.items():
        cmd=['lsof','+D',str(root/'CMakeFiles')]
        run=subprocess.run(cmd,capture_output=True,text=True)
        lsof_runs.append({'command':cmd,'returncode':run.returncode,
                          'stdout':run.stdout,'stderr':run.stderr})
        rows=[ln for ln in run.stdout.splitlines()[1:] if ln.strip() and not ln.startswith('lsof ')]
        lsof_rows.extend(rows)
    return {'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'ps_command':['ps','-Ao','pid,ppid,stat,etime,command'],
            'ps_target_path_matches':path_matches,'lsof_runs':lsof_runs,
            'lsof_non_lsof_open_rows':lsof_rows}

before={'schema':'numi.cardboard.generated_depfile_retirement.before.v1',
        'scope':{str(n):str(TARGETS[n]/'CMakeFiles') for n in NUMBERS},
        'filesystem':statvfs(),'build_artifacts':build_artifacts(),
        'safety_scan':safety_scan(),'files':tree_candidates()}
if before['safety_scan']['ps_target_path_matches'] or before['safety_scan']['lsof_non_lsof_open_rows']:
    raise RuntimeError('active process or open file found in selected build trees; no deletion performed')
# Persist complete pre-removal bytes before the first unlink.
(OUT/'before.json').write_text(json.dumps(before,indent=2,sort_keys=True)+'\n')
all_rows=[row for rows in before['files'].values() for row in rows]
for row in all_rows:
    q=pathlib.Path(row['path'])
    st=q.lstat()
    if q.is_symlink() or not q.is_file() or st.st_dev != row['device'] or st.st_ino != row['inode'] or st.st_size != row['bytes'] or sha(q) != row['sha256']:
        raise RuntimeError(f'candidate changed after inventory; stopped before this file: {q}')
for row in all_rows: pathlib.Path(row['path']).unlink()
left=tree_candidates()
if any(left.values()): raise RuntimeError('new/unremoved .d file remains in selected CMakeFiles tree')
after_artifacts=build_artifacts()
if after_artifacts != before['build_artifacts']:
    raise RuntimeError('source-build artifacts/metadata changed during cleanup')
after={'schema':'numi.cardboard.generated_depfile_retirement.after.v1',
       'filesystem':statvfs(),'remaining_files':left,
       'deleted_count':len(all_rows),
       'deleted_logical_bytes':sum(x['bytes'] for x in all_rows),
       'deleted_allocated_bytes':sum(x['allocated_bytes'] for x in all_rows),
       'build_artifacts':after_artifacts,'safety_scan_after':safety_scan()}
if after['safety_scan_after']['ps_target_path_matches'] or after['safety_scan_after']['lsof_non_lsof_open_rows']:
    raise RuntimeError('process/open-file appeared in selected build trees after cleanup')
if after['filesystem']['available_bytes'] < before['filesystem']['available_bytes']:
    after['available_bytes_delta'] = after['filesystem']['available_bytes']-before['filesystem']['available_bytes']
else: after['available_bytes_delta']=after['filesystem']['available_bytes']-before['filesystem']['available_bytes']
(OUT/'after.json').write_text(json.dumps(after,indent=2,sort_keys=True)+'\n')
summary={'schema':'numi.cardboard.generated_depfile_retirement.summary.v1',
 'build_numbers':list(NUMBERS),'deleted_count':after['deleted_count'],
 'deleted_logical_bytes':after['deleted_logical_bytes'],
 'deleted_allocated_bytes':after['deleted_allocated_bytes'],
 'filesystem_available_bytes_before':before['filesystem']['available_bytes'],
 'filesystem_available_bytes_after':after['filesystem']['available_bytes'],
 'filesystem_available_bytes_delta':after['available_bytes_delta'],
 'source_build_artifacts_verified_before_and_after':True,
 'post_cleanup_safety_scan_clear':True,
 'files':{p.name:sha(p) for p in (OUT/'before.json',OUT/'after.json').__class__.__mro__[:0]}}
# Hash the persisted manifests without making the summary self-referential.
summary['manifest_sha256']={'before.json':sha(OUT/'before.json'),'after.json':sha(OUT/'after.json')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,sort_keys=True))
