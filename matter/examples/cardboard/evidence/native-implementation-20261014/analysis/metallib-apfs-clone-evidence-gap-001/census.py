#!/usr/bin/env python3
"""Read-only current-state inventory for task-owned provenance metallibs."""
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, time

ROOT = pathlib.Path('/Users/home/numi-cardboard-20261006')
EXCLUDED = {'build-cardboard-provenance-154', 'build-cardboard-provenance-157'}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def xattrs(path):
    # The qualified Python build omits os.listxattr; use macOS xattr's stable
    # listing form and bind the complete output bytes, without mutating attrs.
    p = subprocess.run(['/usr/bin/xattr', '-l', str(path)], check=False, capture_output=True)
    if p.returncode:
        return {'error': p.stderr.decode('utf-8', 'replace').strip() or f'xattr exit {p.returncode}'}
    return {'listing_length': len(p.stdout), 'listing_sha256': hashlib.sha256(p.stdout).hexdigest(),
            'listing_utf8': p.stdout.decode('utf-8', 'backslashreplace')}

def acl_lines(path):
    p = subprocess.run(['ls', '-lde', str(path)], check=False, capture_output=True, text=True)
    if p.returncode:
        return {'error': p.stderr.strip() or f'ls exit {p.returncode}'}
    lines = p.stdout.splitlines()
    return [line.rstrip() for line in lines[1:] if line.strip()]

def statrow(path):
    st = os.stat(path, follow_symlinks=False)
    return {
        'path': str(path), 'size': st.st_size, 'sha256': sha(path),
        'mode': oct(st.st_mode & 0o7777), 'uid': st.st_uid, 'gid': st.st_gid,
        'flags': getattr(st, 'st_flags', None), 'nlink': st.st_nlink,
        'blocks_512': st.st_blocks, 'mtime_ns': st.st_mtime_ns,
        'ctime_ns': st.st_ctime_ns, 'birthtime': getattr(st, 'st_birthtime', None),
        'xattrs': xattrs(path), 'acl_lines': acl_lines(path),
    }

def main():
    before = subprocess.check_output(['df', '-k', str(ROOT)], text=True).splitlines()[-1]
    rows, excluded_rows, build_dirs = [], [], []
    for entry in sorted(ROOT.glob('build-cardboard-provenance-*')):
        if not entry.is_dir():
            continue
        build_dirs.append(str(entry))
        target = excluded_rows if entry.name in EXCLUDED else rows
        for path in sorted(entry.rglob('*.metallib')):
            if path.is_file() and not path.is_symlink():
                target.append(statrow(path))
    groups = {}
    for row in rows:
        groups.setdefault(row['sha256'], []).append(row)
    duplicate_groups = {k:v for k,v in groups.items() if len(v) > 1}
    def meta(r):
        return (r['size'], r['mode'], r['uid'], r['gid'], r['flags'], r['xattrs'], r['acl_lines'])
    mismatches = []
    for digest, group in duplicate_groups.items():
        vals = {json.dumps(meta(r), sort_keys=True) for r in group}
        if len(vals) > 1:
            mismatches.append({'sha256':digest, 'paths':[r['path'] for r in group]})
    # One bounded lsof invocation against the exact in-scope file path set.
    allpaths = [r['path'] for r in rows]
    lsof = subprocess.run(['lsof', '-Fpn', '--', *allpaths], capture_output=True, text=True)
    after = subprocess.check_output(['df', '-k', str(ROOT)], text=True).splitlines()[-1]
    report = {
        'schema': 1, 'generated_utc_epoch': int(time.time()),
        'scope': {'root':str(ROOT), 'pattern':'build-cardboard-provenance-*/**/*.metallib', 'excluded_builds':sorted(EXCLUDED)},
        'current_df_before':before, 'current_df_after':after,
        'build_directory_count':len(build_dirs), 'in_scope_count':len(rows),
        'excluded_current_count':len(excluded_rows), 'total_count':len(rows)+len(excluded_rows),
        'unique_in_scope_sha256_count':len(groups), 'duplicate_group_count':len(duplicate_groups),
        'duplicate_path_count':sum(len(v) for v in duplicate_groups.values()),
        'duplicate_extra_path_count':sum(len(v)-1 for v in duplicate_groups.values()),
        'duplicate_group_metadata_mismatches':mismatches,
        'nlink_nonunit_paths':[r['path'] for r in rows if r['nlink'] != 1],
        'lsof_returncode':lsof.returncode, 'lsof_stdout':lsof.stdout, 'lsof_stderr':lsof.stderr,
        'rows':rows, 'excluded_rows':excluded_rows,
        'limitations':['This inventory is current-state only. It does not establish which paths were replaced, their prior bytes or metadata, or recover the compaction operation transcript/receipt.'],
    }
    out = pathlib.Path(__file__).with_name('current-inventory.json')
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('in_scope_count','excluded_current_count','total_count','unique_in_scope_sha256_count','duplicate_group_count','duplicate_extra_path_count','duplicate_group_metadata_mismatches','nlink_nonunit_paths','lsof_returncode','lsof_stderr','current_df_before','current_df_after')}, indent=2))
if __name__ == '__main__': main()
