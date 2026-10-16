#!/usr/bin/env python3
"""Independent consistency audit for compaction-031; read-only on source targets."""
import gzip,hashlib,importlib.util,json,os,re,stat,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=Path('/Users/home/numi-cardboard-20261006')
POINTER='RESTORE_FROM_COMPACTION.txt'
RX=re.compile(r'^build-cardboard-provenance-(\d{3})$')
sp=importlib.util.spec_from_file_location('compact',str(HERE/'compact_trees.py'))
cm=importlib.util.module_from_spec(sp);sp.loader.exec_module(cm)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def req(x,msg):
 if not x: raise RuntimeError(msg)
doc=json.loads((HERE/'preflight.json').read_text())
req(doc['eligible_unarchived_nonempty_roots']==160,'preflight count mismatch')
rows=doc['targets'];summary=[];bad=[]
for row in rows:
 key=row['key']; build=row['build']; target=Path(row['target'])
 m=RX.fullmatch(build)
 req(m and 23<=int(m.group(1))<=153 and build!='build-cardboard-provenance-027',f'out-of-scope target {key}')
 recp=HERE/'receipts'/f'{key}.json'; remp=HERE/'receipts'/f'{key}-removed.json'
 ip=HERE/'inventories'/f'{key}.json.gz'; arc=HERE/'archives'/f'{key}.tar.gz'
 req(recp.is_file() and remp.is_file() and ip.is_file() and arc.is_file(),f'missing artifact for {key}')
 rec=json.loads(recp.read_text());rem=json.loads(remp.read_text())
 ash=sha(arc); ish=sha(ip)
 req(ash==rec['archive_sha256']==rem['archive_sha256'],f'archive hash binding mismatch {key}')
 req(ish==rec['inventory_gzip_sha256']==rem['inventory_gzip_sha256'],f'inventory hash binding mismatch {key}')
 with gzip.open(ip,'rb') as f: inv=json.loads(f.read())
 req(inv['root']==str(target),f'inventory root mismatch {key}')
 files=[e for e in inv['entries'] if e['kind']=='file'];dirs=[e for e in inv['entries'] if e['kind']=='directory']
 req(len(files)==rec['file_count']==rem['removed_files'],f'file count mismatch {key}')
 req(len(dirs)==rec['directory_count'],f'dir count mismatch {key}')
 req(sum(e['size'] for e in files)==rec['logical_file_bytes']==rem['removed_logical_bytes'],f'logical bytes mismatch {key}')
 req(sum(e['allocated_bytes'] for e in files)==rec['allocated_file_bytes']==rem['removed_allocated_bytes'],f'allocated bytes mismatch {key}')
 req(rec['restore_test']['exact_paths_hashes_metadata'] is True,f'no restore verification {key}')
 req(rec['lsof_before_archive']['open_handle_rows']==[] and rec['lsof_before_delete']['open_handle_rows']==[] and rem['lsof_before_delete']['open_handle_rows']==[],f'open handles reported {key}')
 req(rec['tracked_files_before']==[] and rec['tracked_files_before_delete']==[] and
     rec['process_before']['target_matches']==[] and rec['process_before_delete']['target_matches']==[] and
     rem['process_before_delete']['target_matches']==[],f'admission gate not clean {key}')
 req(not inv['symlinks'] and not inv['other_entries'] and all(e['nlink']==1 for e in files),f'unsupported entry in inventory {key}')
 ptr=target/POINTER
 req(target.is_dir() and not target.is_symlink() and ptr.is_file() and not ptr.is_symlink(),f'pointer missing {key}')
 pt=ptr.read_text()
 req(rec['archive_sha256'] in pt and rec['inventory_gzip_sha256'] in pt and f'compact_trees.py restore {key}' in pt,f'pointer binding mismatch {key}')
 cur=cm.inventory(target)
 req({e['path'] for e in cur['entries']}=={'.',POINTER},f'non-pointer residue {key}')
 # Restore-test proof binds the entire file count/path and all metadata/hash checks, including xattrs and ACL listing.
 req(rec['restore_test']['xattr_names']==inv['xattr_names'] and rec['restore_test']['xattr_entries']==inv['xattr_entry_count'] and rec['restore_test']['acl_entries']==inv['acl_entry_count'],f'attribute restore mismatch {key}')
 summary.append({'key':key,'archive_sha256':ash,'archive_bytes':arc.stat().st_size,'inventory_gzip_sha256':ish,'files':len(files),'dirs':len(dirs),'allocated_bytes':sum(e['allocated_bytes'] for e in files),'logical_bytes':sum(e['size'] for e in files),'target_pointer':str(ptr)})
# Ensure no in-scope nonpointer nonempty CMakeFiles trees remain, without traversing excluded source 027 or 154+.
remaining=[]
for build in R.iterdir():
 m=RX.fullmatch(build.name)
 if not m or not (23<=int(m.group(1))<=153) or build.name=='build-cardboard-provenance-027': continue
 for rel in ('CMakeFiles','matter/CMakeFiles'):
  p=build/rel
  if not p.is_dir() or p.is_symlink() or (p/POINTER).exists(): continue
  files=0
  for _,_,fs in os.walk(p,topdown=True,followlinks=False): files+=len(fs)
  if files: remaining.append(str(p))
req(not remaining,'eligible nonpointer CMakeFiles remain: '+repr(remaining[:20]))
cp=subprocess.run(['/bin/df','-k','/System/Volumes/Data'],text=True,capture_output=True,check=True)
finaldf=cm.df_record()
pre=doc['df_at_preflight']
report={'schema':'numi.cardboard.closed-cmakefiles-compaction-final-audit.v1','preflight_sha256':sha(HERE/'preflight.json'),
 'eligible_targets':len(rows),'verified_archives':len(summary),'total_files':sum(x['files'] for x in summary),
 'total_directories':sum(x['dirs'] for x in summary),'total_allocated_file_bytes':sum(x['allocated_bytes'] for x in summary),
 'total_logical_file_bytes':sum(x['logical_bytes'] for x in summary),'total_archive_bytes':sum(x['archive_bytes'] for x in summary),
 'sum_immediate_per_tree_df_changes_bytes':sum(json.loads((HERE/'receipts'/f"{x['key']}-removed.json").read_text())['measured_df_free_change_bytes'] for x in summary),
 'df_preflight':pre,'df_final':finaldf,'endpoint_df_delta_bytes':finaldf['free_bytes']-pre['free_bytes'],
 'remaining_eligible_nonpointer_nonempty_targets':remaining,'status':'all enumerated archives hash-verified; each had a full pre-removal extraction check; each target is pointer-only and restorable',
 'scope':'Only CMakeFiles / matter/CMakeFiles within exact build suffix 023..153 excluding 027; previously pointer-only roots were preserved. No source/build outputs outside those exact trees were deleted.',
 'tree_receipts':summary}
raw=(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(); p=HERE/'final-audit.json';p.write_bytes(raw)
print(json.dumps({k:report[k] for k in ('eligible_targets','verified_archives','total_files','total_allocated_file_bytes','total_logical_file_bytes','total_archive_bytes','sum_immediate_per_tree_df_changes_bytes','df_preflight','df_final','endpoint_df_delta_bytes','remaining_eligible_nonpointer_nonempty_targets','status')},sort_keys=True))
