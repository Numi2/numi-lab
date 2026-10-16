#!/usr/bin/env python3
"""Verify and seal the per-tree archive/inventory/receipt set without changing targets."""
import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/Users/home/numi-cardboard-20261006')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def require(x,m):
 if not x: raise SystemExit(m)
def main():
 preps=sorted((HERE/'receipts').glob('*-prepare.json'))
 removes=sorted((HERE/'receipts').glob('*-remove.json'))
 require(len(preps)==len(removes) and len(preps)>0,'prepare/removal receipt sets differ or are empty')
 targets=[]; totals={'allocated_file_bytes':0,'logical_file_bytes':0,'archive_bytes':0,'regular_files':0,'directories':0,'inventory_entries':0,'xattr_entries':0,'removed_directories':0}
 seen=set()
 for pp in preps:
  d=json.loads(pp.read_text()); name=d['build']; require(name not in seen,'duplicate target'); seen.add(name)
  rp=HERE/'receipts'/f'{name}-remove.json'; require(rp.is_file(),f'missing removal receipt {name}')
  r=json.loads(rp.read_text()); arch=Path(d['archive']); ip=Path(d['inventory']); target=Path(d['target'])
  require(sha(arch)==d['archive_sha256'] and arch.stat().st_size==d['archive_bytes'],f'archive binding failed {name}')
  require(sha(ip)==d['inventory_gzip_sha256'],f'inventory binding failed {name}')
  with gzip.open(ip,'rb') as f: inv=json.loads(f.read())
  require(inv['schema']=='numi.cardboard.buildhost-tree-inventory.v1',f'inventory schema {name}')
  require(inv['root']==str(target),f'inventory root mismatch {name}')
  require(inv['symlinks']==[] and inv['other_entries']==[] and inv['acl_entry_count']==0,f'unsupported source entries {name}')
  require(all(x['flags']==0 for x in inv['entries']),f'flags {name}')
  require(all(x['nlink']==1 for x in inv['entries'] if x['kind']=='file'),f'hardlinks {name}')
  require(d['tracked_files']==[] and d['lsof_gate']['open_handles']==0 and r['lsof_gate']['open_handles']==0,f'preflight handles/tracked {name}')
  require(d['process_gate']['target_mentions_from_build_tools']==[] and r['process_gate']['target_mentions_from_build_tools']==[],f'active process {name}')
  require(d['restore_test']['path_and_metadata_hash_match'] is True,f'restore test missing {name}')
  require(r['remaining_entries']==['.','RESTORE_FROM_COMPACTION.txt'],f'pointer-only target mismatch {name}')
  pointer=target/'RESTORE_FROM_COMPACTION.txt'
  require(pointer.is_file() and d['archive_sha256'] in pointer.read_text() and d['inventory_gzip_sha256'] in pointer.read_text(),f'pointer mismatch {name}')
  # Independently ensure only the expected restore pointer remains.
  names=[]
  for dp,dirs,files in os.walk(target,followlinks=False):
   names.extend(os.path.relpath(os.path.join(dp,x),target) for x in dirs+files)
  require(set(names)=={'RESTORE_FROM_COMPACTION.txt'},f'unexpected live path {name}: {names[:4]}')
  file_rows=[x for x in inv['entries'] if x['kind']=='file']; dir_rows=[x for x in inv['entries'] if x['kind']=='directory']
  item={'build':name,'target':str(target),'archive':str(arch),'archive_sha256':d['archive_sha256'],'archive_bytes':d['archive_bytes'],
    'inventory':str(ip),'inventory_gzip_sha256':d['inventory_gzip_sha256'],'inventory_uncompressed_sha256':d['inventory_uncompressed_sha256'],
    'prepare_receipt':str(pp),'prepare_receipt_sha256':sha(pp),'removal_receipt':str(rp),'removal_receipt_sha256':sha(rp),
    'pointer':str(pointer),'pointer_sha256':sha(pointer),'source_file_count':len(file_rows),'source_directory_count':len(dir_rows),
    'source_allocated_file_bytes':sum(x['allocated_bytes'] for x in file_rows),'source_logical_file_bytes':sum(x['size'] for x in file_rows),
    'source_xattr_entries':inv['xattr_entry_count'],'restore_verified_before_removal':True,
    'restore_comparison_fields':['path set','file SHA-256','kind','mode','uid','gid','size','nanosecond mtime','flags','link count','xattr names and values','ACL count and normalized listing'],
    'birthtime':'recorded in inventory; not reproduced by extraction'}
  targets.append(item)
  totals['allocated_file_bytes']+=item['source_allocated_file_bytes'];totals['logical_file_bytes']+=item['source_logical_file_bytes'];totals['archive_bytes']+=item['archive_bytes'];totals['regular_files']+=item['source_file_count'];totals['directories']+=item['source_directory_count'];totals['inventory_entries']+=len(inv['entries']);totals['xattr_entries']+=item['source_xattr_entries'];totals['removed_directories']+=r['removed_directories']
  require(r['archive_sha256']==d['archive_sha256'] and r['inventory_sha256']==d['inventory_gzip_sha256'],f'cross receipt mismatch {name}')
 require(len(removes)==len(targets),'orphan removal receipt')
 excluded={'build-cardboard-provenance-027':'explicit external-case precaution','build-cardboard-provenance-154':'explicit exclusion','build-cardboard-provenance-157':'explicit exclusion','build-cardboard-provenance-current':'explicit current exclusion'}
 for name in excluded:
  p=BASE/name/'CMakeFiles'
  if p.exists(): require(not (p/'RESTORE_FROM_COMPACTION.txt').exists(),f'excluded path modified: {name}')
 initial=json.loads((HERE/'receipts/build-cardboard-provenance-023-prepare.json').read_text())['df_before']
 crossing=json.loads((HERE/'receipts/build-cardboard-provenance-085-remove.json').read_text())['df_after_remove']
 final_cp=subprocess.check_output(['/bin/df','-k',str(BASE)],text=True).splitlines()[-1].split()
 final_free_kib=int(final_cp[3]); final_free_bytes=final_free_kib*1024
 manifest={'schema':'numi.cardboard.provenance-cmakefiles-compaction.v1','package':'cardboard-provenance-cmakefiles-compaction-001',
  'scope':{'source_root':str(BASE),'allowed_path_pattern':'build-cardboard-provenance-*/CMakeFiles only','excluded_builds':excluded,
    'mutated_path_rule':'Only inventoried regular files were unlinked; only inventoried empty directories below CMakeFiles were rmdir-ed; a restore pointer remains in each compacted root.',
    'untouched_kinds':['source','executables','metallibs/shaders','static libraries','compile_commands.json','non-CMakeFiles build outputs','native external cases']},
  'counts':{'compacted_builds':len(targets),'eligible_builds_left_unmodified':64,'tracked_files_in_compacted_trees':0,'active_build_handle_count':0},
  'aggregate_source_bytes':totals,'targets':targets,
  'free_space_observations':{'initial_df_free_bytes':initial['free'],'threshold_crossing_target':'build-cardboard-provenance-085',
    'threshold_crossing_df_free_bytes':crossing['free'],'threshold_bytes':int(2.5*1024**3),'final_observation_df_free_bytes':final_free_bytes,
    'global_delta_initial_to_threshold_bytes':crossing['free']-initial['free'],
    'caveat':'Concurrent task-owned clone compaction and temporary restoration altered shared df; do not attribute the global delta to these CMakeFiles alone. Per-target removal receipts preserve their own df readings.'},
  'restoration_limits':{'birthtime':'Recorded but filesystem-created birthtime changes during extraction.','ACL':'Every target had zero ACL entries; normalized listing was verified after restore.','hardlinks':'Every archived regular file had nlink=1; archive restoration does not reconstruct shared inode relationships.'},
  'receipt_errata':[{'build':'build-cardboard-provenance-023','field':'remove.df_before_remove','issue':'Contains post-removal df value due first script revision. Use prepare.df_before and remove.measured_free_increase_since_prepare; preserve original receipt.'}],
  'tools':{'inventory_script':str(HERE/'inventory_cmakefiles.py'),'inventory_script_sha256':sha(HERE/'inventory_cmakefiles.py'),
    'compactor_restore_script':str(HERE/'compact_one.py'),'compactor_restore_script_sha256':sha(HERE/'compact_one.py'),
    'inventory_source_pattern':'/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-buildhost-compaction-020/inventory_build_host.py',
    'inventory_source_sha256':sha(Path('/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-buildhost-compaction-020/inventory_build_host.py'))}}
 out=HERE/'archive-manifest.json';out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'manifest':str(out),'sha256':sha(out),'build_count':len(targets),'totals':totals,'initial_free':initial['free'],'threshold_crossing_free':crossing['free'],'final_free':final_free_bytes},sort_keys=True))
if __name__=='__main__':main()
