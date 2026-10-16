#!/usr/bin/env python3
"""Bind the completed 031 compaction archive set and recovery pointers."""
import datetime,hashlib,json,os,stat,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=Path('/Users/home/numi-cardboard-20261006')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def df():
 cp=subprocess.run(['/bin/df','-k','/System/Volumes/Data'],text=True,capture_output=True,check=True)
 cols=cp.stdout.splitlines()[-1].split()
 return {'free_kib':int(cols[3]),'free_bytes':int(cols[3])*1024,'raw':cp.stdout.rstrip()}
def atomic(path,obj):
 path=Path(path); tmp=path.with_name(path.name+'.partial')
 raw=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
 with open(tmp,'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 return hashlib.sha256(raw).hexdigest()
pre=json.loads((HERE/'preflight.json').read_text()); aud=json.loads((HERE/'final-audit.json').read_text())
if len(pre['targets'])!=160 or aud['verified_archives']!=160: raise SystemExit('target count not 160')
if aud['remaining_eligible_nonpointer_nonempty_targets']: raise SystemExit('uncompacted target remains')
if list(HERE.rglob('*.partial')): raise SystemExit('partial file remains')
targets=[]
for row in pre['targets']:
 key=row['key']; prep=HERE/'receipts'/f'{key}.json'; rem=HERE/'receipts'/f'{key}-removed.json'
 inv=HERE/'inventories'/f'{key}.json.gz'; arc=HERE/'archives'/f'{key}.tar.gz'; ptr=Path(row['target'])/'RESTORE_FROM_COMPACTION.txt'
 rec=json.loads(prep.read_text()); rm=json.loads(rem.read_text())
 targets.append({'key':key,'build':row['build'],'relative_target':row['relative_target'],
   'pointer':str(ptr),'archive_relative':str(arc.relative_to(HERE)),'archive_sha256':sha(arc),'archive_bytes':arc.stat().st_size,
   'inventory_relative':str(inv.relative_to(HERE)),'inventory_gzip_sha256':sha(inv),
   'prepare_receipt_relative':str(prep.relative_to(HERE)),'prepare_receipt_sha256':sha(prep),
   'removal_receipt_relative':str(rem.relative_to(HERE)),'removal_receipt_sha256':sha(rem),
   'file_count':rec['file_count'],'directory_count':rec['directory_count'],
   'allocated_file_bytes':rec['allocated_file_bytes'],'logical_file_bytes':rec['logical_file_bytes'],
   'restore_verified_before_unlink':rec['restore_test']['exact_paths_hashes_metadata'],
   'tracked_files':len(rec['tracked_files_before'])+len(rec['tracked_files_before_delete']),
   'open_handles':len(rec['lsof_before_archive']['open_handle_rows'])+len(rec['lsof_before_delete']['open_handle_rows'])+len(rm['lsof_before_delete']['open_handle_rows'])})
# Include all files except the manifest and its seal, which bind them.
files=[]
for base,dirs,names in os.walk(HERE,topdown=True,followlinks=False):
 for d in list(dirs):
  p=Path(base)/d
  if p.is_symlink(): raise SystemExit(f'symlink in package: {p}')
 for name in names:
  p=Path(base)/name
  if p.name in {'package-manifest.json','package-seal.json'}: continue
  if p.is_symlink() or not p.is_file(): raise SystemExit(f'unsupported package member: {p}')
  st=p.stat()
  files.append({'path':p.relative_to(HERE).as_posix(),'bytes':st.st_size,'mode':stat.S_IMODE(st.st_mode),'sha256':sha(p)})
files.sort(key=lambda x:x['path'])
manifest={'schema':'numi.cardboard.closed-build-intermediate-compaction-package.v1',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_root':str(R),'build_suffix_range':[23,153],'explicit_build_exclusions':['build-cardboard-provenance-027'],
 'other_exclusions':['suffixes above 153','current build','source/build outputs outside exact CMakeFiles trees','55 preexisting pointer-only roots','empty CMakeFiles roots'],
 'summary':{k:aud[k] for k in ('eligible_targets','verified_archives','total_files','total_directories','total_allocated_file_bytes','total_logical_file_bytes','total_archive_bytes','sum_immediate_per_tree_df_changes_bytes','endpoint_df_delta_bytes')},
 'preflight_df':aud['df_preflight'],'final_audit_df':aud['df_final'],'final_df_at_seal':df(),
 'targets':targets,'package_files':files,
 'recovery_policy':'All archive and compressed inventory files are retained; each source target contains only a pointer with exact archive and inventory hashes. Restore command is documented in README.md and guarded against live build/open handles.'}
man_sha=atomic(HERE/'package-manifest.json',manifest)
seal={'schema':'numi.cardboard.closed-build-intermediate-compaction-seal.v1',
 'package_manifest_sha256':man_sha,'package_file_count_excluding_manifest_and_seal':len(files),
 'target_count':len(targets),'archive_count':sum(1 for x in files if x['path'].startswith('archives/')),
 'inventory_count':sum(1 for x in files if x['path'].startswith('inventories/')),
 'final_df_before_seal_write':df(),'2GiB_floor_reached':df()['free_bytes']>=2*1024**3,
 'package_manifest_path':str(HERE/'package-manifest.json')}
seal_sha=atomic(HERE/'package-seal.json',seal)
# Replace the seal once with a post-write df reading and self-hash, while keeping the manifest immutable.
seal['final_df_after_package_write']=df(); seal['package_seal_sha256_excluding_self']='not_self_referential'
seal_sha=atomic(HERE/'package-seal.json',seal)
print(json.dumps({'manifest_sha256':man_sha,'seal_sha256':seal_sha,'targets':len(targets),'package_files':len(files),'final_df':seal['final_df_after_package_write'],'2GiB_floor_reached':seal['2GiB_floor_reached']},sort_keys=True))
