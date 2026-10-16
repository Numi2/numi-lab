#!/usr/bin/env python3
import gzip, hashlib, json, os, subprocess
from pathlib import Path

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
D=R/'matter/examples/cardboard/evidence/native-implementation-20261014'
OUT=B/'analysis/publication026-independent-byte-audit-001'
sha=lambda b:hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def git_bytes(*args):
 return subprocess.check_output(['git','-C',str(R),*args])
def strict_load(p):
 def bad(x): raise ValueError('nonstandard JSON constant: '+x)
 return json.loads(Path(p).read_text(),parse_constant=bad)

bind_path=D/'publication-binding-reference026.json'
bind=strict_load(bind_path)
prepared=strict_load(B/'publication026-cases-prepared.json')
summary_bytes=(D/'summary.json').read_bytes()
summary=json.loads(summary_bytes)
prior_bytes=(D/'summary-reference025.json').read_bytes()
prior=json.loads(prior_bytes)
path026=(B/'publication-reference026-pathspec.nul').read_bytes()
path025=(B/'publication-reference025-pathspec.nul').read_bytes()
prior_bind_bytes=(D/'publication-binding-reference025.json').read_bytes()
prior_bind=json.loads(prior_bind_bytes)

assert sha(bind_path.read_bytes())=='eee5d5ad5c72eaaf0a5ab2a44318da14993ff9778f48727828a5b859a2e39390'
assert bind['schema']=='numi.cardboard.reference026-publication-binding.v1'
assert bind['prior_binding_sha256']==sha(prior_bind_bytes)
assert bind['prior_pathspec_sha256']==sha(path025)
assert bind['prior_pathspec_entries']==4070
assert bind['historical_prefix']['sha256']==sha(prior_bytes)
assert bind['historical_prefix']['records']==1398
assert len(prior['records'])==1398
assert summary['records'][:1398]==prior['records']
assert bind['summary_sha256']==sha(summary_bytes)
assert len(summary['records'])==1424
assert len(summary['records'][:1398])==1398
assert len(summary['records'][1398:])==26
assert len(prior_bytes)==16784254
# The committed HEAD is the prior publication. Verify its reference and summary blobs.
head=git_bytes('rev-parse','HEAD').decode().strip()
head_summary=git_bytes('show',f'HEAD:matter/examples/cardboard/evidence/native-implementation-20261014/summary.json')
head_prior_binding=git_bytes('show',f'HEAD:matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference025.json')
assert sha(head_summary)==sha(prior_bytes)
assert sha(head_prior_binding)==sha(prior_bind_bytes)
assert len(prepared['cases'])==26 and len(set(prepared['cases']))==26
assert prepared['cases']==bind['cases']
assert len(set(r['case'] for r in summary['records'][1398:]))==26
assert set(r['case'] for r in summary['records'][1398:])==set(prepared['cases'])
assert bind['new_case_records']==26 and bind['copied_files']==272
assert bind['raw_files']==247 and bind['gzip_files']==25
assert bind['all_archived_bytes_verified'] is True and bind['physical_validation'] is False
assert all(v['status'] not in ('passed','qualified','complete') for v in bind['six_axis_verdicts'].values())
assert bind['six_axis_verdicts']['numerical_convergence']['status']=='unqualified'

# Exact old pathspec prefix bytes must remain unchanged; appended records are the only delta.
assert path026.startswith(path025)
entries=[x.decode('utf-8') for x in path026.split(b'\0') if x]
oldentries=[x.decode('utf-8') for x in path025.split(b'\0') if x]
assert len(oldentries)==4070 and len(set(oldentries))==4070
assert len(entries)==4344 and len(set(entries))==4344
assert entries[:4070]==oldentries

case_rows=summary['records'][1398:]
raw_count=gzip_count=files_count=0
new_repo_paths=set()
allowed_excluded={'inputs','build','base','candidate','__pycache__','archives','restore-check'}
def allowed(rel):
 return not any(part in allowed_excluded or part.startswith(('CMakeFiles','.air')) for part in rel.parts)
def stream_sha(path, decompressed=False):
 h=hashlib.sha256(); n=0
 opener=gzip.open if decompressed else open
 with opener(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):
   n+=len(block); h.update(block)
 return n,h.hexdigest()
for rec in case_rows:
 case=rec['case']; extroot=B/case
 assert rec['external_root']==str(extroot) and extroot.is_dir()
 listed={}
 for row in rec['files']:
  rel=Path(row['name'])
  assert not rel.is_absolute() and '..' not in rel.parts and allowed(rel)
  assert row['name'] not in listed
  listed[row['name']]=row
  src=extroot/rel
  assert row['external_path']==str(src)
  assert src.is_file() and not src.is_symlink()
  n,digest=stream_sha(src)
  assert n==row['bytes'] and digest==row['sha256'], (case,row['name'],'external')
  if 'repository_copy' in row:
   raw_count+=1
   dst=D/row['repository_copy']
   assert dst.is_file() and not dst.is_symlink()
   rn,rd=stream_sha(dst)
   assert rn==n and rd==digest, (case,row['name'],'raw archive')
   assert row['repository_copy']==str((Path(case)/rel).as_posix())
   new_repo_paths.add('matter/examples/cardboard/evidence/native-implementation-20261014/'+row['repository_copy'])
  else:
   gzip_count+=1
   dst=D/row['repository_compressed_copy']
   assert dst.is_file() and not dst.is_symlink()
   cn,cd=stream_sha(dst)
   assert cn==row['compressed_bytes'] and cd==row['compressed_sha256'], (case,row['name'],'compressed archive')
   un,ud=stream_sha(dst,decompressed=True)
   assert un==n and ud==digest, (case,row['name'],'decompressed archive')
   assert row.get('restore') and 'gzip' in row['restore']
   assert row['repository_compressed_copy']==str((Path(case)/rel).as_posix()+'.gz')
   new_repo_paths.add('matter/examples/cardboard/evidence/native-implementation-20261014/'+row['repository_compressed_copy'])
  files_count+=1
 # Recompute eligible source inventory, catching omitted/unexpected files.
 eligible=set()
 for src in extroot.rglob('*'):
  rel=src.relative_to(extroot)
  if src.is_file() and not src.is_symlink() and allowed(rel): eligible.add(rel.as_posix())
 assert set(listed)==eligible, (case,'source inventory differs',len(listed),len(eligible))

assert files_count==272 and raw_count==247 and gzip_count==25
assert path026.endswith(b'\0')
new_path_tail=set(entries[4070:])
expected_tail=set(new_repo_paths)|{
 'matter/examples/cardboard/evidence/native-implementation-20261014/summary-reference025.json',
 'matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference026.json'}
assert new_path_tail==expected_tail
assert len(new_path_tail)==274

# Staged archive entries must equal newly archived paths; preserve the explicitly intended code/docs.
staged=[x.decode() for x in git_bytes('diff','--cached','--name-only','-z').split(b'\0') if x]
staged_set=set(staged)
assert expected_tail.issubset(staged_set)
intentional_extra={
 'matter/examples/cardboard/evidence/native-implementation-20261014/summary.json',
 'matter/examples/cardboard/implementation-20261014.md',
 'matter/examples/cardboard/solver-progress-20261014.md',
 'matter/examples/cardboard/solver-milestone-source149.md',
 'matter/src/metal/base_cgls.metalinc',
 'matter/src/metal/fgmres.metalinc',
 'matter/src/runtime.mm',
}
assert staged_set==expected_tail|intentional_extra, {'unexpected_staged':sorted(staged_set-(expected_tail|intentional_extra)), 'missing':sorted((expected_tail|intentional_extra)-staged_set)}
# Verify exact staged paths for two known unrelated untracked trees are absent.
status=git_bytes('status','--porcelain','--untracked-files=normal','-z').split(b'\0')
untracked=[]
for row in status:
 if row and row[:2]==b'??': untracked.append(row[3:].decode())
assert 'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/' in untracked
assert 'matter/tools/evidence/cohesive-scratch-alias/' in untracked
assert not any(p in expected_tail or p in staged_set for p in untracked)
assert not any('CompilerIdOBJCXX' in p or 'cohesive-scratch-alias' in p for p in entries)
process_rows=subprocess.check_output(['ps','-axo','pid=,ppid=,comm=,args='],text=True).splitlines()
case_markers=['cgls-right-scale-kernel-check-460','cgls-block-right-disabled-parity-462',
              'cgls-block-right-primary48-461','cgls-block-right-primary48-463']
active_case_processes=[line.strip() for line in process_rows if any(m in line for m in case_markers)]
assert not active_case_processes, active_case_processes

report={
 'schema':'numi.cardboard.publication026-independent-byte-audit.v1',
 'verdict':'PASS_BYTE_AND_PREFIX_AUDIT; physical qualification not evaluated',
 'repo':str(R),'head':head,'evidence_root':str(D),
 'binding_sha256':sha(bind_path.read_bytes()),
 'prior_binding_sha256':sha(prior_bind_bytes),
 'prepared_cases_sha256':sha((B/'publication026-cases-prepared.json').read_bytes()),
 'archive_tool_sha256':sha((B/'archive_reference026.py').read_bytes()),
 'pathspec_sha256':sha(path026),'prior_pathspec_sha256':sha(path025),
 'summary_sha256':sha(summary_bytes),'historical_summary_sha256':sha(prior_bytes),
 'summary_record_count':len(summary['records']),'prior_prefix_records':1398,'prior_prefix_exact_json_objects':True,
 'pathspec_entry_count':len(entries),'prior_pathspec_entry_count':len(oldentries),'prior_pathspec_exact_byte_prefix':True,
 'new_case_count':len(case_rows),'file_count':files_count,'raw_file_count':raw_count,'gzip_file_count':gzip_count,
 'external_source_bytes_and_hashes_match':True,'raw_repository_copies_match':True,'compressed_copies_and_restores_match':True,
 'eligible_external_file_inventories_complete':True,'new_pathspec_tail_exact':True,
 'staged_expected_paths_only':True,'unrelated_untracked_not_staged':sorted(untracked),
 'selected_149_460_463_case_processes_active':active_case_processes,
 'six_axis_verdicts_preserved_unqualified':True,
 'qualification_boundary':'This report verifies byte provenance, archive contents, and historical prefix preservation only. It makes no solver, physical, or assembly qualification claim.'
}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
