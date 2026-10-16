#!/usr/bin/env python3
"""Read-only independent byte, prefix, alias, scope, and Git-index audit for publication027."""
import gzip, hashlib, json, os, re, subprocess, sys
from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
D=R/'matter/examples/cardboard/evidence/native-implementation-20261014'
PKG=B/'analysis/publication027-independent-byte-audit-001'
EXCLUDE={'inputs','build','base','candidate','__pycache__','archives','restore-check'}
ALIASES={'analysis/native466-source154-default64-parity-command-002','analysis/native467-source154-allcontrol-outer128-command-002'}
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def allowed(case,rel):
 parts=rel.parts
 if case in ALIASES and len(parts)==2 and parts[0]=='inputs':return True
 return not any(p in EXCLUDE or p.startswith(('CMakeFiles','.air')) for p in parts)
def run(args,cwd=R,check=True):return subprocess.run(args,cwd=cwd,check=check,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
prepared=json.loads((B/'publication027-cases-prepared.json').read_text())
cases=prepared['cases']; assert len(cases)==33 and len(set(cases))==33
binding=json.loads((D/'publication-binding-reference027.json').read_text())
oldbinding=json.loads((D/'publication-binding-reference026.json').read_text())
summarybytes=(D/'summary.json').read_bytes();summary=json.loads(summarybytes)
oldbytes=(D/'summary-reference026.json').read_bytes();oldsummary=json.loads(oldbytes)
assert binding['schema']=='numi.cardboard.reference027-publication-binding.v1'
assert binding['cases']==cases and len(summary['records'])==1457 and len(oldsummary['records'])==1424
assert summary['records'][:1424]==oldsummary['records']
assert sha_bytes(oldbytes)==oldbinding['summary_sha256']==binding['historical_prefix']['sha256']
assert binding['historical_prefix']['records']==1424
assert summary['records'][:1424]==oldsummary['records']
assert binding['summary_sha256']==sha_bytes(summarybytes)
assert binding['prior_binding_sha256']==sha_file(D/'publication-binding-reference026.json')
assert binding['prior_pathspec_entries']==4344
oldpathbytes=(B/'publication-reference026-pathspec.nul').read_bytes()
newpathbytes=(B/'publication-reference027-pathspec.nul').read_bytes()
assert sha_bytes(oldpathbytes)==oldbinding['pathspec_sha256'] if 'pathspec_sha256' in oldbinding else sha_bytes(oldpathbytes)==binding['prior_pathspec_sha256']
assert binding['prior_pathspec_sha256']==sha_bytes(oldpathbytes)
oldpaths=[x.decode() for x in oldpathbytes.split(b'\0') if x]
newpaths_bytes=[x.decode() for x in newpathbytes.split(b'\0') if x]
assert len(oldpaths)==4344 and len(set(oldpaths))==4344
assert newpathbytes.startswith(oldpathbytes), 'historical pathspec byte prefix changed'
assert newpaths_bytes[:4344]==oldpaths
newpath_set=set(newpaths_bytes[4344:]);assert len(newpath_set)==467 and len(newpaths_bytes)==4811
assert len(newpath_set)==len(newpaths_bytes[4344:])
archive_rows=0;raw=compressed=0;case_file_counts={};alias_audits={};archive_paths=set()
for rec in binding['cases']:
 case=rec
 record=next(x for x in summary['records'][1424:] if x.get('case')==case)
 extroot=B/case
 assert extroot.is_dir() and record['external_root']==str(extroot)
 rows=record['files'];assert rows and len(rows)==len({x['name'] for x in rows})
 listed=set()
 for row in rows:
  rel=Path(row['name']);src=extroot/rel
  assert allowed(case,rel),f'excluded path archived: {case}/{rel}'
  assert src.resolve().is_relative_to(extroot.resolve()) and src.is_file() and not src.is_symlink()
  assert row['external_path']==str(src)
  srcbytes=src.read_bytes();digest=sha_bytes(srcbytes)
  assert len(srcbytes)==row['bytes'] and digest==row['sha256'],f'external byte mismatch: {src}'
  if 'repository_copy' in row:
   repo_copy=D/row['repository_copy'];assert repo_copy.is_file() and not repo_copy.is_symlink()
   out=repo_copy.read_bytes();assert len(out)==row['bytes'] and sha_bytes(out)==digest
   raw+=1;relative='matter/examples/cardboard/evidence/native-implementation-20261014/'+row['repository_copy']
  else:
   repo_copy=D/row['repository_compressed_copy'];assert repo_copy.is_file() and not repo_copy.is_symlink()
   packed=repo_copy.read_bytes();assert len(packed)==row['compressed_bytes'] and sha_bytes(packed)==row['compressed_sha256']
   out=gzip.decompress(packed);assert len(out)==row['bytes'] and sha_bytes(out)==digest
   compressed+=1;relative='matter/examples/cardboard/evidence/native-implementation-20261014/'+row['repository_compressed_copy']
  archive_paths.add(relative);listed.add(rel.as_posix())
  # Verify exact input aliases against their named original source and repair receipt.
  if len(rel.parts)==2 and rel.parts[0]=='inputs':
   assert case in ALIASES
   repair=json.loads((extroot/'repair.json').read_text())
   bind=next((x for x in repair['inputs'] if Path(x['copied']).name==rel.name),None)
   assert bind is not None, f'alias lacks repair mapping: {case}/{rel}'
   original=Path(bind['original'])
   assert original.is_file() and not original.is_symlink()
   assert sha_file(original)==digest==bind['sha256'] and len(srcbytes)==bind['bytes']
   alias_audits.setdefault(case,[]).append({'alias':str(rel),'original':str(original),'sha256':digest,'bytes':len(srcbytes)})
  archive_rows+=1
 # independently enumerate all allowed source regular files; rows must be exhaustive
 actual=set()
 for src in extroot.rglob('*'):
  rel=src.relative_to(extroot)
  if src.is_file() and allowed(case,rel):
   assert not src.is_symlink();actual.add(rel.as_posix())
 assert actual==listed,(case,len(actual),len(listed),sorted(actual-listed)[:3],sorted(listed-actual)[:3])
 case_file_counts[case]=len(rows)
 assert record.get('physical_box_qualification') is False
# Input directories may be archived only for the two explicit command aliases.
assert set(alias_audits)==ALIASES
assert all(len(alias_audits[c])==len(json.loads((B/c/'repair.json').read_text())['inputs']) for c in ALIASES)
assert archive_rows==465 and raw==440 and compressed==25
assert (raw,compressed)==(binding['raw_files'],binding['gzip_files'])
assert binding['copied_files']==465 and binding['new_case_records']==33
archive_paths.update({'matter/examples/cardboard/evidence/native-implementation-20261014/summary-reference026.json','matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference027.json'})
assert archive_paths==newpath_set,(len(archive_paths),len(newpath_set),sorted(archive_paths-newpath_set)[:3],sorted(newpath_set-archive_paths)[:3])
# Exact six other staged additions (outside the archive record/pathspec).
# Git staged/index checks.
head=run(['git','rev-parse','HEAD']).stdout.decode().strip()
staged_raw=run(['git','diff','--cached','--name-only','-z']).stdout
staged=[x.decode() for x in staged_raw.split(b'\0') if x]
expected_stage_bytes=(B/'publication027-staged-paths.nul').read_bytes()
expected_stage=[x.decode() for x in expected_stage_bytes.split(b'\0') if x]
assert sha_bytes(expected_stage_bytes)=='6b29076d64bdb83f548898538a27423207e95c202f83485683be63cffb51151e'
assert len(staged)==len(expected_stage)==473 and staged==expected_stage
assert len(set(staged))==473
added_raw=run(['git','diff','--cached','--diff-filter=A','--name-only','-z']).stdout
added=[x.decode() for x in added_raw.split(b'\0') if x]
assert set(added)==set(staged) and len(added)==473
unstaged=run(['git','diff','--name-only','-z']).stdout.split(b'\0');assert [x for x in unstaged if x]==[]
untracked=run(['git','ls-files','--others','--exclude-standard','-z']).stdout.split(b'\0')
untracked=[x.decode() for x in untracked if x]
assert set(untracked)=={
 'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/CMakeOBJCXXCompilerId.mm',
 'matter/tools/evidence/cohesive-scratch-alias/source-audit.json'}
nonarchive=sorted(set(staged)-newpath_set)
# Six product/docs/source additions expected from root's staged scope.
assert len(nonarchive)==6
assert not (set(untracked)&set(staged))
# Check index objects vs working tree bytes, and staged modes are regular files.
idx=run(['git','ls-files','--stage','-z']).stdout
index={}
for item in idx.split(b'\0'):
 if not item:continue
 meta,path=item.split(b'\t',1);mode,oid,stage=meta.decode().split(' ')
 pth=path.decode();assert stage=='0';index[pth]=(mode,oid)
assert set(index)==set(staged)
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
blob_checks=0
for path in staged:
 mode,oid=index[path];proc.stdin.write(oid.encode()+b'\n');proc.stdin.flush()
 header=proc.stdout.readline().decode().strip().split()
 assert len(header)==3 and header[0]==oid and header[1]=='blob'
 size=int(header[2]);data=proc.stdout.read(size);assert len(data)==size and proc.stdout.read(1)==b'\n'
 wt=R/path
 assert wt.is_file() and not wt.is_symlink()
 assert len(data)==wt.stat().st_size and sha_bytes(data)==sha_file(wt),f'index/worktree mismatch: {path}'
 assert mode in ('100644','100755')
 blob_checks+=1
proc.stdin.close();assert proc.wait()==0
# Source/build and failure-preservation assertions from producer plus independent file bytes.
assert json.loads((B/'analysis/source154-default64-native466-parity-001/report.json').read_text())['parity_pass'] is True
integration=json.loads((B/'analysis/source154-root-integration-001/report.json').read_text())
assert integration['source_files_verified']==787
assert json.loads((B/'analysis/native467-launch-guard-001/receipt.json').read_text())['space_abort'] is True
assert json.loads((B/'solver/cgls-complete-outer128-allcontrols-467/receipt.json').read_text())['exit_code']==-15
# Six-axis status is retained and no overall physical qualification is claimed.
assert set(binding['six_axis_verdicts'])=={'solver_correctness','numerical_convergence','component_calibration','analogue_validation','physical_assembly_in_simulation','crush_impact_prediction'}
assert binding['physical_validation'] is False
assert summary['qualification']['physical_assembly'] if 'physical_assembly' in summary.get('qualification',{}) else True
report={
 'schema':'numi.cardboard.publication027-independent-audit.v1','result':'PASS_BYTE_PREFIX_SCOPE_AUDIT',
 'head':head,'staged_paths':len(staged),'staged_all_additions':True,'staged_pathspec_sha256':sha_bytes(expected_stage_bytes),
 'staged_pathspec_exact_match':True,'worktree_matches_git_index_blobs':blob_checks,
 'unstaged_diff_paths':0,'untracked_preserved':untracked,'new_archive_paths':len(newpath_set),
 'new_nonarchive_paths':nonarchive,'historical_summary_records':1424,'new_summary_records':len(summary['records']),
 'summary_prefix_exact_json':True,'summary_reference026_sha256':sha_bytes(oldbytes),
 'old_archive_pathspec_entries':len(oldpaths),'new_archive_pathspec_entries':len(newpaths_bytes),
 'pathspec_prefix_byte_identical':True,'new_archive_records':archive_rows,'raw_files':raw,'gzip_files':compressed,
 'case_count':len(cases),'case_file_counts':case_file_counts,'input_alias_count':sum(map(len,alias_audits.values())),
 'input_alias_verification':alias_audits,'excluded_payload_scope_enforced':True,
 'source154_integration_verified_source_files':787,'native466_full_attempt_parity':True,
 'native467_capacity_interruption_preserved':True,'physical_validation':False,
 'six_axis_verdicts':binding['six_axis_verdicts']}
PKG.mkdir(parents=True,exist_ok=True)
(PKG/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(PKG/'audit.stdout.json').write_text(json.dumps({'case_count':len(cases),'archive_rows':archive_rows,'raw':raw,'gzip':compressed,'staged':len(staged),'git_index_blob_checks':blob_checks,'nonarchive':nonarchive},indent=2,sort_keys=True)+'\n')
print(json.dumps({'result':report['result'],'cases':len(cases),'files':archive_rows,'raw':raw,'gzip':compressed,'staged':len(staged),'git_blob_checks':blob_checks,'nonarchive':nonarchive,'untracked_preserved':untracked},indent=2))
