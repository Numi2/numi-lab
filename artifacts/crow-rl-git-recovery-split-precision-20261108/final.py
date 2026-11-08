from pathlib import Path
import gzip,hashlib,json,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-split-precision-20261108';A=R/'artifacts/crow-rl-split-precision-20261108'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read(p):return objects.read(entries[str(p.relative_to(R))][1])
 plan=json.loads(read(G/'plan.json'));raw=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);sources={f['path']:f for f in m['files']};original=set()
 for f in m['files']:
  key=f['sha256'],f['bytes']
  if key not in original:
   raw=b''.join(content(f,entries,objects));assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'];original.add(key)
  assert not (R/f['path']).exists()
 inventory=json.loads(read(G/'temporary-inventory.json'));temporary=set()
 for f in inventory:
  key=f['sha256'],f['bytes']
  if key not in temporary:
   raw=b''.join(content(sources[f['archive_source']],entries,objects)) if 'archive_source' in f else gzip.decompress(objects.read(entries[f['recovery_blob']][1]))
   assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'];temporary.add(key)
  assert not (R/f['path']).exists()
 bindings=json.loads(read(A/'production-bindings.json'))
 for name,f in bindings['files'].items():
  p=R/name;assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 diff=subprocess.check_output(['git','diff','--name-only',bindings['baseline'],'HEAD','--','.',':(exclude)artifacts/**'],text=True);assert not diff
 status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src','include','docs','tools','python','environments'],text=True);assert not status
scopes=['build/crow-split-precision-build-01','build/crow-split-precision-git-recovery-01']
for s in scopes:assert not (R/s).exists()
workers=[line for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines() if any(line.strip().split(None,1)[-1].startswith(str(R/s)+'/') for s in scopes)];assert not workers
df=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
report=dict(verification_commit=head,remote_commit=remote,manifest_sha256=plan['manifest_sha256'],original_files_verified=len(m['files']),original_unique_contents_verified=len(original),recovery_files_verified=len(inventory),recovery_unique_contents_verified=len(temporary),current_runtime_and_policy_match_committed_baseline=True,production_changes=[],shared_git_provider=subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip(),raw_and_temporary_scopes_absent=scopes,own_native_workers=[],df=df,data_available_GiB=int(df.splitlines()[-1].split()[3])/1024**2,policy_promoted=False,goal_complete=False)
(G/'final-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
