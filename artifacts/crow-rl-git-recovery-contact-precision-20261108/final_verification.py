from pathlib import Path
import gzip,hashlib,json,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-contact-precision-20261108'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read(name):return objects.read(entries[str(G.relative_to(R)/name)][1])
 plan=json.loads(read('plan.json'));raw=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);sources={f['path']:f for f in m['files']};original=set()
 for f in m['files']:
  key=f['sha256'],f['bytes']
  if key not in original:
   data=b''.join(content(f,entries,objects));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];original.add(key)
  assert not (R/f['path']).exists()
 inventory=json.loads(read('temporary-inventory.json'));temporary=set()
 for f in inventory:
  key=f['sha256'],f['bytes']
  if key not in temporary:
   data=b''.join(content(sources[f['archive_source']],entries,objects)) if 'archive_source' in f else gzip.decompress(objects.read(entries[f['recovery_blob']][1]))
   assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];temporary.add(key)
  assert not (R/f['path']).exists()
 old=json.loads(objects.read(entries[plan['runtime_reference_manifest']][1]));old_sources={f['path']:f for f in old['files']};runtime={}
 for current,archived in plan['runtime_pairs']:
  f=old_sources[archived];data=b''.join(content(f,entries,objects));assert hashlib.sha256(data).hexdigest()==f['sha256'] and len(data)==f['bytes'];h=hashlib.sha256((R/current).read_bytes()).hexdigest();assert h==f['sha256'];runtime[current]=h
baseline=plan['production_baseline'];diff=subprocess.check_output(['git','diff','--name-only',baseline,'HEAD','--','.',':(exclude)artifacts/**'],text=True);assert not diff,diff
status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src','include','docs','tools','python','environments'],text=True);assert not status,status
policy=hashlib.sha256((R/'environments/american-crow/action-holds-v1/policies/deployment97.policypack').read_bytes()).hexdigest();assert policy=='6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389'
scopes=['build/crow-contact-precision-build-01','build/crow-contact-precision-git-recovery-01','.numi/retained-runtime/contact-precision-20261108']
for name in scopes:assert not (R/name).exists()
workers=[]
for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
 parts=line.strip().split(None,1)
 if len(parts)==2 and parts[1].startswith((str(R/'build')+'/',str(R/'.numi/retained-runtime')+'/')):workers.append(line)
assert not workers
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(R/'build')],capture_output=True,text=True,timeout=60);assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
df=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
report=dict(verification_commit=head,remote_commit=remote,manifest_sha256=plan['manifest_sha256'],actual_git_source_files_verified=len(m['files']),actual_git_source_unique_contents_verified=len(original),actual_git_recovery_files_verified=len(inventory),actual_git_recovery_unique_contents_verified=len(temporary),current_runtime_sha256=runtime,production_baseline=baseline,production_diff_outside_artifacts=[],protected_policy_sha256=policy,raw_and_temporary_scopes_absent=scopes,own_native_workers=[],build_open_handles=0,df=df,data_available_GiB=int(df.splitlines()[-1].split()[3])/1024**2,policy_or_profile_promoted=False,goal_complete=False)
(G/'final-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
