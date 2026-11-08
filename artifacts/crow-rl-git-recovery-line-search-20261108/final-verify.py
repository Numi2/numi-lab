from pathlib import Path
import gzip,hashlib,json,subprocess,sys
R=Path.cwd();ar='artifacts/crow-rl-line-search-refinement-20261108';wr='artifacts/crow-rl-line-search-witness-20261108';gr='artifacts/crow-rl-git-recovery-line-search-20261108'
A=R/ar;G=R/gr;sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(raw):return hashlib.sha256(raw).hexdigest()
entries=archive.tree()
with archive.Objects() as objects:
 def value(path):return json.loads(objects.read(entries[path][1]))
 m=value(ar+'/payload/manifest.json');originals={f['path']:f for f in m['files']}
 for f in m['files']:
  raw=b''.join(archive.content(f,entries,objects));assert (sha(raw),len(raw))==(f['sha256'],f['bytes']);assert not (R/f['path']).exists()
 temp=value(gr+'/temporary-inventory.json');assert not (R/temp['root']).exists()
 for root in ['build/crow-line-search-witness-build-01','build/crow-line-search-refinement-build-01']:assert not (R/root).exists()
 for f in temp['files']:
  b=f['backing'];raw=b''.join(archive.content(originals[b['original']],entries,objects)) if 'original' in b else gzip.decompress(objects.read(entries[b['gzip']][1]))
  assert (sha(raw),len(raw))==(f['sha256'],f['bytes'])
 prod=value(ar+'/production-final-seal.json')
 for path,f in prod['files'].items():assert (sha((R/path).read_bytes()),(R/path).stat().st_size)==(f['sha256'],f['bytes'])
 common=subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip();assert common==prod['shared_provider']=='/Users/n/MetalRobo-main/.git'
 changed=set(subprocess.check_output(['git','diff','--name-only',prod['base_commit'],'--','CMakeLists.txt','src','include','docs','apps','python','environments'],text=True).splitlines())
 assert changed==set(prod['production_patch_paths'])
 assert sha((R/'environments/american-crow/action-holds-v1/policies/deployment97.policypack').read_bytes())==prod['policy_sha256']=='6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389'
 recovery=value(gr+'/result.json');assert recovery['all_passed'] and recovery['native_commands']==16 and recovery['motor_physical_records_exact']==624 and recovery['unique_candidate_cases_independently_audited']==6
 omission=value(gr+'/omission.json');assert omission['status']=='omitted'
 for f in omission['omitted_files']:assert not (R/f['path']).exists() and entries[f['path']][1]==f['git_blob'];objects.read(f['git_blob'])
 q=value(ar+'/qualification-result.json');component=value(ar+'/component-result.json');w=value(wr+'/result.json');bounded=value(ar+'/integration-bounded-result.json')
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+archive.BRANCH],text=True).split()[0];assert head==remote
df=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
report=dict(schema='crow-line-search-final-verification-v1',verified_commit=head,remote_commit=remote,
 raw_files_verified_absent=len(m['files']),raw_bytes_archived=m['summary']['source_bytes'],new_compressed_bytes=m['summary']['new_compressed_bytes'],
 temporary_files_git_verified_absent=len(temp['files']),temporary_bytes_removed=sum(f['bytes'] for f in temp['files']),
 native_recovery_commands=16,physical_and_numerical_files_exact=recovery['physical_and_numerical_files_exact'],motor_physical_records_exact=624,unique_retained_operator_audits=6,
 production_source_and_runtime_match_observed_candidate=True,production_patch_paths=sorted(changed),protected_policy_unchanged=True,policy_or_profile_promoted=False,
 observer_does_not_change_archived_physical_outputs=w['observations']['archived_physical_files_exact_per_repeat']==12,
 retained_component_predictions_supported=component['all_predictions_supported'],default_crow_physical_files_exact=bounded['default_physical_files_exact'],
 coupled_physical_predictions_supported=q['all_predictions_supported'],coupled_observations=q['observations'],
 shared_git_provider=common,shared_git_provider_protected=True,duplicate_checkout_files_omitted=len(omission['omitted_files']),
 duplicate_checkout_bytes_omitted=omission['omitted_logical_bytes'],free_gib=int(df.splitlines()[1].split()[3])/1024**2,
 archive_checks_passed=True,performance_qualified=False,lifelong_environment_qualified=False,goal_complete=False)
(A/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
