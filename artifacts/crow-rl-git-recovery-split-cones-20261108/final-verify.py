from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys
R=Path.cwd();ar='artifacts/crow-rl-split-cones-20261108';gr='artifacts/crow-rl-git-recovery-split-cones-20261108';A=R/ar;G=R/gr
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(x):return hashlib.sha256(x).hexdigest()
entries=archive.tree()
with archive.Objects() as objects:
 def value(path):return json.loads(objects.read(entries[path][1]))
 manifest=value(ar+'/payload/manifest.json');originals={f['path']:f for f in manifest['files']}
 for f in manifest['files']:
  raw=b''.join(archive.content(f,entries,objects));assert sha(raw)==f['sha256'] and len(raw)==f['bytes']
  assert not (R/f['path']).exists()
 temp=value(gr+'/temporary-inventory.json');assert not (R/temp['root']).exists()
 for f in temp['files']:
  b=f['backing'];raw=b''.join(archive.content(originals[b['original']],entries,objects)) if 'original' in b else gzip.decompress(objects.read(entries[b['gzip']][1]))
  assert sha(raw)==f['sha256'] and len(raw)==f['bytes']
 prod=value(ar+'/production-seal.json')
 for p,s in prod['files'].items():assert sha((R/p).read_bytes())==s['sha256'] and (R/p).stat().st_size==s['bytes']
 common=subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip();assert common==prod['shared_provider']
 assert subprocess.check_output(['git','diff',prod['source_baseline'],'--','src','apps','include','docs','python','environments'])==b''
 recovery=value(gr+'/result.json');assert recovery['all_passed'] and recovery['native_binary_outputs_exact']==72 and recovery['native_commands']==36
 omission=value(gr+'/omission.json');assert omission['status']=='omitted' and all(not (R/f['path']).exists() for f in omission['omitted_files'])
 for f in omission['omitted_files']:assert entries[f['path']][1]==f['git_blob'];objects.read(f['git_blob'])
branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],text=True).split()[0];assert head==remote
df=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
result=dict(schema='crow-split-cones-final-verification-v1',verified_commit=head,remote_commit=remote,raw_files_verified_absent=len(manifest['files']),
 raw_bytes_archived=manifest['summary']['source_bytes'],new_compressed_bytes=manifest['summary']['new_compressed_bytes'],temporary_files_git_verified_absent=len(temp['files']),
 temporary_bytes_removed=sum(f['bytes'] for f in temp['files']),native_recovery_commands=36,native_binary_outputs_exact=72,
 production_source_runtime_policy_unchanged=True,shared_git_provider=common,shared_git_provider_protected=True,
 duplicate_checkout_files_omitted=len(omission['omitted_files']),duplicate_checkout_bytes_omitted=omission['omitted_logical_bytes'],
 free_gib=int(df.splitlines()[1].split()[3])/1024**2,all_passed=True,goal_complete=False)
(A/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
