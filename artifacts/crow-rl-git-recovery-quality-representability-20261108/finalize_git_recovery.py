from pathlib import Path
import gzip,hashlib,json,stat,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-quality-representability-20261108';X=R/'build/crow-quality-representability-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read(name):return objects.read(entries[str(G.relative_to(R)/name)][1])
 plan=json.loads(read('plan.json'));raw=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);sources={f['path']:f for f in m['files']}
 inventory=json.loads(read('temporary-inventory.json'));commands=json.loads(read('execution.json'));checks=json.loads(read('comparisons.json'));rows=json.loads(read('physical-row-comparison.json'))
 assert len(commands)==23 and all(c['returncode']==c['expected_exit_code'] for c in commands)
 assert len(checks)>=80 and all(c.get('byte_exact',c.get('result_fields_exact',False)) for c in checks)
 assert sum(c['physical_records_exact'] for c in rows)==624
 unique=set()
 for f in inventory:
  key=f['sha256'],f['bytes']
  if key not in unique:
   data=b''.join(content(sources[f['archive_source']],entries,objects)) if 'archive_source' in f else gzip.decompress(objects.read(entries[f['recovery_blob']][1]))
   assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];unique.add(key)
  p=R/f['path'];s=p.lstat();assert not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_size==f['bytes'] and stat.S_IMODE(s.st_mode)==f['mode'] and s.st_mtime_ns==f['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 assert {str(p.relative_to(R)) for p in X.rglob('*') if p.is_file()}=={f['path'] for f in inventory}
old=subprocess.run(['ps','-p','50236','-o','pid=,command='],capture_output=True,text=True);assert old.returncode==1 and not old.stdout
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(X)],capture_output=True,text=True,timeout=60)
write(G/'temporary-open-handles-recheck.json',dict(exit_code=handles.returncode,stdout=handles.stdout,stderr=handles.stderr,original_pid=50236,original_pid_present=False))
assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
report=dict(manifest=plan['manifest'],manifest_sha256=plan['manifest_sha256'],restored_files=len(m['files']),restored_bytes=m['summary']['source_bytes'],executed_commands=23,native_commands=20,analysis_commands=3,comparison_records=len(checks),all_comparisons_passed=True,physical_rows_exact=624,temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),new_recovery_blobs=len(list((G/'recovery-blobs').glob('*.gz'))),actual_git_recovery_unique_contents_verified=len(unique),open_handles=0,initial_handle_check_retained='temporary-open-handles.json',policy_or_profile_promoted=False,goal_complete=False,limits='Executed recovery repeats existing observations. Closure was initially blocked by observed handles; separate exact inventory and Git verification passed after that process exited. No compiler rebuild or new ecological qualification.')
write(G/'verification.json',report);print(json.dumps(report),flush=True)
