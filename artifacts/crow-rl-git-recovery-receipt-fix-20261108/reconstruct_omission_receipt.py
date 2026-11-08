"""Recover the missing receipt after omission removed its empty parent."""
from pathlib import Path
import importlib.util,json,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-git-recovery-object-continuation-failure-20261108';base.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
head=a.git('rev-parse','HEAD').decode().strip()
remote=a.git('ls-remote','origin','refs/heads/agent/american-crow-rl-20261107').decode().split()[0]
if head!=remote:raise RuntimeError('Unexpected archival remote state')
entries=a.tree();states={}
for item in a.git('ls-files','-t','-z','--','artifacts/crow-rl-object-continuation-failure-20261108').split(b'\0'):
 if item:states[item[2:].decode()]=item[:1].decode()
files=[]
with a.Objects() as objects:
 for name,(mode,oid) in entries.items():
  if name.startswith('artifacts/crow-rl-object-continuation-failure-20261108/') and not Path(name).exists():
   if states[name]!='S':raise RuntimeError('Missing archive path without skip-worktree: '+name)
   data=objects.read(oid)
   files.append({'path':name,'git_blob':oid,'bytes':len(data),'mode':mode,'content_sha256':a.digest(data),'observed':'absent_with_skip_worktree'})
report={'schema':'numi.reconstructed-checkout-omission-receipt.v1','commit':head,'remote_commit':remote,
 'status':'checkout_omitted_receipt_reconstructed','omitted_files':files,'omitted_logical_bytes':sum(f['bytes'] for f in files),
 'publication_failure':{'type':'FileNotFoundError','message':'Receipt parent was removed by the same empty-directory cleanup before final receipt writing.',
  'requested_receipt':'artifacts/crow-rl-git-recovery-object-continuation-failure-20261108/checkout-omission-receipt.json',
  'operation_stage':'All tracked-file content/open-handle checks, skip-worktree updates and unlinks had completed; failure occurred in final write_json.',
  'original_tool_git_blob':entries['tools/crow_evidence_archive.py'][1]},
 'verification':'Read every missing current-archive content blob from actual committed Git; confirmed each missing path has skip-worktree. No additional deletion or scientific execution.',
 'df_before_unavailable':True,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True),
 'limit':'Reconstructed from committed objects/index and observed absence after failure; no original pre-deletion receipt or immediate df_before survived.'}
a.write_json(base/'checkout-omission-receipt.json',report)
print(json.dumps({'status':report['status'],'omitted_files':len(files),'omitted_bytes':report['omitted_logical_bytes']}))
