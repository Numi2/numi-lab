"""Reanalyze retained recovery once; explain process-local incumbent mismatch."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess
import mlx.core as mx,numpy as np
root=Path.cwd();base=root/'artifacts/crow-rl-ccd-sphere-box-20261108'
tmp=Path('/private/tmp/crow-ccd-sphere-box-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
metadata=json.loads((base/'runtime-recovery.json').read_text())
def write(name,value):
 with (base/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
write('verification-analysis-revision.json',{
 'native_execution_repeated':False,'original_prediction_retained':True,
 'reason':'Strict all-artifact prediction failed on incumbent.policypack. The worker intentionally writes a frozen pre-process baseline from its loaded learner. Fresh process starts at revision1; resumed process starts at checkpoint revision11. Incumbent is a separate evaluation artifact, not used by updates or deployment selection. Verify this exact expected mismatch and restore the original incumbent bytes separately from Git; do not label all canonical artifacts execution-exact.',
 'original_failure_sha256':a.file_hash(base/'verification-failure.json'),
 'original_plan_sha256':a.file_hash(base/'verification-plan.json'),
 'worker_source_sha256':a.file_hash(root/'python/metalrobo/mlx_policy_worker.py'),
 'new_reader_sha256':a.file_hash(Path(__file__)),
 'criteria':'Same exact native error,41model/Adam arrays and metadata,three canonical causal artifacts, all query/contact controls. Resumed incumbent must exactly equal source checkpoint candidate. Original incumbent separately restored/hash-checked from committed Git.'})
try:
 restoration=json.loads((tmp/'runtime-restoration.json').read_text())
 for f in restoration['files']:
  assert a.file_hash(tmp/f['path'])==f['sha256']
 for name in metadata['distance_references']:
  assert a.file_hash(tmp/('query-'+name+'.bin'))==metadata['distance_references'][name]['results_sha256']
 assert 'metal_world_contact=ok ' in (base/'generic-contact.stdout').read_text()
 assert json.loads((base/'generic-contact.receipt.json').read_text())['returncode']==0
 contact=json.loads((tmp/'contact/summary.json').read_text())
 assert contact['status']=='passed_engineering_controls' and contact['accepted_transitions']==7296
 contacts=[];path_identities={}
 for name,record in metadata['contact_references'].items():
  got=json.loads((tmp/'contact'/name/'summary.json').read_text());want=dict(record['summary']);path_identities[name]={}
  for k in ('world_sha256','retina_sha256'):path_identities[name][k]={'original':want.pop(k),'restored':got.pop(k)}
  assert got==want
  for n,f in record['files'].items():assert a.file_hash(tmp/'contact'/name/n)==f['sha256']
  contacts.append(name)
 receipt=json.loads((base/'failure-reproduction.receipt.json').read_text())
 assert receipt['returncode']==1
 assert a.file_hash(base/'failure-reproduction.stdout')==receipt['stdout_sha256']
 assert a.file_hash(base/'failure-reproduction.stderr')==receipt['stderr_sha256']
 assert not (base/'failure-reproduction.stdout').read_bytes()
 stderr=(base/'failure-reproduction.stderr').read_text()
 assert stderr.splitlines()[-1]==metadata['training_reference']['error_line']
 loaded=[s for s in stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 assert loaded and all(s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded)
 target=tmp/'training';assert not list((target/'checkpoints').glob('*'))
 tensors=0;canonical=[]
 for name,f in metadata['training_reference']['files'].items():
  if name=='learner.safetensors':
   x,xm=mx.load(target/name,return_metadata=True);y,ym=mx.load(tmp/'reference/training'/name,return_metadata=True)
   assert xm==ym and x.keys()==y.keys()
   for k in x:assert np.asarray(x[k]).tobytes()==np.asarray(y[k]).tobytes();tensors+=1
  elif name=='incumbent.policypack':
   assert a.file_hash(target/name)==a.file_hash(tmp/'input/source.trainingcheckpoint/candidate.policypack')
  else:
   assert a.file_hash(target/name)==f['sha256'];canonical.append(name)
 entries=a.tree()
 with a.Objects() as objects:
  manifest=json.loads(objects.read(entries[metadata['manifests'][0]][1]))
  f=next(f for f in manifest['files'] if f['path']=='.numi/runs/american-crow/object-ccd-learning-01/training/incumbent.policypack')
  path=tmp/'reference/training/incumbent.policypack'
  with path.open('xb') as stream:
   for block in a.content(f,entries,objects):stream.write(block)
  path.chmod(f['mode']);assert a.file_hash(path)==f['sha256']
 results={'generic_contact_suite_passed':True,'query_streams_exact':list(metadata['distance_references']),
  'contact_arms_physical_arrays_metrics_task_streams_exact':contacts,
  'contact_path_bound_native_checkpoint_identities':path_identities,
  'native_failure_line_exact':True,'error_line':metadata['training_reference']['error_line'],
  'canonical_partial_artifacts_execution_exact':canonical,'partial_model_Adam_arrays_exact':tensors,
  'partial_model_Adam_metadata_exact':True,'new_coupled_checkpoints':0,'behavioral_success':False,
  'incumbent_execution_exact':False,'incumbent_expected_process_initial_policy_exact':True,
  'original_incumbent_git_byte_restoration_exact':True,
  'original_strict_all_artifact_prediction':'contradicted','native_execution_repeated_for_reanalysis':False}
 files=[];directories=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery directory symlink')
   directories.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
 total=sum(f['bytes'] for f in files)
 if total>1024**3:raise RuntimeError('Recovery storage bound exceeded')
 h=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if h.returncode not in (0,1) or h.stdout or h.stderr:raise RuntimeError('Open recovery handles')
 for f in files:
  p=Path(f['path']);s=p.lstat()
  if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or stat.S_IMODE(s.st_mode)!=f['mode'] or a.file_hash(p)!=f['sha256']:
   raise RuntimeError('Recovery file changed before cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'results':results,'restoration':restoration,
  'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'removed_logical_bytes':total}}
 write('verification-v2-before-cleanup.json',report)
 for f in files:Path(f['path']).unlink()
 for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup']['status']='removed';write('verification-v2.json',report)
 print(json.dumps({'status':'passed','failure_reproduced_exact':True,'query_streams':len(metadata['distance_references']),'contact_arms':len(contacts),'model_Adam_arrays':tensors,'temporary_bytes_removed':total}))
except Exception as exc:
 write('verification-v2-failure.json',{'error':str(exc),'type':type(exc).__name__})
 raise
