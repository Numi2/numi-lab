"""One Git-restored native replay, exact scene reconstruction, bounded cleanup."""
from pathlib import Path
import importlib.util,json,os,stat,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-foraging-core-20261107';tmp=Path('/private/tmp/crow-foraging-core-git-recovery-20261107')
plan={'prediction':'Fresh Git-restored cue instrument passes once;134 non-scene outputs byte exact and eight complete scene JSON objects, including both fingerprints, match unchanged native authoring at their respective original/restored paths. Both native executables must load the restored adjacent library without overrides.','budget':'One180second native cue process, one60second authoring-only helper (zero simulation steps), no retries. Read reference arrays directly from Git. Exact-file and open-handle validation before removing temporary files.','scope':'Native authoring/task preservation and runtime recovery on matching device/OS/dependencies. Legacy v13 life restore regressed separately; cue-life restore and learned memory remain unqualified.'}
with (base/'verification-plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);entries=a.tree()
command=[str(tmp/'bin/metalrobo_crow_cue_foraging_check'),str(tmp/'replay')]
try:
 if (tmp/'replay').exists():raise RuntimeError('Native replay already exists; preserve it')
 env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
 p=subprocess.run(command,capture_output=True,text=True,timeout=180,env=env)
 (base/'recovery.stdout').write_text(p.stdout);(base/'recovery.stderr').write_text(p.stderr)
 loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if p.returncode or len(loaded)!=1 or not loaded[0].endswith(str(tmp/'lib/libmetalrobo.dylib')):raise RuntimeError('Recovered cue failed or loaded unexpected library')
 with a.Objects() as objects:
  metadata=json.loads(objects.read(entries[str(base.relative_to(root)/'runtime-recovery.json')][1]));files={}
  for manifest in metadata['manifests']:files.update({f['path']:f for f in json.loads(objects.read(entries[manifest][1]))['files']})
  expected=metadata['native_reference'];actual={str(f.relative_to(tmp/'replay')) for f in (tmp/'replay').rglob('*') if f.is_file()}
  if actual!=set(expected):raise RuntimeError('Native replay output file set differs')
  matches={};relocated={}
  for name,rec in expected.items():
   if rec['comparison']!='byte exact':continue
   dest=tmp/'replay'/name
   if dest.stat().st_size!=rec['bytes'] or a.file_hash(dest)!=rec['sha256']:raise RuntimeError('Recovered native evidence differs: '+name)
   matches[name]={'bytes':rec['bytes'],'sha256':rec['sha256']}
  helper_command=[str(tmp/'bin/scene_relocation_check'),str(tmp/'scene-reconstruction'),metadata['original_output_root'],str(tmp/'replay')]
  authored=subprocess.run(helper_command,capture_output=True,text=True,timeout=60,env=env)
  (base/'scene-authoring.stdout').write_text(authored.stdout);(base/'scene-authoring.stderr').write_text(authored.stderr)
  helper_loaded=[line for line in authored.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
  if authored.returncode or len(helper_loaded)!=1 or not helper_loaded[0].endswith(str(tmp/'lib/libmetalrobo.dylib')):raise RuntimeError('Recovered authoring failed or loaded unexpected library')
  for name,rec in expected.items():
   if rec['comparison']=='byte exact':continue
   reference=b''.join(a.content(files[rec['archive_source']],entries,objects));dest=tmp/'replay'/name;reconstruction=tmp/'scene-reconstruction'
   if json.loads(reference)!=json.loads((reconstruction/'original'/name).read_bytes()) or json.loads(dest.read_bytes())!=json.loads((reconstruction/'replay'/name).read_bytes()):raise RuntimeError('Complete native scene reconstruction differs: '+name)
   arm=Path(name).parent.name;pack_root=reconstruction/('blocked-packs' if arm=='occluded' else 'clear-packs')
   for pack in pack_root.glob('*.mrvpack'):
    if a.file_hash(pack)!=a.file_hash(tmp/'replay'/arm/pack.name):raise RuntimeError('Reconstructed visual content differs')
   relocated[name]={'status':'passed','comparison':'Both complete scene JSON objects, including both fingerprints, match exact native authoring at their respective paths; every reconstructed visual pack byte exact','reference_sha256':rec['sha256'],'replay_sha256':a.file_hash(dest)}
 records=[];dirs=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   path=Path(directory)/name
   if path.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   dirs.append(path)
  for name in names:
   path=Path(directory)/name;s=path.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   records.append({'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),'sha256':a.file_hash(path)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovered runtime handles')
 for rec in records:
  path=Path(rec['path']);s=path.lstat()
  if s.st_size!=rec['bytes'] or s.st_mtime_ns!=rec['mtime_ns'] or stat.S_IMODE(s.st_mode)!=rec['mode'] or a.file_hash(path)!=rec['sha256']:raise RuntimeError('Recovery changed during cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'command':command,'actual_loaded_library':loaded,'library_override':False,'native_summary':json.loads(p.stdout),'runtime_restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'authoring_helper':{'command':helper_command,'actual_loaded_library':helper_loaded},'byte_exact_native_files':matches,'relocated_visual_scene_files':relocated,'temporary_removal':{'status':'validated','files':records,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for rec in records:Path(rec['path']).unlink()
 for path in sorted(dirs,key=lambda p:len(p.parts),reverse=True):path.rmdir()
 tmp.rmdir();report['temporary_removal'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True));(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','byte_exact_native_files':len(matches),'relocated_visual_scenes_exact':len(relocated),'temporary_files_removed':len(records),'temporary_bytes_removed':sum(r['bytes'] for r in records)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),'command':command,'plan':plan},indent=2)+'\n');raise
