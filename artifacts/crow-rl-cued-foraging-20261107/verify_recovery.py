"""Reexecute Git-restored native cue instrument, comparing evidence without reference-array copies."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess,sys
root=Path.cwd();base=root/'artifacts/crow-rl-cued-foraging-20261107';tmp=Path('/private/tmp/crow-cued-foraging-git-recovery-20261107')
plan={'prediction':'Fresh Git-restored cue native runtime completes its unchanged calibration and eight arms. Actual dyld library path must equal restored adjacent library with no library override. Every native file matches its Git reference SHA256/size, except visual-scene JSON whose semantics must match after normalizing only relocated output-root path prefixes. No other fields may differ.','budget':'One180second native process; no retries, preserve any failure. Read references through committed Git objects, without duplicating large reference arrays on disk. Verify every temporary file and open handle before exact-path cleanup.','scope':'Same macOS/device/dependencies; native mechanism and runtime recovery, not learned memory or whole-life restore.'}
existing=sys.argv[1:]==['--analyze-existing']
if sys.argv[1:] and not existing:raise ValueError('Only --analyze-existing is supported')
if not existing:
    if (tmp/'replay').exists() or (base/'recovery.stdout').exists():raise ValueError('Recovery execution already exists; preserve it')
    (base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
amendment={'reason':'The original verifier correctly exposed a comparison error: native scene fingerprints include visual-pack filesystem paths (VisualPresentation.cpp:computeVisualRenderSceneV3Fingerprint); JSON does not expose those paths. String-prefix normalization cannot account for these hashes. Preserve original failure/plan and native logs. Do not rerun or alter the native task.','prediction':'All133 non-scene outputs remain byte exact. Reconstruct the unchanged fixture through native authoring only, then recompute both fingerprints at the original and restored pack paths. Both complete scene JSON objects must match their corresponding native reference, with no ignored fields. Reconstructed visual packs must match every corresponding replay pack byte exactly.','budget':'One bounded authoring-helper compilation and one authoring-only execution; zero additional simulation steps.','analysis_of_existing_replay':existing,'source_commit':'f18e0217d3e6d0a0c132b80c82d6aa9e2027ca17'}
(base/'verification-amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);entries=a.tree()
command=[str(tmp/'bin/metalrobo_crow_cue_foraging_check'),str(tmp/'replay')]
try:
    env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
    if existing:
        p=subprocess.CompletedProcess(command,0,(base/'recovery.stdout').read_text(),(base/'recovery.stderr').read_text())
        if json.loads(p.stdout).get('status')!='passed':raise RuntimeError('Original native replay did not pass')
    else:
        p=subprocess.run(command,capture_output=True,text=True,timeout=180,env=env)
        (base/'recovery.stdout').write_text(p.stdout);(base/'recovery.stderr').write_text(p.stderr)
    loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
    if p.returncode or len(loaded)!=1 or not loaded[0].endswith(str(tmp/'lib/libmetalrobo.dylib')):raise RuntimeError('Recovered process failed or loaded unexpected library')
    with a.Objects() as objects:
        metadata=json.loads(objects.read(entries[str(base.relative_to(root)/'runtime-recovery.json')][1]))
        files={}
        for manifest in metadata['manifests']:files.update({f['path']:f for f in json.loads(objects.read(entries[manifest][1]))['files']})
        if a.git('diff',amendment['source_commit'],'--','include','apps/CrowCameraFixture.hpp','apps/crow_cue_foraging_check.mm'):raise RuntimeError('Authoring source changed since retained native runtime')
        helper=tmp/'bin/scene_relocation_check'
        compile_command=['/usr/bin/clang++','-std=c++23','-O2','-fobjc-arc','-Iinclude','-Imatter/include','-Iapps',str(base/'scene_relocation_check.mm'),'-L'+str(tmp/'lib'),'-lmetalrobo','-Wl,-rpath,@loader_path/../lib','-framework','Foundation','-framework','Metal','-o',str(helper)]
        compiled=subprocess.run(compile_command,capture_output=True,text=True,timeout=120)
        (base/'scene-authoring-build.stdout').write_text(compiled.stdout);(base/'scene-authoring-build.stderr').write_text(compiled.stderr)
        if compiled.returncode:raise RuntimeError('Authoring helper compile failed')
        reconstruction=tmp/'scene-reconstruction'
        helper_command=[str(helper),str(reconstruction),metadata['original_output_root'],str(tmp/'replay')]
        authored=subprocess.run(helper_command,capture_output=True,text=True,timeout=60,env=env)
        (base/'scene-authoring.stdout').write_text(authored.stdout);(base/'scene-authoring.stderr').write_text(authored.stderr)
        helper_loaded=[line for line in authored.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
        if authored.returncode or len(helper_loaded)!=1 or not helper_loaded[0].endswith(str(tmp/'lib/libmetalrobo.dylib')):raise RuntimeError('Authoring helper failed or loaded unexpected library')
        expected=metadata['native_reference'];actual={str(f.relative_to(tmp/'replay')) for f in (tmp/'replay').rglob('*') if f.is_file()}
        if actual!=set(expected):raise RuntimeError('Native recovery output file set differs')
        matches={};relocated={}
        for name,rec in expected.items():
            dest=tmp/'replay'/name
            if rec['comparison']=='byte exact':
                if dest.stat().st_size!=rec['bytes'] or a.file_hash(dest)!=rec['sha256']:raise RuntimeError('Native recovered evidence differs: '+name)
                matches[name]={'bytes':rec['bytes'],'sha256':rec['sha256']}
            else:
                f=files[rec['archive_source']];reference=b''.join(a.content(f,entries,objects))
                if json.loads(reference)!=json.loads((reconstruction/'original'/name).read_bytes()) or json.loads(dest.read_bytes())!=json.loads((reconstruction/'replay'/name).read_bytes()):raise RuntimeError('Native scene path-fingerprint reconstruction differs: '+name)
                arm=Path(name).parent.name;pack_root=reconstruction/('blocked-packs' if arm=='occluded' else 'clear-packs')
                for pack in pack_root.glob('*.mrvpack'):
                    if a.file_hash(pack)!=a.file_hash(tmp/'replay'/arm/pack.name):raise RuntimeError('Reconstructed visual content differs')
                relocated[name]={'status':'passed','comparison':'Both full scene JSON objects, including both fingerprints, match exact native authoring at their respective paths; every reconstructed visual pack byte exact','reference_sha256':rec['sha256'],'replay_sha256':a.file_hash(dest)}
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
        if s.st_size!=rec['bytes'] or s.st_mtime_ns!=rec['mtime_ns'] or a.file_hash(path)!=rec['sha256']:raise RuntimeError('Recovery changed during cleanup')
    report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'amendment':amendment,'command':command,'actual_loaded_library':loaded,'library_override':False,'native_summary':json.loads(p.stdout),'runtime_restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'authoring_helper':{'compile_command':compile_command,'command':helper_command,'source_sha256':a.file_hash(base/'scene_relocation_check.mm'),'actual_loaded_library':helper_loaded},'byte_exact_native_files':matches,'relocated_visual_scene_files':relocated,'temporary_removal':{'status':'validated','files':records,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
    (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for rec in records:Path(rec['path']).unlink()
    for path in sorted(dirs,key=lambda p:len(p.parts),reverse=True):path.rmdir()
    tmp.rmdir();report['temporary_removal'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True))
    (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','byte_exact_native_files':len(matches),'relocated_visual_scenes_exact':len(relocated),'temporary_files_removed':len(records),'temporary_bytes_removed':sum(r['bytes'] for r in records)}))
except Exception as error:
    (base/'verification-amendment-failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),'command':command,'plan':plan},indent=2)+'\n');raise
