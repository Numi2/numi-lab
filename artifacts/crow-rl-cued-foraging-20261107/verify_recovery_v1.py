"""Reexecute Git-restored native cue instrument, comparing evidence without reference-array copies."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-cued-foraging-20261107';tmp=Path('/private/tmp/crow-cued-foraging-git-recovery-20261107')
plan={'prediction':'Fresh Git-restored cue native runtime completes its unchanged calibration and eight arms. Actual dyld library path must equal restored adjacent library with no library override. Every native file matches its Git reference SHA256/size, except visual-scene JSON whose semantics must match after normalizing only relocated output-root path prefixes. No other fields may differ.','budget':'One180second native process; no retries, preserve any failure. Read references through committed Git objects, without duplicating large reference arrays on disk. Verify every temporary file and open handle before exact-path cleanup.','scope':'Same macOS/device/dependencies; native mechanism and runtime recovery, not learned memory or whole-life restore.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);entries=a.tree()
command=[str(tmp/'bin/metalrobo_crow_cue_foraging_check'),str(tmp/'replay')]
try:
    env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
    p=subprocess.run(command,capture_output=True,text=True,timeout=180,env=env)
    (base/'recovery.stdout').write_text(p.stdout);(base/'recovery.stderr').write_text(p.stderr)
    loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
    if p.returncode or len(loaded)!=1 or not loaded[0].endswith(str(tmp/'lib/libmetalrobo.dylib')):raise RuntimeError('Recovered process failed or loaded unexpected library')
    with a.Objects() as objects:
        metadata=json.loads(objects.read(entries[str(base.relative_to(root)/'runtime-recovery.json')][1]))
        files={}
        for manifest in metadata['manifests']:files.update({f['path']:f for f in json.loads(objects.read(entries[manifest][1]))['files']})
        expected=metadata['native_reference'];actual={str(f.relative_to(tmp/'replay')) for f in (tmp/'replay').rglob('*') if f.is_file()}
        if actual!=set(expected):raise RuntimeError('Native recovery output file set differs')
        matches={};relocated={}
        def normalized(value,prefix):
            if isinstance(value,dict):return {k:normalized(v,prefix) for k,v in value.items()}
            if isinstance(value,list):return [normalized(v,prefix) for v in value]
            if isinstance(value,str) and value.startswith(prefix+'/'):return '$OUTPUT/'+value[len(prefix)+1:]
            return value
        for name,rec in expected.items():
            dest=tmp/'replay'/name
            if rec['comparison']=='byte exact':
                if dest.stat().st_size!=rec['bytes'] or a.file_hash(dest)!=rec['sha256']:raise RuntimeError('Native recovered evidence differs: '+name)
                matches[name]={'bytes':rec['bytes'],'sha256':rec['sha256']}
            else:
                f=files[rec['archive_source']];reference=b''.join(a.content(f,entries,objects))
                if normalized(json.loads(reference),metadata['original_output_root'])!=normalized(json.loads(dest.read_bytes()),str(tmp/'replay')):
                    raise RuntimeError('Relocated visual scene semantics differ: '+name)
                relocated[name]={'status':'passed','normalization':'only native output-root path prefix; all geometry/content/fingerprint fields compared'}
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
    report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'command':command,'actual_loaded_library':loaded,'library_override':False,'native_summary':json.loads(p.stdout),'runtime_restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'byte_exact_native_files':matches,'relocated_visual_scene_files':relocated,'temporary_removal':{'status':'validated','files':records,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
    (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for rec in records:Path(rec['path']).unlink()
    for path in sorted(dirs,key=lambda p:len(p.parts),reverse=True):path.rmdir()
    tmp.rmdir();report['temporary_removal'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True))
    (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','byte_exact_native_files':len(matches),'relocated_visual_scenes_exact':len(relocated),'temporary_files_removed':len(records),'temporary_bytes_removed':sum(r['bytes'] for r in records)}))
except Exception as error:
    (base/'verification-failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),'command':command,'plan':plan},indent=2)+'\n');raise
