"""Recover actual Git bytes, execute final updates and one complete native control block."""
from pathlib import Path
import importlib.util,json,os,stat,subprocess,time,hashlib
r=Path.cwd();base=Path('artifacts/crow-rl-post-step-rate-20261108')
tmp=Path('/private/tmp/crow-post-step-rate-recovery-20261108');tmp.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def write(name,value):
    with(base/name).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
plan={'budget':'Exactly2 offline final-update replays, one full12-arm adaptive causal block and one source-compiled native visual-identity oracle; temp<=512MiB. No new training recipe or heldout seed. Timeout180s offline,660s native,120s oracle build/execution.',
      'prediction':'Both53-array final model/Adam states, mode configuration, all scientific optimizer metrics and95 native behavior/binary evidence files reproduce exactly. Native owning functions independently recompute both relocated identity fields in each of12 visual manifests; every other JSON field matches.',
      'scope':'Last-update learner and3-of9-seed adaptive causal recovery, not32-update native life replay or every original heldout execution.'}
write('verification-plan.json',plan)
try:
    entries=a.tree();metadata=json.loads((base/'runtime-recovery.json').read_text());records=[]
    with a.Objects()as objects:
        manifest=json.loads(objects.read(entries[str(base/'payload/manifest.json')][1]));files={f['path']:f for f in manifest['files']}
        def save(relative,data,expected,mode):
            p=tmp/a.safe_path(relative);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(mode)
            assert a.file_hash(p)==expected,relative
            records.append({'path':relative,'sha256':expected,'bytes':len(data),'mode':mode})
        for relative,record in metadata['native'].items():
            f=files[record['archive_source']];save(relative,b''.join(a.content(f,entries,objects)),record['sha256'],f['mode'])
        for relative,record in metadata['python_sources'].items():
            save(relative,objects.read(record['git_blob']),record['sha256'],record['mode'])
        for relative,record in metadata['oracle_sources'].items():
            save(relative,objects.read(record['git_blob']),record['sha256'],record['mode'])
        ports='environments/american-crow/cued-foraging-v1/calibrated-ports.bin'
        record=metadata['git_inputs'][ports];save('input/ports.bin',objects.read(record['git_blob']),record['sha256'],record['mode'])
        for target,source in metadata['files'].items():
            f=files[source];save(target,b''.join(a.content(f,entries,objects)),f['sha256'],f['mode'])
    env={k:v for k,v in os.environ.items()if not k.startswith('DYLD_')}
    env.update(DYLD_PRINT_LIBRARIES='1',PYTHONPATH=str(tmp/'python'),PYTHONDONTWRITEBYTECODE='1',
               METALROBO_LIBRARY=str(tmp/'lib/libmetalrobo.dylib'),METALROBO_METALLIB=str(tmp/'shaders/MetalRobo.metallib'))
    def execute(name,command,timeout):
        write(name+'.command.json',command);start=time.monotonic()
        p=subprocess.run(command,cwd=tmp,capture_output=True,env=env,timeout=timeout)
        (base/(name+'.stdout')).write_bytes(p.stdout);(base/(name+'.stderr')).write_bytes(p.stderr)
        write(name+'.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start})
        assert p.returncode==0,name
        loaded=[x for x in p.stderr.decode().splitlines()if x.startswith('dyld[')and'libmetalrobo.dylib'in x]
        assert loaded and all(x.endswith(str(tmp/'lib/libmetalrobo.dylib'))for x in loaded),name
        return loaded
    offline=execute('learner-recovery',[str(r/'python/.venv/bin/python'),str(tmp/'source/replay-last.py'),str(tmp/'trial'),str(tmp/'replay.json')],180)
    assert json.loads((tmp/'replay.json').read_text())==json.loads((tmp/'input/replay-reference.json').read_text())
    native=execute('native-recovery',[str(tmp/'bin/metalrobo_crow_memory_causal_check'),str(tmp/'evaluation'),
                   str(tmp/'input/adaptive.policypack'),str(tmp/'input/ports.bin'),str(metadata['evaluation_reference']['seed_base'])],660)
    compiler=['clang++','-std=c++20','-fobjc-arc','-I'+str(tmp/'source/include'),'-I'+str(tmp/'source/apps'),
        str(tmp/'source/identity_oracle.mm'),'-L'+str(tmp/'lib'),'-lmetalrobo','-framework','Foundation',
        '-Wl,-rpath,@loader_path/../lib','-o',str(tmp/'bin/identity_oracle')]
    built=subprocess.run(compiler,capture_output=True,text=True,timeout=120)
    write('oracle-build.json',{'command':compiler,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr})
    assert built.returncode==0
    oracle=execute('identity-recovery',[str(tmp/'bin/identity_oracle'),metadata['evaluation_reference']['original_root'],
        str(tmp/'evaluation'),str(tmp/'oracle-assets'),str(metadata['evaluation_reference']['seed_base'])],120)
    identities=json.loads((base/'identity-recovery.stdout').read_text())
    exact=[];relocated=[]
    with a.Objects()as objects:
        for name,ref in metadata['evaluation_reference']['files'].items():
            actual=tmp/'evaluation'/name
            if name.endswith('/visual-scene.json'):
                source=metadata['evaluation_reference']['original_root']+'/'+name
                source=str(Path(source).relative_to(r));f=files[source]
                old=json.loads(b''.join(a.content(f,entries,objects)));new=json.loads(actual.read_text())
                identity_pair=identities[name.split('/')[0]]
                for location,value in (('original',old),('restored',new)):
                    assert value.pop('fingerprint')==identity_pair[location]['manifest']
                    assert value['render_scene'].pop('fingerprint')==identity_pair[location]['render_scene']
                assert old==new,name
                relocated.append({'path':name,'native_owner_identities':identity_pair,'every_other_JSON_field_exact':True})
            else:
                assert a.file_hash(actual)==ref['sha256'],name
                exact.append(name)
    assert len(exact)==95 and len(relocated)==12
    assert {str(p.relative_to(tmp/'evaluation'))for p in(tmp/'evaluation').rglob('*')if p.is_file()}==set(metadata['evaluation_reference']['files'])
    allfiles=[];directories=[]
    for directory,subdirs,names in os.walk(tmp,followlinks=False):
        for name in subdirs:
            p=Path(directory)/name;assert not p.is_symlink();directories.append(p)
        for name in names:
            p=Path(directory)/name;s=p.lstat();assert stat.S_ISREG(s.st_mode)
            allfiles.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
    total=sum(x['bytes']for x in allfiles);assert total<=512*1024**2
    handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
    assert handles.returncode in(0,1)and not handles.stdout and not handles.stderr
    for x in allfiles:
        p=Path(x['path']);s=p.lstat()
        assert s.st_size==x['bytes']and s.st_mtime_ns==x['mtime_ns']and stat.S_IMODE(s.st_mode)==x['mode']and a.file_hash(p)==x['sha256']
    result={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,
        'restored_files':records,'offline_arrays_byte_exact_per_arm':53,'offline_scientific_metrics_exact':True,
        'native_byte_exact_files':exact,'relocated_visual_manifests':relocated,
        'loaded_libraries':{'offline':offline,'native':native,'identity_oracle':oracle},'complete_training_replayed':False,
        'temporary_cleanup':{'status':'validated','files':allfiles,'bytes':total,'open_handles':0}}
    write('verification-before-cleanup.json',result)
    for x in allfiles:Path(x['path']).unlink()
    for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):p.rmdir()
    tmp.rmdir();result['temporary_cleanup']['status']='removed';write('verification.json',result)
    print(json.dumps({'status':'passed','offline_exact_arrays_each':53,'native_exact_files':len(exact),'temporary_bytes_removed':total}))
except Exception as e:
    write('verification-failure.json',{'type':type(e).__name__,'error':str(e)});raise
