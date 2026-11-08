"""Execute scoped native recovery from actual committed Git payload bytes."""
from pathlib import Path
import gzip,importlib.util,json,os,stat,subprocess,time
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-ccd-sweep-20261108'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();mp='artifacts/crow-rl-ccd-128-integration-20261108/payload/manifest.json'
with a.Objects() as obj:manifest=json.loads(obj.read(entries[mp][1]))
files={f['path']:f for f in manifest['files']}
C='build/crow-ccd-articulated-sweep-build-01';P='build/crow-ccd-contact-time-build-01';N='.numi/retained-runtime/ccd-event-capture-20261108'
T=R/'build/crow-ccd-sweep-git-recovery-01';T.mkdir(exist_ok=False)
sources={
 'runtime/libmetalrobo.dylib':N+'/integrated128/libmetalrobo.dylib',
 'runtime/metalrobo/MetalRobo.metallib':N+'/integrated128/metalrobo/MetalRobo.metallib',
 'runtime/metalrobo/RetinalStream.metallib':N+'/integrated128/metalrobo/RetinalStream.metallib',
 'runtime/contact-probe':N+'/integrated128/contact-probe',
 'runtime/sweep-probe':N+'/integrated128/sweep-probe',
 'runtime/CCDSweepProbe.metallib':N+'/integrated128/CCDSweepProbe.metallib',
 'runtime/probe128':C+'/probe128',
 'old/libmetalrobo.dylib':N+'/candidate/libmetalrobo.dylib',
 'old/metalrobo/MetalRobo.metallib':P+'/candidate.metallib',
 'old/probe':P+'/probe',
 'input/cases-probe':C+'/cases-probe','input/cases.bin':C+'/cases.bin','input/grid-expected.bin':C+'/expected-grid.bin',
 'input/projection-probe':C+'/projection-probe','input/baseline.metallib':C+'/baseline.metallib',
 'input/projection-baseline.bin':C+'/projection-baseline.bin','input/projection-candidate.bin':C+'/projection-candidate.bin',
}
for name in ('clamp','refinement'):
    sources[f'input/{name}.ccdsweep']=P+f'/{name}.ccdsweep'
    sources[f'input/{name}-expected.bin']=P+f'/{name}-candidate.bin'
for name in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):
    sources['input/failed32/'+name]=P+'/clamp-physical/'+name
sources['input/clamp-physical.ccdsweep']=P+'/clamp-physical.ccdsweep'
for arm in ('integrated32','admitted128'):
    for name in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):
        sources['input/'+arm+'/'+name]=C+'/'+arm+'/'+name
with a.Objects() as obj:
    for target,source in sources.items():
        f=files[source];p=T/target;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as out:
            for block in a.content(f,entries,obj):out.write(block)
        p.chmod(f['mode']);assert a.file_hash(p)==f['sha256'],source
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.pop('METALROBO_CCD_FAILURE_CAPTURE',None)
env.update(DYLD_LIBRARY_PATH=str(T/'runtime'),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'runtime/metalrobo/MetalRobo.metallib'))
commands=[];loaded=[]
def run(name,cmd,expected=0):
    start=time.monotonic()
    with (T/(name+'.stdout')).open('xb') as out,(T/(name+'.stderr')).open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=120).returncode
    commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start})
    a.write_json(G/'execution-progress.json',commands);print(json.dumps(commands[-1]),flush=True)
    assert code==expected,(name,code)
def exact(actual,expected):assert (T/actual).read_bytes()==(T/expected).read_bytes(),(actual,expected)
run('grid4096',[T/'input/cases-probe',T/'runtime/metalrobo/MetalRobo.metallib',T/'input/cases.bin',T/'grid4096.bin']);exact('grid4096.bin','input/grid-expected.bin')
for name,shader in (('baseline',T/'input/baseline.metallib'),('candidate',T/'runtime/metalrobo/MetalRobo.metallib')):
    run('projection-'+name,[T/'input/projection-probe',shader,T/('projection-'+name+'.bin')]);exact('projection-'+name+'.bin','input/projection-'+name+'.bin')
for name in ('clamp','refinement'):
    run('capture-'+name,[T/'runtime/sweep-probe',T/'old/metalrobo/MetalRobo.metallib',T/'runtime/CCDSweepProbe.metallib',T/f'input/{name}.ccdsweep',T/f'capture-{name}.bin'])
    exact('capture-'+name+'.bin','input/'+name+'-expected.bin')
run('native-contact-owner',[T/'runtime/contact-probe'])
for name,sub in (('integrated32',32),('admitted128',128)):
    run(name,[T/'runtime/probe128',T/name,'0','1',str(sub)])
    for file in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):exact(name+'/'+file,'input/'+name+'/'+file)
    lines=[x for x in (T/(name+'.stderr')).read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
    assert lines and all(x.endswith(str(T/'runtime/libmetalrobo.dylib')) for x in lines);loaded.extend(lines)
env.update(DYLD_LIBRARY_PATH=str(T/'old'),METALROBO_CCD_FAILURE_CAPTURE=str(T/'failed32.ccdsweep'))
env.pop('CROW_PROBE_METALLIB',None)
run('failed32',[T/'old/probe',T/'failed32','0','1','32'],1)
for file in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):exact('failed32/'+file,'input/failed32/'+file)
exact('failed32.ccdsweep','input/clamp-physical.ccdsweep')
lines=[x for x in (T/'failed32.stderr').read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
assert lines and all(x.endswith(str(T/'old/libmetalrobo.dylib')) for x in lines);loaded.extend(lines)
logs=G/'native-logs';logs.mkdir(exist_ok=False)
for p in T.iterdir():
    if p.suffix in ('.stdout','.stderr'):(logs/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
inventory=[{'path':str(p.relative_to(R)),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns} for p in sorted(T.rglob('*')) if p.is_file()]
opened=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(T)],capture_output=True,text=True,timeout=60)
assert opened.returncode in (0,1) and not opened.stdout and not opened.stderr
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:
    p=R/f['path'];s=p.stat();assert (a.file_hash(p),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
receipt={'status':'verified','source_commit':a.git('rev-parse','HEAD').decode().strip(),'actual_git_manifest_oid':entries[mp][1],'all_restored_inputs_sha256_verified':True,'restored_sources':sources,'commands':commands,'loader_lines':loaded,'checks':{'grid4096_exact':True,'projection_before_and_after_fix_exact':True,'two_selected_event_captures_and514_queries_exact':True,'native_contact_owner_including128_actual_substeps_and_invalid_dimensions_passed':True,'full_integrated32_and128_physical_traces_final_states_and_trajectories_exact':True,'negative_primitive32_trace_failure_status_rollback_and_capture_exact':True,'retinal_shader_packaging_sha_verified':True},'scope':'Scoped native recovery, not reexecution of every arm, retinal object-push or all training histories. Original object-push run passed and its complete bytes are separately verified from Git. Numerical gate remains failed.','temporary_cleanup':{'files':inventory,'open_handles':0,'removed_bytes':sum(f['bytes'] for f in inventory),'df_before':before}}
a.write_json(G/'executed-recovery.json',receipt)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in T.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
T.rmdir();receipt['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True));a.write_json(G/'executed-recovery.json',receipt)
print(json.dumps({'status':'verified','executions':len(commands),'temporary_removed_bytes':receipt['temporary_cleanup']['removed_bytes']}),flush=True)
