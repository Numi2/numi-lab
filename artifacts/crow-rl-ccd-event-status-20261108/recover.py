"""Execute committed Git bytes, then remove only the verified temporary restore."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-ccd-event-status-20261108'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();manifest_path='artifacts/crow-rl-ccd-event-status-20261108/payload/manifest.json'
with a.Objects() as objects:m=json.loads(objects.read(entries[manifest_path][1]))
files={f['path']:f for f in m['files']};T=R/'build/crow-ccd-current-git-recovery-01';T.mkdir(exist_ok=False)
V1='build/crow-ccd-translation-build-01';V2='build/crow-ccd-translation-build-02';P='build/crow-beak-hold-build-02';E='build/crow-ccd-event-status-build-01';N='.numi/retained-runtime/beak-hold-ccd-fix-20261108'
sources={'runtime/libmetalrobo.dylib':N+'/libmetalrobo.dylib','runtime/production.metallib':N+'/MetalRobo.metallib',
 'runtime/query.metallib':N+'/CCDSweepProbe.metallib','runtime/sweep-probe':N+'/sweep-probe',
 'runtime/candidate-with-query.metallib':V2+'/candidate.metallib','runtime/event.metallib':E+'/candidate.metallib',
 'runtime/event-probe':E+'/event-status-probe','runtime/beak-probe':P+'/probe-with-runtime',
 'v1/cases-probe':V1+'/cases_probe','v1/cases.bin':V1+'/cases.bin','v1/candidate.metallib':V1+'/candidate.metallib',
 'v1/expected.bin':V1+'/candidate.bin','v2/cases-probe':V2+'/cases_probe','v2/cases.bin':V2+'/cases.bin',
 'v2/expected.bin':V2+'/integrated-grid.bin','v2/query-probe':V2+'/query_probe','v2/query-alpha.f32':V2+'/query-alpha.f32',
 'v2/observed-misses.bin':V2+'/observed-misses.bin','v2/query-expected.bin':V2+'/native-queries.bin',
 'input/calibration.jsonl':P+'/calibration-with-runtime.jsonl','input/event-expected.bin':E+'/candidate.bin'}
for name in ('clamp','open','zero-friction','refinement'):
    sources[f'input/{name}.ccdsweep']=V2+f'/input/{name}.ccdsweep'
    sources[f'input/{name}-expected.bin']=V2+f'/integrated-{name}.bin'
for version,src in [('old',P),('event',E)]:
    for name in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):
        sources[f'input/{version}-{name}']=src+'/clamp/'+name
for version,shader in [('old',N+'/MetalRobo.metallib'),('event',E+'/candidate.metallib')]:
    sources[f'runtime-{version}/libmetalrobo.dylib']=N+'/libmetalrobo.dylib'
    sources[f'runtime-{version}/metalrobo/MetalRobo.metallib']=shader
with a.Objects() as objects:
    for target,source in sources.items():
        f=files[source];p=T/target;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as out:
            for block in a.content(f,entries,objects):out.write(block)
        p.chmod(f['mode']);assert a.file_hash(p)==f['sha256']
env=os.environ.copy();env.pop('METALROBO_METALLIB',None)
env.update(DYLD_LIBRARY_PATH=str(T/'runtime-old'),DYLD_PRINT_LIBRARIES='1')
commands=[]
def run(name,command,expected=0):
    start=time.monotonic()
    with (T/f'{name}.stdout').open('xb') as out,(T/f'{name}.stderr').open('xb') as err:
        code=subprocess.run([str(x) for x in command],env=env,stdout=out,stderr=err,timeout=180).returncode
    commands.append({'name':name,'command':[str(x) for x in command],'exit_code':code,'seconds':time.monotonic()-start})
    assert code==expected,(name,code)
def exact(actual,expected):assert (T/actual).read_bytes()==(T/expected).read_bytes(),(actual,expected)
run('v1-negative-replay',[T/'v1/cases-probe',T/'v1/candidate.metallib',T/'v1/cases.bin',T/'v1-new.bin']);exact('v1-new.bin','v1/expected.bin')
run('v2-production-grid',[T/'v2/cases-probe',T/'runtime/production.metallib',T/'v2/cases.bin',T/'v2-new.bin']);exact('v2-new.bin','v2/expected.bin')
run('v2-native-queries',[T/'v2/query-probe',T/'runtime/candidate-with-query.metallib',T/'v2/observed-misses.bin',T/'v2/query-alpha.f32',T/'queries-new.bin']);exact('queries-new.bin','v2/query-expected.bin')
for name in ('clamp','open','zero-friction','refinement'):
    run('sweep-'+name,[T/'runtime/sweep-probe',T/'runtime/production.metallib',T/'runtime/query.metallib',T/f'input/{name}.ccdsweep',T/f'sweep-{name}.bin'])
    exact(f'sweep-{name}.bin',f'input/{name}-expected.bin')
run('calibration',[T/'runtime/beak-probe','--calibrate']);exact('calibration.stdout','input/calibration.jsonl')
loaded=[]
for version in ('old','event'):
    env['DYLD_LIBRARY_PATH']=str(T/('runtime-'+version))
    run(version+'-clamp',[T/'runtime/beak-probe',T/(version+'-clamp'),'0','1','32'],1)
    for name in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):
        exact(version+'-clamp/'+name,'input/'+version+'-'+name)
    lines=[x for x in (T/f'{version}-clamp.stderr').read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
    assert lines and all(x.endswith(str(T/('runtime-'+version)/'libmetalrobo.dylib')) for x in lines)
    loaded.extend(lines)
run('event-initializer',[T/'runtime/event-probe',T/'runtime/event.metallib',T/'event-new.bin','preserve']);exact('event-new.bin','input/event-expected.bin')
inventory=[{'path':str(p.relative_to(R)),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns}
    for p in sorted(T.rglob('*')) if p.is_file()]
opened=subprocess.run(['/usr/sbin/lsof','-t','+D',str(T)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert opened.returncode in (0,1) and not opened.stdout
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:
    p=R/f['path'];st=p.stat();assert (a.file_hash(p),st.st_size,stat.S_IMODE(st.st_mode),st.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
receipt={'status':'verified','source_commit':a.git('rev-parse','HEAD').decode().strip(),'actual_git_manifest_oid':entries[manifest_path][1],
 'all_restored_inputs_sha256_verified':True,'restored_sources':sources,'commands':commands,'loader_lines':loaded,
 'checks':{'rejected_v1_candidate_8192_records_exact':True,'qualified_v2_production_8192_records_exact':True,'sixteen_native_queries_exact':True,
   'four_retained_real_ccd_replays_exact':True,'calibration_exact':True,'old_and_preserving_shader_clamp_traces_and_all_failure_files_exact':True,
   'preserving_initializer_complete_buffers_exact':True},
 'scope':'Executed scoped recovery of both CCD revisions, all four isolated captures, and the physical clamp under both exact runtimes. Not every study arm, all prior training histories or completed beak capability.',
 'temporary_cleanup':{'files':inventory,'open_handles':0,'df_before':before,'removed_bytes':sum(f['bytes'] for f in inventory)}}
a.write_json(A/'git-recovery.json',receipt)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in T.rglob('*') if p.is_dir()),key=lambda x:len(x.parts),reverse=True):p.rmdir()
T.rmdir();receipt['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True))
a.write_json(A/'git-recovery.json',receipt)
print(json.dumps({'status':'verified','commands':len(commands),'temporary_removed_bytes':receipt['temporary_cleanup']['removed_bytes']}),flush=True)
