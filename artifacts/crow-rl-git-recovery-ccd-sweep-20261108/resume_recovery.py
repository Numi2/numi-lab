"""Finish existing verified native reexecutions; never rerun physics."""
from pathlib import Path
import gzip,importlib.util,json,os,stat,subprocess
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-ccd-sweep-20261108';T=R/'build/crow-ccd-sweep-git-recovery-01'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();mp='artifacts/crow-rl-ccd-128-integration-20261108/payload/manifest.json'
with a.Objects() as obj:m=json.loads(obj.read(entries[mp][1]))
files={f['path']:f for f in m['files']};byhash={f['sha256']:f for f in m['files']};restored=[]
for directory in ('runtime','old','input'):
    for p in (T/directory).rglob('*'):
        if not p.is_file():continue
        f=byhash[a.file_hash(p)]
        assert p.stat().st_size==f['bytes'] and stat.S_IMODE(p.stat().st_mode)==f['mode']
        with a.Objects() as obj:assert b''.join(a.content(f,entries,obj))==p.read_bytes()
        restored.append({'target':str(p.relative_to(T)),'matching_original_git_source':f['path'],'sha256':f['sha256']})
# Bind actual integrated and older libraries to their intended revisions.
N='.numi/retained-runtime/ccd-event-capture-20261108'
for target,source in [('runtime/libmetalrobo.dylib',N+'/integrated128/libmetalrobo.dylib'),('runtime/metalrobo/MetalRobo.metallib',N+'/integrated128/metalrobo/MetalRobo.metallib'),('runtime/metalrobo/RetinalStream.metallib',N+'/integrated128/metalrobo/RetinalStream.metallib'),('old/libmetalrobo.dylib',N+'/candidate/libmetalrobo.dylib'),('old/metalrobo/MetalRobo.metallib','build/crow-ccd-contact-time-build-01/candidate.metallib')]:assert a.file_hash(T/target)==files[source]['sha256']
commands=json.loads((G/'execution-progress.json').read_text());names=['grid4096','projection-baseline','projection-candidate','capture-clamp','capture-refinement','native-contact-owner','integrated32','admitted128','failed32']
assert [c['name'] for c in commands]==names and [c['exit_code'] for c in commands]==[0]*8+[1]
def exact(actual,expected):assert (T/actual).read_bytes()==(T/expected).read_bytes(),(actual,expected)
exact('grid4096.bin','input/grid-expected.bin')
for name in ('baseline','candidate'):exact('projection-'+name+'.bin','input/projection-'+name+'.bin')
for name in ('clamp','refinement'):exact('capture-'+name+'.bin','input/'+name+'-expected.bin')
loaded=[]
for arm in ('integrated32','admitted128'):
    for file in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):exact(arm+'/'+file,'input/'+arm+'/'+file)
    lines=[x for x in (T/(arm+'.stderr')).read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
    assert lines and all(x.endswith(str(T/'runtime/libmetalrobo.dylib')) for x in lines);loaded.extend(lines)
for file in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):exact('failed32/'+file,'input/failed32/'+file)
exact('failed32.ccdsweep','input/clamp-physical.ccdsweep')
lines=[x for x in (T/'failed32.stderr').read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
assert lines and all(x.endswith(str(T/'old/libmetalrobo.dylib')) for x in lines);loaded.extend(lines)
for p in T.iterdir():
    if p.suffix in ('.stdout','.stderr'):assert gzip.decompress((G/'native-logs'/(p.name+'.gz')).read_bytes())==p.read_bytes()
inventory=[{'path':str(p.relative_to(R)),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns} for p in sorted(T.rglob('*')) if p.is_file()]
opened=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(T)],capture_output=True,text=True,timeout=60)
guard={'initial_failure':'All nine native reexecutions and byte comparisons had passed. Initial temporary cleanup lsof assertion failed before receipt/deletion. Its raw lsof result was not logged, so its cause is unknown. No native rerun is performed.','new_open_handle_result':{'exit_code':opened.returncode,'stdout':opened.stdout,'stderr':opened.stderr}}
a.write_json(G/'recovery-cleanup-guard.json',guard)
assert opened.returncode in (0,1) and not opened.stdout and not opened.stderr,guard
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:
    p=R/f['path'];s=p.stat();assert (a.file_hash(p),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
receipt={'status':'verified','source_commit':a.git('rev-parse','HEAD').decode().strip(),'actual_git_manifest_oid':entries[mp][1],'restored_inputs_actual_git_byte_verified':restored,'commands':commands,'loader_lines':loaded,'checks':{'grid4096_exact':True,'projection_before_and_after_fix_exact':True,'two_selected_event_captures_and514_queries_exact':True,'native_contact_owner_including_actual128_substeps_passed':True,'full_integrated32_and128_physical_traces_final_states_and_trajectories_exact':True,'negative_primitive32_trace_failure_status_rollback_and_capture_exact':True,'retinal_shader_packaging_sha_verified':True},'scope':'Scoped native recovery, not reexecution of every arm, retinal object-push or all training histories. Original object-push run passed and its complete bytes are separately verified from Git. Numerical gate remains failed.','cleanup_guard':'recovery-cleanup-guard.json','temporary_cleanup':{'files':inventory,'open_handles':0,'removed_bytes':sum(f['bytes'] for f in inventory),'df_before':before}}
a.write_json(G/'executed-recovery.json',receipt)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in T.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
T.rmdir();receipt['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True));a.write_json(G/'executed-recovery.json',receipt)
print(json.dumps({'status':'verified','executions':len(commands),'temporary_removed_bytes':receipt['temporary_cleanup']['removed_bytes']}),flush=True)
