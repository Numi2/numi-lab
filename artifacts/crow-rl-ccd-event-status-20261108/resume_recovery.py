"""Repair co-installed runtime layout after a detected shader fallback; no grid rerun."""
from pathlib import Path
import importlib.util,json,os,shutil,stat,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-ccd-event-status-20261108';T=R/'build/crow-ccd-current-git-recovery-01'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();mp='artifacts/crow-rl-ccd-event-status-20261108/payload/manifest.json'
with a.Objects() as obj:m=json.loads(obj.read(entries[mp][1]))
mapping={f['path']:f for f in m['files']}
original={str(p.relative_to(T)):{'sha256':a.file_hash(p),'bytes':p.stat().st_size} for p in sorted(T.rglob('*')) if p.is_file()}
def exact(x,y):assert (T/x).read_bytes()==(T/y).read_bytes(),(x,y)
# Retain the failed attempt; all large outputs already have exact Git backing.
matches={}
for name in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):
    exact('old-clamp/'+name,'input/event-'+name);matches[name]='build/crow-ccd-event-status-build-01/clamp/'+name
for suffix in ('.stdout','.stderr'):shutil.copy2(T/('old-clamp'+suffix),A/('recovery-attempt-1-clamp'+suffix))
a.write_json(A/'recovery-attempt-1.json',{'status':'physical shader isolation rejected','reason':'METALROBO_METALLIB is not read by MetalWorld. Library discovery did not find a co-installed shader and could fall back to the build shader. Whole contact-status mismatch exposed the defect.',
  'authority':'src/metal/MetalWorld.mm defaultMetallibPath and ensure initialization library selection.',
  'failed_contact_status_compared_to_original_trial':True,'all_failed_attempt_physical_outputs_match_preserving_shader_capture':matches,
  'temporary_inventory_before_layout_repair':original,'no_evidence_removed':True,
  'revision':'Restore each shader at the owning library-relative metalrobo/MetalRobo.metallib path, pin its Git bytes, and verify actual loaded dylib path. Repeat only affected physical calibration/clamp checks; earlier explicit-URL grids and sweeps are retained.'})
# Existing direct-URL probes were isolated: compare their complete outputs again.
exact('v1-new.bin','v1/expected.bin');exact('v2-new.bin','v2/expected.bin');exact('queries-new.bin','v2/query-expected.bin')
for name in ('clamp','open','zero-friction','refinement'):exact('sweep-'+name+'.bin','input/'+name+'-expected.bin')
N='.numi/retained-runtime/beak-hold-ccd-fix-20261108';E='build/crow-ccd-event-status-build-01'
sources={}
with a.Objects() as obj:
    for version,shader in [('old',N+'/MetalRobo.metallib'),('event',E+'/candidate.metallib')]:
        for target,src in [('libmetalrobo.dylib',N+'/libmetalrobo.dylib'),('metalrobo/MetalRobo.metallib',shader)]:
            p=T/('runtime-'+version)/target;p.parent.mkdir(parents=True,exist_ok=True)
            f=mapping[src]
            with p.open('xb') as out:
                for block in a.content(f,entries,obj):out.write(block)
            p.chmod(f['mode']);assert a.file_hash(p)==f['sha256']
            sources[str(p.relative_to(T))]={'source':src,'sha256':f['sha256']}
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env['DYLD_PRINT_LIBRARIES']='1'
commands=[];loaded=[]
def run(name,cmd,expected):
    start=time.monotonic()
    with (T/(name+'.stdout')).open('xb') as out,(T/(name+'.stderr')).open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=180).returncode
    commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start})
    a.write_json(A/'recovery-continuation-execution.json',commands);assert code==expected,(name,code)
for version in ('old','event'):
    env['DYLD_LIBRARY_PATH']=str(T/('runtime-'+version))
    if version=='old':
        run('coinstalled-calibration',[T/'runtime/beak-probe','--calibrate'],0)
        exact('coinstalled-calibration.stdout','input/calibration.jsonl')
    run('coinstalled-'+version+'-clamp',[T/'runtime/beak-probe',T/('coinstalled-'+version+'-clamp'),'0','1','32'],1)
    for name in ('trace.tsv','failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure-contact-status.bin','failure.json'):
        exact('coinstalled-'+version+'-clamp/'+name,'input/'+version+'-'+name)
    lines=[x for x in (T/('coinstalled-'+version+'-clamp.stderr')).read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
    assert lines and all(x.endswith(str(T/('runtime-'+version)/'libmetalrobo.dylib')) for x in lines)
    loaded.extend(lines)
    for suffix in ('.stdout','.stderr'):shutil.copy2(T/('coinstalled-'+version+'-clamp'+suffix),A/('recovery-'+version+'-clamp'+suffix))
run('event-initializer',[T/'runtime/event-probe',T/'runtime/event.metallib',T/'event-new.bin','preserve'],0)
exact('event-new.bin','input/event-expected.bin')
inventory=[{'path':str(p.relative_to(R)),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),'mtime_ns':p.stat().st_mtime_ns}
    for p in sorted(T.rglob('*')) if p.is_file()]
opened=subprocess.run(['/usr/sbin/lsof','-t','+D',str(T)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert opened.returncode in (0,1) and not opened.stdout
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:
    p=R/f['path'];st=p.stat();assert (a.file_hash(p),st.st_size,stat.S_IMODE(st.st_mode),st.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
receipt={'status':'verified','actual_git_manifest_oid':entries[mp][1],'source_commit':a.git('rev-parse','HEAD').decode().strip(),
 'direct_url_probes_retained_from_attempt_1':['v1 candidate8192','v2 production8192','v2 query16','four captured sweeps257queries each'],
 'coinstalled_runtime_sources':sources,'actual_loader_lines':loaded,'continuation_commands':commands,
 'all_restored_inputs_sha256_verified':True,
 'checks':{'rejected_v1_candidate_8192_records_exact':True,'qualified_v2_production_8192_records_exact':True,'sixteen_native_queries_exact':True,
   'four_retained_real_ccd_replays_exact':True,'calibration_exact':True,'old_and_preserving_shader_clamp_traces_and_all_failure_files_exact':True,
   'preserving_initializer_complete_buffers_exact':True},
 'scope':'Two CCD candidate revisions, direct real-pair replays, and the physical clamp under both actual Git-restored co-installed runtimes. Not every trial arm or complete beak manipulation.',
 'temporary_cleanup':{'files':inventory,'open_handles':0,'df_before':before,'removed_bytes':sum(f['bytes'] for f in inventory)}}
a.write_json(A/'git-recovery.json',receipt)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in T.rglob('*') if p.is_dir()),key=lambda x:len(x.parts),reverse=True):p.rmdir()
T.rmdir();receipt['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True))
a.write_json(A/'git-recovery.json',receipt)
print(json.dumps({'status':'verified','temporary_removed_bytes':receipt['temporary_cleanup']['removed_bytes'],'coinstalled_runtime_sources':sources}),flush=True)
