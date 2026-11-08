"""Execute only actual Git-restored native evidence; bound and record temporary cleanup."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,stat,time
R=Path.cwd();A=Path('artifacts/crow-rl-beak-hold-capture-20261108')
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
entries=a.tree();m=json.loads((A/'payload/manifest.json').read_text());mapping=json.loads((A/'runtime-recovery.json').read_text())
T=Path('build/crow-beak-hold-git-recovery-01');T.mkdir(exist_ok=False)
sources={f'runtime/{n}':v['archive_source'] for n,v in mapping['native'].items()}
for n in ('clamp','open','zero-friction','refinement'):
    for suffix in ('.ccdsweep','-replay.bin','-replay.json'):
        sources[f'input/{n}{suffix}']=f'build/crow-beak-hold-build-01/capture-{n}{suffix}'
sources['input/clamp-trace.tsv']='build/crow-beak-hold-build-01/capture-clamp/trace.tsv'
sources['input/calibration.jsonl']='build/crow-beak-hold-build-01/calibration-v5.jsonl'
with a.Objects() as objects:
    for name,src in sources.items():
        f=next(f for f in m['files'] if f['path']==src);p=T/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as out:
            for block in a.content(f,entries,objects):out.write(block)
        p.chmod(f['mode']);assert a.file_hash(p)==f['sha256']
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str((T/'runtime').resolve()),DYLD_PRINT_LIBRARIES='1',
    METALROBO_METALLIB=str((T/'runtime/MetalRobo.metallib').resolve()),
    METALROBO_CCD_FAILURE_CAPTURE=str((T/'new-failure.ccdsweep').resolve()))
def execute(name,cmd,timeout):
    with (T/f'{name}.stdout').open('xb') as out,(T/f'{name}.stderr').open('xb') as err:
        code=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=timeout).returncode
    return code
assert execute('calibration',[str((T/'runtime/probe').resolve()),'--calibrate'],30)==0
assert (T/'calibration.stdout').read_bytes()==(T/'input/calibration.jsonl').read_bytes()
assert execute('clamp',[str((T/'runtime/probe').resolve()),str((T/'clamp').resolve()),'0','1','32'],180)==1
assert (T/'clamp/trace.tsv').read_bytes()==(T/'input/clamp-trace.tsv').read_bytes()
assert (T/'new-failure.ccdsweep').read_bytes()==(T/'input/clamp.ccdsweep').read_bytes()
for n in ('clamp','open','zero-friction','refinement'):
    cmd=[str((T/'runtime/sweep-probe').resolve()),str((T/'runtime/MetalRobo.metallib').resolve()),
         str((T/'runtime/CCDSweepProbe.metallib').resolve()),str((T/f'input/{n}.ccdsweep').resolve()),str((T/f'{n}.bin').resolve())]
    assert execute(f'replay-{n}',cmd,60)==0
    assert (T/f'{n}.bin').read_bytes()==(T/f'input/{n}-replay.bin').read_bytes()
    assert json.loads((T/f'replay-{n}.stdout').read_text())==json.loads((T/f'input/{n}-replay.json').read_text())
loaded=[s for s in (T/'clamp.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
assert loaded and all(s.endswith(str((T/'runtime/libmetalrobo.dylib').resolve())) for s in loaded)
files=[{'path':str(p),'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode),
        'mtime_ns':p.stat().st_mtime_ns} for p in sorted(T.rglob('*')) if p.is_file()]
result=subprocess.run(['/usr/sbin/lsof','-t','+D',str(T)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert result.returncode in (0,1) and not result.stdout
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in files:
    p=Path(f['path']);s=p.stat();assert (a.file_hash(p),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
receipt={'status':'verified','actual_committed_git_bytes':True,'calibration_exact':True,
    'physical_clamp_accepted_trace_and_protected_failure_exact':True,'all_four_isolated_replays_byte_exact':True,
    'actual_loader_lines':loaded,'restored_sources':sources,'temporary_cleanup':{'files':files,'open_handles':0,'df_before':before},
    'scope':'One entire failing clamp trajectory and four isolated CCD sweeps; not every parent arm or a successful manipulation capability.'}
a.write_json(A/'verification.json',receipt)
for f in files:Path(f['path']).unlink()
for p in sorted((p for p in T.rglob('*') if p.is_dir()),key=lambda x:len(x.parts),reverse=True):p.rmdir()
T.rmdir();receipt['temporary_cleanup']['df_after']=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
receipt['temporary_cleanup']['removed_bytes']=sum(f['bytes'] for f in files);receipt['temporary_cleanup']['status']='removed'
a.write_json(A/'verification.json',receipt)
print(json.dumps({'status':'verified','temporary_removed_bytes':receipt['temporary_cleanup']['removed_bytes']}))
