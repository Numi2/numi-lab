from pathlib import Path
import csv,gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-quality-representability-20261108';A=R/'artifacts/crow-rl-quality-representability-20261108';X=R/'build/crow-quality-representability-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((G/'plan.json').read_text());entries=tree()
with Objects() as objects:
    raw=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw)
    assert m['summary']['source_bytes']<=plan['restore_max_bytes'];X.mkdir()
    for f in m['files']:
        p=X/f['path'];p.parent.mkdir(parents=True,exist_ok=True);data=b''.join(content(f,entries,objects))
        assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];p.write_bytes(data);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
    for name in ['fp64_reference.py','explore.py','failed_start_explore.py','substep7_reference.py']:
        p=X/A.relative_to(R)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(objects.read(entries[str(A.relative_to(R)/name)][1]))
print(json.dumps(dict(restored_files=len(m['files']),restored_bytes=m['summary']['source_bytes'])),flush=True)
B=X/'build/crow-quality-representability-build-01';N=B/'native-trial-01';Y=X/'generated';Y.mkdir()
T0=X/'.numi/retained-runtime/quality-representability-base-20261108';T1=X/'.numi/retained-runtime/quality-adjacent-refinement-20261108';commands=[];checks=[]
def run(name,args,runtime=None,expected=0,capture=None,cwd=None):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ['METALROBO_QUALITY_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
    if runtime:env.update(DYLD_LIBRARY_PATH=str(runtime),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(runtime/'metalrobo/MetalRobo.metallib'))
    if capture:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(capture)
    start=time.monotonic()
    with (Y/(name+'.stdout')).open('xb') as out,(Y/(name+'.stderr')).open('xb') as err:
        try:code=subprocess.run([str(a) for a in args],env=env,stdout=out,stderr=err,cwd=cwd or R,timeout=120).returncode
        except subprocess.TimeoutExpired:code=None
    rec=dict(name=name,argv=[str(a) for a in args],returncode=code,expected_exit_code=expected,seconds=time.monotonic()-start)
    if runtime:
        lines=[s for s in (Y/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];rec['loader_lines']=lines
        assert lines and all(s.endswith(str(runtime/'libmetalrobo.dylib')) for s in lines),rec
    commands.append(rec);write(G/'execution.json',commands);print(json.dumps(rec),flush=True);assert code==expected,rec
def exact(actual,reference):
    rec=dict(actual=str(actual.relative_to(X)),reference=str(reference.relative_to(X)),byte_exact=actual.read_bytes()==reference.read_bytes(),sha256=sha(actual));checks.append(rec);write(G/'comparisons.json',checks);assert rec['byte_exact'],rec
def directory_exact(actual,reference,exclude=()):
    for p in sorted(reference.iterdir()):
        if p.is_file() and p.name not in exclude:exact(actual/p.name,p)
probe=B/'warm-validation-01/warm-probe'
cases=[('old-expanded',T0,B/'capture-one.bin',True),('corrected-legacy',T1,B/'capture-one.bin',False),('corrected-original',T1,B/'capture-one.bin',True)]
cases += [(prefix+'-free-'+str(factor).replace('.','p'),runtime,B/('free-'+str(factor).replace('.','p')+'.bin'),True) for factor in [.81,.93,1.07,1.19] for prefix,runtime in [('old',T0),('corrected',T1)]]
for name,runtime,capture,scratch in cases:
    args=[probe,capture,runtime/'metalrobo/MetalRobo.metallib',Y/name]
    if scratch:args += [B/'original-warm-v.f32','--polish-scratch']
    run(name,args);directory_exact(Y/name,N/name)
run('warm-neighbor',[probe,B/'capture-one.bin',T0/'metalrobo/MetalRobo.metallib',Y/'warm-neighbor',B/'neighbor-v.f32'])
directory_exact(Y/'warm-neighbor',B/'warm-validation-01/neighbor')
tsv=[]
for name,option,count in [('motor','--native-drive',576),('zero','--zero-gain',48)]:
    run(name,[T1/'probe',T1/'metalrobo/MetalRobo.metallib',Y/(name+'.tsv'),option],T1)
    expected=list(csv.DictReader((N/('corrected-'+name+'.tsv')).open(),delimiter='\t'));actual=list(csv.DictReader((Y/(name+'.tsv')).open(),delimiter='\t'))
    assert len(expected)==len(actual)==count
    passed=[{k:v for k,v in a.items() if k!='gpu_ms'} for a in expected]==[{k:v for k,v in a.items() if k!='gpu_ms'} for a in actual]
    tsv.append(dict(case=name,physical_records_exact=count if passed else 0,excluded_columns=['gpu_ms']));write(G/'physical-row-comparison.json',tsv);assert passed
run('default',[B/'reference/fixture-default',Y/'default','0','1','32'],T1)
directory_exact(Y/'default',N/'corrected-default',exclude=['runtime.json'])
for name,runtime,chunk in [('old-crow',T0,1),('corrected-crow-one',T1,1),('corrected-crow-eight',T1,8)]:
    run(name,[B/'reference/crow-fixture',Y/name,'0',str(chunk),'32','2'],runtime,1,Y/(name+'.bin'))
    directory_exact(Y/name,N/name);exact(Y/(name+'.bin'),N/(name+'.bin'))
run('ir-owner',[T1/'ir-probe'],T1);run('contact-owner',[T1/'contact-probe',T1/'metalrobo/MetalRobo.metallib'],T1)
for script,result,outputs in [('explore.py','analysis.json',['neighbor-v.f32','neighbor-gradient.f64']),('failed_start_explore.py','failed-start-analysis.json',['failed-start-neighbor-v.f32','failed-start-gradient.f64']),('substep7_reference.py','substep7-fp64-analysis.json',['substep7-fp64-reference-v.f64','substep7-fp64-reference-impulse.f64','substep7-fp64-reference-gradient.f64'])]:
    expected={name:(B/name).read_bytes() for name in outputs}
    run('analysis-'+script,[R/'python/.venv/bin/python',X/A.relative_to(R)/script],cwd=X)
    for name,data in expected.items():assert (B/name).read_bytes()==data
    with Objects() as objects:old=json.loads(objects.read(entries[str(A.relative_to(R)/result)][1]))
    new=json.loads((X/A.relative_to(R)/result).read_text());old.pop('seconds',None);new.pop('seconds',None);assert old==new,(script,old,new)
    checks.append(dict(analysis=script,result_fields_exact=True,excluded_fields=['seconds'] if script!='substep7_reference.py' else [],exact_array_files=len(outputs)));write(G/'comparisons.json',checks)
known={(f['sha256'],f['bytes']):f['path'] for f in m['files']};inventory=[];(G/'recovery-blobs').mkdir()
for p in sorted(X.rglob('*')):
    if not p.is_file():continue
    s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink();h=sha(p)
    f=dict(path=str(p.relative_to(R)),bytes=s.st_size,sha256=h,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns)
    if (h,s.st_size) in known:f['archive_source']=known[(h,s.st_size)]
    else:
        dest=G/'recovery-blobs'/(h+'.gz')
        if not dest.exists():dest.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
        f['recovery_blob']=str(dest.relative_to(R))
    inventory.append(f)
assert sum(f['bytes'] for f in inventory)<=plan['temporary_budget_bytes'];write(G/'temporary-inventory.json',inventory)
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(X)],capture_output=True,text=True,timeout=60)
write(G/'temporary-open-handles.json',dict(exit_code=handles.returncode,stdout=handles.stdout,stderr=handles.stderr));assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
report=dict(manifest=plan['manifest'],manifest_sha256=plan['manifest_sha256'],restored_files=len(m['files']),restored_bytes=m['summary']['source_bytes'],executed_commands=len(commands),native_commands=20,analysis_commands=3,comparison_records=len(checks),all_comparisons_passed=True,physical_rows_exact=624,temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),new_recovery_blobs=len(list((G/'recovery-blobs').glob('*.gz'))),open_handles=0,policy_or_profile_promoted=False,goal_complete=False,limits='Executed recovery repeats existing numerical and physical observations, including failures. Compiler output is restored and hash verified; no rebuild or fresh ecological qualification is claimed.')
assert len(commands)==23
write(G/'verification.json',report);print(json.dumps(report),flush=True)
