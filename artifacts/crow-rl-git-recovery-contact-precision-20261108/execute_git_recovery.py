from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-contact-precision-20261108';A=R/'artifacts/crow-rl-contact-precision-20261108';X=R/'build/crow-contact-precision-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((G/'plan.json').read_text());entries=tree()
with Objects() as objects:
    raw=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);assert m['summary']['source_bytes']<=plan['restore_max_bytes'];X.mkdir()
    for f in m['files']:
        p=X/f['path'];p.parent.mkdir(parents=True,exist_ok=True);data=b''.join(content(f,entries,objects));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];p.write_bytes(data);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
    for name,(mode,oid) in entries.items():
        if name.startswith(str(A.relative_to(R))+'/') and '/payload/' not in name:
            p=X/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(objects.read(oid));p.chmod(int(mode,8)&0o777)
print(json.dumps(dict(restored_files=len(m['files']),restored_bytes=m['summary']['source_bytes'])),flush=True)
B=X/'build/crow-contact-precision-build-01';T=X/'.numi/retained-runtime/contact-precision-20261108';Y=X/'generated';Y.mkdir();commands=[];checks=[]
def run(name,args,cwd):
    start=time.monotonic();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with (Y/(name+'.stdout')).open('xb') as out,(Y/(name+'.stderr')).open('xb') as err:
        try:code=subprocess.run([str(a) for a in args],cwd=cwd,env=env,stdout=out,stderr=err,timeout=120).returncode
        except subprocess.TimeoutExpired:code=None
    rec=dict(name=name,argv=[str(a) for a in args],returncode=code,seconds=time.monotonic()-start);commands.append(rec);write(G/'execution.json',commands);print(json.dumps(rec),flush=True);assert code==0,rec
def strip_seconds(value):
    if isinstance(value,dict):return {k:strip_seconds(v) for k,v in value.items() if k!='seconds'}
    if isinstance(value,list):return [strip_seconds(v) for v in value]
    return value
analyses=[('diagnose.py','diagnosis.json'),('block_search.py','block-search-result.json'),('lattice_search.py','lattice-search-result.json'),('lattice_certificate.py','lattice-certificate.json'),('chebyshev_search.py','chebyshev-search-result.json'),('dual_mix.py','mixed-dual-certificate.json'),('verify_certificate.py','certificate-verification.json'),('prove_local_box.py','local-box-proof.json'),('summarize.py','study-result.json')]
for script,result in analyses:
    run('analysis-'+Path(script).stem,[R/'python/.venv/bin/python',X/A.relative_to(R)/script],X)
    with Objects() as objects:expected=json.loads(objects.read(entries[str(A.relative_to(R)/result)][1]))
    actual=json.loads((X/A.relative_to(R)/result).read_text());assert strip_seconds(actual)==strip_seconds(expected),script
    checks.append(dict(script=script,report=result,scientific_fields_exact=True,excluded_fields=['seconds']));write(G/'comparisons.json',checks)
def exact(actual,expected):
    passed=actual.read_bytes()==expected.read_bytes();checks.append(dict(actual=str(actual.relative_to(X)),expected=str(expected.relative_to(X)),byte_exact=passed,sha256=sha(actual)));write(G/'comparisons.json',checks);assert passed
for name,capture,warm in [('original',B/'substep7-capture.bin',None),('rounded',B/'substep7-capture.bin',B/'rounded-reference-v.f32'),('chebyshev',B/'substep7-capture.bin',B/'chebyshev-v.f32'),('control',B/'control-substep2.bin',B/'control-neighbor-v.f32')]:
    cmd=[B/'warm-probe',capture,T/'MetalRobo.metallib',Y/name]
    if warm:cmd.append(warm)
    run('native-'+name,cmd,R)
    for p in sorted((B/'native-01'/name).iterdir()):exact(Y/name/p.name,p)
known={(f['sha256'],f['bytes']):f['path'] for f in m['files']};inventory=[];(G/'recovery-blobs').mkdir()
for p in sorted(X.rglob('*')):
    if not p.is_file():continue
    s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink();h=sha(p);f=dict(path=str(p.relative_to(R)),bytes=s.st_size,sha256=h,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns)
    if (h,s.st_size) in known:f['archive_source']=known[(h,s.st_size)]
    else:
        dest=G/'recovery-blobs'/(h+'.gz')
        if not dest.exists():dest.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
        f['recovery_blob']=str(dest.relative_to(R))
    inventory.append(f)
assert len(commands)==13 and sum(f['bytes'] for f in inventory)<=plan['temporary_budget_bytes'];write(G/'temporary-inventory.json',inventory)
report=dict(schema='crow-contact-precision-executed-git-recovery-v1',manifest=plan['manifest'],manifest_sha256=plan['manifest_sha256'],restored_files=len(m['files']),restored_bytes=m['summary']['source_bytes'],native_commands=4,scientific_analysis_commands=8,report_audit_commands=1,comparison_records=len(checks),all_scientific_and_defined_bytes_match=True,temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),new_recovery_blobs=len(list((G/'recovery-blobs').glob('*.gz'))),closure_handle_check='Pending separate process after this executor exits; no cleanup authorized by this report alone',scope='Exact replay of retained numerical/native observations and local proof, not new ecological evidence or whole-system qualification.',policy_or_profile_promoted=False,goal_complete=False)
write(G/'verification.json',report);print(json.dumps(report),flush=True)
