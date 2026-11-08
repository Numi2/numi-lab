from pathlib import Path
import csv,hashlib,json,os,shutil,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-representability-20261108';B=R/'build/crow-quality-representability-build-01'
T0=R/'.numi/retained-runtime/quality-representability-base-20261108';T1=R/'.numi/retained-runtime/quality-adjacent-refinement-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
if sys.argv[1]=='prepare':
    sys.path.insert(0,str(R/'tools'))
    from crow_evidence_archive import tree,Objects,content
    entries=tree();ref=B/'reference';ref.mkdir()
    manifest='artifacts/crow-rl-quality-first-failure-20261108/payload/manifest.json'
    with Objects() as objects:
        m=json.loads(objects.read(entries[manifest][1]))
        for name in ['crow-fixture','fixture-default']:
            match=[f for f in m['files'] if f['path'].endswith('/reference/build/crow-native-drive-ir-build-01/'+name)]
            assert len(match)==1,(name,len(match));f=match[0]
            data=b''.join(content(f,entries,objects));assert hashlib.sha256(data).hexdigest()==f['sha256']
            p=ref/name;p.write_bytes(data);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
    raw=(B/'capture-one.bin').read_bytes();d=struct.unpack_from('<16I8f',raw,32);nv,rows,blocks=d[2:5]
    free_offset=256+blocks*128+4*(nv*nv+rows*nv+rows)
    (B/'original-warm-v.f32').write_bytes(raw[free_offset+nv*4:free_offset+nv*8])
    variants=[]
    for factor in [.81,.93,1.07,1.19]:
        variant=bytearray(raw);free=np.frombuffer(raw,dtype='<f4',count=nv,offset=free_offset)
        variant[free_offset:free_offset+nv*4]=(free*factor).astype('<f4').tobytes()
        name='free-'+str(factor).replace('.','p');p=B/(name+'.bin');p.write_bytes(variant)
        variants.append(dict(name=name,factor=factor,path=str(p.relative_to(R)),sha256=sha(p)))
    files=[A/n for n in ['native_trial.py','apply_refinement.py','adjacent_neighbor.metal.inc','adjacent_refinement.metal.inc']]
    files += [R/n for n in ['include/metalrobo/unified_quality_shared.h','src/metal/MetalUnifiedQualitySolver.metal','src/metal/MetalUnifiedQualitySolver.mm','src/metal/MetalWorld.mm']]
    files += [T0/'libmetalrobo.dylib',T0/'metalrobo/MetalRobo.metallib',B/'reference/crow-fixture',B/'reference/fixture-default',B/'capture-one.bin']
    plan=dict(schema='crow-quality-adjacent-native-trial-plan-v1',purpose='Bounded native implementation and engineering validation of adjacent-FP32 stationarity refinement',
      implementation=dict(maximum_velocities=64,maximum_refinements=8,trigger='Completed Newton budget, valid feasible state, scaled residual <=16 times unchanged optimality tolerance, full Hessian scratch available',candidate='All +/- one-ULP singles and distinct-coordinate pairs; generalized-Hessian ranking distributed over SIMD32; fully reproject winner; accept only strict native residual improvement, existing feasibility gate and tight objective roundoff bound',default_path='Already-converged problems do not execute refinement; insufficient legacy scratch safely skips it',excluded='No host physics, crow-specific index rules, tolerance/iteration/configuration changes, padding fix, learner or policy promotion'),
      predictions=dict(original_corrected_status=0,original_corrected_independent_residual_max=1e-10,legacy_scratch_status=7,legacy_capture_exact=True,finite_native_physical_rows_exact=576,zero_native_physical_rows_exact=48,default_physical_files_exact=10,variants_corrected_status=0,crow='At least the formerly failing problem advances; completion is a measured outcome, not assumed'),
      numerical_variants=variants,inputs={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},
      controls=['Old runtime with original and full Hessian scratch','Corrected shader with legacy one-float scratch','Four predeclared free-velocity variants, old/corrected shaders','Old/corrected 576 motor and 48 zero-gain cases','Old/corrected default 640 transitions','Actual coupled crow, chunk 1 and 8','Constraint IR and native contact owner checks'],
      budgets=dict(native_command_seconds=120,build_seconds=300,raw_bytes_max=100000000,crow_lanes=4,crow_control_ticks=160,crow_substeps=32),
      stop='Stop unexpected host error, GPU error, malformed record, failed build or unsafe resource/binding result. Preserve ordinary solver rejection, record failed predictions and continue finite controls; no automatic setting relaxation.',
      limits='One exposed crow fixture plus numerical variants and existing regressions. No ecological held-out behavior, external validation, or whole-environment qualification.')
    write(A/'native-trial-plan.json',plan);print(json.dumps(plan));sys.exit(0)
assert sys.argv[1]=='run'
plan=json.loads((A/'native-trial-plan.json').read_text())
out=B/'native-trial-01';out.mkdir();records=[]
def run(name,command,runtime=None,expected=0,capture=None,timeout=120,library=False):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ['METALROBO_QUALITY_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
    if runtime:
        env.update(DYLD_LIBRARY_PATH=str(runtime),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(runtime/'metalrobo/MetalRobo.metallib'))
    if capture:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(capture)
    t=time.monotonic()
    with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
        try:p=subprocess.run([str(x) for x in command],env=env,cwd=R,stdout=stdout,stderr=stderr,timeout=timeout);code=p.returncode
        except subprocess.TimeoutExpired:code=None
    rec=dict(name=name,argv=[str(x) for x in command],returncode=code,seconds=time.monotonic()-t,runtime=str(runtime) if runtime else None)
    if library:
        lines=[s for s in (out/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
        rec['loader_lines']=lines
        assert lines and all(s.endswith(str(runtime/'libmetalrobo.dylib')) for s in lines),rec
    records.append(rec);write(out/'execution.json',records);print(json.dumps(rec),flush=True)
    assert code in expected if isinstance(expected,list) else code==expected,rec
    return (out/(name+'.stdout')).read_text()
run('full-release-build',['cmake','--build','build','--config','Release','-j','6'],timeout=300)
T1.mkdir();(T1/'metalrobo').mkdir()
for src,dst in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/bin/metalrobo_motor_contact_probe','probe'),('build/bin/metalrobo_constraint_ir_probe','ir-probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe')]:shutil.copy2(R/src,T1/dst)
source=(A/'fp64_reference.py').read_text().split('initial=evaluate(failed)')[0].replace("B=R/'build/crow-quality-first-failure-build-01/repair-01'","B=R/'build/crow-quality-representability-build-01'")
ns={};exec(compile(source,'retained-independent-fp64','exec'),ns)
probe=B/'warm-validation-01/warm-probe';numerical=[]
for name,runtime,case,scratch in [('old-expanded',T0,B/'capture-one.bin',True),('corrected-legacy',T1,B/'capture-one.bin',False),('corrected-original',T1,B/'capture-one.bin',True)]+[(runtime_name+'-'+v['name'],runtime,R/v['path'],True) for v in plan['numerical_variants'] for runtime_name,runtime in [('old',T0),('corrected',T1)]]:
    cmd=[probe,case,runtime/'metalrobo/MetalRobo.metallib',out/name]
    if scratch:cmd += [B/'original-warm-v.f32','--polish-scratch']
    result=json.loads(run(name,cmd));result['case']=name
    if result['code']==0:
        raw=case.read_bytes();nv=ns['nv'];rows=ns['rows'];blocks=len(ns['block']);free_offset=256+blocks*128+4*(nv*nv+rows*nv+rows)
        old_free=ns['free'].copy();ns['free'][:]=np.frombuffer(raw,dtype='<f4',count=nv,offset=free_offset)
        velocity=np.fromfile(out/name/'velocity.f32',dtype='<f4').astype(float);val=ns['evaluate'](velocity)
        impulses=np.fromfile(out/name/'impulse.f32',dtype='<f4').astype(float)
        result.update(independent_scaled_optimality=val[-1],independent_impulse_max_difference=float(np.max(np.abs(impulses-val[2]))))
        ns['free'][:]=old_free
    if name=='corrected-legacy':result['capture_exact']=(out/name/'capture.bin').read_bytes()==case.read_bytes()
    numerical.append(result);write(A/'native-numerical-result.json',numerical)
physical=[]
for kind,option,count in [('motor','--native-drive',576),('zero','--zero-gain',48)]:
    for name,runtime in [('old',T0),('corrected',T1)]:
        run(name+'-'+kind,[runtime/'probe',runtime/'metalrobo/MetalRobo.metallib',out/(name+'-'+kind+'.tsv'),option],runtime=runtime,library=True)
    old=list(csv.DictReader((out/('old-'+kind+'.tsv')).open(),delimiter='\t'));new=list(csv.DictReader((out/('corrected-'+kind+'.tsv')).open(),delimiter='\t'))
    assert len(old)==len(new)==count
    exact=[{k:v for k,v in row.items() if k!='gpu_ms'} for row in old]==[{k:v for k,v in row.items() if k!='gpu_ms'} for row in new]
    physical.append(dict(case=kind,records=count,physical_exact=exact,excluded_columns=['gpu_ms']));write(A/'native-physical-result.json',physical)
for name,runtime in [('old',T0),('corrected',T1)]:
    run(name+'-default',[B/'reference/fixture-default',out/(name+'-default'),'0','1','32'],runtime=runtime,library=True)
default=[]
for p in sorted((out/'old-default').iterdir()):
    if p.name!='runtime.json' and p.is_file():default.append(dict(name=p.name,byte_exact=p.read_bytes()==(out/'corrected-default'/p.name).read_bytes(),sha256=sha(p)))
write(A/'default-physical-result.json',default)
for name,runtime,chunk in [('old-crow',T0,1),('corrected-crow-one',T1,1),('corrected-crow-eight',T1,8)]:
    run(name,[B/'reference/crow-fixture',out/name,'0',str(chunk),'32','2'],runtime=runtime,expected=[0,1],capture=out/(name+'.bin'),library=True)
run('ir-owner',[T1/'ir-probe'],runtime=T1,library=True)
run('contact-owner',[T1/'contact-probe',T1/'metalrobo/MetalRobo.metallib'],runtime=T1,library=True)
write(A/'native-trial-completion.json',dict(executed_commands=len(records),runtime={str(p.relative_to(R)):sha(p) for p in T1.rglob('*') if p.is_file()},raw_bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),policy_or_profile_promoted=False,goal_complete=False))
