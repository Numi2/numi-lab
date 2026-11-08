from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,time
import numpy as np
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-128-integration-20261108'
T=R/'.numi/retained-runtime/ccd-event-capture-20261108/integrated128'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'])
for p,h in json.loads((A/'plan.json').read_text())['inputs'].items():assert sha(R/p)==h,p
(T/'metalrobo').mkdir(parents=True,exist_ok=False)
for src,dst in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/shaders/CCDSweepProbe.metallib','CCDSweepProbe.metallib'),('build/bin/metalrobo_ccd_sweep_probe','sweep-probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe'),('build/bin/metalrobo_crow_object_push_check','object-push-check')]:shutil.copy2(R/src,T/dst)
commands=[]
def execute(name,cmd,expected=0,env=None):
    start=time.monotonic()
    with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=180).returncode
    commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start});write(B/'integration-execution.json',commands);print(json.dumps(commands[-1]),flush=True)
    if expected is not None:assert code==expected,(name,code)
    return code
execute('compile128',['clang++','-std=c++20','-O3','-fno-fast-math','-fobjc-arc','-I',R/'include',B/'fixture128.mm',T/'libmetalrobo.dylib','-Wl,-rpath,@loader_path/runtime','-framework','Foundation','-o',B/'probe128'])
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
execute('calibration128',[B/'probe128','--calibrate'],env=env)
assert (B/'calibration128.stdout').read_bytes()==(B/'calibrate-unused-poses.stdout').read_bytes()
execute('integrated-projection',[B/'projection-probe',T/'metalrobo/MetalRobo.metallib',B/'integrated-projection.bin'])
assert (B/'integrated-projection.bin').read_bytes()==(B/'projection-candidate.bin').read_bytes()
execute('integrated-grid',[B/'cases-probe',T/'metalrobo/MetalRobo.metallib',B/'cases.bin',B/'integrated-grid.bin'])
assert (B/'integrated-grid.bin').read_bytes()==(B/'expected-grid.bin').read_bytes()
execute('integrated-contact',[T/'contact-probe'],env=env)
for name,sub,reference in [('integrated32',32,'clamp'),('integrated64',64,'refinement')]:
    execute(name,[B/'probe128',B/name,'0','1',str(sub)],env=env)
    for file in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):
        assert (B/name/file).read_bytes()==(B/reference/file).read_bytes(),(name,file)
    lines=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    assert lines and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in lines)
env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/'admitted128.ccdsweep')
code=execute('admitted128',[B/'probe128',B/'admitted128','0','1','128'],None,env)
report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'projection_and4096_grid_exact':True,
 'host_invalid0_and129_and_GPU128_boundary_passed':True,'legacy32_and64_complete_trace_final_states_scene_trajectory_and_summary_exact':True,
 'native_contact_regression_passed':True,'admitted128_exit_code':code,'commands':commands,
 'runtime':{str(p.relative_to(R)):sha(p) for p in T.rglob('*') if p.is_file()},'protected_profile_promoted':False}
if code==0:
    def rows(name):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'trace.tsv').open(),delimiter='\t')]
    coarse=rows('refinement');fine=rows('admitted128');assert len(coarse)==len(fine)==640
    error=np.array([sum((c[k]-f[k])**2 for k in 'xyz')**.5 for c,f in zip(coarse,fine)])
    initial=json.loads((B/'admitted128/initial-geometry.json').read_text());lanes=[]
    for lane in range(4):
        q=[r for r in fine if r['lane']==lane];hold=[r for r in q if 64<=r['tick']<=95];release=[r for r in q if r['tick']>=128];anchor=next(r for r in q if r['tick']==31)
        m={'minimum_lift_m':min(r['z']-initial[lane]['position'][2] for r in hold),'maximum_slip_m':max(sum((r['rel_'+a]-anchor['rel_'+a])**2 for a in 'xyz')**.5 for r in hold),
          'opposed_samples':sum(r['upper_Jn']>1e-9 and r['lower_Jn']>1e-9 for r in hold),'maximum_penetration_before96_m':max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in q if r['tick']<96)),
          'release_drop_m':min(r['z'] for r in hold)-q[-1]['z'],'minimum_release_gap_m':min(min(r['upper_gap_m'],r['lower_gap_m']) for r in release)}
        m['mechanical_gates_pass']=m['minimum_lift_m']>=.02 and m['maximum_slip_m']<=.005 and m['opposed_samples']>=16 and m['maximum_penetration_before96_m']<=.0008 and m['release_drop_m']>=.03 and m['minimum_release_gap_m']>=.001
        lanes.append(m)
    phases={f'{lo}:{hi}':float(max(e for e,r in zip(error,coarse) if lo<=r['tick']<=hi)) for lo,hi in ((0,31),(32,63),(64,95),(96,127),(128,159))}
    report.update(all640_transitions_accepted=True,maximum_64vs128_position_error_m=float(error.max()),phase_errors_m=phases,
      refinement5mm_pass=bool(error.max()<=.005),fine_lanes=lanes,all_fine_mechanical_gates_pass=all(m['mechanical_gates_pass'] for m in lanes))
else:report['fine_failure']=json.loads((B/'admitted128/failure.json').read_text()) if (B/'admitted128/failure.json').exists() else 'No physical trace; inspect typed stderr.'
write(A/'results.json',report)
env.pop('METALROBO_CCD_FAILURE_CAPTURE',None)
execute('integrated-object-push',[T/'object-push-check',B/'integrated-object-push'],env=env)
report.update(native_object_push_regression_passed=True,commands=commands)
write(A/'results.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('commands','runtime')}),flush=True)
