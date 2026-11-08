"""Bounded first-event capture and exact replay, not a new motor trial."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
from oracle import cast,rotate,calibrate
R=Path.cwd();B=R/'build/crow-ccd-event-capture-build-01';A=R/'artifacts/crow-rl-ccd-event-capture-20261108'
T=R/'.numi/retained-runtime/ccd-event-capture-20261108/candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def seal():
    assert not subprocess.check_output(['git','status','--porcelain'])
    (T/'metalrobo').mkdir(parents=True,exist_ok=False)
    for src,dst in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/shaders/CCDSweepProbe.metallib','CCDSweepProbe.metallib'),('build/bin/metalrobo_ccd_sweep_probe','sweep-probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe')]:shutil.copy2(R/src,T/dst)
    write(A/'calibration.json',calibrate())
    inputs=[p for p in B.rglob('*') if p.is_file() and p.suffix not in ('.stdout','.stderr') and '__pycache__' not in p.parts]
    inputs+=[p for p in T.rglob('*') if p.is_file()]
    inputs +=[R/p for p in ('src/metal/Collision.metal','src/metal/CCDFailureCapture.inc','apps/ccd_sweep_probe.mm','include/metalrobo/engine_types.h')]
    write(A/'plan.json',{'purpose':'Exact replication and instrument qualification of retained event-budget failures. No physics modification, learning or capability promotion.',
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'prediction':{'unchanged_trace_rollback_world_stamp_and_available_contact_status':True,'successful_open_control_exact':True,
        'old8192_grid_and_old_unresolved_sweep_exact':True,'selected_event_capture_replays_first_record_exact':True,
        'hypothesis_not_admission_gate':'Selected lower time is zero despite a positive upper impact time, repeatedly reusing an overly coarse safe endpoint.',
        'fixed_box_only_FP64_vs_257_native_gap_queries_max_error_m':5e-7},
      'units':'Exact retained4-lane clamp32 and refinement64 failures, open32 control; diagnostic replications, not independent new conditions.',
      'trials':[{'name':'clamp','arm':0,'chunk':1,'substeps':32,'expected_exit':1},{'name':'refinement','arm':0,'chunk':1,'substeps':64,'expected_exit':1},{'name':'open','arm':1,'chunk':1,'substeps':32,'expected_exit':0}],
      'runtime_selection':'Native dylib explicitly loaded by DYLD_LIBRARY_PATH; its co-installed metalrobo/MetalRobo.metallib is authoritative. METALROBO_METALLIB is not used.',
      'inputs':{str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},
      'budget':{'physical_executions':3,'pair_replays':3,'grid_executions':1,'contact_regressions':1,'timeout_seconds_each':180,'new_raw_max_bytes':33554432,
        'stop':'One execution per condition, no retry or physics tuning. Retain mismatches, missing captures and negative hypothesis. Further physics revision requires a new sealed plan and unused confirmation conditions.'},
      'analysis':'Compare every predeclared byte control before using a selected-event capture. Classify literal quaternion endpoints; use the independently calibrated FP64 translating-sphere/AABB reference only for fixed box orientation. Keep native query records and all rejected observations. Capture failure cannot weaken physics admission.'})
    shutil.copy2(__file__,A/'study.py');shutil.copy2(B/'oracle.py',A/'oracle.py')
    print(json.dumps({'plan_sha256':sha(A/'plan.json')}),flush=True)
def run():
    assert not subprocess.check_output(['git','status','--porcelain'])
    plan=json.loads((A/'plan.json').read_text())
    for p,v in plan['inputs'].items():assert sha(R/p)==v['sha256'],p
    env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1')
    commands=[];results={}
    def execute(name,cmd,expected=0):
        start=time.monotonic()
        with (B/f'{name}.stdout').open('xb') as out,(B/f'{name}.stderr').open('xb') as err:
            code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=180).returncode
        commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start})
        write(B/'execution.json',commands);print(json.dumps(commands[-1]),flush=True);assert code==expected,(name,code)
    for trial in plan['trials']:
        name=trial['name'];env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'.ccdsweep'))
        execute(name,[B/'probe',B/name,str(trial['arm']),'1',str(trial['substeps'])],trial['expected_exit'])
        loader=[s for s in (B/f'{name}.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
        assert loader and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loader)
        expected=B/'expected'/name
        for p in expected.iterdir():assert p.read_bytes()==(B/name/p.name).read_bytes(),(name,p.name)
        results[name]={'available_physical_byte_controls_exact':True,'actual_loader_lines':loader,'capture_exists':(B/(name+'.ccdsweep')).exists()}
        assert (B/(name+'.ccdsweep')).exists()==(name!='open')
    env.pop('METALROBO_CCD_FAILURE_CAPTURE',None)
    for name in ('clamp','refinement','old'):
        execute('replay-'+name,[T/'sweep-probe',T/'metalrobo/MetalRobo.metallib',T/'CCDSweepProbe.metallib',B/(name+'.ccdsweep'),B/(name+'-replay.bin')])
        report=json.loads((B/f'replay-{name}.stdout').read_text());write(B/(name+'-replay.json'),report)
        if name=='old':
            continue
        assert report['record_source']=='protected-selected-event' and report['stored_record_matches'][0]
        record=report['stored_record'];results[name]['event']=report
        lower,upper=record['interval_distance'][:2]
        results[name]['coarse_zero_lower_hypothesis_supported']=lower==0 and upper>0
        shapes=report['shapes'];box=next(s for s in shapes if s['type']==2);sphere=next(s for s in shapes if s['type']==0)
        fixed=box['start']['rotation']==box['end']['rotation'];results[name]['fixed_box_orientation']=fixed
        if fixed:
            q=np.array(box['start']['rotation'],dtype=float);q/=np.linalg.norm(q);q[:3]*=-1
            start=np.array(sphere['start']['center_radius'][:3])-np.array(box['start']['center_radius'][:3])
            end=np.array(sphere['end']['center_radius'][:3])-np.array(box['end']['center_radius'][:3])
            p=rotate(q,start);d=rotate(q,end-start);half=np.array(box['dimensions'][:3]);radius=sphere['dimensions'][0];tolerance=report['ccd_parameters'][1]
            reference=cast(p,d,half,radius,tolerance)
            native=np.fromfile(B/(name+'-replay.bin'),dtype='<f4',offset=128).reshape(257,4)
            gaps=np.array([np.linalg.norm(np.maximum(np.abs(p+alpha*d)-half,0.))-radius for alpha in np.linspace(0,1,257)])
            error=float(np.max(np.abs(gaps-native[:,1])))
            results[name].update(FP64_reference=reference,native_query_max_absolute_error_m=error)
            assert error<=5e-7 and bool(np.all(native[:,2]==0))
    execute('grid',[B/'cases-probe',T/'metalrobo/MetalRobo.metallib',B/'cases.bin',B/'grid.bin'])
    assert (B/'grid.bin').read_bytes()==(B/'expected-grid.bin').read_bytes()
    assert (B/'old-replay.bin').read_bytes()==(B/'expected-old-replay.bin').read_bytes()
    execute('contact',[T/'contact-probe'])
    write(A/'results.json',{'instrument_qualified':True,'physics_changed':False,'physical_capability_qualified':False,
      'results':results,'grid8192_and_old_unresolved_replay_exact':True,'native_contact_regression_passed':True,'commands':commands,
      'next':'Use literal selected-event and native query evidence to derive an event-time repair; do not raise event limits.'})
    print(json.dumps({'instrument_qualified':True,'results':{k:{n:v for n,v in d.items() if n!='event'} for k,d in results.items()}}),flush=True)
{'seal':seal,'run':run}[sys.argv[1]]()
