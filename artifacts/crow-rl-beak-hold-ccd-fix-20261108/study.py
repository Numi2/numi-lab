"""New physical conditions and a geometry-motivated wider native jaw release."""
from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,sys,time
R=Path('/Users/n/Developer/numi-crow-rl-20261107');B=R/'build/crow-beak-hold-build-02';A=R/'artifacts/crow-rl-beak-hold-ccd-fix-20261108'
trials=[('clamp',0,1,32),('open',1,1,32),('no-contact',2,1,32),('zero-friction',3,1,32),('partition',0,8,32),('refinement',0,1,64)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def seal():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
    integration=json.loads((R/'artifacts/crow-rl-ccd-translation-no-floor-20261108/integration.json').read_text())
    assert integration['native_contact_regression_passed'] and integration['native_object_push_controls']['physics_rejections']==0
    assert [json.loads(s) for s in (B/'calibration-with-runtime.jsonl').read_text().splitlines()]==[{'calibration':'passed','fp64_gap_known_values':3,'rotated_frame_known_values':1},{'compiled_fixture_calibration':'passed','articulation_bodies':17,'scene_bodies_excluded':2}]
    A.mkdir(exist_ok=False)
    files=[B/'probe-with-runtime.mm',B/'probe-with-runtime',B/'calibration-with-runtime.jsonl',Path(__file__),
        R/'src/metal/Collision.metal',R/'build/lib/libmetalrobo.dylib',R/'build/shaders/MetalRobo.metallib']
    plan={'schema':'crow-native-beak-contact-unused-conditions-v3','purpose':'mechanical qualification on four unused initial object poses; predefined motor commands, no learning or biology',
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
      'parent':'artifacts/crow-rl-beak-hold-capture-20261108','physics_revision':'artifacts/crow-rl-ccd-translation-no-floor-20261108',
      'model_revision':'Use qualified translating sphere-box CCD correction. Keep crow geometry, rigid surrogate mass/radius/material and clamp/lift gesture. Open jaw to native action+0.6 at96 instead of returning to default0.23rad; the larger opening gives a forward-sloping lower beak under the held head pitch, predicting gravity-driven release. Remove unsupported direct-action world checkpoint request; retain actual physical end states and explicitly qualify no retinal-policy continuation.',
      'question':'Does physical clamp/lift/release survive four unused initial object poses, matched contact/friction controls, chunk partition and timestep refinement?',
      'parameters':{'sphere_radius_m':.004,'sphere_mass_kg':.00025,'default_jaw_rad':.23,'control_hz':50,'ticks':160,'lanes':4,'seed':2651511101,
        'initial_pose_design':'From(0.215,0,0.26685), x±0.00025 and z±0.00015 in a full2x2 grid; y is±0.00015 according to x sign. Initial gap checked independently for every lane.',
        'actions':'Pitch0 before32, -0.4 afterward. Clamp jaw-0.35 through95, +0.6 from96; open control jaw0 throughout. All18 other components0. Existing actuator dynamics, native effort limits and root support unchanged.',
        'ownership':'Only initial q/v/scene upload, then ordinary resident solved contacts and native action authority. No attachment, latch, teleport, gravity exemption, food ledger or reset.'},
      'prediction':{'each_positive_lane':{'minimum_lift_ticks64_95_m':.020,'maximum_head_relative_slip_from_tick31_m':.005,
          'opposed_normal_impulse_samples_ticks64_95_at_least':16,'release_drop_by_tick159_from_minimum_hold_height_m':.030,
          'both_jaw_gaps_ticks128_159_m_at_least':.001,'maximum_sampled_penetration_ticks0_95_m':.0008},
        'controls':'Open and disabled-contact arms fail the hold/lift criterion in every matched lane. Disabled contact reports exactly zero jaw/object impulses. Zero object friction is predicted to fail hold/lift; if it succeeds, friction-dependence is contradicted and geometric trapping must be assessed.',
        'chunk1_vs8_final_q_v_scene':'byte exact; no world checkpoint claim',
        '32_vs64_maximum_paired_object_position_difference_over_all160ticks_m':.005},
      'validity':['All six runs and3840transitions accepted with no physical termination or native rejection.','All four initial signed gaps>20micrometres; all trace values finite.',
          'Actual native dylib loader and exact shader/source identities pinned.','Four distinct deterministic physical initial conditions are not independent biological replicates.','Impulses sample the final microstep only; not integrated full-tick forces.'],
      'analysis':'Report every lane/arm and all failed predicates. Qualify the complete predefined mechanical gesture only if primary, causal controls and numerical checks all pass. No p-values; no selection or threshold changes. A failed validity gate is inconclusive; valid out-of-prediction outcomes are contradicted.',
      'trials':[{'name':n,'arm':a,'chunk':c,'substeps':s} for n,a,c,s in trials],
      'runtime_inputs':{str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files},
      'budget':{'runs':6,'transitions':3840,'timeout_seconds_each':180,'trial_output_maximum_bytes':33554432,
         'stop':'One execution per listed condition. Preserve failures and typed accepted rollback states. No retries, parameter tuning, learned policy promotion or production environment release. Further revision requires a new sealed plan and unused conditions.'},
      'limits':['Object starts between jaws: ground pickup/approach untested.','Rigid surrogate is not ingested food; mass transfer/swallowing absent.','Estimated geometry/friction/actuation lack biological calibration.','Diagnostic has no retinal-policy/learner/checkpoint continuation qualification.','Broad lifelong and social environment remains unfinished.']}
    write(A/'plan.json',plan)
    shutil.copy2(__file__,A/'study.py');shutil.copy2(B/'probe-with-runtime.mm',A/'probe.mm')
    (A/'.gitattributes').write_text('*.stdout -text -diff\n*.stderr -text -diff\n')
    print(json.dumps({'plan_sha256':sha(A/'plan.json')}))
def run():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
    plan=json.loads((A/'plan.json').read_text())
    for p,v in plan['runtime_inputs'].items():assert sha(R/p)==v['sha256'],p
    rows=[]
    for n,arm,chunk,sub in trials:
        out=B/n;assert not out.exists()
        command=['/usr/bin/time','-lp','/usr/bin/env','DYLD_PRINT_LIBRARIES=1',f'DYLD_LIBRARY_PATH={R}/build/lib',
            f'METALROBO_METALLIB={R}/build/shaders/MetalRobo.metallib',f'METALROBO_CCD_FAILURE_CAPTURE={B}/{n}.ccdsweep',
            str(B/'probe-with-runtime'),str(out),str(arm),str(chunk),str(sub)]
        start=time.monotonic();timed_out=False
        with (B/f'{n}.stdout').open('xb') as stdout,(B/f'{n}.stderr').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=R,stdout=stdout,stderr=stderr,start_new_session=True)
            try:code=process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait(timeout=10)
        row={'name':n,'command':command,'exit_code':code,'timed_out':timed_out,'seconds':time.monotonic()-start}
        rows.append(row);write(B/'execution.json',{'plan_sha256':sha(A/'plan.json'),'results':rows});print(json.dumps(row),flush=True)
        trial_size=sum(p.stat().st_size for name,*_ in trials for p in (B/name).rglob('*') if p.is_file())
        if timed_out or trial_size>33554432:raise RuntimeError('Bounded stop; original process group terminal; preserve all outputs')
    write(A/'execution-summary.json',{'all_runs_exit_zero':all(x['exit_code']==0 for x in rows),'results':rows})
os.chdir(R)
{'seal':seal,'run':run}[sys.argv[1]]()
