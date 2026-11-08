#!/usr/bin/env python3
"""Bounded native beak-contact study; no host physics or policy fitting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/Users/n/Developer/numi-crow-rl-20261107')
BUILD = ROOT / 'build/crow-beak-hold-build-01'
OUT = ROOT / 'artifacts/crow-rl-beak-hold-repair-20261108'
TRIALS = [('repair-clamp', 0, 1, 32), ('repair-open', 1, 1, 32),
          ('repair-no-contact', 2, 1, 32), ('repair-zero-friction', 3, 1, 32),
          ('repair-chunk-eight', 0, 8, 32), ('repair-refinement', 0, 1, 64)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def seal():
    OUT.mkdir(exist_ok=False)
    paths = [BUILD/'probe-v5', BUILD/'probe-v5.mm', BUILD/'inventory',
             BUILD/'inventory.mm', BUILD/'inventory.json', BUILD/'calibration-v5.jsonl',
             ROOT/'build/lib/libmetalrobo.dylib', ROOT/'build/shaders/MetalRobo.metallib']
    assert [json.loads(x) for x in (BUILD/'calibration-v5.jsonl').read_text().splitlines()] == [{'calibration':'passed','fp64_gap_known_values':3,'rotated_frame_known_values':1},{'compiled_fixture_calibration':'passed','articulation_bodies':17,'scene_bodies_excluded':2}]
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=ROOT)
    plan = {
      'schema': 'crow-native-beak-contact-exploration-v2',
      'instrument_revision': 'Parent artifacts/crow-rl-beak-hold-20261108: all six attempts exited before physics at initial FP64 kinematics due global rather than articulation-local output span. Only observer span/status and compiled-fixture calibration repaired; mechanical parameters, actions and thresholds unchanged. Repeated unobserved conditions, not independent confirmation.',
      'purpose': 'exploration: single authored mechanical condition, four identical determinism lanes; no statistical replicates',
      'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'question': 'Can ordinary upper/lower beak contacts retain, lift, and release an initially between-jaws free sphere?',
      'model': 'Existing active-head articulated crow; 4 mm radius, 0.25 g free rigid surrogate; no latch, attachment, food ledger, gravity exemption, state edits or learned policy.',
      'alternative': 'Open geometry, insufficient contact/friction, numerical overlap, or a scheduling artifact explains apparent hold.',
      'instrument_calibration': {'path': str(BUILD/'calibration-v5.jsonl'), 'sha256': sha(BUILD/'calibration-v5.jsonl'),
         'scope': 'FP64 analytic signed gap and body-frame transform known values only; contact attribution separately tested by contact-disabled control.'},
      'runtime_inputs': {str(p.relative_to(ROOT)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in paths},
      'schedule': {'environment_count':4,'ticks':160,'control_hz':50,'seed':2651511101,
         'initial_jaw_rad':0.23,'initial_center_m':[0.215,0,0.26685],
         'action_pitch': '0 through tick31, -0.4 from32',
         'action_jaw': '-0.35 through95 except open arm; 0 from96; native actuator scales/default reference unchanged'},
      'trials': [{'name':n,'arm':a,'chunk':c,'substeps':s,
          'command':['build/crow-beak-hold-build-01/probe-v5',f'build/crow-beak-hold-build-01/{n}',str(a),str(c),str(s)]} for n,a,c,s in TRIALS],
      'prediction': {
         'primary_clamp': {'each_lane_minimum_lift_ticks64_95_m':0.020,
            'each_lane_maximum_head_relative_slip_from_tick31_m':0.005,
            'each_lane_opposed_positive_normal_contact_samples_ticks64_95_at_least':16,
            'each_lane_drop_after_release_from_minimum_held_height_to_tick159_m_at_least':0.030},
         'controls': 'Open and disabled-contact arms must not satisfy primary hold/lift; disabled contact must report exactly zero object/beak impulses. Zero-friction is a mechanism ablation, with predicted failure to hold/lift; failure of this prediction requires examining geometric trapping.',
         'numerical': {'32_vs_64_maximum_paired_sphere_position_difference_m':0.005,
            'chunk1_vs8_final_q_v_scene_world':'byte exact',
            'maximum_jaw_penetration_ticks0_95_m':0.0008}},
      'validity': ['all 3840 declared transitions accepted; no physics rejection or typed failure',
         'initial independent signed gaps each >20 micrometres',
         'finite samples and four identical lanes',
         'actual loader identifies retained native library; shader identity pinned',
         'final microstep impulses are contact evidence samples, not integrated full-tick force'],
      'analysis': 'Report all arms, every lane, direct geometry/trajectory and impulses. No p-values: same condition duplicated in four lanes. Model-supported only if primary/control/numerical predicates all pass; otherwise contradicted or inconclusive according to validity. Descriptive refinement comparison does not establish convergence.',
      'budget': {'trials':6,'transitions':3840,'timeout_seconds_per_trial':180,'maximum_output_bytes':67108864,
         'stopping':'One execution per declared condition, no retries or parameter tuning. Preserve failure logs and remaining-arm status. Stop if evidence exceeds budget or worker remains alive. Further revisions require new sealed plan and unused conditions.'},
      'limits': ['Supported torso and predefined native actions; not learned behavior.',
         'Object starts between jaws: not ground pickup or approach.',
         'Rigid non-food surrogate: no swallow, mass transfer, nutrition, or biological calibration.',
         'Current geometry/materials/actuation are estimated; no biological performance claim.']}
    write(OUT/'plan.json',plan)
    shutil.copy2(BUILD/'probe-v5.mm',OUT/'probe.mm')
    shutil.copy2(Path(__file__),OUT/'study.py')
    (OUT/'.gitattributes').write_text('*.stdout -text -diff\n*.stderr -text -diff\n')
    write(OUT/'environment.json',{'uname':subprocess.check_output(['uname','-a'],text=True).strip(),
        'sw_vers':subprocess.check_output(['sw_vers'],text=True),
        'clang':subprocess.check_output(['clang++','--version'],text=True),
        'df':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)})
    print(json.dumps({'sealed_plan':str(OUT/'plan.json'),'sha256':sha(OUT/'plan.json')}))

def run():
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)==b''
    plan=json.loads((OUT/'plan.json').read_text())
    for p,identity in plan['runtime_inputs'].items():
        assert sha(ROOT/p)==identity['sha256'], p
    env=os.environ.copy()
    env.update(DYLD_LIBRARY_PATH=str(ROOT/'build/lib'),DYLD_PRINT_LIBRARIES='1',
               METALROBO_METALLIB=str(ROOT/'build/shaders/MetalRobo.metallib'))
    results=[]
    for name,arm,chunk,substeps in TRIALS:
        assert not (BUILD/name).exists()
        command=['/usr/bin/time','-lp',str(BUILD/'probe-v5'),str(BUILD/name),str(arm),str(chunk),str(substeps)]
        start=time.monotonic()
        with (BUILD/f'{name}.stdout').open('wb') as stdout,(BUILD/f'{name}.stderr').open('wb') as stderr:
            try:
                code=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=180).returncode
            except subprocess.TimeoutExpired:
                code='timeout'
        results.append({'name':name,'exit_code':code,'elapsed_seconds':time.monotonic()-start,'command':command})
        write(BUILD/'execution-v2.json',{'plan_sha256':sha(OUT/'plan.json'),'results':results})
        print(json.dumps(results[-1]),flush=True)
        size=sum(p.stat().st_size for p in BUILD.rglob('*') if p.is_file())
        if size>67108864 or code=='timeout':
            raise RuntimeError('Bounded stop; inspect live workers before any recovery')
    assert all(x['exit_code']==0 for x in results), 'One or more trials failed; preserved'

os.chdir(ROOT)
{'seal':seal,'run':run}[sys.argv[1]]()
