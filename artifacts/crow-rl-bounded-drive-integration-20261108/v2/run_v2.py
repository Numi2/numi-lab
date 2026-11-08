from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time

R=Path.cwd()
B=Path('build/crow-bounded-drive-integration-build-01')
A=Path('artifacts/crow-rl-bounded-drive-integration-20261108/v2')
T=Path('.numi/retained-runtime/bounded-drive-integration-v2-20261108')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')


assert not A.exists() and not T.exists()
cal=json.loads((B/'calibration-v3.stdout').read_text())
assert cal['mass_error']<1e-14 and cal['constitutive_error']<1e-9 and cal['conditions']==128
assert not (B/'calibration-v3.stderr').read_bytes()
A.mkdir(parents=True)
(T/'metalrobo').mkdir(parents=True)
for source,target in [(Path('build/bin/metalrobo_bounded_drive_probe'),T/'probe'),
                      (Path('build/lib/libmetalrobo.dylib'),T/'libmetalrobo.dylib'),
                      (Path('build/shaders/MetalRobo.metallib'),T/'metalrobo/MetalRobo.metallib')]:
    shutil.copy2(source,target)
source_paths=[Path(p) for p in ('CMakeLists.txt','include/metalrobo/MetalWorld.hpp',
    'include/metalrobo/engine_types.h','src/metal/MetalWorld.metal',
    'src/metal/MetalWorld.mm','src/metal/MetalWorldContact.metal',
    'apps/bounded_drive_probe.cpp')]
inputs=source_paths+[T/'probe',T/'libmetalrobo.dylib',T/'metalrobo/MetalRobo.metallib',B/'calibration-v3.stdout']
plan={
 'question':'Does native smooth bounded motor prediction reproduce the independently solved constitutive equations across scalar/coupled/multiple articulations and CCD modes, while preserving the existing clamp-then-augment mode?',
 'parent':'crow-rl-implicit-drive-audit-20261108: native saturation defect confirmed; endpoint-only motor row rejected by hybrid q/torque gates.',
 'model':'Physical ABA acceleration and Cholesky mass include physical armature, no motor damping/stiffness inertia. A native box/L1 coordinate solve produces bounded motor torque and acceleration; source-state mass/free acceleration remain fixed across remaining-time prediction and selected-time solve, each with its own actual duration. CCD position prediction includes smooth motor acceleration. Endpoint contacts use physical mass, and position is not reintegrated across impact.',
 'profile_semantics':'Gains, effort limits and source-speed motor envelope unchanged; controller scaling and source-velocity dry friction/dead zone retained. No environment or policy parameters tuned.',
 'conditions':{
    'one_microstep_native_environments':514,
    'component_rows':976,
    'paired_grid':'4 mechanism variants x8 states x4 durations x2 actuation modes x2 CCD modes =512 accepted environment steps,896 component rows.',
    'variants':['single scalar hinge','coupled coaxial two-DoF articulation','two independent one-DoF articulations','coupled two-DoF articulation with dry friction'],
    'capacity':'Two fixed-root star articulations with20 independent scalar joints each (42 bodies total), CCD disabled/hybrid =2 additional steps,80 component rows; exercises response columns32..39.',
    'replication':'First scalar-state sequence reuses exposed earlier diagnosis/bounded conditions. These are engineering regression checks.',
    'unused_conditions':'Second-joint states, inertial coupling, dry-friction mechanism and full40-DoF configuration have no prior native motor observations.',
    'independence':'Manufactured deterministic conditions, no independent biological/learning replicates or population inference.'},
 'calibration':cal,
 'reference':'Native FP64 mass matrices equal analytic mechanism masses; exhaustive independent five-region active-set enumeration checks coupled bounded motors, including the stationary dry-friction dead zone. Legacy reference solves the original fully augmented mass with pre-clamped command. Forty independent hinges use the calibrated scalar closed form.',
 'criteria':{'all_rows_maximum_absolute_velocity_reference_error_rad_s':2e-6,
             'all_rows_maximum_q_integration_error_rad':2e-7,
             'all_rows_maximum_torque_reference_error_Nm':1e-7,
             'native_rejections':0,
             'bounded_aba_and_contact_mass_implicit_bits':0,
             'legacy_aba_and_contact_mass_implicit_bits':1},
 'validity':'All514 accepted,976 component rows; r.dispatched/published true, actual Apple GPU device identified; no scene bodies, geometry or external forces in mechanisms. Runtime/shader/executable/input hashes match. Missing native rows or rejection is inconclusive; no automatic rerun.',
 'budget':{'native_invocations':1,'timeout_seconds':180,'new_trial_output_max_bytes':1500000,'stop':'One invocation, analyze every declared condition; no retries, tuning, extra seeds or promotion.'},
 'limits':'New native mode initially admits contact graphs with at most40 total generalized velocities. This trial has no geometric contact, motor/contact simultaneous coupling, task energy integration, reset/partition/checkpoint, real crow release, end-to-end throughput or learning qualification.',
 'inputs':{str(p):sha(p) for p in inputs},
 'protected_policy_sha256':sha(Path('environments/american-crow/action-holds-v1/policies/deployment97.policypack'))
}
plan['amendment']={'parent_seal':'3476e8ab','parent_analysis':'../partial-analysis.json','prior':'Original plan completed512 of514 microsteps. The invalid80-body capacity fixture was rejected before its two steps. Completed bounded cases include2.058839697e-6 rad/s error, contradicting the unchanged2e-6 gate. A subsequent41-body single-tree draft also failed host-only calibration (ABA limit32 per articulation); no new native observations occurred.','source_change':'Internal motor convergence32eps to4eps, and fused multiply-add in coordinate RHS and residual. Maximum256 sweeps and all physical coefficients unchanged. Wide fixture uses two20-joint stars, within per-articulation and global body limits; its mass remains diagonal.','units':'All512 small-mechanism conditions are exposed engineering regressions. The40-DoF layout is unused native validation; neither rejected topology produced a physical step.','acceptance_gates_changed':False}
write(A/'plan.json',plan)
shutil.copy2(B/'run_v2.py',A/'run_v2.py')
cache=Path('build/CMakeCache.txt').read_text().splitlines()
write(B/'build-v2-provenance.json',{'source_inputs':{str(p):sha(p) for p in source_paths},
  'command':'cmake --build build --target metalrobo_bounded_drive_probe -j4',
  'cmake':[s for s in cache if s.startswith(('CMAKE_BUILD_TYPE:','CMAKE_CXX_COMPILER:','CMAKE_OBJCXX_COMPILER:'))],
  'clang':subprocess.check_output(['clang++','--version'],text=True)})
subprocess.run(['git','add',*[str(p) for p in source_paths],str(A)],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
subprocess.run(['git','commit','--quiet','-m','Tighten native motor convergence and seal regression and capacity trial'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for p,h in plan['inputs'].items():assert sha(Path(p))==h
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(R/T),DYLD_PRINT_LIBRARIES='1')
cmd=[str(R/T/'probe'),str(R/T/'metalrobo/MetalRobo.metallib'),str(R/B/'native-v2.tsv')]
start=time.monotonic()
with (B/'native-v2.stdout').open('xb') as out,(B/'native-v2.stderr').open('xb') as err:
    code=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=180).returncode
receipt={'sealed_head':head,'command':cmd,'exit_code':code,'seconds':time.monotonic()-start}
write(B/'execution-v2.json',receipt)
report={'execution':receipt,'policy_or_profile_promoted':False,'full_crow_qualified':False}
if code!=0:
    report.update(verdict='inconclusive_native_instrument_failure',failure='Native invocation rejected; partial TSV and typed stderr retained, no rerun.')
    write(A/'results.json',report);print(json.dumps(report),flush=True);raise SystemExit(1)
rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/'native-v2.tsv').open(),delimiter='\t')]
assert len(rows)==976
groups={}
for bounded in (0,1):
    selected=[r for r in rows if r['bounded']==bounded]
    groups[str(bounded)]={
      'component_rows':len(selected),
      'maximum_velocity_reference_error':max(abs(r['final_v']-r['reference_v']) for r in selected),
      'maximum_q_integration_error':max(abs(r['final_q']-(r['q']+r['h']*r['final_v'])) for r in selected),
      'maximum_torque_reference_error':max(abs(r['motor_torque']-r['reference_torque']) for r in selected),
      'mass_flags_match_mode':all(r['aba_implicit']==1-bounded and r['contact_mass_implicit']==1-bounded for r in selected)}
checks={}
for key,g in groups.items():
    checks[key]=g['maximum_velocity_reference_error']<=2e-6 and g['maximum_q_integration_error']<=2e-7 and g['maximum_torque_reference_error']<=1e-7 and g['mass_flags_match_mode']
loads=[s for s in (B/'native-v2.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
assert loads and all(s.endswith(str(R/T/'libmetalrobo.dylib')) for s in loads)
device_lines=[s for s in (B/'native-v2.stdout').read_text().splitlines() if s.startswith('device=')]
assert len(device_lines)==64 and all(s.startswith('device=Apple ') for s in device_lines)
report.update(verdict='native_bounded_drive_constitutive_trial_passed' if all(checks.values()) else 'native_bounded_drive_prediction_contradicted',
              native_microsteps=514,component_rows=len(rows),groups=groups,checks=checks,loader_lines=loads,
              actual_device=device_lines[0].split(' variant=')[0],
              limits=plan['limits'])
write(A/'results.json',report)
write(B/'analysis-v2.json',report)
print(json.dumps(report,indent=2),flush=True)
