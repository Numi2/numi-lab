from pathlib import Path
import csv, hashlib, json, shutil, subprocess
R=Path.cwd(); B=R/'build/crow-ccd-event-status-build-01'; A=R/'artifacts/crow-rl-ccd-event-status-20261108'
P=R/'build/crow-beak-hold-build-02'; PA=R/'artifacts/crow-rl-beak-hold-ccd-fix-20261108'
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Preserve the exact successfully loaded runtime before rebuilding its shader.
T=R/'.numi/retained-runtime/beak-hold-ccd-fix-20261108'; T.mkdir(parents=True,exist_ok=False)
runtime={}
for name,src in {'MetalRobo.metallib':'build/shaders/MetalRobo.metallib','CCDSweepProbe.metallib':'build/shaders/CCDSweepProbe.metallib',
    'libmetalrobo.dylib':'build/lib/libmetalrobo.dylib','contact-probe':'build/bin/metalrobo_metal_world_contact_probe',
    'object-push-check':'build/bin/metalrobo_crow_object_push_check','sweep-probe':'build/bin/metalrobo_ccd_sweep_probe'}.items():
    shutil.copy2(R/src,T/name);runtime[name]={'archive_source':str((T/name).relative_to(R)),'sha256':sha(T/name)}
write(PA/'runtime-recovery.json',runtime)
results={}
for name in ('clamp','open','no-contact','zero-friction','partition','refinement'):
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((P/name/'trace.tsv').open(),delimiter='\t')]
    initial=json.loads((P/name/'initial-geometry.json').read_text());lanes=[]
    for lane in range(4):
        q=[x for x in rows if int(x['lane'])==lane];hold=[x for x in q if 64<=x['tick']<=95];release=[x for x in q if 128<=x['tick']<=159]
        anchor=next((x for x in q if x['tick']==31),None)
        m={'samples':len(q),'last_tick':q[-1]['tick'],'full_hold_window':len(hold)==32,'full_release_window':len(release)==32,
           'maximum_sampled_penetration_before96_m':max(0.,-min(min(x['upper_gap_m'],x['lower_gap_m']) for x in q if x['tick']<96))}
        if hold and anchor:
            lift=min(x['z']-initial[lane]['position'][2] for x in hold)
            slip=max(sum((x['rel_'+a]-anchor['rel_'+a])**2 for a in 'xyz')**.5 for x in hold)
            opposed=sum(x['upper_Jn']>1e-9 and x['lower_Jn']>1e-9 for x in hold)
            m.update(minimum_lift_m=lift,maximum_relative_slip_m=slip,opposed_samples=opposed,
                hold_prefix_pass=len(hold)==32 and lift>=.02 and slip<=.005 and opposed>=16)
        if release and hold:
            m.update(release_drop_m=min(x['z'] for x in hold)-q[-1]['z'],minimum_release_gap_m=min(min(x['upper_gap_m'],x['lower_gap_m']) for x in release))
        lanes.append(m)
    results[name]={'complete':not (P/name/'failure.json').exists(),'lanes':lanes}
    if not results[name]['complete']:results[name]['failure']=json.loads((P/name/'failure.json').read_text())
write(PA/'analysis.json',{'verdict':'inconclusive: native rejections invalidate the full gesture','qualified':False,'arms':results,
  'first_failure_mechanism':'Unresolved. Later unconditional CCD initializers clear event counters, so final zero counters cannot exclude an event scheduling failure.',
  'limits':['Accepted prefixes are descriptive, not completed manipulation.','No-contact mask disables all object contacts, including the ground.','Microstep impulse samples are not integrated full-tick forces.','No retinal policy, ground pickup, ingestion or biological calibration.']})
A.mkdir(exist_ok=False)
src=R/'src/metal/Collision.metal';text=src.read_text()
start=text.index('kernel void mr_world_initialize_ccd_event_state(')
position=text.index('    MRCCDEventStateGPU state = {};',start)
text=text[:position]+'''    // A failed environment is skipped by later statically encoded substeps.
    // Preserve its first event failure instead of clearing the diagnostic
    // counters while its accepted physical state is being rolled back.
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
'''+text[position:];src.write_text(text)
cmake=R/'CMakeLists.txt';text=cmake.read_text();marker='add_executable(\n    metalrobo_deterministic_broadphase_probe'
assert text.count(marker)==1
text=text.replace(marker,'''add_executable(metalrobo_ccd_event_status_probe apps/ccd_event_status_probe.mm)
add_dependencies(metalrobo_ccd_event_status_probe metalrobo_shaders)
target_include_directories(metalrobo_ccd_event_status_probe PRIVATE include)
target_compile_options(metalrobo_ccd_event_status_probe PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(metalrobo_ccd_event_status_probe PRIVATE
    "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
set_target_properties(metalrobo_ccd_event_status_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

'''+marker);cmake.write_text(text)
shutil.copy2(__file__,A/'prepare.py')
write(A/'plan.json',{'purpose':'Repair first-failure diagnostic preservation; no contact/integration physics change.',
 'parent':'artifacts/crow-rl-beak-hold-ccd-fix-20261108','source_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'checks':['Production initializer healthy outputs byte exact against the retained baseline, both hybrid and disabled CCD, two substep generations.',
   'All 13 non-success codes and out-of-range lanes preserve entire A/B cursor, cluster and status bytes, twice per mode.',
   'Repeat initializer candidate outputs byte exact.',
   'One exact clamp failure replication: trace and accepted rollback q/v/scene byte exact; retain first event counters without changing failure tick/codes.',
   'One exact open-jaw successful control: trace and final q/v/scene byte exact.',
   'Native contact regression passes.'],
 'inputs':{str(p.relative_to(R)):sha(p) for p in (src,R/'apps/ccd_event_status_probe.mm',R/'CMakeLists.txt',P/'probe-with-runtime',T/'MetalRobo.metallib')},
 'stop':'One baseline initializer, candidate and repeat; one clamp failure, one open control and one native contact regression, each timeout180s. No retries or physics tuning. Maximum new evidence32MiB.',
 'diagnostic_prediction':'Counters in failed lanes remain visible after subsequent statically encoded substeps. Do not infer the original mechanism from previously cleared counters.'})
print(json.dumps({'plan_sha256':sha(A/'plan.json'),'runtime':runtime,'partial_results':results}))
