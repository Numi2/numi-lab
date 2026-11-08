"""Execute the presealed native grid and captured projected sweeps once."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
R=Path.cwd();B=R/'build/crow-ccd-translation-build-02';A=R/'artifacts/crow-rl-ccd-translation-no-floor-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text())
assert not subprocess.check_output(['git','status','--porcelain'])
for p,v in plan['inputs'].items():assert sha(R/p)==v['sha256'],p
receipts=[]
def run(n,cmd):
    t=time.monotonic()
    with (B/f'{n}.stdout').open('xb') as out,(B/f'{n}.stderr').open('xb') as err:
        code=subprocess.run(cmd,stdout=out,stderr=err,timeout=180).returncode
    receipts.append({'name':n,'command':cmd,'exit_code':code,'seconds':time.monotonic()-t})
    write(B/'execution.json',{'plan_sha256':sha(A/'plan.json'),'commands':receipts})
    assert code==0,(n,code)
run('compile',['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-c',str(B/'candidate_probe.metal'),'-o',str(B/'candidate.air')])
run('link',['xcrun','-sdk','macosx','metallib',str(B/'candidate.air'),'-o',str(B/'candidate.metallib')])
run('compile-query',['clang++','-std=c++20','-O3','-fobjc-arc','-I',str(R/'include'),str(B/'query_probe.mm'),'-framework','Foundation','-framework','Metal','-o',str(B/'query_probe')])
for n,shader in [('baseline','baseline.metallib'),('candidate','candidate.metallib'),('repeat','candidate.metallib')]:
    run(n,[str(B/'cases_probe'),str(B/shader),str(B/'cases.bin'),str(B/f'{n}.bin')])
captures={}
for n in ('clamp','open','zero-friction','refinement'):
    run(f'capture-{n}',[str(B/'sweep_probe'),str(B/'candidate.metallib'),str(B/'candidate.metallib'),str(B/f'input/{n}.ccdsweep'),str(B/f'{n}.bin')])
    v=json.loads((B/f'capture-{n}.stdout').read_text())
    old=(B/f'input/{n}-replay.bin').read_bytes();new=(B/f'{n}.bin').read_bytes()
    captures[n]={'record':v['replay_records'][0], 'all_query_bytes_exact':old[128:]==new[128:],
       'both_captured_events_resolve':all(x['status']==0 and not (x['flags']&128) for x in v['replay_records'])}
run('parent-queries',[str(B/'query_probe'),str(B/'candidate.metallib'),str(B/'observed-misses.bin'),str(B/'query-alpha.f32'),str(B/'native-queries.bin')])
run('parent-regression',[str(B/'cases_probe'),str(B/'candidate.metallib'),str(B/'observed-misses.bin'),str(B/'parent-regression.bin')])
old=(B/'baseline.bin').read_bytes();new=(B/'candidate.bin').read_bytes()
assert len(old)==len(new)==4096*2*64
oldints=np.frombuffer(old,dtype='<u4').reshape(-1,16);newints=np.frombuffer(new,dtype='<u4').reshape(-1,16)
oldfloat=np.frombuffer(old,dtype='<f4').reshape(-1,16);newfloat=np.frombuffer(new,dtype='<f4').reshape(-1,16)
groups=json.loads((B/'groups.json').read_text());oracle=json.loads((B/'oracle.json').read_text())
control_failures=[];unsafe=[];eligible=[];gray=[];pair_mismatches=[];times=[]
for i,(group,pred) in enumerate(zip(groups,oracle)):
    control=group in ('stationary','overlap','rotating')
    for order in range(2):
        j=i*2+order;flags=int(newints[j,14]);status=int(newints[j,15])
        if control:
            if old[j*64:(j+1)*64]!=new[j*64:(j+1)*64]:control_failures.append(j)
            continue
        minimum=pred['minimum_unsigned_box_distance_minus_radius_m'];tol=float(np.float32(1e-5))
        if abs(minimum-tol)<=.5e-6:
            gray.append(j);continue
        eligible.append(j)
        reference_crosses=pred['first_alpha'] is not None and minimum<tol-.5e-6
        if reference_crosses and status==0 and not(flags&(2|4|8)):
            unsafe.append({'record':j,'group':group,'reference':pred,'native_flags':flags,'native_toi':newfloat[j,4:6].tolist()})
        if reference_crosses and status==0 and flags&4:
            times.append({'record':j,'first_reference_alpha':pred['first_alpha'],'native_lower_alpha':float(newfloat[j,4])/float(np.float32(.000625)),'native_upper_alpha':float(newfloat[j,5])/float(np.float32(.000625))})
    if (newints[i*2,14]&14,newints[i*2,15])!=(newints[i*2+1,14]&14,newints[i*2+1,15]):pair_mismatches.append(i)
baseline_unresolved=sum(bool(oldints[j,14]&128) for j in eligible);candidate_unresolved=sum(bool(newints[j,14]&128) for j in eligible)
success=float(sum(newints[j,15]==0 for j in eligible)/len(eligible))
reduction=1-candidate_unresolved/baseline_unresolved if baseline_unresolved else 1.
queries=np.fromfile(B/'native-queries.bin',dtype='<f4').reshape(4,4,4)
references=json.loads((B/'query-reference.json').read_text())
query_error=max(abs(float(queries[i,3,1])-p['minimum_unsigned_box_distance_minus_radius_m']) for i,p in enumerate(references))
regression=np.fromfile(B/'parent-regression.bin',dtype='<u4').reshape(8,16)
query_checks={'parent_minimum_native_gap_below_tolerance':bool(np.all(queries[:,3,1]<float(np.float32(1e-5))-.5e-6)),'parent_native_FP64_minimum_error_bound':query_error<=2.5e-7,'parent_query_status_valid':bool(np.all(queries[:,:,2]==0) and np.all(queries[:,:,3]==1)),'exposed_grazing_impacts_repaired':bool(np.all(regression[:,15]==0) and np.all((regression[:,14]&4)!=0))}
checks={**query_checks,'captured_pairs_resolve':all(v['both_captured_events_resolve'] for v in captures.values()),
    'captured_distance_queries_exact':all(v['all_query_bytes_exact'] for v in captures.values()),
    'zero_unsafe_successful_no_impact':not unsafe,'exact_controls':not control_failures,
    'repeat_exact':new==(B/'repeat.bin').read_bytes(), 'success_fraction_at_least_99_percent':success>=.99,
    'unresolved_reduction_at_least_50_percent':reduction>=.50}
report={'schema':'crow-native-ccd-translation-no-floor-v2','parent_query_maximum_absolute_error_m':query_error,'parent_query_values':queries.tolist(),'prediction_verdict':'supported' if all(checks.values()) else 'contradicted',
   'production_replacement_authorized_by_this_result':False,'checks':checks,'captures':captures,
   'eligible_records':len(eligible),'gray_records':gray,'baseline_unresolved':baseline_unresolved,'candidate_unresolved':candidate_unresolved,
   'eligible_success_fraction':success,'unresolved_reduction_fraction':reduction,'control_failures':control_failures,
   'unsafe_successful_no_impact_records':unsafe,'pair_order_classification_mismatches':pair_mismatches,
   'reference_crossing_time_comparisons':times,
   'candidate_source_sha256':sha(B/'Collision.candidate.metal'),'candidate_shader_sha256':sha(B/'candidate.metallib'),
   'limits':'Synthetic projected geometry and exact retained sweeps. The rotating-box legacy path and event scheduler still require qualification against narrow grazing contacts and full physical release before production selection.'}
write(B/'analysis-full.json',report)
brief=dict(report);brief['reference_crossing_time_comparisons_count']=len(times);brief.pop('reference_crossing_time_comparisons')
write(A/'analysis.json',brief)
write(A/'runtime.json',{'baseline_sha256':sha(B/'baseline.metallib'),'candidate_sha256':sha(B/'candidate.metallib'),
    'native_case_executable_sha256':sha(B/'cases_probe'),'native_capture_executable_sha256':sha(B/'sweep_probe'),'commands':receipts})
print(json.dumps({'verdict':report['prediction_verdict'],'checks':checks,'success_fraction':success,'baseline_unresolved':baseline_unresolved,'candidate_unresolved':candidate_unresolved,'unsafe_no_impact':len(unsafe),'control_failures':len(control_failures),'captures':captures}),flush=True)
