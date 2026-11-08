from pathlib import Path
import csv,hashlib,json,os,subprocess,time
import numpy as np
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-articulated-sweep-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'])
plan=json.loads((A/'plan.json').read_text())
for p,v in plan['inputs'].items():assert sha(R/p)==v['sha256'],p
commands=[]
def run(name,cmd,env=None,expected=0):
    start=time.monotonic()
    with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=180).returncode
    commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start});write(B/'execution.json',commands);print(json.dumps(commands[-1]),flush=True)
    if expected is not None:assert code==expected,(name,code)
    return code
for name,shader in [('baseline','baseline.metallib'),('candidate','candidate.metallib'),('repeat','candidate.metallib')]:run('projection-'+name,[B/'projection-probe',B/shader,B/('projection-'+name+'.bin')])
old=np.fromfile(B/'projection-baseline.bin',dtype='<u4').reshape(2,2,128,2,20);new=np.fromfile(B/'projection-candidate.bin',dtype='<u4').reshape(2,2,128,2,20)
keep=[i for i in range(20) if i!=2]
v=np.float32(.5)+np.arange(128,dtype=np.float32)*np.float32(.001);expected=np.float32(.000625)*v*np.float32(.25)
checks={'hybrid_own_flag_does_not_change_geometry_or_margin':bool(np.array_equal(new[1,:,:,0,4:],new[1,:,:,1,4:]) and np.array_equal(new[1,:,:,0,2],new[1,:,:,1,2])),
 'own_CCD_source_flag_retained':bool(np.all(new[:,:,:,0,3]==0) and np.all(new[:,:,:,1,3]!=0)),
 'flagged_future_FK_exact':bool(np.array_equal(old[:,:,:,1],new[:,:,:,1])),
 'speculative_unflagged_only_margin_changes':bool(np.array_equal(old[0,:,:,0][:,:,keep],new[0,:,:,0][:,:,keep])),
 'speculative_margin_matches_known_generalized_speed_envelope':bool(np.allclose(new.view('<f4')[0,:,:,0,2],expected,rtol=2e-7,atol=0)),
 'projection_repeat_exact':(B/'projection-candidate.bin').read_bytes()==(B/'projection-repeat.bin').read_bytes(),
 'baseline_hybrid_motion_omission_demonstrated':bool(np.all(old[1,1,:,0,4]!=old[1,1,:,1,4]))}
run('grid',[B/'cases-probe',B/'candidate.metallib',B/'cases.bin',B/'grid.bin'])
checks['contact_time_grid4096_exact']=(B/'grid.bin').read_bytes()==(B/'expected-grid.bin').read_bytes()
report={'projection_checks':checks,'physical_capability_qualified':False,'production_replaced':False,'commands':commands}
write(A/'results.json',report)
assert all(checks.values()),checks
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(B/'runtime'),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(B/'runtime/metalrobo/MetalRobo.metallib'))
results={}
for trial in plan['trials']:
    name=trial['name'];env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'.ccdsweep'))
    code=run(name,[B/'probe',B/name,str(trial['arm']),str(trial['chunk']),str(trial['substeps'])],env,None)
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'trace.tsv').open(),delimiter='\t')]
    initial=json.loads((B/name/'initial-geometry.json').read_text());metrics=[]
    for lane in range(4):
        q=[r for r in rows if r['lane']==lane];hold=[r for r in q if 64<=r['tick']<=95];release=[r for r in q if r['tick']>=128];anchor=next((r for r in q if r['tick']==31),None)
        x={'samples':len(q),'finite':all(np.isfinite(z) for r in q for z in r.values()),'all160ticks':len(q)==160}
        if q:x.update(last_tick=q[-1]['tick'],maximum_penetration_before96_m=max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in q if r['tick']<96)))
        if hold and anchor:
            x.update(minimum_lift_m=min(r['z']-initial[lane]['position'][2] for r in hold),maximum_slip_m=max(sum((r['rel_'+a]-anchor['rel_'+a])**2 for a in 'xyz')**.5 for r in hold),opposed_samples=sum(r['upper_Jn']>1e-9 and r['lower_Jn']>1e-9 for r in hold))
            x['hold_pass']=len(hold)==32 and x['minimum_lift_m']>=.02 and x['maximum_slip_m']<=.005 and x['opposed_samples']>=16
            if release:x.update(release_drop_m=min(r['z'] for r in hold)-q[-1]['z'],minimum_release_gap_m=min(min(r['upper_gap_m'],r['lower_gap_m']) for r in release))
        metrics.append(x)
    loaded=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    assert loaded and all(s.endswith(str(B/'runtime/libmetalrobo.dylib')) for s in loaded)
    assert 'Crow probe metallib: '+env['CROW_PROBE_METALLIB'] in (B/(name+'.stderr')).read_text()
    results[name]={'exit_code':code,'lanes':metrics,'actual_loader':loaded,'rows':len(rows)}
    p=B/name/'failure.json'
    if p.exists():results[name]['failure']=json.loads(p.read_text())
    report.update(physical_results=results,commands=commands);write(A/'results.json',report)
accepted=all(v['exit_code']==0 for v in results.values())
physical_checks={'all_native_executions_accepted':accepted,'all_values_finite':all(x['finite'] for v in results.values() for x in v['lanes'])}
if accepted:
    positive=('clamp','partition','refinement')
    # Chunk8 samples every eighth tick: compare final states, and apply sampled
    # geometric gates only to both fully sampled positive arms.
    physical_checks.update(each_fully_sampled_positive_lane_mechanical_gates=all(x['hold_pass'] and x['maximum_penetration_before96_m']<=.0008 and x['release_drop_m']>=.03 and x['minimum_release_gap_m']>=.001 for n in ('clamp','refinement') for x in results[n]['lanes']),
      all_negative_controls_fail_hold=all(not x.get('hold_pass',False) for n in ('open','no-contact','zero-friction') for x in results[n]['lanes']),
      final_chunk_partition_exact=all((B/'clamp'/n).read_bytes()==(B/'partition'/n).read_bytes() for n in ('final-q.f32','final-v.f32','final-scene.bin')))
    c=np.array([[r['x'],r['y'],r['z']] for r in [{k:float(v) for k,v in z.items()} for z in csv.DictReader((B/'clamp/trace.tsv').open(),delimiter='\t')]])
    f=np.array([[r['x'],r['y'],r['z']] for r in [{k:float(v) for k,v in z.items()} for z in csv.DictReader((B/'refinement/trace.tsv').open(),delimiter='\t')]])
    error=float(np.max(np.linalg.norm(c-f,axis=1)));report['maximum_paired_32vs64_position_error_m']=error;physical_checks['full_trajectory_refinement_within5mm']=error<=.005
report.update(physical_checks=physical_checks,mechanical_gesture_qualified=all(physical_checks.values()),physical_capability_qualified=False,
 limits='Even a complete mechanical gesture is not learned manipulation, ground pickup, ingestion, retinal control or biological calibration.',commands=commands)
write(A/'results.json',report);print(json.dumps({'projection_checks':checks,'physical_checks':physical_checks,'maximum_error_m':report.get('maximum_paired_32vs64_position_error_m')}),flush=True)
