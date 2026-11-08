from pathlib import Path
import csv, hashlib, json, os, subprocess, time

R = Path.cwd()
A = R / 'artifacts/crow-zero-friction-solver-20261108'
B = R / 'build/crow-zero-friction-build-01'
O = B / 'observations'
plan = json.loads((A / 'inputs.json').read_text())
for name, expected in plan['inputs'].items():
    assert hashlib.sha256((R / name).read_bytes()).hexdigest() == expected, name
O.mkdir(exist_ok=False)
records = []
def run(name, argv, env, timeout):
    start = time.monotonic()
    with (O / (name + '.stdout')).open('xb') as out, (O / (name + '.stderr')).open('xb') as err:
        try:
            code = subprocess.run([str(x) for x in argv], cwd=R, env=env,
                                  stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
    records.append(dict(name=name, returncode=code, seconds=time.monotonic()-start))
    (A/'execution.json').write_text(json.dumps(records, indent=2, sort_keys=True)+'\n')
    print(json.dumps(records[-1]),flush=True)
    return code

env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}
for name in ('metalrobo_metal_unified_quality_probe', 'metalrobo_metal_world_contact_probe'):
    if run(name, [R/'build/bin'/name], env, 300) != 0:
        raise SystemExit('Native solver probe failed; fixture run not admitted')
for name, arm in [('frictionless',3), ('normal',0)]:
    local = {**env, 'DYLD_LIBRARY_PATH':str(B/'runtime'), 'DYLD_PRINT_LIBRARIES':'1',
             'CROW_PROBE_METALLIB':str(B/'runtime/metalrobo/MetalRobo.metallib'),
             'METALROBO_QUALITY_FAILURE_CAPTURE':str(O/(name+'-capture.bin'))}
    run(name, [B/'crow-fixture', O/name, str(arm), '1', '32', '2', 'api-default'], local, 600)
    assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib'))
               for line in (O/(name+'.stderr')).read_text().splitlines())

result = {'schema':'crow-zero-friction-result-v1', 'cases':{}, 'reached_held_out_navigation':False}
for name in ('frictionless','normal'):
    d=O/name
    summary=json.loads((d/'summary.json').read_text()) if (d/'summary.json').exists() else None
    failure=json.loads((d/'failure.json').read_text()) if (d/'failure.json').exists() else None
    rows=[]
    if (d/'trace.tsv').exists():
        for r in csv.DictReader((d/'trace.tsv').open(), delimiter='\t'):
            try:
                if None in r or any(v is None for v in r.values()): continue
                for v in r.values(): float(v)
                rows.append(r)
            except (ValueError,TypeError): continue
    max_tangent=max([0.]+[abs(float(row[k])) for row in rows for k in ('upper_Jt','lower_Jt')])
    jaw_penetration=max([0.]+[-float(row[k]) for row in rows for k in ('upper_gap_m','lower_gap_m')])
    ground_penetration=max([0.]+[.004-float(row['z']) for row in rows])
    held=None if not summary else sum(l['minimum_lift_64_95_m']>=.01 and
        l['maximum_head_relative_slip_m']<=.0005 and l['opposed_contact_samples']>=28
        for l in summary['lanes'])
    result['cases'][name]=dict(summary=summary,failure=failure,trace_rows=len(rows),
        max_sampled_tangential_impulse=max_tangent,max_sampled_jaw_penetration_m=jaw_penetration,
        max_sampled_ground_penetration_m=ground_penetration,held_count=held)
gold=R/'build/crow-grasp-resume-build-01/reference/normal-gold'
result['normal_physical_files_exact']=all((O/'normal'/p.name).is_file() and
    p.read_bytes()==(O/'normal'/p.name).read_bytes() for p in gold.iterdir())
result['physics_complete']=all(c['summary'] and c['summary']['accepted_transitions']==640 and
    c['summary']['physics_rejections']==0 and c['trace_rows']==640 and c['failure'] is None
    and c['max_sampled_jaw_penetration_m']<=.0008 and c['max_sampled_ground_penetration_m']<=.0008
    for c in result['cases'].values())
result['frictionless_tangent_zero']=result['cases']['frictionless']['max_sampled_tangential_impulse']<=1e-12
result['all_validation_passed']=result['physics_complete'] and result['frictionless_tangent_zero'] and result['normal_physical_files_exact']
(A/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True),flush=True)
