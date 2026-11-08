from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-representability-20261108';B=R/'build/crow-quality-representability-build-01'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=(A/'fp64_reference.py').read_text().split('initial=evaluate(failed)')[0]
source=source.replace("B=R/'build/crow-quality-first-failure-build-01/repair-01'","B=R/'build/crow-quality-representability-build-01'")
if sys.argv[1]=='prepare':
    rounded=np.fromfile(B/'fp64-reference-v.f64',dtype='<f8').astype('<f4')
    (B/'rounded-reference-v.f32').write_bytes(rounded.tobytes())
    plan=dict(schema='crow-quality-native-warm-validation-plan-v1',purpose='Engineering replay of one exposed numerical problem; test native acceptance without editing solver or tolerances',
      inputs={str(p.relative_to(R)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in [A/'warm_probe.mm',A/'warm_validation.py',B/'capture-one.bin',B/'rounded-reference-v.f32',B/'neighbor-v.f32',R/'build/shaders/MetalRobo.metallib']},
      predicted=[dict(case='original',code=7,newton=20,pcg=77,capture_exact=True),dict(case='rounded',code=7),dict(case='neighbor',code=0,independent_residual_max=1e-10)],
      budgets=dict(cases=3,command_timeout_seconds=120,new_output_bytes_max=5000000),policy_promoted=False,
      decision='Native acceptance of neighbor supports implementing generic native refinement; rejection requires FP32 arithmetic diagnosis before any solver change')
    (A/'warm-validation-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    print(json.dumps(plan));sys.exit(0)
assert sys.argv[1]=='run'
plan=json.loads((A/'warm-validation-plan.json').read_text())
for p,m in plan['inputs'].items():assert digest(R/p)==m['sha256'],p
out=B/'warm-validation-01';out.mkdir()
records=[]
def run(name,command):
    t=time.monotonic();p=subprocess.run(command,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
    rec=dict(name=name,argv=command,returncode=p.returncode,seconds=time.monotonic()-t);records.append(rec)
    (out/'execution.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
    assert p.returncode==0,rec
    return p.stdout.decode()
run('build',['xcrun','clang++','-std=c++17','-fobjc-arc','-Iinclude',str(A/'warm_probe.mm'),'-framework','Metal','-framework','Foundation','-o',str(out/'warm-probe')])
ns={};exec(compile(source,'retained-independent-fp64','exec'),ns)
results=[]
for name,warm in [('original',None),('rounded',B/'rounded-reference-v.f32'),('neighbor',B/'neighbor-v.f32')]:
    cmd=[str(out/'warm-probe'),str(B/'capture-one.bin'),str(R/'build/shaders/MetalRobo.metallib'),str(out/name)]
    if warm is not None:cmd.append(str(warm))
    result=json.loads(run(name,cmd));result['case']=name
    if result['code']==0:
        v=np.fromfile(out/name/'velocity.f32',dtype='<f4').astype(float)
        val=ns['evaluate'](v);p=np.fromfile(out/name/'impulse.f32',dtype='<f4').astype(float)
        result.update(independent_scaled_optimality=val[-1],independent_impulse_max_difference=float(np.max(np.abs(p-val[2]))),warm_output_exact=(out/name/'velocity.f32').read_bytes()==warm.read_bytes())
    if name=='original':result['capture_exact']=(out/name/'capture.bin').read_bytes()==(B/'capture-one.bin').read_bytes()
    results.append(result)
report=dict(schema='crow-quality-native-warm-validation-result-v1',cases=results,predictions_satisfied=[all(c.get(k)==v for k,v in prediction.items() if k!='independent_residual_max') and (c.get('independent_scaled_optimality',float('inf'))<=prediction['independent_residual_max'] if 'independent_residual_max' in prediction else True) for c,prediction in zip(results,plan['predicted'])],limits='One exposed problem. Warm states are diagnostics; no transition is fed back to the world.',policy_or_profile_promoted=False)
(A/'warm-validation-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report),flush=True)
