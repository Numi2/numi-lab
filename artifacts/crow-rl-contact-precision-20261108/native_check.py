from pathlib import Path
import hashlib,json,os,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-contact-precision-20261108';B=R/'build/crow-contact-precision-build-01';T=R/'.numi/retained-runtime/contact-precision-20261108'
source=(A/'reference-source.py').read_text().split('initial=evaluate(failed)')[0].replace("B=R/'build/crow-quality-representability-build-01'","B=R/'build/crow-contact-precision-build-01'")
ns={};exec(compile(source,'retained-independent-reference','exec'),ns)
if sys.argv[1]=='prepare':
    T.mkdir(parents=True);p=T/'MetalRobo.metallib';import shutil;shutil.copy2(R/'build/shaders/MetalRobo.metallib',p)
    (B/'rounded-reference-v.f32').write_bytes(np.fromfile(B/'reference-v.f64',dtype='<f8').astype('<f4').tobytes())
    files=[A/'native_check.py',B/'warm-probe',B/'substep7-capture.bin',B/'rounded-reference-v.f32',B/'chebyshev-v.f32',B/'control-substep2.bin',B/'control-neighbor-v.f32',p]
    plan=dict(schema='crow-contact-native-precision-check-plan-v1',purpose='Native engineering replay and arithmetic cross-check of retained/warm candidate states, without source/configuration changes',inputs={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},predictions=dict(original_capture_exact=True,original_code=7,rounded_code=7,chebyshev_code=7,control_code=0,control_velocity_exact=True),budget=dict(cases=4,seconds_each=120,raw_bytes_max=10000000),stop='Unexpected process/GPU/structural failure; ordinary solver rejection is retained. Independently evaluate every accepted substep-seven output; an acceptance outside the gate is a discrepancy.',limits='No native world transition, settings change, policy promotion or fresh ecological result.')
    (A/'native-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(json.dumps(plan));sys.exit(0)
assert sys.argv[1]=='run';plan=json.loads((A/'native-plan.json').read_text())
for name,meta in plan['inputs'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==meta['sha256']
out=B/'native-01';out.mkdir();records=[]
for name,capture,warm in [('original',B/'substep7-capture.bin',None),('rounded',B/'substep7-capture.bin',B/'rounded-reference-v.f32'),('chebyshev',B/'substep7-capture.bin',B/'chebyshev-v.f32'),('control',B/'control-substep2.bin',B/'control-neighbor-v.f32')]:
    cmd=[str(B/'warm-probe'),str(capture),str(T/'MetalRobo.metallib'),str(out/name)]
    if warm:cmd.append(str(warm))
    start=time.monotonic();p=subprocess.run(cmd,capture_output=True,timeout=120);(out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
    rec=dict(case=name,argv=cmd,returncode=p.returncode,seconds=time.monotonic()-start)
    if p.returncode==0:
        rec.update(json.loads(p.stdout))
        if name=='original':rec['capture_exact']=(out/name/'capture.bin').read_bytes()==capture.read_bytes()
        if rec['code']==0:
            velocity=np.fromfile(out/name/'velocity.f32',dtype='<f4').astype(float)
            if name=='control':rec['velocity_exact']=(out/name/'velocity.f32').read_bytes()==warm.read_bytes()
            else:rec.update(independent_scaled_optimality=ns['evaluate'](velocity)[-1],accepted_point_independently_inside_gate=ns['evaluate'](velocity)[-1]<=1e-10)
    records.append(rec);(A/'native-result.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n');print(json.dumps(rec),flush=True);assert p.returncode==0
