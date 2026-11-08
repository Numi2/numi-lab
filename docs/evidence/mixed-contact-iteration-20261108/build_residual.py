from pathlib import Path
import json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108')
s=Path('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metal').read_text()
a='''        laneNormalResidual = max(
            laneNormalResidual,
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f)
        );'''
b=a+'''
        contact.friction.xyz = float3(normalEquation,
            contact.impulses.x > kConeEpsilon ? abs(normalEquation) : max(-normalEquation, 0.0f),
            dot(localRows[0].direction.xyz, relative));'''
assert s.count(a)==1,s.count(a);s=s.replace(a,b);(p/'residual.metal').write_text(s)
old=json.loads((p/'trace-build.json').read_text());logs=[]
for item in old:
 args=[x.replace('trace.metal','residual.metal').replace('trace.air','residual.air').replace('trace.metallib','residual.metallib') for x in item['command']]
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':args,'exit':r.returncode,'output':r.stdout});(p/'residual-build.json').write_text(json.dumps(logs,indent=2));print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
s=(p/'sweep_probe.cpp').read_text();a='''<<"],\\"target\\":"<<r.targetVelocity''';b='''<<"],\\"row_residual\\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<"],\\"target\\":"<<r.targetVelocity'''
assert s.count(a)==1;s=s.replace(a,b);(p/'residual_probe.cpp').write_text(s)
