from pathlib import Path
import json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108');s=Path('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metal').read_text()
a='''            const MRWave32PreconditionerGPU preconditioner =
                preconditioners[
                    constraintBase + localConstraint
                ];
            float3 proposed = previous + float3('''
b='''            if (iteration == 0u) {
                contact.targetVelocityAndPreSolveNormal.xyz = float3(rhs.x, previous.x, localRows[0].targetVelocity);
            } else if (iteration == 1u) {
                contact.friction.xyz = float3(rhs.x, previous.x, localRows[0].targetVelocity);
            }
            const MRWave32PreconditionerGPU preconditioner =
                preconditioners[
                    constraintBase + localConstraint
                ];
            float3 proposed = previous + float3('''
assert s.count(a)==1;s=s.replace(a,b);(p/'iteration.metal').write_text(s)
old=json.loads(Path('/Users/numi/ccd-frictionless-oblique-20261108/build.json').read_text());logs=[]
for item in old:
 args=[]
 for x in item['command']:
  for oldname,newname in [('candidate.metal','iteration.metal'),('candidate.air','iteration.air'),('candidate.metallib','iteration.metallib')]:x=x.replace('/Users/numi/ccd-frictionless-oblique-20261108/'+oldname,str(p/newname))
  args.append(x)
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':args,'exit':r.returncode,'output':r.stdout});(p/'iteration-build.json').write_text(json.dumps(logs,indent=2));print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
