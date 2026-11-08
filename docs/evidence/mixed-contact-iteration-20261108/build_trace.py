from pathlib import Path
import json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108')
s=Path('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metal').read_text()
a='''            preconditioner.row2.w = relaxation;
            preconditioners[constraintBase + localConstraint] =
                preconditioner;'''
b='''            preconditioner.row2.w = relaxation;
            contacts[constraintBase + localConstraint].targetVelocityAndPreSolveNormal.xyz =
                float3(relaxation, diagonalNormalResponse, absoluteNormalRowSum);
            preconditioners[constraintBase + localConstraint] =
                preconditioner;'''
assert s.count(a)==1;s=s.replace(a,b);(p/'trace.metal').write_text(s)
old=json.loads(Path('/Users/numi/ccd-frictionless-oblique-20261108/build.json').read_text());logs=[]
for item in old:
 args=[]
 for x in item['command']:
  x=x.replace('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metal',str(p/'trace.metal')).replace('/Users/numi/ccd-frictionless-oblique-20261108/candidate.air',str(p/'trace.air')).replace('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metallib',str(p/'trace.metallib'))
  args.append(x)
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':args,'exit':r.returncode,'output':r.stdout});(p/'trace-build.json').write_text(json.dumps(logs,indent=2));print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
