from pathlib import Path
import json,subprocess
s=Path("candidate-2e-5-cap128.metal").read_text()
start=s.index("    float laneNormalResidual = 0.0f;",s.index("const float groupError = simd_max(max(finalNormal, finalTangent));"))
needle="""        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;"""
pos=s.index(needle,start)
instrument="""        // Research-only row diagnostics, stored after the last solve sweep.
        const MRWave32PreconditionerGPU diagnosticPre = preconditioners[constraintBase + localConstraint];
        const float rhsU = localRows[1].targetVelocity - dot(localRows[1].direction.xyz, relative) - localRows[1].regularization * contact.impulses.y;
        const float rhsV = localRows[2].targetVelocity - dot(localRows[2].direction.xyz, relative) - localRows[2].regularization * contact.impulses.z;
        const float3 projectedDiagnostic = projectFrictionCone(float3(contact.impulses.x, contact.impulses.y + diagnosticPre.row1.w * rhsU, contact.impulses.z + diagnosticPre.row1.w * rhsV), cone);
        contact.friction.xyz = float3(normalEquation, length(projectedDiagnostic.yz - contact.impulses.yz) / diagnosticPre.row1.w, dot(localRows[0].direction.xyz, relative));
"""
s=s[:pos]+instrument+s[pos:]
Path("candidate-2e-5-rowtrace.metal").write_text(s)
old=json.loads(Path("cap-build.json").read_text())[2:];out=[]
for item in old:
 args=[x.replace("candidate-2e-5-cap128", "candidate-2e-5-rowtrace") for x in item["command"]]
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out.append(dict(command=args,exit=p.returncode,output=p.stdout));Path("rowtrace-build.json").write_text(json.dumps(out,indent=2)+"\n");print(p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
p=Path("sustained-v3.mm").read_text()
needle2="std::cout<<\"{\\\"flags\\\":\"<<c.flags"
replacement="std::cout<<\"{\\\"row_trace\\\":[\"<<c.friction.x<<\",\"<<c.friction.y<<\",\"<<c.friction.z<<\"],\\\"flags\\\":\"<<c.flags"
assert p.count(needle2)==1
Path("sustained-v4.mm").write_text(p.replace(needle2,replacement))
