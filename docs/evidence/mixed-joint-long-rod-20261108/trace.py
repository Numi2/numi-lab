from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108")
source=work/"long.metal"
assert hashlib.sha256(source.read_bytes()).hexdigest()==(root/"artifacts.sha256").read_text().splitlines()[0].split()[0]
s=source.read_text()
start=s.index("// Research-only joint block.")
needle="                device MRContactConstraintGPU& c=contacts[constraintBase+ids[i]];\n"
pos=s.index(needle,start)+len(needle)
s=s[:pos]+"                if (iteration==0u) c.friction.w=0.0f;\n"+s[pos:]
needle="                    contacts[constraintBase+ids[i]].impulses.xyz=next;\n"
pos=s.index(needle,start)+len(needle)
s=s[:pos]+"                    contacts[constraintBase+ids[i]].friction.w=float(iteration+1u);\n"+s[pos:]
start=s.index("    float laneNormalResidual = 0.0f;",s.index("const float groupError = simd_max(max(finalNormal, finalTangent));"))
needle="        const float limitU =\n            cone.effectiveFrictionU * contact.impulses.x;"
pos=s.index(needle,start)
instrument="""
        const MRWave32PreconditionerGPU pre=preconditioners[constraintBase+localConstraint];
        const float u=localRows[1].targetVelocity-dot(localRows[1].direction.xyz,relative)-localRows[1].regularization*contact.impulses.y;
        const float v=localRows[2].targetVelocity-dot(localRows[2].direction.xyz,relative)-localRows[2].regularization*contact.impulses.z;
        const float3 projected=projectFrictionCone(float3(contact.impulses.x,contact.impulses.y+pre.row1.w*u,contact.impulses.z+pre.row1.w*v),cone);
        const float marker=contact.friction.w;
        contact.friction=float4(normalEquation,length(projected.yz-contact.impulses.yz)/pre.row1.w,pre.row1.w,marker);
        contact.response=float4(u,v,cone.effectiveFrictionU,cone.effectiveFrictionV);
        contact.targetVelocityAndPreSolveNormal=float4(dot(localRows[1].direction.xyz,relative),dot(localRows[2].direction.xyz,relative),localRows[1].regularization,localRows[2].regularization);
"""
s=s[:pos]+instrument+s[pos:]
shader=work/"trace.metal";air=work/"trace.air";library=work/"trace.metallib";shader.write_text(s)
link=json.loads((root.parent/"mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)],
 [str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link],
 ["/Users/numi/Work/mixed-full-operator-artifacts-20261108/probe",str(library),"2"]]
records=[]
for k,cmd in enumerate(cmds):
 p=subprocess.run(cmd,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 records.append({"command":cmd,"returncode":p.returncode,"stderr":p.stderr,"stdout":p.stdout if k<2 else "saved as trace.json"})
 (root/"trace-build.json").write_text(json.dumps(records,indent=2)+"\n")
 if k==2:(root/"trace-three.json").write_text(p.stdout)
 if p.returncode:raise SystemExit(p.returncode)
(root/"trace-artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (shader,air,library)))
