from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108");out=root/"witness";out.mkdir(exist_ok=True)
base=work/"trace.metal"
assert hashlib.sha256(base.read_bytes()).hexdigest()==(root/"trace-artifacts.sha256").read_text().splitlines()[0].split()[0]
s=base.read_text();start=s.index("        const MRWave32PreconditionerGPU pre=preconditioners[constraintBase+localConstraint];",s.index("float laneNormalResidual = 0.0f;",s.index("const float groupError = simd_max(max(finalNormal, finalTangent));")))
end=s.index("        const float limitU =",start)
insert="""
        const float marker=contact.friction.w;
        if (typedRodConstraint(contact)) {
            const uint wi=constraintWitnessIndices[constraintBase+localConstraint];
            if (wi!=MR_INVALID_INDEX) {
                const MRRodToolWitnessGPU w=rodWitnesses[wi];
                const MRRodColliderGPU c=rodColliders[w.identity.z];
                contact.friction=float4(float(c.nodeA),float(c.nodeB),float(c.edgeIndex),marker);
                contact.response=float4(float(c.rodIndex),float(w.identity.z),w.rodPointAndWeight.w,c.radiusAndOffsets.x);
            }
        } else contact.friction=float4(-1.0f,-1.0f,-1.0f,marker);
"""
s=s[:start]+insert+s[end:]
shader=work/"witness.metal";air=work/"witness.air";library=work/"witness.metallib";shader.write_text(s)
link=json.loads((root.parent/"mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)],
 [str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]]
records=[]
def run(label,cmd):
 print(label,flush=True)
 p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (out/(label+".stdout")).write_text(p.stdout);(out/(label+".stderr")).write_text(p.stderr)
 records.append({"label":label,"command":[str(x) for x in cmd],"returncode":p.returncode})
 (out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise SystemExit(p.returncode)
for i,cmd in enumerate(cmds):run("build-"+str(i),cmd)
for n in (5,9,17):run("rod%d"%n,[work/("rod%d"%n),library,"2"])
(out/"artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (base,shader,air,library)))
