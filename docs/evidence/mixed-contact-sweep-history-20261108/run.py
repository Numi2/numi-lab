import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[2]
work = Path("/Users/numi/Work/mixed-contact-sweep-history-artifacts-20261108")
work.mkdir(parents=True, exist_ok=True)
source = Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metal")
control = Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metallib")
probe = Path("/Users/numi/Work/mixed-full-operator-artifacts-20261108/probe")
expected = (root.parent / "mixed-normal-block-20261108/inputs.sha256").read_text().splitlines()
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected[0].split()[0]
assert hashlib.sha256(control.read_bytes()).hexdigest() == expected[1].split()[0]
link = json.loads((root.parent / "mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
airs = [Path(x) for x in link if x.endswith(".air") and not x.endswith("/baseline.air")]
prior_hashes = (root.parent / "mixed-full-operator-20261108/air-inputs.sha256").read_text()
current_hashes = "".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in airs)
assert current_hashes == prior_hashes, "linked AIR inputs changed"
(root/"inputs.sha256").write_text(current_hashes + "".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (source,control,probe)))
text = source.read_text()
needle = "        const float groupError = simd_max(max(finalNormal, finalTangent));\n"
assert text.count(needle) == 1
capture = """
        if (rodWork && completedIterations == CAPTURE_SWEEP) {
            for (uint lt = 0u; lt < work.tileCount; ++lt) {
                const MRContactTileGPU tile = tiles[tileBase + work.firstTile + lt];
                if (localLane >= tile.constraintCount) continue;
                const uint id = tileConstraintIndices[constraintBase + tile.partialOffset + localLane];
                device MRContactConstraintGPU& c = contacts[constraintBase + id];
                device const MREvaluatedConstraintIRRowGPU* r = evaluatedRows + rowBase + 3u * id;
                const MRWave32PreconditionerGPU p = preconditioners[constraintBase + id];
                const float3 v = typedRelativePointVelocity(id, c, bodies, dispatch.articulationIndex,
                    pointJacobians, pointJacobianBase, dispatch.nv, articulationVelocity, rodNodes,
                    rodEdges, rodColliders, rodWitnesses, constraintWitnessIndices, constraintBase);
                const float3 j = c.impulses.xyz;
                const float3 rv = float3(dot(r[0].direction.xyz,v),dot(r[1].direction.xyz,v),dot(r[2].direction.xyz,v));
                const float3 rhs = float3(r[0].targetVelocity-rv.x-r[0].regularization*j.x,
                    r[1].targetVelocity-rv.y-r[1].regularization*j.y,
                    r[2].targetVelocity-rv.z-r[2].regularization*j.z);
                c.friction = float4(j, float(completedIterations));
                c.response = float4(rhs, p.row1.w);
                c.targetVelocityAndPreSolveNormal = float4(rv, p.row2.w);
            }
        }
"""
def run(name, cmd, output):
    print(name, flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (root/(name+".stderr")).write_text(p.stderr)
    if output: (root/(name+".json")).write_text(p.stdout)
    else: (root/(name+".log")).write_text(p.stdout)
    if p.returncode: raise RuntimeError(name+" exit "+str(p.returncode))
    return p
run("control", [probe,control,"2"], True)
for sweep in (1,32,96,120,124,125,126,127,128):
    name="sweep-%03d"%sweep
    shader=work/(name+".metal"); air=work/(name+".air"); library=work/(name+".metallib")
    shader.write_text(text.replace(needle,needle+capture.replace("CAPTURE_SWEEP",str(sweep)+"u")))
    run(name+"-compile",["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",repo/"include","-c",shader,"-o",air],False)
    command=[str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]
    assert str(air) in command and str(library) in command
    run(name+"-link",command,False)
    with (root/"artifacts.sha256").open("a") as f:
        for p in (shader,air,library): f.write(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n")
    run(name,[probe,library,"2"],True)
