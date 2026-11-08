import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[2]
src = Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metal")
expected = (root.parent / "mixed-normal-block-20261108/inputs.sha256").read_text().splitlines()[0].split()[0]
assert hashlib.sha256(src.read_bytes()).hexdigest() == expected
source = src.read_text()
start = source.index("    float laneNormalResidual = 0.0f;", source.index("const float groupError = simd_max(max(finalNormal, finalTangent));"))
needle = """        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;"""
pos = source.index(needle, start)
insert = """
        // Diagnostic-only capture after the last solver sweep.
        const MRWave32PreconditionerGPU dp = preconditioners[constraintBase + localConstraint];
        const float u = localRows[1].targetVelocity - dot(localRows[1].direction.xyz, relative) - localRows[1].regularization * contact.impulses.y;
        const float v = localRows[2].targetVelocity - dot(localRows[2].direction.xyz, relative) - localRows[2].regularization * contact.impulses.z;
        const float3 projected = projectFrictionCone(float3(contact.impulses.x, contact.impulses.y + dp.row1.w * u, contact.impulses.z + dp.row1.w * v), cone);
        contact.friction = float4(normalEquation, length(projected.yz - contact.impulses.yz) / dp.row1.w, dot(localRows[0].direction.xyz, relative), dp.row1.w);
        contact.response = float4(u, v, cone.effectiveFrictionU, cone.effectiveFrictionV);
        contact.targetVelocityAndPreSolveNormal = float4(dot(localRows[1].direction.xyz, relative), dot(localRows[2].direction.xyz, relative), localRows[1].regularization, localRows[2].regularization);
"""
(root / "trace.metal").write_text(source[:pos] + insert + source[pos:])
link = json.loads((root.parent / "mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
commands = [
 ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-I", str(repo / "include"), "-c", str(root / "trace.metal"), "-o", str(root / "trace.air")],
 [str(root / "trace.air") if x.endswith("/baseline.air") else str(root / "trace.metallib") if x.endswith("/baseline.metallib") else x for x in link],
]
records=[]
for cmd in commands:
    p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    records.append({"command":cmd,"returncode":p.returncode,"output":p.stdout})
    (root/"build.json").write_text(json.dumps(records,indent=2)+"\n")
    if p.returncode: raise SystemExit(p.returncode)
(root/"input-sha256.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (src,root/"trace.metal",root/"trace.metallib",Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metallib"),Path("/Users/numi/Work/mixed-full-operator-artifacts-20261108/probe"))))
