from pathlib import Path
import json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108');s=Path('/Users/numi/ccd-frictionless-oblique-20261108/candidate.metal').read_text()
a='''        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        const float3 warm = projectFrictionCone('''
b='''        // Isolated research candidate: scalar normal and isotropic tangent steps.
        const MREvaluatedConstraintIRConeGPU splitCone = evaluatedCones[constraintBase + localConstraint];
        if (splitCone.effectiveFrictionU == splitCone.effectiveFrictionV &&
            splitCone.effectiveFrictionU >= 0.0f && splitCone.maximumNormalImpulse <= 0.0f) {
            float splitScale = 0.0f;
            for (uint splitRow = 0u; splitRow < 3u; ++splitRow)
                for (uint splitCol = 0u; splitCol < 3u; ++splitCol)
                    splitScale = max(splitScale, abs(effective[splitRow][splitCol]));
            const float splitFloor = kContactMatrixRegularization * splitScale;
            const float splitNormal = effective[0][0] + splitFloor;
            const float splitTangent = max(abs(effective[1][1]) + abs(effective[1][2]),
                                            abs(effective[2][1]) + abs(effective[2][2])) + splitFloor;
            if (splitNormal > kMatrixFloor && splitTangent > kMatrixFloor &&
                isfinite(splitNormal) && isfinite(splitTangent)) {
                preconditioner.row0.w = 1.0f / splitNormal;
                preconditioner.row1.w = 1.0f / splitTangent;
            }
        }
        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        const float3 warm = projectFrictionCone('''
assert s.count(a)==1,s.count(a);s=s.replace(a,b)
a='''            float3 proposed = previous + float3(
                dot(preconditioner.row0.xyz, rhs),
                dot(preconditioner.row1.xyz, rhs),
                dot(preconditioner.row2.xyz, rhs)
            );'''
b='''            float3 proposed;
            if (preconditioner.row0.w > 0.0f && preconditioner.row1.w > 0.0f) {
                proposed = previous + float3(preconditioner.row0.w * rhs.x,
                                             preconditioner.row1.w * rhs.y,
                                             preconditioner.row1.w * rhs.z);
            } else {
                proposed = previous + float3(
                    dot(preconditioner.row0.xyz, rhs),
                    dot(preconditioner.row1.xyz, rhs),
                    dot(preconditioner.row2.xyz, rhs));
            }'''
assert s.count(a)==1,s.count(a);s=s.replace(a,b);(p/'split.metal').write_text(s)
old=json.loads((p/'trace-build.json').read_text());logs=[]
for item in old:
 args=[x.replace('trace.metal','split.metal').replace('trace.air','split.air').replace('trace.metallib','split.metallib') for x in item['command']]
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':args,'exit':r.returncode,'output':r.stdout});(p/'split-build.json').write_text(json.dumps(logs,indent=2));print(r.returncode,r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
