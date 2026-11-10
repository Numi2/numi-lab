from pathlib import Path
import difflib,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-wing-aero-alignment-20261110';B=R/'build/crow-v15-wing-aero-alignment-20261110'
old=(R/'src/metal/FlappingWingDynamics.metal').read_text();s=old
needle='''        const float3 chord = rotate(wingOrientation, float3(1.0f, 0.0f, 0.0f));'''
replacement='''        // Research-only v15 wing: sweep and flap share the shoulder; the
        // child anchor preserves its authored pose at neutral flap 0.35 rad.
        // This specialized formula is not valid for arbitrary robot models.
        if (!hasSweep || !hasPronation) continue;
        const float4 neutralInverse = axisAngle(hingeAxis, -0.35f);
        const float3 shoulder = rotate(rootOrientation, wing.rootJointParentAnchor.xyz);
        const float3 flapChild = rotate(neutralInverse, wing.rootJointChildAnchor.xyz);
        const float3 wingOffset = rotate(rootOrientation, rotate(sweepRotation, rotate(hingeRotation, -flapChild)));
        const float3 rootToWingOrigin = shoulder + wingOffset;
        const float3 rootToWingCOM = rootToWingOrigin + rotate(wingOrientation, wing.bodyCenterOfMass.xyz);
        const float3 chord = rotate(wingOrientation, float3(1.0f, 0.0f, 0.0f));'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
start=s.index('            const float3 neutralPoint = float3(');end=s.index('            const float3 incomingAir =',start)
s=s[:start]+'''            // Neutral stations start at the shoulder and extend to twice
            // the authored center-to-shoulder span. Re-express the authored
            // neutral geometry in the actual distal-wing frame.
            const float3 neutralPoint = float3(
                wing.rootToCenterAndArea.x,
                wing.rootJointParentAnchor.y + 2.0f * eta *
                    (wing.rootToCenterAndArea.y - wing.rootJointParentAnchor.y),
                wing.rootToCenterAndArea.z
            );
            const float3 neutralCOM = wing.rootJointParentAnchor.xyz -
                wing.rootJointChildAnchor.xyz;
            const float3 localPoint = rotate(neutralInverse, neutralPoint - neutralCOM);
            const float3 fromWingOrigin = rotate(wingOrientation, localPoint);
            const float3 rootToPoint = rootToWingOrigin + fromWingOrigin;
            const float3 fromShoulder = rootToPoint - shoulder;
            const float3 sweepStrokeVelocity = cross(
                sweepAxisWorld * sweepAngularRate, fromShoulder);
            const float3 flapStrokeVelocity = cross(
                hingeAxisWorld * flapAngularRate, fromShoulder);
            const float3 strokeVelocity = flapStrokeVelocity + sweepStrokeVelocity +
                cross(pronationAxisWorld * pronationAngularRate, fromWingOrigin);
            const float3 pointVelocity = rootVelocity +
                cross(rootAngularVelocity, rootToPoint) + strokeVelocity;
''' +s[end:]
start=s.index('        const float4 rootJointRotation = hasSweep');end=s.index('        const float3 wingTorque =',start)
s=s[:start]+s[end:]
(B/'FlappingWingDynamics.metal').write_text(s);(A/'aero.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/FlappingWingDynamics.metal',tofile='b/research-v15-aero.metal')))
cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-c',str(B/'FlappingWingDynamics.metal'),'-o',str(B/'aero.air')];subprocess.run(cmd,check=True)
airs=[str(x) for x in (R/'build/shaders').glob('MetalRobo*.air') if x.name not in ('MetalRoboFlappingWingDynamics.air','MetalRoboLocomotionTask.air')];airs.extend([str(B/'aero.air'),str(R/'build/crow-v15-toe-flight-screen-20261110/tracking.air')]);subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,'-o',str(B/'tracking.metallib')],check=True)
