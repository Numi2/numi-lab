from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-approach-reference-fix-20261110';B=R/'build/crow-v15-approach-reference-fix-20261110';base=(R/'build/crow-v15-foot-handoff-20261110/handoff.metal').read_text();s=base
old='''state.navigation.z>=5.0f && state.threatTeacher.w<1.5f) {
            // Bounded approach speed from accepted foot-to-goal error.
            actorCommand.x=clamp(0.10f+0.65f*state.nutrition.z-0.60f*state.commandExtension.z,0.00f,0.20f);'''
new='''state.navigation.z>=4.0f && state.threatTeacher.w<1.5f && state.nutrition.z<0.40f) {
            // Slow toward the physical foot joint using accepted state.
            actorCommand.x=clamp(0.10f+0.80f*state.nutrition.z-0.80f*state.commandExtension.z,-0.05f,0.20f);'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        if (state.navigation.z >= float(kCrowNavigationWaypointCount)) {
            uint course=MR_INVALID_INDEX;''';new='''        if (state.navigation.z >= float(kCrowNavigationWaypointCount-1u)) {
            uint course=MR_INVALID_INDEX;'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                        const float feetX=0.5f*(bodyStates[bodyBase+lf].position.x+bodyStates[bodyBase+rf].position.x);
                        state.nutrition.z=clamp(finalGoal.x+0.015f-feetX,-0.25f,0.25f);'''
new='''                        const MRBodyStateGPU left=bodyStates[bodyBase+lf],right=bodyStates[bodyBase+rf];
                        const float3 leftJoint=left.position.xyz+rotate(left.orientation,contactGroups[0].kinematicReference.xyz);
                        const float3 rightJoint=right.position.xyz+rotate(right.orientation,contactGroups[1].kinematicReference.xyz);
                        const float feetX=0.5f*(leftJoint.x+rightJoint.x);
                        state.nutrition.z=clamp(finalGoal.x+0.015f-feetX,-0.50f,0.50f);'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                if (phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&''';new='''                if (state.navigation.z >= float(kCrowNavigationWaypointCount) && phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                if (loadedFeet>=2u && abs(state.nutrition.z)<0.045f &&''';new='''                if (state.navigation.z >= float(kCrowNavigationWaypointCount) && loadedFeet>=2u && abs(state.nutrition.z)<0.045f &&'''
assert s.count(old)==1;s=s.replace(old,new)
name='referencefix';src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/handoff.metal',tofile='b/referencefix.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
