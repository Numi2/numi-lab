from pathlib import Path
import subprocess,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-contact-release-hold-j-20261110';B=R/'build/crow-v15-contact-release-hold-j-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'build/crow-v15-rebound-contact-flare-20261110/teacher.metal').read_text();s=base
needle='''        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action < 15u)
            navigationTeacherRequested = 0.0f;'''
replacement=needle+'''
        if (fixedFlightController && state.threatTeacher.w >= 1.5f && action >= 7u && action <= 12u) {
            const uint joint=(action-7u)%3u;
            const float delta=clamp(-state.threatGeometry.z-0.05f*state.threatGeometry.w,-0.18f,0.0f);
            navigationTeacherRequested=joint==0u ? -0.10f+delta/0.70f : joint==1u ? 0.18f : -0.08f-delta/0.65f;
        }'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
old="navigationTeacherRequested=joint==0u ? (swing>=0.0f ? -0.30f*swing : -0.042f*swing + 0.536f*0.35f*max(-swing,0.0f)) : joint==1u ? 0.40f*lift - 0.35f*max(-swing,0.0f) : -0.15f*lift + 0.577f*0.35f*max(-swing,0.0f);"
new="navigationTeacherRequested=joint==0u ? (swing>=0.0f ? -0.34f*swing : -0.042f*swing + 0.536f*0.45f*max(-swing,0.0f)) : joint==1u ? 0.55f*lift - 0.45f*max(-swing,0.0f) : -0.236f*lift + 0.577f*0.45f*max(-swing,0.0f);"
assert s.count(old)==1;s=s.replace(old,new)
oldRecovery="if (landingPhase >= 1.2f && landingPhase < 1.5f && toeTimers == 0u && state.nutrition.x + state.nutrition.y < 0.25f) landingPhase=1.0f;"
newRecovery="if (landingPhase >= 1.2f && landingPhase < 1.5f && (length(yawFrameLinear.xy)>0.16f || (pos.z<perch.z+0.17f && rootLinearVelocity.z < -0.25f) || (pos.z>perch.z+0.42f && rootLinearVelocity.z>0.10f))) landingPhase=1.0f;"
assert s.count(oldRecovery)==1;s=s.replace(oldRecovery,newRecovery)
oldTakeover="const bool privilegedCommand = (dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f);"
newTakeover="const bool privilegedCommand = (dispatch.sampling.w != 0u && (dispatch.assistance.x < 0.5f || state.episode.x >= 2100u));"
assert s.count(oldTakeover)==1;s=s.replace(oldTakeover,newTakeover)
oldPlacement="const float xError = landingTarget.x+0.045f-feetX;"
newPlacement="const float xError = landingTarget.x+0.020f-feetX;"
assert s.count(oldPlacement)==1;s=s.replace(oldPlacement,newPlacement)
oldClosing="const bool closing=state.navigation.z>=5.0f;"
newClosing="const bool closing=state.navigation.z>=5.0f && state.threatTeacher.w<2.5f;"
assert s.count(oldClosing)==1;s=s.replace(oldClosing,newClosing)
oldDuration="const uint supportDuration = (state.status.w >> 25u) & 127u;"
newDuration=oldDuration
assert s.count(oldDuration)==1;s=s.replace(oldDuration,newDuration)
oldCounter='state.navigation.z>=5.0f && state.threatTeacher.w<1.5f'
newCounter='state.navigation.z>=5.0f && state.threatTeacher.w<2.5f'
assert s.count(oldCounter)==1;s=s.replace(oldCounter,newCounter)
oldPhase="if (landingPhase > 0.5f && landingPhase < 1.5f && loadedFeet >= 2u && supportDuration >= 25u &&\n                    abs(xError) < 0.06f && abs(yError) < 0.16f &&\n                    length(yawFrameLinear.xy) < 0.08f &&\n                    abs(rootLinearVelocity.z) < 0.12f) landingPhase=2.0f;"
newPhase=oldPhase+"\n                if (landingPhase < 1.5f) state.threatMetadata.x=0u;\n                else if (landingPhase < 2.5f) {\n                    if (state.threatMetadata.x==0u || state.threatMetadata.x==MR_INVALID_INDEX) state.threatMetadata.x=state.episode.x;\n                    if (state.episode.x >= state.threatMetadata.x+200u && state.nutrition.x>0.10f && state.nutrition.y>0.10f && state.nutrition.x+state.nutrition.y>2.2f) landingPhase=3.0f;\n                }"
assert s.count(oldPhase)==1;s=s.replace(oldPhase,newPhase)
(B/'teacher.metal').write_text(s);(A/'pose-hold.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/teacher.metal')));subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air','MetalRoboCollision.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air'),str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True)
