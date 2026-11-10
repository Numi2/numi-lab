from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-alternating-foot-release-20261110';B=R/'build/crow-v15-alternating-foot-release-20261110';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'build/crow-v15-student-landing-controller-20261110/teacher.metal').read_text();needle='''        const float navigationTeacherBlend = fixedFlightController
            ? 1.0f''';insert='''        // After the contact-conditioned release, lift one foot only while
        // the opposite foot is loaded. The normal actuator drive applies it.
        if (fixedFlightController && state.episode.z==10u && state.navigation.z>=5.0f && state.threatTeacher.w>=2.5f && action>=7u && action<=12u && state.threatMetadata.x!=MR_INVALID_INDEX) {
            const uint leg=(action-7u)/3u, joint=(action-7u)%3u;
            const float elapsed=(float(state.episode.x)-float(state.threatMetadata.x+200u))*dispatch.timing.x;
            const float u=leg==0u ? (elapsed-0.20f)/1.00f : (elapsed-1.40f)/1.00f;
            const float otherLoad=leg==0u ? state.nutrition.y : state.nutrition.x;
            const float lift=(u>0.0f && u<1.0f && otherLoad>0.20f) ? sin(M_PI_F*u) : 0.0f;
            const float delta=joint==0u ? -0.34f*lift : joint==1u ? 0.55f*lift : -0.236f*lift;
            navigationTeacherRequested=clamp(navigationTeacherRequested+delta,-1.0f,1.0f);
        }
        const float navigationTeacherBlend = fixedFlightController
            ? 1.0f''';assert base.count(needle)==1;s=base.replace(needle,insert);(B/'teacher.metal').write_text(s);(A/'release.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/teacher.metal',tofile='b/foot-release.metal')))
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(B/'teacher.air')],check=True);other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air','MetalRoboCollision.air'));subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(x) for x in other],str(R/'build/crow-v15-hybrid-ccd-failure-20261110/collision-sat.air'),str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(B/'teacher.air'),'-o',str(B/'teacher.metallib')],check=True);print(hashlib.sha256((B/'teacher.metallib').read_bytes()).hexdigest())
