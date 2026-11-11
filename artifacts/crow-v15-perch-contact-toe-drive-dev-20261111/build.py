from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perch-contact-toe-drive-dev-20261111';B=R/'build/crow-v15-perch-contact-toe-drive-dev-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'artifacts/crow-v15-landing-descent-dev-20261111/candidate.metal').read_text();s=base
old='    // Experimental bounded toe position drives; no force is injected.'
new="""    // Development-only physical two-segment drives. Counter input below is
    // restricted to accepted toe/foot contact with the course perch.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        state.episode.z == 10u && program.counts0.w >= 5u && dispatch.counts.w == 38u) {
        for (uint side=0u;side<2u;++side) {
            const uint ramp=(state.status.w>>(side*6u)) & 63u;
            const uint frontRamp=ramp>2u ? ramp-2u : 0u;
            for (uint toe=0u;toe<4u;++toe) {
                const float proximal=toe==3u ? min(.04f*float(ramp),1.15f) : min(.03f*float(frontRamp),1.05f);
                for(uint segment=0u;segment<2u;++segment) {
                    const uint dof=22u+side*8u+toe*2u+segment;
                    effortTrajectory[pass.controlStep*worldDispatch.effortStepStride + environment*worldDispatch.effortEnvironmentStride + dof]=proximal*(segment==0u ? 1.0f : .75f);
                }
            }
        }
    }
"""+old
assert s.count(old)==1;s=s.replace(old,new)
old='        if (program.counts0.w>=5u && dispatch.counts.w==38u) {\n            const float seconds=float(state.episode.x)*dispatch.timing.x;\n            const uint previous=state.status.w;'
new="""        if (program.counts0.w>=5u && dispatch.counts.w==38u) {
            uint perch=MR_INVALID_INDEX;
            for (uint ri=0u;ri<program.counts1.w;++ri) {
                const MRTaskRewardOperatorGPU operation=rewards[ri];
                if (operation.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS || operation.source.x==MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) { perch=operation.source.z+4u; break; }
            }
            bool loaded[2]={false,false};
            if (state.navigation.z>=5.0f && perch!=MR_INVALID_INDEX) {
                for (uint ci=0u;ci<activeContacts;++ci) {
                    const MRContactConstraintGPU cc=contacts[contactBase+ci];
                    if (cc.impulses.x<=0.0f || (cc.bodyA!=perch && cc.bodyB!=perch)) continue;
                    const uint b=cc.bodyA==perch ? cc.bodyB : cc.bodyA;
                    if (b==robotFirst+10u || (b>=robotFirst+17u && b<=robotFirst+24u)) loaded[0]=true;
                    if (b==robotFirst+13u || (b>=robotFirst+25u && b<=robotFirst+32u)) loaded[1]=true;
                }
            }
            uint packed=0u;
            for(uint side=0u;side<2u;++side) {
                const uint oldRamp=(state.status.w>>(side*6u)) & 63u;
                const uint ramp=loaded[side] ? min(oldRamp+1u,34u) : (oldRamp>0u ? oldRamp-1u : 0u);
                packed |= ramp << (side*6u);
            }
            state.status.w=(state.status.w & ~4095u) | packed;
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            const uint previous=state.status.w;"""
assert s.count(old)==1;s=s.replace(old,new)
(A/'candidate.metal').write_text(s);(A/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/separated.metal',tofile='b/height.metal')));air=B/'tracking.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'candidate.metal'),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/'tracking.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);(A/'hashes.json').write_text(json.dumps({'shader_sha256':hashlib.sha256(ml.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((A/'candidate.metal').read_bytes()).hexdigest(),'frozen_flight_source_sha256':hashlib.sha256((F/'CrowFlightControllerResearchV2.metal').read_bytes()).hexdigest()},indent=2)+'\n')
