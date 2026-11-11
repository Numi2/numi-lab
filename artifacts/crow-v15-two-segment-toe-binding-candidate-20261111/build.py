from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-toe-binding-candidate-20261111';B=R/'build/crow-v15-two-segment-toe-binding-candidate-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'artifacts/crow-v15-two-segment-walk-frozen-preflight-20261111/walking-wrapper.metal').read_text();s=base
old='''    // Experimental bounded toe position drives; no force is injected.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        state.episode.z == 10u && program.counts0.w >= 5u && dispatch.counts.w == 30u) {'''
assert s.count(old)==1
new='''    // Two-segment position targets, gated by accepted foot contact counters.
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
    // Retain the one-segment candidate for its original 30-velocity body.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        state.episode.z == 10u && program.counts0.w >= 5u && dispatch.counts.w == 30u) {'''
s=s.replace(old,new)
old='''        if (program.counts0.w>=5u && dispatch.counts.w==30u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            uint packed=0u;
            if (state.navigation.z>=5.0f && seconds<55.0f) {
                for (uint side=0u;side<2u;++side) {
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=length(force)>0.05f && true;
                    const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint next=loaded ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);
                }
            }'''
new='''        if (program.counts0.w>=5u && (dispatch.counts.w==30u || dispatch.counts.w==38u)) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            uint packed=0u;
            if (state.navigation.z>=5.0f && (dispatch.counts.w==38u || seconds<55.0f)) {
                for (uint side=0u;side<2u;++side) {
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=dispatch.counts.w==38u ? state.nutrition[side]>0.05f : length(force)>0.05f;
                    const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint limit=dispatch.counts.w==38u ? 42u : 34u;
                    const uint next=loaded ? min(previous+1u,limit) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);
                }
            }'''
assert s.count(old)==1;s=s.replace(old,new);(A/'candidate-wrapper.metal').write_text(s);(A/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/two-segment-walk.metal',tofile='b/contact-driven-toes.metal')));air=B/'tracking.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'candidate-wrapper.metal'),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/'tracking.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'hashes.json').write_text(json.dumps({'shader_sha256':h(ml),'wrapper_sha256':h(A/'candidate-wrapper.metal'),'frozen_controller_sha256':h(F/'CrowFlightControllerResearchV2.metal')},indent=2)+'\n');print('built',flush=True)
