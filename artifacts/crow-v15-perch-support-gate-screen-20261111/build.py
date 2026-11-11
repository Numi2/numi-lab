from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perch-support-gate-screen-20261111';B=R/'build/crow-v15-perch-support-gate-screen-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110'
base=(R/'artifacts/crow-v15-perch-aligned-descent-dev-20261111/candidate.metal').read_text()
old="""                uint loadedFeet=0u;
                for (uint gi=0u; gi<program.counts0.w; ++gi) {
                    const MRTaskContactGroupGPU group=contactGroups[gi];
                    if ((group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u &&
                        compactContact[compactBase+group.members.w] > 0.05f) ++loadedFeet;
                }
                if (landingPhase > 0.5f && loadedFeet >= 2u &&
                    abs(xError) < 0.06f && abs(yError) < 0.16f &&
                    length(yawFrameLinear.xy) < 0.08f &&
                    abs(rootLinearVelocity.z) < 0.12f) landingPhase=2.0f;"""
new="""                uint loadedFeet=0u;
                float contactNormal=0.0f;
                for (uint gi=0u; gi<program.counts0.w; ++gi) {
                    const MRTaskContactGroupGPU group=contactGroups[gi];
                    if ((group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u) {
                        const float force=compactContact[compactBase+group.members.w];
                        if (force > 0.05f) ++loadedFeet;
                        contactNormal += max(force,0.0f);
                    }
                }
                // Development heuristic only: normal magnitude is not a
                // measurement of vertical load or opposing grasp.
                const bool stableContact=landingPhase>0.5f && loadedFeet>=2u && contactNormal>=1.0f &&
                    (state.status.w&63u)>=3u && ((state.status.w>>6u)&63u)>=3u &&
                    abs(xError)<0.06f && abs(yError)<0.16f && abs(pitch)<0.15f &&
                    abs(baseAngular.y)<0.25f && length(yawFrameLinear.xy)<0.08f &&
                    abs(rootLinearVelocity.z)<0.12f;
                const uint readyMask=63u<<23u;
                const uint readyCount=stableContact ? min(((state.status.w>>23u)&63u)+1u,15u) : 0u;
                state.status.w=(state.status.w & ~readyMask) | (readyCount<<23u);
                if (readyCount>=15u) landingPhase=2.0f;"""
assert base.count(old)==1
s=base.replace(old,new)
(A/'candidate.metal').write_text(s)
(A/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/aligned.metal',tofile='b/support_gate.metal')))
air=B/'tracking.air';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(A/'candidate.metal'),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/'tracking.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);(A/'hashes.json').write_text(json.dumps({'shader_sha256':hashlib.sha256(ml.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((A/'candidate.metal').read_bytes()).hexdigest(),'frozen_flight_source_sha256':hashlib.sha256((F/'CrowFlightControllerResearchV2.metal').read_bytes()).hexdigest()},indent=2)+'\n')
