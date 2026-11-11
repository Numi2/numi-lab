from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-v1-sustained-rear-contact-gate-20261111';B=R/'build/crow-v15-v1-sustained-rear-contact-gate-20261111';base=(R/'build/crow-v15-v1-centered-perch-approach-20261111/teacher.metal').read_text();s=base
old='if (landingPhase > 0.5f && loadedFeet >= 2u && toeRampsReady &&';new='if (landingPhase > 0.5f && loadedFeet >= 2u && toeRampsReady && ((state.status.w >> 28u) & 15u) >= 10u &&';assert s.count(old)==1;s=s.replace(old,new)
old='state.status.w=packed | launchRetractLatch | lowLoad1 | lowLoad2 | flightReleaseLatch;';new='''uint rearStable=0u;
            if (state.navigation.z>=5.0f) {
                bool bothRearAndLoaded=true;
                for (uint side=0u; side<2u; ++side) {
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 rearForce=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    bothRearAndLoaded &= compactContact[compactBase+contactGroups[side].members.w]>0.05f && length(rearForce)>0.05f;
                }
                rearStable=bothRearAndLoaded ? min(((state.status.w>>28u)&15u)+1u,15u) : 0u;
            }
            state.status.w=packed | launchRetractLatch | lowLoad1 | lowLoad2 | flightReleaseLatch | (rearStable<<28u);''';assert s.count(old)==1;s=s.replace(old,new)
(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/centered.metal',tofile='b/rear-gated.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
