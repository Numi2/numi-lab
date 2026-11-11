from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-v1-contact-gated-grip-20261111';B=R/'build/crow-v15-v1-contact-gated-grip-20261111';base=(R/'build/crow-v15-v1-alignment-gated-descent-20261111/teacher.metal').read_text();s=base
old='''const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=length(force)>0.05f && true;''';new='''const bool loaded=compactContact[compactBase+contactGroups[side].members.w]>0.05f;''';assert s.count(old)==1;s=s.replace(old,new)
old='''if (landingPhase > 0.5f && loadedFeet >= 2u &&
                    abs(xError) < 0.06f''';new='''const bool toeRampsReady=(state.status.w & 63u)>=20u && ((state.status.w>>6u)&63u)>=20u;
                if (landingPhase > 0.5f && loadedFeet >= 2u && toeRampsReady &&
                    abs(xError) < 0.06f''';assert s.count(old)==1;s=s.replace(old,new)
(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/aligned.metal',tofile='b/foot_grip.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
