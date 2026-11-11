from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-v1-rear-first-toe-control-20261111';B=R/'build/crow-v15-v1-rear-first-toe-control-20261111';base=(R/'build/crow-v15-v1-sustained-rear-contact-gate-20261111/teacher.metal').read_text();s=base
old='min(0.025f*float(ramp),0.85f)';new='min(0.025f*float(ramp>17u ? ramp-17u : 0u),0.85f)';assert s.count(old)==1;s=s.replace(old,new)
old='const uint next=(loaded || grace>0u) ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);';new='''const uint rearWrench=compactBase+contactGroups[3u+side].reference.y;
                    const float3 rearForce=float3(compactContact[rearWrench],compactContact[rearWrench+1u],compactContact[rearWrench+2u]);
                    const bool canAdvance=previous<17u || length(rearForce)>0.05f;
                    const uint next=(loaded || grace>0u) ? (canAdvance ? min(previous+1u,51u) : previous) : (previous>0u ? previous-1u : 0u);''';assert s.count(old)==1;s=s.replace(old,new)
(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/contact-gated.metal',tofile='b/rear-first.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
