from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-v1-toe-contact-memory-20261111';B=R/'build/crow-v15-v1-toe-contact-memory-20261111';base=(R/'build/crow-v15-v1-load-relieved-landing-20261111/teacher.metal').read_text();old='''const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint next=loaded ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);''';new='''const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint graceShift=16u+side*6u;
                    const uint oldGrace=(state.status.w>>graceShift)&63u;
                    const uint grace=loaded ? 15u : (oldGrace>0u ? oldGrace-1u : 0u);
                    const uint next=(loaded || grace>0u) ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= (next << (side*6u)) | (grace << graceShift);''';assert base.count(old)==1;s=base.replace(old,new);(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/wing_relief.metal',tofile='b/contact_memory.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
