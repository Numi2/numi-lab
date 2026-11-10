from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-planar-sensor-20261110';B=R/'build/crow-v15-planar-sensor-20261110';base=(R/'build/crow-v15-foot-sensor-20261110/route.metal').read_text()
needle='                    packed |= uint(mm+512) << 13u;';assert base.count(needle)==1
replacement=needle+"""
                    const float footY=0.5f*(bodyStates[bodyBase+contactGroups[0].reference.x].position.y+bodyStates[bodyBase+contactGroups[1].reference.x].position.y);
                    const int y4mm=int(round(clamp(goal.y-footY,-1.0f,1.0f)*250.0f));
                    packed |= uint(y4mm+256) << 23u;"""
s=base.replace(needle,replacement);(B/'route.metal').write_text(s);(A/'sensor-y.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/x-sensor.metal',tofile='b/xy-sensor.metal')))
air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'route.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
