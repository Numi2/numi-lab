from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-planar-guidance-20261110';B=R/'build/crow-v15-planar-guidance-20261110';base=(R/'build/crow-v15-foot-speed-mapping-20261110/route.metal').read_text()
old='                    packed |= uint(mm+512) << 13u;';assert base.count(old)==1
new=old+'''
                    const float footY=0.5f*(bodyStates[bodyBase+contactGroups[0].reference.x].position.y+bodyStates[bodyBase+contactGroups[1].reference.x].position.y);
                    const int y4mm=int(round(clamp(goal.y-footY,-1.0f,1.0f)*250.0f));
                    packed |= uint(y4mm+256) << 23u;'''
s=base.replace(old,new)
old='            oracleCommand.x=0.10f+clamp(0.45f*footError-0.7f*state.commandExtension.z,-0.12f,0.12f);';assert s.count(old)==1
new='''            const float footErrorY=float(int((state.status.w>>23u)&511u)-256)*0.004f;
            const float heading=-state.commandExtension.x;
            const float c=cos(heading), sy=sin(heading);
            const float forwardError=c*footError+sy*footErrorY;
            const float lateralError=-sy*footError+c*footErrorY;
            oracleCommand.x=0.10f+clamp(0.45f*forwardError-0.7f*state.commandExtension.z,-0.12f,0.12f);
            oracleCommand.y=clamp(0.45f*lateralError-0.7f*state.threatTeacher.x,-0.12f,0.12f);'''
s=s.replace(old,new)
old='                if (phase < 0.5f && abs(finalGoal.x-footX)<0.025f && abs(finalGoal.y-pos.y)<0.08f &&';assert s.count(old)==1
new='''                const float footY=0.5f*(bodyStates[bodyBase+contactGroups[0].reference.x].position.y+bodyStates[bodyBase+contactGroups[1].reference.x].position.y);
                if (phase < 0.5f && abs(finalGoal.x-footX)<0.025f && abs(finalGoal.y-footY)<0.08f &&'''
s=s.replace(old,new);assert s.count('45.0f')==2;s=s.replace('45.0f','55.0f');(B/'route.metal').write_text(s);(A/'planar-guidance.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/speed-mapping.metal',tofile='b/planar-guidance.metal')))
air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'route.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
for filename in ('run.py','analyze.py'):
 s=(R/'artifacts/crow-v15-foot-speed-mapping-20261110'/filename).read_text().replace('crow-v15-foot-speed-mapping-20261110','crow-v15-planar-guidance-20261110')
 if filename=='analyze.py':s=s.replace('t<2250','t<2750').replace('t>=2250','t>=2750')
 (A/filename).write_text(s)
