from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-foot-guidance-20261110';B=R/'build/crow-v15-foot-guidance-20261110';base=(R/'build/crow-v15-foot-sensor-20261110/route.metal').read_text()
old='        const float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;';assert base.count(old)==1
new="""        float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;
        if (fixedFlightController && !privilegedCommand && state.episode.z==10u && dispatch.counts.w==30u && state.navigation.z>=5.0f && state.threatTeacher.w<1.5f) {
            // Delayed, quantized simulation-state feedback; experiment only.
            const float footError=float(int((state.status.w>>13u)&1023u)-512)*0.001f;
            oracleCommand.x=clamp(1.2f*footError,-0.18f,0.18f);
            if(state.threatTeacher.w<0.5f) oracleCommand.w=1.20f;
        }"""
s=base.replace(old,new)
old2='                if (phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&\n                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;';assert s.count(old2)==1
new2="""                const float footX=0.5f*(bodyStates[bodyBase+contactGroups[0].reference.x].position.x+bodyStates[bodyBase+contactGroups[1].reference.x].position.x);
                if (phase < 0.5f && abs(finalGoal.x-footX)<0.025f && abs(finalGoal.y-pos.y)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;"""
s=s.replace(old2,new2);(B/'route.metal').write_text(s);(A/'guidance.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/sensor.metal',tofile='b/guidance.metal')))
air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'route.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
