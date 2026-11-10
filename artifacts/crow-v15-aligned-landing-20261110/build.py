from pathlib import Path
import subprocess,difflib,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-aligned-landing-20261110';B=R/'build/crow-v15-aligned-landing-20261110';base=(R/'build/crow-v15-causal-route-20261110/route.metal').read_text();old='''        const float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;''';new='''        float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;
        // Require goal alignment before permitting descent on the experimental navigation task.
        if (fixedFlightController && !privilegedCommand && state.episode.z==10u &&
            state.navigation.z>=5.0f && state.threatTeacher.w<0.5f)
            oracleCommand.w=max(oracleCommand.w,1.25f);''';assert base.count(old)==1;s=base.replace(old,new)
old='''                if (phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;''';new='''                if (phase>0.5f && phase<1.5f &&
                    (length(finalGoal.xy-pos.xy)>0.10f || length(yawFrameLinear.xy)>0.16f)) phase=0.0f;
                if (phase < 0.5f && length(finalGoal.xy-pos.xy)<0.08f &&
                    length(yawFrameLinear.xy)<0.08f) phase=1.0f;''';assert s.count(old)==1;s=s.replace(old,new);src=B/'route.metal';src.write_text(s);(A/'alignment.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/causal-route.metal',tofile='b/aligned-landing.metal')));air=B/'route.air';lib=B/'route.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
