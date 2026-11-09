from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-toe-reactive-20261109';B=R/'build/crow-v15-toe-reactive-20261109';src0=R/'build/crow-v15-grip-search-20261109/symmetric-launch-wings-on-push.metal';base=src0.read_text()
end='''    }
}

inline float actuatorEffortEnvelope('''
act='''    }
    // Experimental bounded toe position drives; no force is injected.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        state.episode.z == 10u && program.counts0.w >= 5u && dispatch.counts.w == 30u) {
        const float seconds=float(state.episode.x)*dispatch.timing.x;
        const bool closing=state.navigation.z>=5.0f && seconds<55.0f;
        for (uint side=0u;side<2u;++side) {
            const uint ramp=(state.status.w>>(side*6u)) & 63u;
            for (uint toe=0u;toe<4u;++toe) {
                const float target=!closing ? 0.0f : toe==3u ? 0.85f : min(0.025f*float(ramp),0.85f);
                effortTrajectory[pass.controlStep*worldDispatch.effortStepStride + environment*worldDispatch.effortEnvironmentStride + 22u + side*4u + toe]=target;
            }
        }
    }
}

inline float actuatorEffortEnvelope('''
assert base.count(end)==1
old='''        state.nutrition.xy=float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);'''
new='''        state.nutrition.xy=float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);
        if (program.counts0.w>=5u && dispatch.counts.w==30u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            uint packed=0u;
            if (state.navigation.z>=5.0f && seconds<55.0f) {
                for (uint side=0u;side<2u;++side) {
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=length(force)>0.05f && SENSOR_ENABLED;
                    const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint next=loaded ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);
                }
            }
            state.status.w=packed;
        }'''
assert base.count(old)==1
for name,enabled in [('active','true'),('sensor-zero','false')]:
 s=base.replace(end,act).replace(old,new.replace('SENSOR_ENABLED',enabled));src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/symmetric-launch.metal',tofile='b/toe-'+name+'.metal')))
 air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
