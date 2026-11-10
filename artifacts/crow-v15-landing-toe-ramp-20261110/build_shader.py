from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-landing-toe-ramp-20261110";B=R/"build/crow-v15-landing-toe-ramp-20261110";base=(R/"build/crow-v15-gait-coordinated-confirmation-20261110/push-trigger.metal").read_text()
old="""        const bool closing=state.navigation.z>=5.0f && seconds<55.0f;
        for (uint side=0u;side<2u;++side) {
            const uint ramp=(state.status.w>>(side*6u)) & 63u;
            for (uint toe=0u;toe<4u;++toe) {
                const float target=!closing ? 0.0f : toe==3u ? 0.85f : min(0.025f*float(ramp),0.85f);"""
new="""        const bool closing=state.navigation.z>=5.0f;
        for (uint side=0u;side<2u;++side) {
            const uint ramp=(state.status.w>>(side*6u)) & 63u;
            const uint rearRamp=(state.status.w>>(13u+side*6u)) & 63u;
            for (uint toe=0u;toe<4u;++toe) {
                const float target=!closing ? 0.0f : toe==3u ? min(0.020f*float(rearRamp),0.85f) : min(0.025f*float(ramp),0.85f);"""
assert base.count(old)==1;s=base.replace(old,new)
old="""            if (state.navigation.z>=5.0f && seconds<55.0f) {
                for (uint side=0u;side<2u;++side) {
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=length(force)>0.05f && true;
                    const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint next=loaded ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);
                }
            }"""
new="""            if (state.navigation.z>=5.0f) {
                for (uint side=0u;side<2u;++side) {
                    const uint rearPrevious=(state.status.w>>(13u+side*6u))&63u;
                    const float footLoad=side==0u ? state.nutrition.x : state.nutrition.y;
                    const uint rearNext=(footLoad>0.05f || rearPrevious>0u) ? min(rearPrevious+1u,43u) : 0u;
                    packed |= rearNext << (13u+side*6u);
                    const uint w=compactBase+contactGroups[3u+side].reference.y;
                    const float3 force=float3(compactContact[w],compactContact[w+1u],compactContact[w+2u]);
                    const bool loaded=length(force)>0.05f;
                    const uint previous=(state.status.w>>(side*6u))&63u;
                    const uint next=loaded ? min(previous+1u,34u) : (previous>0u ? previous-1u : 0u);
                    packed |= next << (side*6u);
                }
            }"""
assert s.count(old)==1;s=s.replace(old,new);src=B/"toe-ramp.metal";src.write_text(s);(A/"toe-ramp.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/confirmed-gait.metal",tofile="b/toe-ramp.metal")));air=B/"toe-ramp.air";lib=B/"toe-ramp.metallib";subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(hashlib.sha256(lib.read_bytes()).hexdigest())
