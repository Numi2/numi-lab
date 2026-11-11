from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-journey-bounded-flight-screen-20261111";B=R/"build/crow-v15-journey-bounded-flight-screen-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110";base=(R/"artifacts/crow-v15-hip-roll-walk-launch-controls-20261111/gait-080.metal").read_text()
needle="""        const float4 actorCommand = float4(
            0.10f+0.30f*actorLatent.x,0.30f*actorLatent.y,
            0.15f*actorLatent.z,0.975f+0.325f*actorLatent.w);"""
replacement="""        // Obstacle-free bounded-command development screen only.
        // Keeps the exact hierarchical journey path and frozen flight law;
        // commands replace the actor, so this is not autonomous navigation.
        const uint flightMode=environment%5u;
        const float rate=flightMode==1u ? .03f : flightMode==2u ? -.03f : 0.0f;
        const float elapsed=max(0.0f,float(state.episode.x)*dispatch.timing.x-7.2f);
        const float desiredHeading=clamp(rate*elapsed+state.commandExtension.x,-.15f,.15f);
        const float height=flightMode==3u ? 1.05f : flightMode==4u ? .65f : .85f;
        const float4 actorCommand=float4(.35f,0.0f,desiredHeading,height);"""
assert base.count(needle)==1;s=base.replace(needle,replacement);source=A/"tracking.metal";source.write_text(s);(A/"tracking.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/journey-gait-080.metal",tofile="b/bounded-flight-screen.metal")));air=B/"tracking.air";subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(source),"-o",str(air)],check=True);replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str(replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")];ml=B/"tracking.metallib";subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(air),"-o",str(ml)],check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"source_sha256":h(source),"shader_sha256":h(ml),"frozen_flight_sha256":h(F/"CrowFlightControllerResearchV2.metal")},indent=2)+"\n")
