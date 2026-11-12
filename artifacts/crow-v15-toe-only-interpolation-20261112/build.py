from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-toe-only-interpolation-20261112";B=R/"build/crow-v15-toe-only-interpolation-20261112";S=R/"artifacts/crow-v15-matched-binary-gait-screen-20261112";M=R/"build/crow-v15-matched-binary-gait-screen-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"artifacts/crow-v15-tiny-swing-toe-screen-20261112/task.metal").read_text()
needle="if (program.interactionTiming.w>=0.5f && walkingSeconds>=4.0f"
assert old.count(needle)==1
s=old.replace(needle,"if (walkingSeconds>=4.0f")
needle="""    if (true) {
        for (uint i=0u; i<dispatch.counts.w; ++i) {
            const uint e=environment*worldDispatch.effortEnvironmentStride+i;
            effortTrajectory[worldDispatch.effortStepStride+e]=effortTrajectory[e];
        }
    }
}"""
replacement="""    for (uint i=0u; i<dispatch.counts.w; ++i) {
        const bool rearToe = i==28u || i==29u || i==36u || i==37u;
        const float seconds=float(state.episode.x)*dispatch.timing.x;
        const bool smooth=program.interactionTiming.w>=0.5f && rearToe && seconds>=4.0f && seconds<6.0f && state.episode.x>0u;
        if (!smooth) {
            const uint e=environment*worldDispatch.effortEnvironmentStride+i;
            effortTrajectory[worldDispatch.effortStepStride+e]=effortTrajectory[e];
        }
    }
}"""
assert s.count(needle)==1
s=s.replace(needle,replacement)
(A/"task.metal").write_text(s);(A/"toe-interpolation.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/tiny-toe.metal",tofile="b/toe-interpolation.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"World.metal"),"-o",str(B/"World.air")],check=True)
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"task.metal"),"-o",str(B/"task.air")],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str((B/"World.air") if p.name=="MetalRoboWorld.air" else replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"task.air"),"-o",str(B/"support-aware.metallib")],check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"host":h(M/"libmetalrobo.dylib"),"metallib":h(B/"support-aware.metallib"),"task_source":h(A/"task.metal")},indent=2)+"\n")
