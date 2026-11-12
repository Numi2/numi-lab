from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-small-swing-toe-screen-20261112";B=R/"build/crow-v15-small-swing-toe-screen-20261112";S=R/"artifacts/crow-v15-matched-binary-gait-screen-20261112";M=R/"build/crow-v15-matched-binary-gait-screen-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(S/"task.metal").read_text()
needle="const float proximal=max(perchTarget,flightTarget);"
replacement="""float proximal=max(perchTarget,flightTarget);
                const float walkingSeconds=float(state.episode.x)*dispatch.timing.x;
                const float carrier=sin(kTwoPi*(walkingSeconds+5.0f/1000.0f)/0.50f)*(side==0u ? 1.0f : -1.0f);
                const bool oppositeSupported=(state.status.w & (side==0u ? (1u<<21) : (1u<<20)))!=0u;
                if (program.interactionTiming.w>=0.5f && walkingSeconds>=4.0f && walkingSeconds<6.0f && toe==3u && carrier>0.0f && oppositeSupported && (state.status.w & (side==0u ? (1u<<20) : (1u<<21)))==0u)
                    proximal=min(proximal,-.05f*smoothstep(0.0f,.30f,carrier));"""
assert old.count(needle)==1;s=old.replace(needle,replacement)
needle2="if (program.interactionTiming.w < 0.5f || state.episode.x == 0u || float(state.episode.x)*dispatch.timing.x < 4.0f || float(state.episode.x)*dispatch.timing.x >= 6.0f) {"
assert s.count(needle2)==1;s=s.replace(needle2,"if (true) {")
(A/"task.metal").write_text(s);(A/"toe-swing.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/gait.metal",tofile="b/toe-swing.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"task.metal"),"-o",str(B/"task.air")],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str((R/"build/crow-v15-walk-exact-bypass-20261112/World.air") if p.name=="MetalRoboWorld.air" else replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"task.air"),"-o",str(B/"support-aware.metallib")],check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"host":h(M/"libmetalrobo.dylib"),"metallib":h(B/"support-aware.metallib"),"task_source":h(A/"task.metal")},indent=2)+"\n")
