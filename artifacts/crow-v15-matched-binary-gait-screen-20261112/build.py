from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-matched-binary-gait-screen-20261112";B=R/"build/crow-v15-matched-binary-gait-screen-20261112";E=R/"artifacts/crow-v15-walk-exact-bypass-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110";H=R/"build/crow-v15-substep-target-interpolation-screen-20261112";J=R/"build/crow-v15-hip-roll-journey-integration-20261111"
old=(E/"task5.metal").read_text();needle="if (state.episode.x == 0u || float(state.episode.x)*dispatch.timing.x < 4.0f || float(state.episode.x)*dispatch.timing.x >= 6.0f) {";assert old.count(needle)==1
new=old.replace(needle,"if (program.interactionTiming.w < 0.5f || state.episode.x == 0u || float(state.episode.x)*dispatch.timing.x < 4.0f || float(state.episode.x)*dispatch.timing.x >= 6.0f) {");(A/"task.metal").write_text(new)
assert new.count("interactionTiming.w")==1
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"task.metal"),"-o",str(B/"task.air")],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str((R/"build/crow-v15-walk-exact-bypass-20261112/World.air") if p.name=="MetalRoboWorld.air" else replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"task.air"),"-o",str(B/"matched.metallib")],check=True)
orig=(R/"src/core/TaskProgram.cpp").read_text();needle="    const float resetProbability =\n";assert orig.count(needle)==1
src="#include <cstdlib>\n"+orig.replace(needle,"    staged->header.interactionTiming.w = std::getenv(\"CROW_EXPERIMENT_GAIT_SMOOTH\") != nullptr ? 1.0f : 0.0f;\n"+needle);(B/"TaskProgram.cpp").write_text(src)
(A/"host.patch").write_text("".join(difflib.unified_diff(orig.splitlines(True),src.splitlines(True),fromfile="a/TaskProgram.cpp",tofile="b/TaskProgram.cpp")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/core/TaskProgram.cpp") in x);args=shlex.split(line);obj=B/"TaskProgram.cpp.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"TaskProgram.cpp");subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(J/"c_api.cpp.o");args[args.index("CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o")]=str(obj)
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 key="CMakeFiles/metalrobo.dir/"+n+".o";assert key in args;args[args.index(key)]=str(H/"MetalWorld.mm.o" if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"host":h(B/"libmetalrobo.dylib"),"metallib":h(B/"matched.metallib"),"world_air":h(R/"build/crow-v15-walk-exact-bypass-20261112/World.air"),"task_source":h(A/"task.metal")},indent=2)+"\n")
