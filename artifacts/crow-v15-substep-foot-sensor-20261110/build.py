from pathlib import Path
import difflib,subprocess,shlex,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-substep-foot-sensor-20261110";B=R/"build/crow-v15-substep-foot-sensor-20261110";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
old=(R/"build/crow-v15-guide-midheight-20261110/teacher.metal").read_text();s=old
a="""    MRTaskStateGPU state = taskStates[environment];
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||"""
b="""    MRTaskStateGPU state = taskStates[environment];
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
        state.episode.z == 10u && state.navigation.z >= 5.0f && program.counts0.w >= 2u && dispatch.counts.w == 30u) {
        float2 accumulated = pass.physicsSubstep == 0u ? float2(0.0f) : as_type<float2>(state.threatMetadata.zw);
        const MRMetalWorldContactStatusGPU cs = contactStatuses[environment];
        if (cs.code == MR_STEP_SUCCESS) {
            device const MRTaskContactGroupGPU* groups = taskTable<MRTaskContactGroupGPU>(arena, program.offsets0.w);
            device const uint* members = taskTable<uint>(arena, program.offsets1.x);
            const uint base=environment*contactDispatch.constraintStride;
            for (uint i=0u;i<min(cs.activeContacts,contactDispatch.constraintCapacity);++i) {
                const MRContactConstraintGPU c=contacts[base+i];
                const bool ra=c.bodyA>=program.articulation.x && c.bodyA<program.articulation.x+program.articulation.y;
                const bool rb=c.bodyB>=program.articulation.x && c.bodyB<program.articulation.x+program.articulation.y;
                if (ra==rb) continue;
                const uint body=ra ? c.bodyA : c.bodyB;
                if (bodyMember(body,groups[0],members)) accumulated.x += abs(c.impulses.x);
                if (bodyMember(body,groups[1],members)) accumulated.y += abs(c.impulses.x);
            }
        }
        state.threatMetadata.zw=as_type<uint2>(accumulated);
        taskStates[environment]=state;
        return;
    }
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||"""
assert s.count(a)==1;s=s.replace(a,b);(B/"teacher.metal").write_text(s);(A/"sensor.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/teacher.metal",tofile="b/teacher.metal")))
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/"teacher.metal"),"-o",str(B/"teacher.air")],check=True)
other=sorted(p for p in (R/"build/shaders").glob("MetalRobo*.air") if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air"))
subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(x) for x in other],str(R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),str(B/"teacher.air"),"-o",str(B/"teacher.metallib")],check=True)
old=(R/"build/crow-v15-substep-contact-observer-20261110/MetalWorld.mm").read_text();s=old
a="""            const bool taskTracksImpactContacts =
                nativeTask &&
                !config.taskProgram.impactEvents().empty();""";b="""            const bool taskTracksImpactContacts =
                nativeTask &&
                (!config.taskProgram.impactEvents().empty() || std::getenv("CROW_SUBSTEP_FOOT_SENSOR") != nullptr);""";assert s.count(a)==1;s=s.replace(a,b)
a='std::fprintf(stderr, "CROW_TASK_STATE %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g\\n", lane, x.episode.x, x.status.w, x.navigation.z, x.nutrition.x, x.nutrition.y, x.threatTeacher.w, x.nutrition.w, x.commandExtension.z);';b='std::fprintf(stderr, "CROW_TASK_STATE %u %u %u %.9g %.9g %.9g %.9g %.9g %.9g %u %u\\n", lane, x.episode.x, x.status.w, x.navigation.z, x.nutrition.x, x.nutrition.y, x.threatTeacher.w, x.nutrition.w, x.commandExtension.z, x.threatMetadata.z, x.threatMetadata.w);';assert s.count(a)==1;s=s.replace(a,b);src=B/"MetalWorld.mm";src.write_text(s);(A/"host.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile="a/MetalWorld.mm",tofile="b/MetalWorld.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x);args=shlex.split(line);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,"-I"+str(R/"src/metal"));args.insert(1,"-I"+str(R/"artifacts/crow-clearance-observer-20261109"));subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(R/"build/crow-v15-heavy-wing-route-20261110/c_api.cpp.o");args[args.index("CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o")]=str(obj);subprocess.run(args,cwd=R/"build",check=True);print(hashlib.sha256((B/"libmetalrobo.dylib").read_bytes()).hexdigest())
