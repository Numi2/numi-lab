from pathlib import Path
import subprocess,shlex,difflib
R=Path.cwd();A=R/"artifacts/crow-v15-causal-launch-20261110";B=R/"build/crow-v15-causal-launch-20261110";base=(R/"build/crow-v15-grip-search-20261109/complete-observer/MetalWorld.mm").read_text();s=base
needle="""bool privatePersistentBuffer(const std::size_t index) {
    switch (index) {""";pos=s.index(needle);end=s.index("\n}",pos);part=s[pos:end];assert part.count("    case kTaskState:\n")==1;s=s[:pos]+part.replace("    case kTaskState:\n","")+s[end:]
needle="""            const auto end = std::chrono::steady_clock::now();""";insert="""            if (std::getenv("CROW_TASK_STATE_DIAGNOSTIC") != nullptr) {
                id<MTLBuffer> stateBuffer = pending->context->buffers[kTaskState];
                auto* states = static_cast<const MRTaskStateGPU*>(stateBuffer.contents);
                if (states != nullptr) for (std::uint32_t lane = 0; lane < pending->diagnostics.layout.dispatch.environmentCount; ++lane) {
                    const auto& x = states[lane];
                    std::fprintf(stderr, "CROW_TASK_STATE %u %u %u %.9g %.9g %.9g\\n", lane, x.episode.x, x.status.w, x.navigation.z, x.nutrition.x, x.nutrition.y);
                }
            }
""";assert s.count(needle)==1;s=s.replace(needle,insert+needle);src=B/"MetalWorld.mm";src.write_text(s);(A/"observer.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/MetalWorld.mm",tofile="b/MetalWorld-task-state-diagnostic.mm")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/metal/MetalWorld.mm") in x);args=shlex.split(line);obj=B/"MetalWorld.mm.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,"-I"+str(R/"src/metal"));args.insert(1,"-I"+str(R/"artifacts/crow-clearance-observer-20261109"));subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(R/"build/crow-v15-toe-support-20261109/c_api.cpp.o");args[args.index("CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o")]=str(obj);subprocess.run(args,cwd=R/"build",check=True)
